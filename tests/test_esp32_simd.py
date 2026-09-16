"""Verify the S3 adapter contract on host; Xtensa execution needs a board.

The independent stub checks ABI units/alignment and deliberately reads the
assembly pipeline's guard region. It does not emulate the S3 instructions.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from esp32_denoiser.export import IntegerDenoiser, export_model
from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig
from esp32_denoiser.quantization import configure_qat


WRAPPER = r'''
#include "denoiser.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static unsigned aligned_calls, unaligned_calls, invalid_calls;
static volatile int8_t touched;
int32_t esp_nn_dot_s8_aligned_esp32s3(const int8_t *a, const int8_t *b, int bytes) {
    int32_t sum = 0;
    if (((uintptr_t)a & 15) || ((uintptr_t)b & 15) || bytes < 16 || (bytes & 15)) {
        ++invalid_calls; return 0;
    }
    ++aligned_calls;
    for (int i = 0; i < bytes; ++i) sum += (int32_t)a[i] * b[i];
    return sum;
}
int32_t esp_nn_dot_s8_unaligned_esp32s3(const int8_t *a, const int8_t *b, int blocks) {
    int32_t sum = 0;
    if (((uintptr_t)a & 15) || blocks < 1) { ++invalid_calls; return 0; }
    ++unaligned_calls;
    const int8_t *base = (const int8_t *)((uintptr_t)b & ~(uintptr_t)15);
    /* Read every vector the official pipelined load schedule can touch. */
    int vectors = (blocks & 1) ? blocks + 1 : blocks + 2;
    for (int i = 0; i < vectors * 16; ++i) touched = base[i];
    int input_vectors = blocks + ((blocks & 1) ? 0 : 1);
    for (int i = 0; i < input_vectors * 16; ++i) touched = a[i];
    for (int i = 0; i < blocks * 16; ++i) sum += (int32_t)a[i] * b[i];
    return sum;
}
unsigned count_aligned(void) { return aligned_calls; }
unsigned count_unaligned(void) { return unaligned_calls; }
unsigned count_invalid(void) { return invalid_calls; }
void clear_counts(void) { aligned_calls = unaligned_calls = invalid_calls = 0; }
int process(const uint8_t *blob, size_t bytes, const int8_t *frames, unsigned count, int8_t *out) {
    edn_model model;
    if (edn_init(&model, blob, bytes)) return -1;
    void *state = malloc(model.state_bytes);
    if (!state) return -1;
    int status = edn_reset(&model, state, model.state_bytes);
    for (unsigned i = 0; !status && i < count; ++i)
        status = edn_process_frame(&model, state, model.state_bytes,
                                  frames + i*model.input_channels, out + i*model.output_channels);
    free(state);
    return status;
}
'''


@pytest.fixture(scope="module")
def simd_adapter(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("A C compiler is required for adapter verification")
    directory = tmp_path_factory.mktemp("simd_adapter")
    root = Path(__file__).resolve().parents[1] / "firmware/esp32_denoiser"
    wrapper = directory / "wrapper.c"
    wrapper.write_text(WRAPPER)
    library = directory / "adapter.so"
    subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-DEDN_USE_S3_SIMD=1", "-shared", "-fPIC", "-I", str(root),
                    str(root / "denoiser.c"), str(root / "frequency.c"), str(wrapper), "-o", str(library)],
                   check=True, capture_output=True, text=True)
    runtime = ctypes.CDLL(str(library))
    runtime.process.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                ctypes.c_uint, ctypes.c_void_p]
    runtime.edn_neural_backend.restype = ctypes.c_char_p
    return runtime


def test_device_known_answer_cases_and_backend_name(simd_adapter):
    simd_adapter.clear_counts()
    assert simd_adapter.edn_backend_self_test() == 0
    assert simd_adapter.count_aligned() == 4 * 13
    assert simd_adapter.count_unaligned() == 4 * 13 * 16
    assert simd_adapter.count_invalid() == 0
    assert b"s3_simd" in simd_adapter.edn_neural_backend()


@pytest.mark.parametrize("width", [8, 31, 64, 513])
def test_nonzero_streaming_parity_with_blob_and_feature_misalignment(tmp_path, simd_adapter, width):
    torch.manual_seed(771)
    model = SpectralTCN(SpectralTCNConfig(width=width, dilations=(1, 2)))
    with torch.no_grad():
        model.head.weight.normal_(0, 0.15)
        model.head.bias.normal_(0, 0.05)
    model = configure_qat(model).eval()
    path = tmp_path / "nonzero.bin"
    export_model(model, path, max_bytes=4_000_000)
    data = path.read_bytes()
    rng = np.random.default_rng(982)
    count = 12
    # A deliberately shifted input base also changes alignment every frame.
    frame_storage = np.empty(count * model.feature_size + 1, dtype=np.int8)
    frames = frame_storage[1:].reshape(count, model.feature_size)
    frames[:] = rng.integers(-128, 128, frames.shape, dtype=np.int8)
    frames[0, ::2], frames[0, 1::2] = -128, 127
    expected = IntegerDenoiser(path).process(frames)
    assert np.count_nonzero(expected) > expected.size // 4
    for offset in (0, 1, 7, 15):
        storage = ctypes.create_string_buffer(len(data) + 32)
        address = (ctypes.addressof(storage) + 15) // 16 * 16 + offset
        ctypes.memmove(address, data, len(data))
        actual = np.empty_like(expected)
        simd_adapter.clear_counts()
        assert simd_adapter.process(address, len(data), frames.ctypes.data,
                                     count, actual.ctypes.data) == 0
        np.testing.assert_array_equal(actual, expected)
        assert simd_adapter.count_invalid() == 0
        # The 387-input projection uses SIMD even when narrow/oversized hidden
        # rows correctly use scalar fallback; offset one forces unaligned rows.
        assert simd_adapter.count_unaligned() > 0
        if width == 64 and offset == 0:
            assert simd_adapter.count_aligned() > 0


def test_frequency_graph_uses_simd_with_exact_global_flatten_order(tmp_path, simd_adapter):
    from esp32_denoiser.frequency_export import IntegerFrequencyDenoiser, export_frequency_model
    from esp32_denoiser.frequency_model import FrequencyUNet
    from esp32_denoiser.frequency_quantization import configure_frequency_qat

    torch.manual_seed(669)
    model = FrequencyUNet()
    with torch.no_grad():
        model.head.weight.normal_(std=0.25)
        model.head.bias.normal_(std=0.06)
    configure_frequency_qat(model, hidden_exponent=-6)
    path = tmp_path / "frequency.bin"
    export_frequency_model(model, path)
    data = path.read_bytes()
    frames = np.random.default_rng(721).integers(-128, 128, (12, 3, 257), dtype=np.int8)
    expected = IntegerFrequencyDenoiser(data).process(frames)
    assert np.count_nonzero(expected) > expected.size // 3
    lib = simd_adapter
    lib.ednf_model_handle_bytes.restype = ctypes.c_size_t
    lib.ednf_workspace_bytes.argtypes = [ctypes.c_void_p]
    lib.ednf_workspace_bytes.restype = ctypes.c_size_t
    lib.ednf_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ednf_reset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ednf_process_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.c_void_p, ctypes.c_void_p]
    for offset in (0, 1):
        blob = ctypes.create_string_buffer(bytes(offset) + data)
        handle = ctypes.create_string_buffer(lib.ednf_model_handle_bytes())
        assert lib.ednf_init(handle, ctypes.addressof(blob)+offset, len(data)) == 0
        length = lib.ednf_workspace_bytes(handle)
        state = ctypes.create_string_buffer(length)
        assert lib.ednf_reset(handle, state, length) == 0
        lib.clear_counts()
        actual = np.empty_like(expected)
        for i, frame in enumerate(frames):
            assert lib.ednf_process_frame(handle, state, length, frame.ctypes.data, actual[i].ctypes.data) == 0
        np.testing.assert_array_equal(actual, expected)
        assert lib.count_invalid() == 0
        assert lib.count_unaligned() > 0
        if offset == 0:
            assert lib.count_aligned() > 0
