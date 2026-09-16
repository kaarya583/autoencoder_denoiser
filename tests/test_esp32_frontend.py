"""Verify the complete C audio path and ESP-DSP's FFT ordering contract.

The ESP path uses an independent host DFT stub with vendor-compatible bit
reversal. These are correctness checks, not measurements of ESP32 throughput.
"""

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from esp32_denoiser.evaluate import IntegerWaveformEnhancer
from esp32_denoiser.export import export_model
from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig
from esp32_denoiser.quantization import configure_qat


WRAPPER = r"""
#include "audio.h"
#include <stdlib.h>
static edn_model model;
static edn_audio_state audio;
static void *neural_state;
int test_init(const void *blob, size_t bytes) {
    if (edn_init(&model, blob, bytes)) return -1;
    neural_state = calloc(1, model.state_bytes);
    if (!neural_state) return -1;
    return edn_audio_init(&audio, &model, neural_state, model.state_bytes);
}
int test_process(const float *input, float *output) {
    return edn_audio_process(&audio, input, output);
}
int test_reset(void) { return edn_audio_reset(&audio); }
int test_fft_alignment(void) { return ((uintptr_t)audio.fft) % 16; }
void test_close(void) { free(neural_state); neural_state = NULL; }
"""

FFT_HEADER = r"""
#define ESP_OK 0
extern unsigned char dsps_fft2r_initialized;
extern int dsps_fft_w_table_size;
int dsps_fft2r_init_fc32(float *table, int size);
int dsps_fft2r_fc32(float *data, int size);
int dsps_bit_rev_fc32(float *data, int size);
"""

FFT_STUB = r"""
#include "dsps_fft2r.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
unsigned char dsps_fft2r_initialized;
int dsps_fft_w_table_size;
static unsigned reversed(unsigned value) {
    unsigned result = 0, bit;
    for (bit = 0; bit < 9; ++bit) { result = 2*result + (value & 1); value >>= 1; }
    return result;
}
int dsps_fft2r_init_fc32(float *table, int size) {
    (void)table;
    if (size != 512) return -1;
    dsps_fft2r_initialized = 1; dsps_fft_w_table_size = size;
    return ESP_OK;
}
int dsps_fft2r_fc32(float *data, int size) {
    float result[1024];
    unsigned frequency, sample;
    if (size != 512 || ((uintptr_t)data) % 16) return -1;
    for (frequency = 0; frequency < 512; ++frequency) {
        double real = 0, imag = 0;
        for (sample = 0; sample < 512; ++sample) {
            double angle = -6.2831853071795864769 * frequency * sample / 512;
            double cosine = cos(angle), sine = sin(angle);
            real += data[2*sample]*cosine - data[2*sample+1]*sine;
            imag += data[2*sample]*sine + data[2*sample+1]*cosine;
        }
        /* ESP-DSP's forward complex FFT emits bit-reversed output. */
        result[2*reversed(frequency)] = (float)real;
        result[2*reversed(frequency)+1] = (float)imag;
    }
    memcpy(data, result, sizeof(result));
    return ESP_OK;
}
int dsps_bit_rev_fc32(float *data, int size) {
    float result[1024];
    unsigned frequency;
    if (size != 512) return -1;
    for (frequency = 0; frequency < 512; ++frequency) {
        result[2*frequency] = data[2*reversed(frequency)];
        result[2*frequency+1] = data[2*reversed(frequency)+1];
    }
    memcpy(data, result, sizeof(result));
    return ESP_OK;
}
"""


@pytest.fixture(params=("host_radix_fft", "esp_dsp_contract"))
def audio_runtime(request, tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("A C99 compiler is required for frontend verification")
    source = Path(__file__).resolve().parents[1] / "firmware/esp32_denoiser"
    wrapper = tmp_path / "wrapper.c"
    wrapper.write_text(WRAPPER)
    command = [compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
               "-I", str(source), str(source / "denoiser.c"), str(source / "audio.c"), str(wrapper)]
    if request.param == "esp_dsp_contract":
        (tmp_path / "sdkconfig.h").write_text("/* Host ESP-DSP contract: scalar neural backend. */\n")
        (tmp_path / "dsps_fft2r.h").write_text(FFT_HEADER)
        stub = tmp_path / "fft_stub.c"
        stub.write_text(FFT_STUB)
        command += ["-DESP_PLATFORM", "-I", str(tmp_path), str(stub)]
    library_path = tmp_path / "frontend.so"
    subprocess.run(command + ["-lm", "-o", str(library_path)], check=True, capture_output=True, text=True)
    library = ctypes.CDLL(str(library_path))
    library.test_init.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.test_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.test_reset.argtypes = []
    library.test_fft_alignment.argtypes = []
    library.test_close.argtypes = []
    library.test_close.restype = None
    yield library
    library.test_close()


def _process(runtime, audio):
    padded = np.pad(audio, (0, (-len(audio)) % 256 + 256))
    output = np.empty_like(padded)
    for offset in range(0, len(padded), 256):
        assert runtime.test_process(padded[offset:].ctypes.data, output[offset:].ctypes.data) == 0
    # The first output hop is analysis overlap; the zero input hop flushes it.
    return output[256:256 + len(audio)].copy()


@pytest.mark.parametrize("nontrivial", (False, True))
def test_full_c_frontend_matches_torch_dsp(audio_runtime, tmp_path, nontrivial):
    torch.manual_seed(75)
    model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2), mask_scale=1.5))
    with torch.no_grad():
        if nontrivial:
            model.head.weight.normal_(0, 0.2)
            model.head.bias.normal_(0, 0.1)
        # Shipped constants must drive C DSP, including nondefault values.
        model.window.mul_(0.9)
        model.erb_lower_weight.mul_(0.8)
        model.erb_upper_weight.mul_(0.8)
    binary = tmp_path / "model.bin"
    export_model(configure_qat(model), binary)
    blob = ctypes.create_string_buffer(binary.read_bytes())
    assert audio_runtime.test_init(blob, len(binary.read_bytes())) == 0
    assert audio_runtime.test_fft_alignment() == 0
    rng = np.random.default_rng(41)
    audio = rng.normal(0, 0.07, 1027).astype(np.float32)
    audio += (0.04 * np.cos(np.arange(len(audio)) * 0.23) + 0.02).astype(np.float32)
    audio[-1] = 0.4  # Expose incomplete final-frame flushing.
    expected = IntegerWaveformEnhancer(binary)(torch.from_numpy(audio)[None])[0].numpy()
    actual = _process(audio_runtime, audio)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    if nontrivial:
        assert np.max(np.abs(actual - audio)) > 0.01
    else:
        np.testing.assert_allclose(actual, audio, atol=2e-6, rtol=2e-5)
    assert audio_runtime.test_reset() == 0
    np.testing.assert_array_equal(_process(audio_runtime, audio), actual)
    assert audio_runtime.test_reset() == 0
    np.testing.assert_array_equal(_process(audio_runtime, np.zeros(257, dtype=np.float32)), 0)
