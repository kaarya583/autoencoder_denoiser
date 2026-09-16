"""The C frequency graph must match the serialized streaming integer oracle."""
import ctypes
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest
import torch

from esp32_denoiser.frequency_export import HEADER, LAYER, IntegerFrequencyDenoiser, export_frequency_model
from esp32_denoiser.frequency_model import FrequencyUNet, FrequencyUNetConfig
from esp32_denoiser.frequency_quantization import configure_frequency_qat


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("A C compiler is required to verify the runtime")
    directory = tmp_path_factory.mktemp("frequency_c")
    source = Path(__file__).resolve().parents[1] / "firmware/esp32_denoiser/frequency.c"
    library = directory / "frequency.so"
    subprocess.run([cc, "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared",
                    "-fPIC", str(source), str(source.with_name("denoiser.c")), "-o", str(library)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    lib.ednf_model_handle_bytes.restype = ctypes.c_size_t
    lib.ednf_workspace_bytes.argtypes = [ctypes.c_void_p]
    lib.ednf_workspace_bytes.restype = ctypes.c_size_t
    lib.ednf_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ednf_reset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ednf_process_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                        ctypes.c_void_p, ctypes.c_void_p]
    return lib


def make_blob(tmp_path, default=False):
    torch.manual_seed(886)
    config = FrequencyUNetConfig() if default else FrequencyUNetConfig(
        encoder_channels=(3, 4, 5), global_width=7,
        local_dilations=(1, 3), global_dilations=(1, 2, 4))
    model = FrequencyUNet(config)
    with torch.no_grad():
        model.head.weight.normal_(std=0.3)
        model.head.bias.normal_(std=0.07)
        for block in (*model.local_blocks, *model.global_blocks):
            block.pointwise.weight.mul_(4)
    configure_frequency_qat(model, hidden_exponent=-6)
    path = tmp_path / "frequency.bin"
    report = export_frequency_model(model, path)
    return path.read_bytes(), report


@pytest.mark.parametrize("default", [False, True])
def test_exact_nonzero_streaming_state_reset_and_workspace(tmp_path, runtime, default):
    data, report = make_blob(tmp_path, default)
    oracle = IntegerFrequencyDenoiser(data)
    rng = np.random.default_rng(88)
    frames = rng.integers(-128, 128, (29, 3, 257), dtype=np.int8)
    frames[0] = 0
    expected = oracle.process(frames)
    assert np.count_nonzero(expected) > expected.size // 3
    # Offset one ensures the parser never relies on native aligned integer reads.
    blob = ctypes.create_string_buffer(b"x" + data)
    handle = ctypes.create_string_buffer(runtime.ednf_model_handle_bytes())
    assert runtime.ednf_init(handle, ctypes.addressof(blob) + 1, len(data)) == 0
    required = runtime.ednf_workspace_bytes(handle)
    assert report["neural_history_bytes"] < required < 48 * 1024
    if default:
        assert required == 34_844
    state = ctypes.create_string_buffer(required)
    assert runtime.ednf_reset(handle, state, required - 1) == -1
    assert runtime.ednf_reset(handle, state, required) == 0
    actual = np.empty_like(expected)
    for i, frame in enumerate(frames):
        assert runtime.ednf_process_frame(handle, state, required,
                                           frame.ctypes.data, actual[i].ctypes.data) == 0
    np.testing.assert_array_equal(actual, expected)
    assert runtime.ednf_reset(handle, state, required) == 0
    assert runtime.ednf_process_frame(handle, state, required,
                                       frames[0].ctypes.data, actual[0].ctypes.data) == 0
    np.testing.assert_array_equal(actual[0], expected[0])
    state[0:4] = struct.pack("<I", 0xffffffff)
    assert runtime.ednf_process_frame(handle, state, required,
                                       frames[0].ctypes.data, actual[0].ctypes.data) == -1


def test_c_parser_rejects_corrupt_graph_ranges_state_and_dsp(tmp_path, runtime):
    data, _ = make_blob(tmp_path)
    handle = ctypes.create_string_buffer(runtime.ednf_model_handle_bytes())
    mutations = [(0, b"X"), (8, struct.pack("<I", 2)), (23, b"\x01"),
                 (27, b"\x01"), (36, struct.pack("<I", 65537)),
                 (HEADER.size + 2, b"\x03"),
                 (HEADER.size + 16, struct.pack("<I", 0xfffffff0)),
                 (HEADER.size + 28, b"\x01")]
    dsp = struct.unpack_from("<I", data, 32)[0]
    mutations += [(dsp + 12, struct.pack("<I", 0x7fc00000)),
                  (dsp + 16, struct.pack("<I", 0x7f800000))]
    for offset, value in mutations:
        malformed = bytearray(data)
        malformed[offset:offset+len(value)] = value
        blob = ctypes.create_string_buffer(bytes(malformed))
        assert runtime.ednf_init(handle, blob, len(malformed)) == -1
    for malformed in (data[:20], data[:-1], bytes(100_000)):
        blob = ctypes.create_string_buffer(malformed)
        assert runtime.ednf_init(handle, blob, len(malformed)) == -1
