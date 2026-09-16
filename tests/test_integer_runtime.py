"""Deployment checks exercise nonzero masks and persistent temporal state."""

import ctypes
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest
import torch

from esp32_denoiser.export import HEADER, IntegerDenoiser, export_model, requantize
from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig
from esp32_denoiser.quantization import calibrate_hidden_exponent, configure_qat, round_away


class CModel(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint8)),
                ("model_bytes", ctypes.c_size_t), ("state_bytes", ctypes.c_size_t),
                ("input_channels", ctypes.c_uint16), ("hidden_channels", ctypes.c_uint16),
                ("output_channels", ctypes.c_uint16), ("blocks", ctypes.c_uint16),
                ("input_exponent", ctypes.c_int8), ("hidden_exponent", ctypes.c_int8),
                ("output_exponent", ctypes.c_int8),
                ("dsp_data", ctypes.POINTER(ctypes.c_uint8)), ("dsp_bytes", ctypes.c_size_t)]


@pytest.fixture(scope="module")
def c_runtime(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("A C99 compiler is required to verify the deployment runtime")
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp("integer_runtime")
    library = directory / "denoiser.so"
    subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-shared", "-fPIC", str(root / "firmware/esp32_denoiser/denoiser.c"),
                    "-o", str(library)], check=True, capture_output=True, text=True)
    runtime = ctypes.CDLL(str(library))
    runtime.edn_init.argtypes = [ctypes.POINTER(CModel), ctypes.c_void_p, ctypes.c_size_t]
    runtime.edn_reset.argtypes = [ctypes.POINTER(CModel), ctypes.c_void_p, ctypes.c_size_t]
    runtime.edn_process_frame.argtypes = [ctypes.POINTER(CModel), ctypes.c_void_p,
                                          ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
    return runtime


def _model(hidden_exponent=-4):
    torch.manual_seed(34)
    model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2, 4)))
    # An identity-initialized head would make all parity checks trivially zero.
    with torch.no_grad():
        model.head.weight.normal_(0, 0.3)
        model.head.bias.normal_(0, 0.1)
        for block in model.blocks:
            block.pointwise.weight.mul_(5)
    return configure_qat(model, input_exponent=-7, hidden_exponent=hidden_exponent).eval()


@pytest.mark.parametrize("hidden_exponent", [-4, -6])
def test_exact_c_integer_parity_and_qat(tmp_path, c_runtime, hidden_exponent):
    model = _model(hidden_exponent)
    path = tmp_path / "model.bin"
    report = export_model(model, path)
    oracle = IntegerDenoiser(path)
    rng = np.random.default_rng(456)
    frames = rng.integers(-128, 128, (80, model.feature_size), dtype=np.int8)
    expected = oracle.process(frames)
    assert np.count_nonzero(expected) > expected.size // 3
    assert all(bool((history < 0).any()) for history in oracle.history)
    blob = ctypes.create_string_buffer(path.read_bytes())
    cm = CModel()
    assert c_runtime.edn_init(ctypes.byref(cm), blob, len(path.read_bytes())) == 0
    state = ctypes.create_string_buffer(report["state_bytes"])
    assert c_runtime.edn_reset(ctypes.byref(cm), state, len(state)) == 0
    actual = np.empty_like(expected)
    for index, frame in enumerate(frames):
        assert c_runtime.edn_process_frame(ctypes.byref(cm), state, len(state),
                                            frame.ctypes.data, actual[index].ctypes.data) == 0
    np.testing.assert_array_equal(actual, expected)
    with torch.inference_mode():
        inputs = torch.from_numpy(frames.astype(np.float32).T.copy())[None] / 128
        fake = model.forward_features(inputs)[0].T * 128
    np.testing.assert_allclose(round_away(fake).numpy(), expected, atol=1, rtol=0)
    # Independent, chunked calls must preserve identical FIFO semantics.
    oracle.reset()
    split = np.concatenate([oracle.process(frames[:13]), oracle.process(frames[13:])])
    np.testing.assert_array_equal(split, expected)
    assert c_runtime.edn_reset(ctypes.byref(cm), state, len(state)) == 0
    replay = np.empty_like(expected[0])
    assert c_runtime.edn_process_frame(ctypes.byref(cm), state, len(state),
                                       frames[0].ctypes.data, replay.ctypes.data) == 0
    np.testing.assert_array_equal(replay, expected[0])


def test_default_model_real_export_size(tmp_path):
    model = configure_qat(SpectralTCN(), input_exponent=-7)
    path = tmp_path / "default.bin"
    report = export_model(model, path)
    assert path.stat().st_size == report["model_bytes"] < 99_000
    assert report["weight_bytes"] == 83_392
    assert report["bias_bytes"] == 5_384
    assert report["state_bytes"] == 8_280
    assert report["dsp_bytes"] == 3_988
    assert report["format"] == "EDNSI8-v2"
    assert IntegerDenoiser(path).mask_scale == 2.0
    assert report["parameter_payload_reduction"] > 119
    with pytest.raises(ValueError, match="exceeding"):
        export_model(model, tmp_path / "too_small.bin", max_bytes=100)
    legacy = configure_qat(SpectralTCN(SpectralTCNConfig(activation_mode="relu")))
    with pytest.raises(ValueError, match="signed hardtanh"):
        export_model(legacy, tmp_path / "legacy.bin")


def test_rounding_and_saturation():
    values = np.array([-257, -255, -5, -3, -1, 0, 1, 3, 5, 255, 257], dtype=np.int32)
    np.testing.assert_array_equal(requantize(values, -1), [-128, -128, -3, -2, -1, 0, 1, 2, 3, 127, 127])
    np.testing.assert_array_equal(requantize(np.array([-2, -1, 0, 1, 2]), 7), [-128, -128, 0, 127, 127])


def test_runtime_rejects_bad_blob_and_short_state(tmp_path, c_runtime):
    path = tmp_path / "model.bin"
    report = export_model(_model(), path)
    data = path.read_bytes()
    old_version = data[:8] + struct.pack("<I", 1) + data[12:]
    for malformed in [data[:10], b"BADMAGIC" + data[8:], data[:-1], old_version]:
        blob = ctypes.create_string_buffer(malformed)
        assert c_runtime.edn_init(ctypes.byref(CModel()), blob, len(malformed)) == -1
    malformed = bytearray(data)
    struct.pack_into("<I", malformed, HEADER.size + 12, len(data) + 1)
    blob = ctypes.create_string_buffer(bytes(malformed))
    assert c_runtime.edn_init(ctypes.byref(CModel()), blob, len(malformed)) == -1
    blob = ctypes.create_string_buffer(data)
    cm = CModel()
    assert c_runtime.edn_init(ctypes.byref(cm), blob, len(data)) == 0
    state = ctypes.create_string_buffer(report["state_bytes"] - 1)
    assert c_runtime.edn_reset(ctypes.byref(cm), state, len(state)) == -1


def test_qat_keeps_training_gradients_and_checks_accumulator(tmp_path):
    model = _model()
    output = model.forward_features(torch.rand(1, 387, 16) - 0.5)
    output.square().mean().backward()
    assert model.head.weight.grad is not None
    assert bool(torch.isfinite(model.head.weight.grad).all())
    assert bool((model.head.weight.grad != 0).any())
    with torch.no_grad():
        model.head.bias.fill_(1e20)
    with pytest.raises(ValueError, match="overflow"):
        export_model(model, tmp_path / "overflow.bin")


def test_calibration_restores_mode_and_supports_exact_export(tmp_path):
    torch.manual_seed(19)
    model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2)))
    model.train()
    exponent = calibrate_hidden_exponent(model, [torch.randn(1, 4096) * 0.1])
    assert model.training
    assert -12 <= exponent < -4  # Avoid the excessively coarse default grid.
    assert not model.input_proj._forward_hooks
    configure_qat(model, hidden_exponent=exponent)
    report = export_model(model, tmp_path / "calibrated.bin")
    assert report["hidden_exponent"] == exponent
    with pytest.raises(ValueError, match="before configure_qat"):
        calibrate_hidden_exponent(model, [torch.zeros(1, 512)])


@pytest.mark.parametrize("branch_bias, expected_exponent", [(0.0, -4), (-9.0, -3)])
def test_calibration_ignores_clipped_extrema_but_keeps_signed_branch_range(branch_bias, expected_exponent):
    model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1,)))
    with torch.no_grad():
        model.input_proj.weight.zero_()
        model.input_proj.bias.fill_(100)  # Hardtanh only exposes +6.
        block = model.blocks[0]
        block.depthwise.weight.zero_()
        block.depthwise.bias[::2] = -100  # ReLU6 discards negative extrema.
        block.depthwise.bias[1::2] = 100  # ReLU6 only exposes +6.
        block.pointwise.weight.zero_()
        block.pointwise.bias.fill_(branch_bias)
    exponent = calibrate_hidden_exponent(model, [torch.zeros(1, 512)])
    assert exponent == expected_exponent
    # In the negative-branch case, +6 + (-9) = -3. Observing only clipped
    # states would choose a grid that truncates -9 and breaks cancellation.
    assert not model.input_activation._forward_hooks
    assert not block.depth_activation._forward_hooks
    assert not block.pointwise._forward_hooks
    assert not block.output_activation._forward_hooks
