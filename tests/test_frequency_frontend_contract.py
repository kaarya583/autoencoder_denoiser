"""Inspect full-bin feature layout and real-FFT endpoint behavior in C audio."""

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from esp32_denoiser.frequency_export import IntegerFrequencyDenoiser, export_frequency_model
from esp32_denoiser.frequency_model import FrequencyUNet, FrequencyUNetConfig
from esp32_denoiser.frequency_quantization import configure_frequency_qat
from esp32_denoiser.quantization import round_away
from test_esp32_frontend import FFT_HEADER, FFT_STUB
from test_esp32_frequency_frontend import initialize


@pytest.fixture(params=("host", "esp_contract"))
def inspecting_frontend(request, tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("A C compiler is required for frontend verification")
    source = Path(__file__).resolve().parents[1] / "firmware/esp32_denoiser"
    wrapper = tmp_path / "inspect.c"
    wrapper.write_text('#include "frequency_audio.h"\n'
                       'const int8_t *features(ednf_audio_state *s) { return s->features; }\n'
                       'const int8_t *deltas(ednf_audio_state *s) { return s->deltas; }\n')
    command = [compiler, "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
               "-I", str(source), str(wrapper), str(source / "frequency_audio.c"),
               str(source / "frequency.c"), str(source / "denoiser.c")]
    if request.param == "esp_contract":
        (tmp_path / "sdkconfig.h").write_text("/* Scalar host contract. */\n")
        (tmp_path / "dsps_fft2r.h").write_text(FFT_HEADER)
        (tmp_path / "fft_stub.c").write_text(FFT_STUB)
        command += ["-DESP_PLATFORM", "-I", str(tmp_path), str(tmp_path / "fft_stub.c")]
    path = tmp_path / "inspect.so"
    subprocess.run(command + ["-lm", "-o", str(path)], check=True, capture_output=True)
    library = ctypes.CDLL(str(path))
    for name in ("ednf_model_handle_bytes", "ednf_audio_state_bytes"):
        getattr(library, name).restype = ctypes.c_size_t
    library.ednf_workspace_bytes.argtypes = [ctypes.c_void_p]
    library.ednf_workspace_bytes.restype = ctypes.c_size_t
    library.ednf_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.ednf_audio_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.ednf_audio_reset.argtypes = [ctypes.c_void_p]
    library.ednf_audio_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    for name in ("features", "deltas"):
        getattr(library, name).argtypes = [ctypes.c_void_p]
        getattr(library, name).restype = ctypes.POINTER(ctypes.c_int8)
    return library


def test_exact_fullbin_layout_integer_deltas_and_fft_endpoints(inspecting_frontend, tmp_path):
    torch.manual_seed(87)
    model = FrequencyUNet(FrequencyUNetConfig(encoder_channels=(4, 6, 8), global_width=8,
                                             local_dilations=(1, 2), global_dilations=(1, 2),
                                             mask_scale=1.5))
    with torch.no_grad():
        model.window.mul_(0.9)
        model.head.weight.normal_(std=0.25)
        model.head.bias.copy_(torch.tensor([-0.2, 0.15]))
    configure_frequency_qat(model, hidden_exponent=-6)
    binary = tmp_path / "model.bin"
    export_frequency_model(model, binary)
    oracle = IntegerFrequencyDenoiser(binary)
    library = inspecting_frontend
    context, owners = initialize(library, binary)
    assert owners
    rng = np.random.default_rng(876)
    timeline = np.arange(1031)
    for samples in (
        np.full(1031, 0.13, np.float32),
        (0.13 * (-1.0) ** timeline).astype(np.float32),
        (0.13 + 0.07 * (-1.0) ** timeline + 0.09 * np.sin(timeline * 0.13)
         + rng.normal(0, 0.004, 1031)).astype(np.float32),
    ):
        samples[-1] += 0.2  # Require the final incomplete hop to be flushed.
        assert library.ednf_audio_reset(context) == 0
        oracle.reset()
        padded = np.pad(samples, (0, (-len(samples)) % 256 + 256))
        previous = torch.zeros(1, 256)
        ola = torch.zeros(1, 256)
        ola_weight = torch.zeros(1, 256)
        expected_chunks, actual_chunks = [], []
        with torch.inference_mode():
            for chunk in padded.reshape(-1, 256):
                actual = np.empty(256, np.float32)
                assert library.ednf_audio_process(context, chunk.ctypes.data, actual.ctypes.data) == 0
                spectrum, features = model.frame_features(torch.cat((previous, torch.from_numpy(chunk)[None]), -1)[:, None])
                expected_features = round_away(features[0, :, 0] * 128).clamp(-128, 127).to(torch.int8).numpy()
                c_features = np.ctypeslib.as_array(library.features(context), shape=(771,)).copy().reshape(3, 257)
                np.testing.assert_allclose(c_features.astype(np.int16), expected_features, atol=1, rtol=0)
                c_deltas = np.ctypeslib.as_array(library.deltas(context), shape=(514,)).copy()
                # Feed the actual C-quantized features to isolate neural parity
                # from harmless FFT rounding at a feature quantization boundary.
                np.testing.assert_array_equal(c_deltas, oracle.step(c_features))
                deltas = torch.from_numpy(c_deltas.copy()).float()[None, :, None] / 128
                reconstructed = torch.fft.irfft(model.apply_mask(spectrum, deltas), n=512)[:, 0] * model.window
                weight = model.window.square()
                expected = (ola + reconstructed[:, :256]) / (ola_weight + weight[:256]).clamp_min(1e-8)
                expected_chunks.append(expected[0].numpy())
                actual_chunks.append(actual)
                ola, ola_weight = reconstructed[:, 256:], weight[256:][None]
                previous = torch.from_numpy(chunk.copy())[None]
        expected = np.concatenate(expected_chunks)[256:256 + len(samples)]
        actual = np.concatenate(actual_chunks)[256:256 + len(samples)]
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
        assert np.max(np.abs(actual - samples)) > 0.01
