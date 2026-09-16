"""Full frequency-model DSP parity, including the ESP-DSP ABI contract."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from esp32_denoiser.frequency_evaluate import FrequencyIntegerWaveformEnhancer
from esp32_denoiser.frequency_export import export_frequency_model
from esp32_denoiser.frequency_model import FrequencyUNet, FrequencyUNetConfig
from esp32_denoiser.frequency_quantization import configure_frequency_qat
from test_esp32_frontend import FFT_HEADER, FFT_STUB


@pytest.fixture(params=("host", "esp_contract"))
def frontend(request, tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("A C compiler is required for frontend verification")
    source = Path(__file__).resolve().parents[1] / "firmware/esp32_denoiser"
    command = [cc, "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
               str(source / "frequency_audio.c"), str(source / "frequency.c"), str(source / "denoiser.c")]
    if request.param == "esp_contract":
        (tmp_path / "sdkconfig.h").write_text("/* Scalar host contract. */\n")
        (tmp_path / "dsps_fft2r.h").write_text(FFT_HEADER)
        (tmp_path / "fft_stub.c").write_text(FFT_STUB)
        command += ["-DESP_PLATFORM", "-I", str(tmp_path), str(tmp_path / "fft_stub.c")]
    library = tmp_path / "frontend.so"
    subprocess.run(command + ["-lm", "-o", str(library)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    lib.ednf_model_handle_bytes.restype = ctypes.c_size_t
    lib.ednf_workspace_bytes.argtypes = [ctypes.c_void_p]
    lib.ednf_workspace_bytes.restype = ctypes.c_size_t
    lib.ednf_audio_state_bytes.restype = ctypes.c_size_t
    lib.ednf_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ednf_audio_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ednf_audio_reset.argtypes = [ctypes.c_void_p]
    lib.ednf_audio_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.ednf_audio_process_pcm16.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return lib


def initialize(lib, path):
    blob = ctypes.create_string_buffer(path.read_bytes())
    model = ctypes.create_string_buffer(lib.ednf_model_handle_bytes())
    assert lib.ednf_init(model, blob, path.stat().st_size) == 0
    workspace = ctypes.create_string_buffer(lib.ednf_workspace_bytes(model))
    raw = ctypes.create_string_buffer(lib.ednf_audio_state_bytes() + 15)
    audio = (ctypes.addressof(raw) + 15) // 16 * 16
    assert lib.ednf_audio_init(audio + 1, model, workspace, len(workspace)) == -1
    assert lib.ednf_audio_init(audio, model, workspace, len(workspace)) == 0
    return audio, (raw, workspace, model, blob)


def process(lib, audio, samples, pcm=False):
    assert lib.ednf_audio_reset(audio) == 0
    dtype = np.int16 if pcm else np.float32
    padded = np.pad(samples, (0, (-len(samples)) % 256 + 256)).astype(dtype)
    output = np.empty_like(padded)
    fn = lib.ednf_audio_process_pcm16 if pcm else lib.ednf_audio_process
    for i in range(0, len(padded), 256):
        assert fn(audio, padded[i:i+256].ctypes.data, output[i:i+256].ctypes.data) == 0
    return output[256:256+len(samples)]


@pytest.mark.parametrize("identity", [True, False])
def test_full_waveform_nondefault_dsp_pcm_silence_and_reset(tmp_path, frontend, identity):
    torch.manual_seed(191)
    model = FrequencyUNet(FrequencyUNetConfig(encoder_channels=(4, 6, 8), global_width=8,
                           local_dilations=(1, 2), global_dilations=(1, 2, 4), mask_scale=1.75))
    with torch.no_grad():
        model.window.mul_(0.9)
        if not identity:
            model.head.weight.normal_(std=0.35)
            model.head.bias.normal_(std=0.1)
    configure_frequency_qat(model, hidden_exponent=-6)
    path = tmp_path / "model.bin"
    export_frequency_model(model, path)
    audio, owners = initialize(frontend, path)
    assert owners  # Keep all caller-owned buffers alive for the complete stream.
    rng = np.random.default_rng(124)
    samples = (rng.normal(0, 0.15, 1793) + 0.15*np.sin(np.arange(1793)*0.4)).astype(np.float32)
    actual = process(frontend, audio, samples)
    with torch.inference_mode():
        expected = FrequencyIntegerWaveformEnhancer(path, backend="numpy")(torch.from_numpy(samples)[None])[0].numpy()
    relative_rmse = np.sqrt(np.mean((actual-expected)**2)/np.mean(expected**2))
    assert relative_rmse < 0.003
    if identity:
        np.testing.assert_allclose(actual, samples, atol=1e-6, rtol=1e-5)
    else:
        assert np.max(np.abs(actual-samples)) > 0.02
    np.testing.assert_array_equal(process(frontend, audio, samples), actual)
    np.testing.assert_array_equal(process(frontend, audio, np.zeros(800, np.float32)), 0)
    pcm = np.clip(np.rint(samples*32768), -32768, 32767).astype(np.int16)
    integer_output = process(frontend, audio, pcm, pcm=True)
    float_output = process(frontend, audio, pcm.astype(np.float32)/32768)
    expected_pcm = np.clip(np.rint(float_output*32768), -32768, 32767).astype(np.int16)
    np.testing.assert_array_equal(integer_output, expected_pcm)
