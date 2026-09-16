"""The shipped PCM16 path must be measured in addition to unclipped masks."""

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pytest
import soundfile as sf
import torch

from esp32_denoiser.embedded import EmbeddedWaveformEnhancer, main
from esp32_denoiser.evaluate import IntegerWaveformEnhancer, evaluate_manifest
from esp32_denoiser.export import export_model
from esp32_denoiser.metrics import si_sdr
from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig
from esp32_denoiser.quantization import configure_qat


pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="C99 compiler required")


def _export(tmp_path: Path, gain: float) -> Path:
    model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2)))
    with torch.no_grad():
        model.head.bias[:257].fill_((gain - 1) / model.config.mask_scale)
    binary = tmp_path / "model.bin"
    export_model(configure_qat(model), binary)
    return binary


def test_pcm16_applies_exact_input_quantization_and_output_clipping(tmp_path):
    binary = _export(tmp_path, gain=2)
    pcm = EmbeddedWaveformEnhancer(binary)
    floating = EmbeddedWaveformEnhancer(binary, "float32")
    try:
        rng = np.random.default_rng(85)
        audio = torch.from_numpy(rng.normal(0, 0.2, (1, 1537)).astype(np.float32))
        audio[0, :4] = torch.tensor([-1.5, 1.5, -0.8, 0.8])
        encoded = (audio * 32768).round().clamp(-32768, 32767) / 32768
        reference = floating(encoded)
        expected = (reference * 32768).round().clamp(-32768, 32767) / 32768
        actual = pcm(audio)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert pcm.io_statistics["input_clipped_samples"] == 2
        assert pcm.io_statistics["output_at_rail_samples"] >= 4
        assert actual.max() <= 32767 / 32768 and actual.min() >= -1
        torch.testing.assert_close(pcm(audio), actual, atol=0, rtol=0)
        torch.testing.assert_close(pcm(torch.zeros(1, 257)), torch.zeros(1, 257), atol=0, rtol=0)
        pcm.close()
        with pytest.raises(RuntimeError, match="closed"):
            pcm(audio)
    finally:
        pcm.close()
        floating.close()


def test_pcm16_quality_delta_across_input_levels_and_full_float_parity(tmp_path):
    binary = _export(tmp_path, gain=0.75)
    pcm = EmbeddedWaveformEnhancer(binary)
    floating = EmbeddedWaveformEnhancer(binary, "float32")
    reference = IntegerWaveformEnhancer(binary)
    try:
        generator = torch.Generator().manual_seed(47)
        clean = torch.randn(1, 4097, generator=generator) * 0.1
        noisy = clean + torch.randn(1, 4097, generator=generator) * 0.03
        for gain in (1, 0.1, 0.01):
            enhanced_float = floating(noisy * gain)
            enhanced_pcm = pcm(noisy * gain)
            torch.testing.assert_close(enhanced_float, reference(noisy * gain), atol=2e-6, rtol=2e-5)
            difference = (si_sdr(enhanced_pcm, clean * gain) - si_sdr(enhanced_float, clean * gain)).abs()
            assert difference.item() < 0.05
        assert pcm.io_statistics["input_clipped_samples"] == 0
        assert pcm.io_statistics["output_at_rail_samples"] == 0
    finally:
        pcm.close()
        floating.close()


def test_embedded_cli_compares_identical_utterances_and_labels_host_timing(tmp_path, monkeypatch):
    binary = _export(tmp_path, gain=0.75)
    rng = np.random.default_rng(59)
    clean = rng.normal(0, 0.1, 1087).astype(np.float32)
    noisy = clean + rng.normal(0, 0.03, len(clean)).astype(np.float32)
    for name, waveform in (("clean", clean), ("noisy", noisy)):
        sf.write(tmp_path / f"{name}.wav", waveform, 16000, subtype="FLOAT")
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(json.dumps({"id": "p100_001", "speaker": "p100", "sample_rate": 16000,
                                    "samples": len(clean), "clean": "clean.wav", "noisy": "noisy.wav"}) + "\n")
    report = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["embedded", "--integer-model", str(binary),
                                     "--manifest", str(manifest), "--output", str(report),
                                     "--compare-reference"])
    main()
    result = json.loads(report.read_text())
    assert result["summary"]["utterances"] == 1
    assert result["model"]["io_format"] == "pcm16"
    assert result["model"]["io_statistics"]["samples"] == len(clean)
    assert "not ESP32" in result["timing"]["measurement"]
    assert abs(result["comparison"]["versus_unclipped_c_si_sdri_difference_db"]) < 0.05
    assert abs(result["comparison"]["versus_torch_dsp_si_sdri_difference_db"]) < 0.05
    assert "BOTH input and output" in result["comparison"]["interpretation"]
    assert result["comparison"]["float32_full_c_preservation"]["enhanced"]["projection_gain"]["mean"] < 0.8


def test_reused_full_c_evaluation_reports_only_current_pcm_counts_and_levels(tmp_path):
    binary = _export(tmp_path, gain=2)
    timeline = np.arange(1031)
    clean = (0.7 * np.sin(timeline * 0.09)).astype(np.float32)
    noisy = clean + 0.03 * np.cos(timeline * 0.71).astype(np.float32)
    for name, waveform in (("clean", clean), ("noisy", noisy)):
        sf.write(tmp_path / f"{name}.wav", waveform, 16000, subtype="FLOAT")
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(json.dumps({"id": "clip", "speaker": "fixture", "sample_rate": 16000,
                                    "samples": len(clean), "clean": "clean.wav", "noisy": "noisy.wav"}) + "\n")
    pcm = EmbeddedWaveformEnhancer(binary)
    floating = EmbeddedWaveformEnhancer(binary, "float32")
    try:
        # Previous calls must not leak samples/rails into the new report.
        pcm(torch.full((1, 513), 1.5))
        first = evaluate_manifest(pcm, manifest)
        second = evaluate_manifest(pcm, manifest)
        unclipped = evaluate_manifest(floating, manifest)
        assert first["io_statistics"] == second["io_statistics"]
        assert first["io_statistics"]["samples"] == len(clean)
        assert first["io_statistics"]["input_clipped_samples"] == 0
        assert first["io_statistics"]["output_at_rail_samples"] > 100
        assert first["preservation"]["enhanced"]["pcm16_out_of_range_samples"] == 0
        assert first["preservation"]["enhanced"]["pcm16_rail_or_exceeds_samples"] > 100
        assert unclipped["preservation"]["enhanced"]["projection_gain"]["mean"] == pytest.approx(2, abs=.01)
        assert unclipped["preservation"]["enhanced"]["peak_abs_max"] > 1.3
        assert unclipped["preservation"]["enhanced"]["pcm16_out_of_range_samples"] > 100
    finally:
        pcm.close()
        floating.close()


@pytest.mark.parametrize("mutation", ("model", "audio"))
def test_reference_passes_freeze_model_bytes_and_reject_changed_audio(tmp_path, monkeypatch, mutation):
    import esp32_denoiser.embedded as embedded
    from test_esp32_evaluate import _manifest

    binary = _export(tmp_path, gain=.75)
    expected_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    manifest = _manifest(tmp_path)
    report = tmp_path / "report.json"
    calls = 0

    def replace_after_first_pass(*args, **kwargs):
        nonlocal calls
        result = evaluate_manifest(*args, **kwargs)
        calls += 1
        if calls == 1:
            if mutation == "model":
                binary.write_bytes(b"another training process replaced this path")
            else:
                audio_path = tmp_path / "noisy_0.wav"
                waveform, rate = sf.read(audio_path, dtype="float32")
                sf.write(audio_path, waveform * .5, rate, subtype="FLOAT")
        return result

    monkeypatch.setattr(embedded, "evaluate_manifest", replace_after_first_pass)
    monkeypatch.setattr("sys.argv", ["embedded", "--integer-model", str(binary),
                                     "--manifest", str(manifest), "--output", str(report),
                                     "--compare-reference"])
    if mutation == "audio":
        with pytest.raises(ValueError, match="identical manifest and audio"):
            main()
        assert not report.exists()
    else:
        main()
        result = json.loads(report.read_text())
        assert calls == 3
        assert result["model"]["model_sha256"] == expected_hash
        assert abs(result["comparison"]["versus_unclipped_c_si_sdri_difference_db"]) < .05
