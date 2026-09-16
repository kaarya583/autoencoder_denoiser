"""Waveform parity checks include the exported constants and real C inference."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf
import torch

from esp32_denoiser.evaluate import IntegerWaveformEnhancer, evaluate_manifest, load_checkpoint, main
from esp32_denoiser.export import export_model
from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig
from esp32_denoiser.quantization import configure_qat


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _nontrivial_model():
    torch.manual_seed(42)
    model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2, 4), mask_scale=1.5))
    with torch.no_grad():
        model.head.weight.normal_(0, 0.25)
        model.head.bias.normal_(0, 0.1)
        for block in model.blocks:
            block.pointwise.weight.mul_(5)
        # Nondefault DSP constants expose accidental use of a fresh model's
        # defaults, rather than the constants actually shipped in the blob.
        model.window.mul_(0.9)
        model.erb_lower_weight.mul_(0.8)
        model.erb_upper_weight.mul_(0.8)
    return configure_qat(model).eval()


@pytest.mark.skipif(shutil.which("cc") is None, reason="C99 compiler required")
def test_actual_c_waveform_matches_qat_and_numpy_with_exported_constants(tmp_path):
    model = _nontrivial_model()
    binary = tmp_path / "model.bin"
    export_model(model, binary)
    c_model = IntegerWaveformEnhancer(binary, backend="c")
    numpy_model = IntegerWaveformEnhancer(binary, backend="numpy")
    assert len(list(c_model.parameters())) == 0
    assert c_model.mask_scale == 1.5
    torch.testing.assert_close(c_model.window, model.window)
    torch.testing.assert_close(c_model.erb_lower_weight, model.erb_lower_weight)
    audio = torch.randn(2, 1901) * 0.1
    with torch.inference_mode():
        expected = model(audio)
        actual = c_model(audio)
        oracle = numpy_model(audio)
        replay = c_model(audio)
    assert (actual - audio).abs().max() > 0.01
    torch.testing.assert_close(actual, oracle, atol=0, rtol=0)
    torch.testing.assert_close(actual, replay, atol=0, rtol=0)
    # The floating convolution used for QAT can round an accumulator on a
    # boundary differently by one output code; assess its waveform consequence.
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)


def test_integer_identity_handles_short_audio_and_silence(tmp_path):
    model = configure_qat(SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2))))
    binary = tmp_path / "identity.bin"
    export_model(model, binary)
    integer = IntegerWaveformEnhancer(binary, backend="numpy")
    for count in (1, 255, 256, 257):
        audio = torch.randn(1, count) * 0.1
        torch.testing.assert_close(integer(audio), audio, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(integer(torch.zeros(1, 513)), torch.zeros(1, 513))
    with pytest.raises(ValueError, match="finite"):
        integer(torch.tensor([[float("nan")]]))


def _manifest(tmp_path: Path) -> Path:
    records = []
    generator = np.random.default_rng(9)
    for index, samples in enumerate((701, 1533)):
        clean = generator.normal(0, 0.1, samples).astype(np.float32)
        noisy = clean + generator.normal(0, 0.03, samples).astype(np.float32)
        for role, audio in (("clean", clean), ("noisy", noisy)):
            sf.write(tmp_path / f"{role}_{index}.wav", audio, 16000, subtype="FLOAT")
        records.append({"id": f"p100_{index:03d}", "speaker": "p100", "sample_rate": 16000,
                        "samples": samples, "clean": f"clean_{index}.wav", "noisy": f"noisy_{index}.wav"})
    manifest = tmp_path / "val.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    return manifest


def test_checkpoint_and_binary_evaluation_report_equal_utterances(tmp_path, monkeypatch):
    model = configure_qat(SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2)))).eval()
    checkpoint = tmp_path / "qat.pt"
    torch.save({"model": model.state_dict(), "model_config": asdict(model.config),
                "phase": "qat", "epoch": 2}, checkpoint)
    reloaded, provenance = load_checkpoint(checkpoint)
    assert provenance["precision"] == "fake-quantized PyTorch simulation"
    manifest = _manifest(tmp_path)
    float_result = evaluate_manifest(reloaded, manifest)
    binary = tmp_path / "model.bin"
    export_model(model, binary)
    report = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["evaluate", "--integer-model", str(binary),
                                     "--backend", "numpy", "--manifest", str(manifest),
                                     "--output", str(report), "--audio-dir", str(tmp_path / "audio")])
    main()
    result = json.loads(report.read_text())
    assert result["summary"]["utterances"] == 2
    assert result["summary"]["valid_utterances"] == 2
    assert result["summary"]["weighting"] == "equal per utterance"
    assert result["summary"]["si_sdri"] == pytest.approx(0, abs=1e-5)
    assert result["summary"]["si_sdri"] == pytest.approx(float_result["summary"]["si_sdri"], abs=1e-5)
    assert result["timing"]["audio_seconds"] == (701 + 1533) / 16000
    assert "not ESP32" in result["timing"]["measurement"]
    assert result["timing"]["processing_rtf"] > 0
    assert len(list((tmp_path / "audio").glob("*.wav"))) == 6
    assert len(result["model"]["model_sha256"]) == 64


@pytest.mark.parametrize("corruption", ("short_length", "long_length", "audio_hash"))
def test_full_evaluation_rejects_stale_audio_before_inference(tmp_path, corruption):
    manifest = _manifest(tmp_path)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    if corruption == "audio_hash":
        records[0]["clean_sha256"] = "0" * 64
    else:
        records[0]["samples"] += -1 if corruption == "short_length" else 1
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))

    def unexpected_inference(_):
        raise AssertionError("Corrupt audio metadata must fail before inference")

    with pytest.raises(ValueError, match="length|SHA256"):
        evaluate_manifest(unexpected_inference, manifest)


def test_report_binds_complete_audio_and_exposes_gain_and_polarity(tmp_path):
    manifest = _manifest(tmp_path)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in records:
        for role in ("clean", "noisy"):
            row[role + "_sha256"] = hashlib.sha256((tmp_path / row[role]).read_bytes()).hexdigest()
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    reference = evaluate_manifest(lambda noisy: noisy, manifest)
    attenuated = evaluate_manifest(lambda noisy: -0.25 * noisy, manifest)
    assert attenuated["summary"]["si_sdri"] == pytest.approx(0, abs=1e-10)
    actual = attenuated["preservation"]["enhanced"]
    original = reference["preservation"]["enhanced"]
    assert actual["projection_gain"]["mean"] == pytest.approx(-0.25 * original["projection_gain"]["mean"])
    assert actual["rms_ratio"]["mean"] == pytest.approx(0.25 * original["rms_ratio"]["mean"])
    assert actual["normalized_waveform_l1"]["mean"] > original["normalized_waveform_l1"]["mean"]
    assert attenuated["source_verification"] == {
        "manifest_unchanged": True, "whole_file_lengths_verified": True,
        "audio_files_hashed": 4, "declared_audio_hashes_verified": 4}
    assert attenuated["utterances"][0]["audio_sha256"]["noisy"] == records[0]["noisy_sha256"]


def test_manifest_mutation_cannot_relabel_an_evaluation(tmp_path):
    manifest = _manifest(tmp_path)

    def mutate_manifest(noisy):
        manifest.write_text(manifest.read_text() + "\n")
        return noisy

    with pytest.raises(ValueError, match="manifest changed during"):
        evaluate_manifest(mutate_manifest, manifest)


def test_checkpoint_hash_is_of_loaded_bytes_not_replaced_path(tmp_path, monkeypatch):
    model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2)))
    checkpoint = tmp_path / "float.pt"
    torch.save({"model": model.state_dict(), "model_config": asdict(model.config), "phase": "float"}, checkpoint)
    original = checkpoint.read_bytes()
    load = torch.load

    def replace_during_load(source, **kwargs):
        checkpoint.write_bytes(b"concurrent replacement")
        return load(source, **kwargs)

    monkeypatch.setattr(torch, "load", replace_during_load)
    _, provenance = load_checkpoint(checkpoint)
    assert provenance["model_sha256"] == hashlib.sha256(original).hexdigest()
