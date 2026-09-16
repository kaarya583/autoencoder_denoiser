"""Check optional metric plumbing without bundling or downloading speech."""

import json
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import soundfile as sf
import torch

from esp32_denoiser.evaluate import evaluate_manifest
from esp32_denoiser.quality import PerceptualMetrics, summarize_perceptual


class TooShort(RuntimeError):
    pass


class NoSpeech(RuntimeError):
    pass


def _packages(monkeypatch, pesq, stoi):
    packages = {"pesq": SimpleNamespace(pesq=pesq, BufferTooShortError=TooShort,
                                       NoUtterancesError=NoSpeech),
                "pystoi": SimpleNamespace(stoi=stoi)}
    monkeypatch.setattr("esp32_denoiser.quality.import_module", packages.__getitem__)


def _manifest(tmp_path):
    audio = np.sin(np.arange(8000) * 0.1).astype(np.float32) * 0.1
    for role in ("clean", "noisy"):
        sf.write(tmp_path / f"{role}.wav", audio, 16000, subtype="FLOAT")
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(json.dumps({"id": "p232_001", "speaker": "p232", "sample_rate": 16000,
                                   "samples": len(audio), "source_split": "test",
                                   "clean": "clean.wav", "noisy": "noisy.wav"}) + "\n")
    return manifest


def test_missing_optional_dependency_fails_before_model_inference(tmp_path, monkeypatch):
    def unavailable(name):
        raise ImportError(name)
    monkeypatch.setattr("esp32_denoiser.quality.import_module", unavailable)
    calls = []
    with pytest.raises(RuntimeError, match="pip install pesq==0.0.4 pystoi==0.4.1"):
        evaluate_manifest(lambda audio: calls.append(audio), _manifest(tmp_path), perceptual=True)
    assert calls == []
    assert evaluate_manifest(torch.nn.Identity(), _manifest(tmp_path))["summary"]["valid_utterances"] == 1


def test_same_gain_and_alignment_no_clipping_and_standard_metric_arguments(monkeypatch):
    calls = []
    def pesq(rate, reference, degraded, mode):
        assert (rate, mode) == (16000, "wb")
        calls.append((reference.copy(), degraded.copy()))
        return 2.0
    def stoi(reference, degraded, rate, extended):
        assert (rate, extended) == (16000, False)
        calls.append((reference.copy(), degraded.copy()))
        return 0.8
    _packages(monkeypatch, pesq, stoi)
    clean = np.array([2.0, -2.0, 1.0, -1.0])
    noisy, enhanced = clean * 0.5, clean * 2.0
    result = PerceptualMetrics()(clean, noisy, enhanced)
    assert result["perceptual_common_gain"] == 0.25
    for reference, degraded in calls[:2]:
        np.testing.assert_array_equal(reference, clean * 0.25)
        np.testing.assert_array_equal(degraded, noisy * 0.25)
    for reference, degraded in calls[2:]:
        np.testing.assert_array_equal(reference, clean * 0.25)
        np.testing.assert_array_equal(degraded, enhanced * 0.25)
    np.testing.assert_array_equal(clean, [2.0, -2.0, 1.0, -1.0])
    assert result["perceptual_errors"] == {}


def test_silence_short_speech_warnings_and_programming_failures_are_explicit(monkeypatch):
    def pesq(*args):
        raise TooShort("short clip")
    def stoi(*args, **kwargs):
        warnings.warn("Not enough STFT frames", RuntimeWarning)
        return 1e-5
    _packages(monkeypatch, pesq, stoi)
    quality = PerceptualMetrics()
    silent = quality(np.zeros(10), np.ones(10), np.zeros(10))
    assert set(silent["perceptual_errors"].values()) == {"silent_reference"}
    short = quality(np.array([0., 0.1, -0.1]), np.zeros(3), np.zeros(3))
    assert short["pesq_wb_noisy"] is None and short["stoi_enhanced"] is None
    assert short["perceptual_errors"]["pesq_wb_noisy"] == "TooShort"
    assert "Not enough" in short["perceptual_errors"]["stoi_noisy"]
    with pytest.raises(ValueError, match="finite"):
        quality([1, np.nan], [1, 2], [1, 2])
    def broken(*args):
        raise MemoryError("allocation failed")
    quality.pesq = broken
    with pytest.raises(MemoryError):
        quality([0.1, -0.1], [0.1, 0.1], [0.2, -0.1])


def test_paired_equal_utterance_averages_do_not_skip_only_bad_enhancements():
    rows = [
        {"samples": 10, "pesq_wb_noisy": 1., "pesq_wb_enhanced": 2., "stoi_noisy": .2, "stoi_enhanced": .8},
        {"samples": 100000, "pesq_wb_noisy": 3., "pesq_wb_enhanced": 4., "stoi_noisy": .6, "stoi_enhanced": .4},
        {"samples": 20, "pesq_wb_noisy": 5., "pesq_wb_enhanced": None, "stoi_noisy": .8, "stoi_enhanced": None},
    ]
    summary = summarize_perceptual(rows)
    assert summary["pesq_wb"] == {"valid_utterances": 2, "invalid_utterances": 1,
                                   "noisy": 2., "enhanced": 3., "improvement": 1.}
    assert summary["stoi"]["improvement"] == pytest.approx(.2)
    assert summarize_perceptual([])["pesq_wb"]["enhanced"] is None


def test_perceptual_evaluation_reuses_each_enhanced_waveform_once(tmp_path, monkeypatch):
    _packages(monkeypatch, lambda *args: 4., lambda *args, **kwargs: 1.)
    calls = []
    def enhance(audio):
        calls.append(audio.shape)
        return audio
    result = evaluate_manifest(enhance, _manifest(tmp_path), perceptual=True)
    assert len(calls) == 1
    assert result["utterances"][0]["pesq_wb_enhanced"] == 4.
    assert result["perceptual"]["summary"]["stoi"]["improvement"] == 0.
    assert result["perceptual"]["implementation"]["selection_metric"] is False
