"""Metric-only BLAS limits must preserve results and the caller's execution settings."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from threadpoolctl import ThreadpoolController
import torch

import esp32_denoiser.evaluate as evaluation


def _manifest(tmp_path):
    time = np.arange(48_013, dtype=np.float64) / 16000
    clean = (.1 * np.sin(2 * np.pi * 430 * time)).astype(np.float32)
    noisy = (clean + .035 * np.sin(2 * np.pi * 1600 * time)).astype(np.float32)
    for role, audio in (("clean", clean), ("noisy", noisy)):
        sf.write(tmp_path / f"{role}.wav", audio, 16000, subtype="FLOAT")
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(json.dumps(dict(id="p226_001", speaker="p226", sample_rate=16000,
                                       samples=len(clean), source_split="train",
                                       clean="clean.wav", noisy="noisy.wav")) + "\n")
    return manifest


def _thread_settings(controller):
    return {library["filepath"]: (library["user_api"], library["num_threads"])
            for library in controller.info()}


def test_metric_limits_preserve_scores_and_restore_inference_settings(tmp_path, monkeypatch):
    controller = ThreadpoolController()
    before = _thread_settings(controller)
    calls = {"inference": 0, "perceptual": 0, "preservation": 0}
    expected_limited = False

    def check_scope():
        current = _thread_settings(controller)
        for path, (api, threads) in before.items():
            assert current[path] == (api, 1 if expected_limited and api == "blas" else threads)

    class Enhancer(torch.nn.Module):
        def forward(self, audio):
            assert _thread_settings(controller) == before
            calls["inference"] += 1
            return .7 * audio

    original_preservation = evaluation.preservation_metrics

    def preservation(*args):
        check_scope()
        calls["preservation"] += 1
        return original_preservation(*args)

    def metric(reference, degraded):
        check_scope()
        calls["perceptual"] += 1
        # Uses actual BLAS reduction, while avoiding optional PESQ data/dependencies.
        return float(np.dot(reference, degraded) / np.dot(reference, reference))

    packages = {"pesq": SimpleNamespace(pesq=lambda rate, ref, deg, mode: metric(ref, deg),
                                       BufferTooShortError=RuntimeError, NoUtterancesError=RuntimeError),
                "pystoi": SimpleNamespace(stoi=lambda ref, deg, rate, extended: metric(ref, deg))}
    monkeypatch.setattr("esp32_denoiser.quality.import_module", packages.__getitem__)
    monkeypatch.setattr(evaluation, "preservation_metrics", preservation)
    enhancer = Enhancer()
    manifest = _manifest(tmp_path)
    # Compare against the original unlimited execution of exactly the same metrics.
    monkeypatch.setattr(evaluation, "ThreadpoolController",
                        lambda: SimpleNamespace(limit=lambda **kwargs: nullcontext()))
    reference = evaluation.evaluate_manifest(enhancer, manifest, perceptual=True)
    expected_limited = True
    monkeypatch.setattr(evaluation, "ThreadpoolController", ThreadpoolController)
    limited = evaluation.evaluate_manifest(enhancer, manifest, perceptual=True)
    assert calls == {"inference": 2, "perceptual": 8, "preservation": 2}
    assert enhancer.training and _thread_settings(controller) == before
    for key in ("si_sdr_noisy", "si_sdr_enhanced", "si_sdri"):
        assert limited["summary"][key] == pytest.approx(reference["summary"][key], abs=1e-12)
    first, second = reference["utterances"][0], limited["utterances"][0]
    for role in ("noisy", "enhanced"):
        assert second["preservation"][role] == pytest.approx(first["preservation"][role], abs=1e-12)
        for name in ("pesq_wb", "stoi"):
            assert second[f"{name}_{role}"] == pytest.approx(first[f"{name}_{role}"], abs=1e-12)
    assert limited["metric_execution"]["blas_threads"] == 1


def test_metric_failure_restores_thread_limits_and_model_mode(tmp_path, monkeypatch):
    controller = ThreadpoolController()
    before = _thread_settings(controller)
    enhancer = torch.nn.Identity().train()

    def broken(*args):
        for library in controller.info():
            if library["user_api"] == "blas":
                assert library["num_threads"] == 1
        raise ValueError("intentional metric failure")

    monkeypatch.setattr(evaluation, "preservation_metrics", broken)
    with pytest.raises(ValueError, match="intentional metric failure"):
        evaluation.evaluate_manifest(enhancer, _manifest(tmp_path))
    assert enhancer.training and _thread_settings(controller) == before
