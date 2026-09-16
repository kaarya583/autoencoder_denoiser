"""Verified baseline conversion and fresh-optimizer continuation invariants."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch

from esp32_denoiser.evaluate import load_checkpoint
from esp32_denoiser.frequency_model import FrequencyUNet, FrequencyUNetConfig
from esp32_denoiser.train import TrainConfig, train
from esp32_denoiser.warm_start import prepare_deep_filter_warm_start
from test_frequency_factory import _pilot_data


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _source(root, *, batch_norm=True, phase="float", kind="frequency_unet"):
    torch.manual_seed(85)
    model = FrequencyUNet(FrequencyUNetConfig(encoder_channels=(2, 3, 4), global_width=4,
        local_dilations=(1, 2), global_dilations=(1, 2), encoder_batch_norm=batch_norm))
    with torch.no_grad():
        model.head.weight.normal_(std=0.2)
        model.head.bias.normal_(std=0.05)
        for _ in range(3):
            model(torch.randn(2, 1027) * 0.1)
    payload = {"model": model.state_dict(), "model_config": asdict(model.config),
               "phase": phase, "model_kind": kind, "epoch": 17, "best_si_sdri": 7.1,
               "optimizer": {"must_not_be_copied": True},
               "provenance": {"manifest_sha256": {"train": "training-hash", "val": "validation-hash"},
                              "test_used_for_selection": False}}
    path = root / "source.pt"
    torch.save(payload, path)
    return path, model.eval(), payload


def test_conversion_preserves_waveforms_common_weights_and_provenance(tmp_path):
    source, baseline, source_payload = _source(tmp_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "initializations" / "frequency_deep_filter.pt"
    rng_before = torch.random.get_rng_state().clone()
    report = prepare_deep_filter_warm_start(source, destination, expected_source_sha256=digest)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, atol=0, rtol=0)
    converted, metadata = load_checkpoint(destination)
    assert metadata["model_kind"] == "frequency_deep_filter"
    payload = torch.load(destination, weights_only=False)
    assert payload["initialization_only"] is True and payload["epoch"] == 0
    assert payload["phase"] == "float"
    assert not {"optimizer", "scheduler", "scaler", "best_si_sdri"} & payload.keys()
    provenance = payload["provenance"]
    assert provenance["source_checkpoint_sha256"] == digest
    assert provenance["source_checkpoint_epoch"] == 17
    assert provenance["source_recorded_best_si_sdri"] == 7.1
    assert provenance["source_provenance"] == source_payload["provenance"]
    assert provenance["validation_performed"] is False
    assert provenance["waveform_parity"]["exact_float32_waveform_equality"] is True
    assert report["initialization_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert report["resume_optimizer"] is False
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(converted.state_dict()[key], value, atol=0, rtol=0)
    assert not torch.count_nonzero(converted.filter_head.weight)
    assert not torch.count_nonzero(converted.filter_head.bias)
    audio = torch.randn(2, 2311) * 0.1
    with torch.inference_mode():
        torch.testing.assert_close(converted(audio), baseline(audio), atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        prepare_deep_filter_warm_start(source, destination)


@pytest.mark.parametrize("options,expected", [
    ({"phase": "qat"}, "float checkpoint"),
    ({"kind": "spectral_tcn"}, "frequency_unet"),
    ({"batch_norm": False}, "BatchNorm baseline"),
])
def test_wrong_source_rejected_before_destination_creation(tmp_path, options, expected):
    source, _, _ = _source(tmp_path, **options)
    destination = tmp_path / "uncreated" / "initialization.pt"
    with pytest.raises(ValueError, match=expected):
        prepare_deep_filter_warm_start(source, destination)
    assert not destination.parent.exists()


def test_source_hash_is_checked_before_and_after_conversion(tmp_path, monkeypatch):
    source, _, _ = _source(tmp_path)
    destination = tmp_path / "uncreated" / "initialization.pt"
    with pytest.raises(ValueError, match="expected frozen source"):
        prepare_deep_filter_warm_start(source, destination, expected_source_sha256="0" * 64)
    original_read = Path.read_bytes
    reads = 0

    def changing_read(path):
        nonlocal reads
        data = original_read(path)
        if path == source:
            reads += 1
            if reads == 2:
                return data + b"changed-after-load"
        return data

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    with pytest.raises(ValueError, match="changed during conversion"):
        prepare_deep_filter_warm_start(source, destination)
    assert reads == 2
    assert not destination.parent.exists()


def test_initialization_checkpoint_requires_fresh_optimizer_and_trains(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    source, _, _ = _source(tmp_path)
    destination = tmp_path / "initialization.pt"
    report = prepare_deep_filter_warm_start(source, destination)
    paths = _pilot_data(tmp_path)
    config = TrainConfig(train_manifest=paths["train"], val_manifest=paths["val"],
        output_dir=str(tmp_path / "run"), model_kind="frequency_deep_filter",
        model_options=report["model_config"], resume=str(destination), epochs=1,
        batch_size=2, crop_seconds=0.048, workers=0, eval_batch_size=2,
        max_steps_per_epoch=1, max_hours=0.01, amp=False)
    with pytest.raises(ValueError, match="resume_optimizer=False"):
        train(config)
    config.resume_optimizer = False
    result = train(config)
    assert result["epoch"] == 1
    trained = torch.load(Path(config.output_dir) / "last.pt", weights_only=False)
    assert trained["model_kind"] == "frequency_deep_filter"
    assert trained["provenance"]["resume_checkpoint_sha256"] == report["initialization_sha256"]
    assert torch.count_nonzero(trained["model"]["filter_head.weight"]) > 0


def test_broader_continuation_configs_are_matched_except_architecture_and_paths():
    root = Path(__file__).resolve().parents[1] / "configs"
    baseline = json.loads((root / "esp32_frequency_bn_broad_float.json").read_text())
    filtered = json.loads((root / "esp32_frequency_deep_filter_broad_float.json").read_text())
    assert baseline["resume"].endswith("initializations/frequency_bn_base.pt")
    assert filtered["resume"].endswith("initializations/frequency_deep_filter.pt")
    for config in (baseline, filtered):
        assert config["resume_optimizer"] is False
        assert config["learning_rate"] == 0.0002
        # A lower primary score after broadening the data must not stop either
        # matched arm before its planned training budget is exhausted.
        assert config["epochs"] > 0 and config["patience"] >= config["epochs"]
        assert config["synthetic_probability"] == 0.5 and config["epoch_samples"] == 10802
        for key in ("model_kind", "output_dir", "resume"):
            config.pop(key)
    assert baseline == filtered
