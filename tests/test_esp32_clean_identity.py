"""Clean-identity exposure, reproducible resumes and truthful teacher recipes."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from esp32_denoiser.data import PairedAudioDataset
from esp32_denoiser.distillation import FrozenTeacher, checkpoint_clean_identity_probability
from esp32_denoiser.extra_data import DynamicMixtureDataset
from esp32_denoiser.train import TrainConfig, train
from test_esp32_distillation import broad_teacher_files, teacher_files
from test_esp32_hybrid import hybrid_config


@pytest.mark.parametrize("value", [-.01, 1.01, float("nan"), float("inf"), True, None, "0.15"])
def test_invalid_probability_fails_before_io(tmp_path, value):
    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="clean_identity_probability"):
        train(TrainConfig("missing-train", "missing-val", str(output), clean_identity_probability=value))
    assert not output.exists()


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_both_actual_training_datasets_obey_probability_and_validation_stays_noisy(hybrid_config, monkeypatch, probability):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    counts = {"paired": 0, "synthetic": 0, "validation": 0}
    paired_get, synthetic_get = PairedAudioDataset.__getitem__, DynamicMixtureDataset.__getitem__

    def paired(dataset, index):
        result = paired_get(dataset, index)
        if dataset.random_crop:
            counts["paired"] += 1
            assert dataset.clean_identity_prob == probability
            assert torch.equal(result["noisy"], result["clean"]) == bool(probability)
        else:
            counts["validation"] += 1
            assert dataset.clean_identity_prob == 0 and not torch.equal(result["noisy"], result["clean"])
        return result

    def synthetic(dataset, index):
        result = synthetic_get(dataset, index)
        counts["synthetic"] += 1
        assert dataset.clean_identity_prob == probability
        assert result["mixture"]["clean_identity"] == bool(probability)
        assert torch.equal(result["noisy"], result["clean"]) == bool(probability)
        return result

    monkeypatch.setattr(PairedAudioDataset, "__getitem__", paired)
    monkeypatch.setattr(DynamicMixtureDataset, "__getitem__", synthetic)
    config = replace(hybrid_config, clean_identity_probability=probability, epoch_samples=32, max_steps_per_epoch=8)
    train(config)
    assert min(counts.values()) > 0
    directory = Path(config.output_dir)
    assert json.loads((directory / "config.json").read_text())["clean_identity_probability"] == probability
    assert json.loads((directory / "provenance.json").read_text())["clean_identity_probability"] == probability
    checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint_clean_identity_probability(checkpoint["train_config"], checkpoint["provenance"]) == probability


def test_old_checkpoint_default_resumes_identically_and_changed_exposure_requires_new_optimizer(hybrid_config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert hybrid_config.clean_identity_probability == .03
    train(hybrid_config)
    directory = Path(hybrid_config.output_dir)
    original_path = directory / "last.pt"
    original = torch.load(original_path, map_location="cpu", weights_only=False)
    legacy = dict(original, train_config=dict(original["train_config"]), provenance=dict(original["provenance"]))
    legacy["train_config"].pop("clean_identity_probability")
    legacy["provenance"].pop("clean_identity_probability")
    legacy_path = directory / "legacy.pt"
    torch.save(legacy, legacy_path)
    recorded_config = (directory / "config.json").read_bytes()
    with pytest.raises(ValueError, match="requires resume_optimizer=False"):
        train(replace(hybrid_config, epochs=2, resume=str(original_path), clean_identity_probability=.15))
    assert (directory / "config.json").read_bytes() == recorded_config
    for name, path in (("modern", original_path), ("legacy", legacy_path)):
        train(replace(hybrid_config, epochs=2, resume=str(path), output_dir=str(directory / name)))
    modern = torch.load(directory / "modern/last.pt", map_location="cpu", weights_only=False)
    old = torch.load(directory / "legacy/last.pt", map_location="cpu", weights_only=False)
    for name, value in modern["model"].items():
        torch.testing.assert_close(value, old["model"][name], rtol=0, atol=0)
    assert old["train_config"]["clean_identity_probability"] == old["provenance"]["clean_identity_probability"] == .03
    train(replace(hybrid_config, resume=str(original_path), resume_optimizer=False,
                  clean_identity_probability=.15, output_dir=str(directory / "new_recipe")))
    changed = torch.load(directory / "new_recipe/last.pt", map_location="cpu", weights_only=False)
    assert changed["epoch"] == 1 and changed["provenance"]["clean_identity_probability"] == .15


def test_teacher_provenance_reports_nondefault_identity_for_paired_and_broad_sources(broad_teacher_files):
    paths, checkpoint = broad_teacher_files
    old = FrozenTeacher(paths["checkpoint"], paths["val"])
    assert old.provenance["clean_identity_probability"] == .03
    assert old.provenance["added_training_sources"]["clean_identity_probability"] == .03
    checkpoint["train_config"]["clean_identity_probability"] = .15
    checkpoint["provenance"]["clean_identity_probability"] = .15
    torch.save(checkpoint, paths["checkpoint"])
    teacher = FrozenTeacher(paths["checkpoint"], paths["val"])
    assert teacher.provenance["clean_identity_probability"] == .15
    assert teacher.provenance["added_training_sources"]["clean_identity_probability"] == .15


@pytest.mark.parametrize("mutation", ["missing_provenance", "mismatch", "invalid", "removed_config"])
def test_teacher_identity_recipe_must_be_consistent(teacher_files, mutation):
    paths, checkpoint = teacher_files
    checkpoint["train_config"]["clean_identity_probability"] = .15
    checkpoint["provenance"]["clean_identity_probability"] = .15
    if mutation == "missing_provenance":
        checkpoint["provenance"].pop("clean_identity_probability")
    elif mutation == "mismatch":
        checkpoint["provenance"]["clean_identity_probability"] = .03
    elif mutation == "invalid":
        checkpoint["train_config"]["clean_identity_probability"] = float("nan")
    else:
        checkpoint["train_config"].pop("clean_identity_probability")
    torch.save(checkpoint, paths["checkpoint"])
    with pytest.raises(ValueError, match="clean_identity_probability"):
        FrozenTeacher(paths["checkpoint"], paths["val"])


def test_real_nondefault_teacher_checkpoint_passes_strict_broad_lineage(broad_teacher_files, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    paths, _ = broad_teacher_files
    output = paths["checkpoint"].parent / "identity_teacher"
    config = TrainConfig(str(paths["train"]), str(paths["val"]), str(output),
                         clean_identity_probability=.15, epochs=1, batch_size=1, eval_batch_size=1,
                         crop_seconds=.048, width=4, dilations=(1, 2), workers=0, amp=False, max_hours=.01,
                         synthetic_speech_manifest=str(paths["speech_train"]),
                         synthetic_noise_manifest=str(paths["noise_train"]),
                         synthetic_probability=.5, epoch_samples=2)
    train(config)
    teacher = FrozenTeacher(output / "best.pt", paths["val"])
    assert teacher.provenance["added_training_sources"]["clean_identity_probability"] == .15
    assert teacher.provenance["clean_identity_probability"] == .15


def test_clean_exposure_arm_differs_only_in_probability_and_destination():
    root = Path(__file__).resolve().parents[1] / "configs"
    control = json.loads((root / "esp32_zero_bias_broad_distillation_control_float.json").read_text())
    changed = json.loads((root / "esp32_zero_bias_broad_clean_float.json").read_text())
    assert changed.pop("clean_identity_probability") == .15
    assert changed.pop("output_dir") == "/content/esp32_runs/float_zero_bias_broad_clean"
    control.pop("output_dir")
    assert changed == control
    assert changed["resume"] == "/content/esp32_runs/broad_kd_inputs/student.pt"
    assert changed["epochs"] == changed["patience"] == 40
    assert changed["learning_rate"] == 1e-4 and changed["distillation_weight"] == 0
