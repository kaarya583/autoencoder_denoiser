"""Training-only fixed teacher gain, provenance, and KD resume integration."""
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from esp32_denoiser.distillation import FrozenTeacher, calibrate_distillation_weight
from esp32_denoiser.teacher_gain import calibrate_teacher_output_gain
from esp32_denoiser.train import TrainConfig, train
from test_esp32_distillation import teacher_files, broad_teacher_files, _student_config, _calibration_batch, ScaleWaveform


def write_calibration(paths, result):
    path = paths["checkpoint"].parent / "teacher_gain.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return path


def test_training_clean_fit_corrects_known_gain_and_preserves_rng_and_weights(broad_teacher_files, monkeypatch):
    paths, _ = broad_teacher_files
    teacher = FrozenTeacher(paths["checkpoint"], paths["val"])
    weights = {name: value.clone() for name, value in teacher.model.state_dict().items()}
    monkeypatch.setattr(teacher.model, "forward", lambda waveform: waveform * 3)
    rng = torch.get_rng_state().clone()
    result = calibrate_teacher_output_gain(teacher, examples_per_source=8, crop_seconds=.04, batch_size=2, seed=19)
    assert result["output_gain"] == pytest.approx(1/3, abs=1e-7)
    assert result["fit"]["valid_examples"] == 2  # Distinct available recordings, not padded to 16 examples.
    assert result["diagnostics"]["raw_mean_projection_gain"] == pytest.approx(3)
    assert result["diagnostics"]["corrected_mean_projection_gain"] == pytest.approx(1)
    assert abs(result["diagnostics"]["normalized_mse_after"]) < 1e-12
    assert {row["source"] for row in result["selected_examples"]} == {"paired_train", "speech_train"}
    assert {row["speaker"] for row in result["selected_examples"]} == {"p225", "101"}
    assert teacher.output_gain == 1 and teacher.provenance["output_gain_calibration"] is None
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in teacher.model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    repeated = calibrate_teacher_output_gain(teacher, examples_per_source=8, crop_seconds=.04, batch_size=2, seed=19)
    assert repeated == result
    path = write_calibration(paths, result)
    corrected = FrozenTeacher(paths["checkpoint"], paths["val"], gain_calibration=path)
    monkeypatch.setattr(corrected.model, "forward", lambda waveform: waveform * 3)
    waveform = torch.randn(2, 700) * .1
    torch.testing.assert_close(corrected.predict(waveform), waveform, rtol=1e-6, atol=1e-7)
    assert corrected.provenance["output_gain_calibration"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not corrected.predict(waveform).requires_grad
    with pytest.raises(ValueError, match="uncorrected"):
        calibrate_teacher_output_gain(corrected)


@pytest.mark.parametrize("mutation,reason", [
    ("checkpoint", "identity mismatch"), ("manifest", "identity mismatch"),
    ("sources", "training sources mismatch"), ("heldout", "unaudited training crop"),
    ("fit", "least-squares fit"), ("audio", "source audio changed"),
])
def test_gain_calibration_rejects_changed_identity_or_unapproved_crops(teacher_files, mutation, reason):
    paths, _ = teacher_files
    teacher = FrozenTeacher(paths["checkpoint"], paths["val"])
    result = calibrate_teacher_output_gain(teacher, examples_per_source=1)
    if mutation == "checkpoint":
        result["teacher_checkpoint_sha256"] = "0" * 64
    elif mutation == "manifest":
        result["teacher_manifest_sha256"]["train"] = "0" * 64
    elif mutation == "sources":
        result["training_sources"]["paired_train"]["manifest"] = str(paths["val"])
    elif mutation == "heldout":
        result["selected_examples"][0]["id"] = "p226_001"
    elif mutation == "fit":
        result["output_gain"] *= 2
    else:
        result["selected_examples"][0]["audio_sha256"] = "0" * 64
    path = write_calibration(paths, result)
    with pytest.raises(ValueError, match=reason):
        FrozenTeacher(paths["checkpoint"], paths["val"], gain_calibration=path)


def test_calibration_refuses_changed_training_manifest_or_zero_teacher(teacher_files, monkeypatch):
    paths, _ = teacher_files
    teacher = FrozenTeacher(paths["checkpoint"], paths["val"])
    monkeypatch.setattr(teacher.model, "forward", torch.zeros_like)
    with pytest.raises(ValueError, match="nonzero teacher responses"):
        calibrate_teacher_output_gain(teacher)
    paths["train"].write_text(paths["train"].read_text() + "\n")
    with pytest.raises(ValueError, match="manifest changed"):
        calibrate_teacher_output_gain(teacher)


def test_gain_metadata_reaches_kd_calibration_and_resume_rejects_changed_file(teacher_files, monkeypatch):
    paths, _ = teacher_files
    teacher = FrozenTeacher(paths["checkpoint"], paths["val"])
    gain = calibrate_teacher_output_gain(teacher, examples_per_source=1)
    path = write_calibration(paths, gain)
    corrected = FrozenTeacher(paths["checkpoint"], paths["val"], gain_calibration=path)
    monkeypatch.setattr(corrected.model, "forward", lambda waveform: waveform * .8)
    coefficient = calibrate_distillation_weight(
        ScaleWaveform(), corrected, [_calibration_batch()],
        lambda estimate, clean, lengths: (estimate-clean).square().mean())
    assert coefficient["teacher_output_gain"] == corrected.output_gain
    assert coefficient["teacher_gain_calibration_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    config = _student_config(paths, teacher_gain_calibration=str(path))
    train(config)
    checkpoint_path = paths["checkpoint"].parent / "student/last.pt"
    saved = torch.load(checkpoint_path, weights_only=False)
    assert saved["provenance"]["distillation"]["teacher"]["output_gain_calibration"]["sha256"] == coefficient["teacher_gain_calibration_sha256"]
    path.write_text(path.read_text() + "\n")  # Same gain/JSON, different immutable calibration identity.
    with pytest.raises(ValueError, match="gain calibration changed"):
        train(replace(config, resume=str(checkpoint_path), epochs=2))


def test_gain_calibration_requires_active_teacher_before_training_io(tmp_path):
    config = TrainConfig("missing_train", "missing_val", str(tmp_path / "run"), teacher_gain_calibration="gain.json")
    with pytest.raises(ValueError, match="requires an active teacher"):
        train(config)
    assert not (tmp_path / "run").exists()
