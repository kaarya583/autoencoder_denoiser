"""Teacher lineage checks and detached, padding-aware response supervision."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from esp32_denoiser.distillation import FrozenTeacher, calibrate_distillation_weight, response_distillation_loss, teacher_quality_gate
from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig
from esp32_denoiser.train import TrainConfig, train


@pytest.fixture
def teacher_files(tmp_path):
    paths = {}
    for split, speaker in (("train", "p225"), ("val", "p226")):
        clean = (.1 * np.sin(np.arange(800) * .13)).astype(np.float32)
        record = {"id": speaker + "_001", "speaker": speaker, "sample_rate": 16000,
                  "samples": len(clean), "source_split": "train"}
        for role, audio in (("clean", clean), ("noisy", clean + .03 * np.sin(np.arange(800) * .41))):
            path = tmp_path / f"{speaker}_{role}.wav"
            sf.write(path, audio, 16000, subtype="FLOAT")
            record[role] = str(path)
        paths[split] = tmp_path / f"{split}.jsonl"
        paths[split].write_text(json.dumps(record) + "\n")
    model = SpectralTCN(SpectralTCNConfig(width=4, dilations=(1, 2)))
    provenance = {"test_used_for_selection": False, "model_kind": "spectral_tcn",
                  "train_utterances": 1, "validation_utterances": 1,
                  "train_speakers": ["p225"], "validation_speakers": ["p226"],
                  "manifest_sha256": {split: hashlib.sha256(path.read_bytes()).hexdigest() for split, path in paths.items()}}
    checkpoint = {"model_kind": "spectral_tcn", "model_config": asdict(model.config), "model": model.state_dict(),
                  "phase": "float", "epoch": 1, "provenance": provenance,
                  "train_config": {"train_manifest": str(paths["train"]), "val_manifest": str(paths["val"]), "resume": None}}
    paths["checkpoint"] = tmp_path / "teacher.pt"
    torch.save(checkpoint, paths["checkpoint"])
    return paths, checkpoint


def test_fresh_teacher_is_frozen_finite_and_preserves_input_shape(teacher_files, monkeypatch):
    paths, _ = teacher_files
    teacher = FrozenTeacher(paths["checkpoint"], paths["val"], "cpu")
    assert not any(parameter.requires_grad for parameter in teacher.model.parameters())
    assert teacher.provenance["checkpoint_sha256"] == hashlib.sha256(paths["checkpoint"].read_bytes()).hexdigest()
    teacher.model.train()  # predict must always reassert eval mode.
    audio = torch.randn(2, 768, requires_grad=True) * .1
    result = teacher.predict(audio)
    assert result.shape == audio.shape and torch.isfinite(result).all()
    assert result.dtype == torch.float32 and not result.requires_grad and not teacher.model.training
    student = result.clone().requires_grad_()
    response_distillation_loss(student, result, torch.tensor([768, 650])).backward()
    assert student.grad is not None and all(parameter.grad is None for parameter in teacher.model.parameters())
    monkeypatch.setattr(teacher.model, "forward", lambda value: torch.full_like(value, float("nan")))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        teacher.predict(audio)
    monkeypatch.setattr(teacher.model, "forward", lambda value: value[:, :-1])
    with pytest.raises(ValueError, match="shape"):
        teacher.predict(audio)


@pytest.mark.parametrize("mutation,reason", [
    ("resume", "fresh"), ("test_selection", "test-based"),
    ("unknown_kind", "model_kind"), ("synthetic", "both configured"),
])
def test_old_or_unaudited_teacher_lineage_is_rejected(teacher_files, mutation, reason):
    paths, checkpoint = teacher_files
    if mutation == "resume":
        checkpoint["train_config"]["resume"] = "old-test-selected.pt"
    elif mutation == "test_selection":
        checkpoint["provenance"]["test_used_for_selection"] = True
    elif mutation == "unknown_kind":
        checkpoint["model_kind"] = "gated_old_unet"
    else:
        checkpoint["train_config"]["synthetic_speech_manifest"] = "extra.jsonl"
    torch.save(checkpoint, paths["checkpoint"])
    with pytest.raises(ValueError, match=reason):
        FrozenTeacher(paths["checkpoint"], paths["val"], "cpu")


def test_teacher_rejects_selection_leakage_and_changed_manifests(teacher_files):
    paths, _ = teacher_files
    # Teacher selection == student selection is allowed; teacher training is not.
    with pytest.raises(ValueError, match="student selection speaker"):
        FrozenTeacher(paths["checkpoint"], paths["train"], "cpu")
    paths["train"].write_text(paths["train"].read_text() + "\n")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        FrozenTeacher(paths["checkpoint"], paths["val"], "cpu")


def test_teacher_selection_must_come_from_official_training_origin(teacher_files):
    paths, _ = teacher_files
    row = json.loads(paths["val"].read_text())
    row["source_split"] = "test"
    paths["val"].write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="teacher selection.*training-origin"):
        FrozenTeacher(paths["checkpoint"], paths["val"], "cpu")


@pytest.fixture
def broad_teacher_files(teacher_files):
    from esp32_denoiser.extra_data import SOURCES, _asset_identity
    paths, checkpoint = teacher_files
    directory = paths["checkpoint"].parent / "extra/manifests"
    directory.mkdir(parents=True)
    preparation = {"version": 1, "sample_rate": 16000, "splits": {}}
    for kind in ("speech", "noise"):
        for split, index in (("train", 101), ("val", 202)):
            member = (f"LibriSpeech/train-clean-100/{index}/10/{index}-10-0000.flac" if kind == "speech" else
                      f"musan/noise/free-sound/noise-{index}.wav")
            identifier, group, speaker = _asset_identity(kind, member)
            path = directory / f"{kind}_{split}{'.flac' if kind == 'speech' else '.wav'}"
            audio = np.sin(np.arange(640) * (index / 1000 + (kind == "noise"))) * .1
            sf.write(path, audio, 16000)
            row = {"id": identifier, "group": group, "speaker": speaker, "kind": kind, "split": split,
                   "source": SOURCES[kind].source, "license": SOURCES[kind].license, "member": member,
                   "path": path.name, "samples": 640, "sample_rate": 16000,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            key = f"{kind}_{split}"
            paths[key] = directory / f"{key}.jsonl"
            paths[key].write_text(json.dumps(row) + "\n")
            preparation["splits"][key] = {"manifest_sha256": hashlib.sha256(paths[key].read_bytes()).hexdigest(),
                                           "records": 1, "groups": [group]}
    paths["preparation"] = directory / "provenance.json"
    paths["preparation"].write_text(json.dumps(preparation))
    checkpoint["train_config"].update(synthetic_speech_manifest=str(paths["speech_train"]),
                                      synthetic_noise_manifest=str(paths["noise_train"]),
                                      synthetic_probability=.5, crop_seconds=3.0, epoch_samples=2)
    checkpoint["provenance"]["added_training_sources"] = {
        "speech_manifest_sha256": preparation["splits"]["speech_train"]["manifest_sha256"],
        "noise_manifest_sha256": preparation["splits"]["noise_train"]["manifest_sha256"],
        "speech_recordings": 1, "noise_recordings": 1,
        "synthetic_probability": .5, "samples_per_epoch": 2}
    source = Path(__file__).resolve().parents[1] / "esp32_denoiser"
    checkpoint["provenance"]["source_sha256"] = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("extra_data.py", "mixtures.py", "train.py")}
    torch.save(checkpoint, paths["checkpoint"])
    return paths, checkpoint


def _replace_source_manifest(paths, key, row):
    """Keep preparation consistent so independent overlap checks are exercised."""
    paths[key].write_text(json.dumps(row) + "\n")
    preparation = json.loads(paths["preparation"].read_text())
    preparation["splits"][key] = {"manifest_sha256": hashlib.sha256(paths[key].read_bytes()).hexdigest(),
                                  "records": 1, "groups": [row["group"]]}
    paths["preparation"].write_text(json.dumps(preparation))


def test_verified_fresh_broad_teacher_records_exact_partitions_and_stays_frozen(broad_teacher_files):
    paths, _ = broad_teacher_files
    teacher = FrozenTeacher(paths["checkpoint"], paths["val"])
    added = teacher.provenance["added_training_sources"]
    assert "audited LibriSpeech/MUSAN" in teacher.provenance["lineage"]
    assert added["synthetic_probability"] == .5 and added["samples_per_epoch"] == 2
    assert added["snr_db"] == [-5, 20] and added["peak_limit"] == .99
    for kind in ("speech", "noise"):
        for split in ("train", "val"):
            entry = added["partitions"][kind][split]
            assert entry["manifest"] == str(paths[f"{kind}_{split}"].resolve())
            assert entry["manifest_sha256"] == hashlib.sha256(paths[f"{kind}_{split}"].read_bytes()).hexdigest()
            assert entry["recordings"] == 1
    assert "202" in teacher.validation_speakers  # Extra held-out speakers cannot calibrate KD.
    prediction = teacher.predict(torch.randn(1, 800) * .1)
    assert prediction.shape == (1, 800) and torch.isfinite(prediction).all()
    assert not prediction.requires_grad and not any(p.requires_grad for p in teacher.model.parameters())
    assert "not rehashed" in added["verification_scope"]
    assert added["source_code"]["verified_against_current_files"] == ["extra_data.py", "mixtures.py"]


def test_actual_hybrid_training_checkpoint_passes_broad_teacher_audit(broad_teacher_files):
    paths, _ = broad_teacher_files
    output = paths["checkpoint"].parent / "fresh_broad_run"
    config = TrainConfig(str(paths["train"]), str(paths["val"]), str(output),
                         epochs=1, batch_size=1, eval_batch_size=1, crop_seconds=.048,
                         width=4, dilations=(1, 2), workers=0, amp=False, max_hours=.01,
                         synthetic_speech_manifest=str(paths["speech_train"]),
                         synthetic_noise_manifest=str(paths["noise_train"]),
                         synthetic_probability=.5, epoch_samples=2)
    train(config)
    teacher = FrozenTeacher(output / "best.pt", paths["val"])
    added = teacher.provenance["added_training_sources"]
    assert added["samples_per_epoch"] == 2 and added["crop_seconds"] == .048
    assert teacher.provenance["recorded_manifest_hashes_verified"]


@pytest.mark.parametrize("mutation,reason", [
    ("train_hash", "manifest hash/count mismatch"), ("count", "manifest hash/count mismatch"),
    ("val_hash", "preparation provenance mismatch"), ("val_partition", "val partition"),
    ("missing_added", "complete known added_training_sources"), ("missing_preparation", "Missing adjacent"),
    ("recipe", "Unknown synthetic training recipe"), ("sampling", "sampling recipe differs"),
    ("missing_paired_hash", "complete paired manifest hash"),
    ("missing_source_hash", "lacks recorded source SHA256"),
    ("changed_mixer", "Unknown synthetic mixer source hash"),
])
def test_broad_teacher_requires_complete_untampered_source_lineage(broad_teacher_files, mutation, reason):
    paths, checkpoint = broad_teacher_files
    if mutation == "train_hash":
        paths["speech_train"].write_text(paths["speech_train"].read_text() + "\n")
    elif mutation == "count":
        checkpoint["provenance"]["added_training_sources"]["speech_recordings"] = 2
    elif mutation == "val_hash":
        paths["noise_val"].write_text(paths["noise_val"].read_text() + "\n")
    elif mutation == "val_partition":
        row = json.loads(paths["speech_val"].read_text())
        row["split"] = "train"
        _replace_source_manifest(paths, "speech_val", row)
    elif mutation == "missing_added":
        checkpoint["provenance"].pop("added_training_sources")
    elif mutation == "missing_preparation":
        paths["preparation"].unlink()
    elif mutation == "recipe":
        checkpoint["train_config"]["synthetic_snr_db"] = [10, 30]
    elif mutation == "sampling":
        checkpoint["train_config"]["synthetic_probability"] = .75
    elif mutation == "missing_paired_hash":
        checkpoint["provenance"]["manifest_sha256"].pop("train")
    elif mutation == "missing_source_hash":
        checkpoint["provenance"]["source_sha256"].pop("extra_data.py")
    else:
        checkpoint["provenance"]["source_sha256"]["mixtures.py"] = "0" * 64
    torch.save(checkpoint, paths["checkpoint"])
    with pytest.raises(ValueError, match=reason):
        FrozenTeacher(paths["checkpoint"], paths["val"])


@pytest.mark.parametrize("overlap", ["id", "group", "path", "sha256"])
def test_broad_teacher_rejects_cross_partition_overlap_even_with_consistent_sidecar(broad_teacher_files, overlap):
    from esp32_denoiser.extra_data import _asset_identity
    paths, _ = broad_teacher_files
    training = json.loads(paths["speech_train"].read_text())
    validation = json.loads(paths["speech_val"].read_text())
    if overlap == "id":
        validation.update({key: training[key] for key in ("id", "group", "speaker", "member")})
    elif overlap == "group":
        validation["member"] = "LibriSpeech/train-clean-100/101/10/101-10-0001.flac"
        validation["id"], validation["group"], validation["speaker"] = _asset_identity("speech", validation["member"])
    else:
        validation[overlap] = training[overlap]
    _replace_source_manifest(paths, "speech_val", validation)
    with pytest.raises(ValueError, match=f"{overlap} overlap"):
        FrozenTeacher(paths["checkpoint"], paths["val"])


def test_legacy_teacher_requires_matching_sidecar_counts_and_speakers(teacher_files):
    paths, checkpoint = teacher_files
    sidecar = paths["checkpoint"].parent / "provenance.json"
    provenance = checkpoint.pop("provenance")
    checkpoint.pop("model_kind")
    torch.save(checkpoint, paths["checkpoint"])
    with pytest.raises(ValueError, match="adjacent provenance"):
        FrozenTeacher(paths["checkpoint"], paths["val"], "cpu")
    # Original sidecars need not contain hashes/kind, but their observed source
    # counts and speaker identities are mandatory and must be exact.
    provenance.pop("manifest_sha256")
    provenance.pop("model_kind")
    sidecar.write_text(json.dumps(provenance))
    teacher = FrozenTeacher(paths["checkpoint"], paths["val"], "cpu")
    assert teacher.provenance["model_kind"] == "spectral_tcn"
    assert not teacher.provenance["recorded_manifest_hashes_verified"]
    assert teacher.provenance["legacy_sidecar_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    provenance["train_utterances"] = 2
    sidecar.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="train_utterances"):
        FrozenTeacher(paths["checkpoint"], paths["val"], "cpu")
    provenance["train_utterances"] = 1
    provenance["train_speakers"] = ["p999"]
    sidecar.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="train_speakers"):
        FrozenTeacher(paths["checkpoint"], paths["val"], "cpu")


def test_response_loss_detaches_teacher_and_ignores_padding():
    generator = torch.Generator().manual_seed(912)
    student = (torch.randn(2, 1200, generator=generator) * .1).requires_grad_()
    teacher = (torch.randn(2, 1200, generator=generator) * .1).requires_grad_()
    lengths = torch.tensor([700, 1200])
    loss = response_distillation_loss(student, teacher, lengths)
    expected = torch.stack([response_distillation_loss(student[i:i+1, :length], teacher[i:i+1, :length],
                                                      torch.tensor([length])) for i, length in enumerate(lengths)]).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    loss.backward()
    assert torch.isfinite(student.grad).all() and torch.count_nonzero(student.grad[0, 700:]) == 0
    assert teacher.grad is None
    altered = teacher.detach().clone()
    altered[0, 700:] = float("nan")
    torch.testing.assert_close(response_distillation_loss(student, altered, lengths), loss)


def test_optional_quality_gate_and_empty_cohort_are_explicit():
    generator = torch.Generator().manual_seed(142)
    clean = torch.randn(2, 1200, generator=generator) * .1
    noise = torch.randn(2, 1200, generator=generator) * .1
    noisy = clean + noise
    teacher = torch.stack((clean[0], noise[1]))
    lengths = torch.tensor([1200, 1200])
    gate = teacher_quality_gate(teacher, clean, noisy, lengths, minimum_improvement_db=1)
    assert gate.tolist() == [True, False]
    student = noisy.clone().requires_grad_()
    gated = response_distillation_loss(student, teacher, lengths, example_mask=gate)
    single = response_distillation_loss(student[:1], teacher[:1], lengths[:1])
    torch.testing.assert_close(gated, single)
    empty = response_distillation_loss(student, teacher, lengths, example_mask=torch.tensor([False, False]))
    assert empty.item() == 0 and empty.requires_grad
    empty.backward()
    assert torch.count_nonzero(student.grad) == 0


@pytest.mark.parametrize("settings,reason", [
    ({"distillation_weight": .1}, "configured together"),
    ({"teacher_checkpoint": "teacher.pt"}, "configured together"),
    ({"distillation_weight": float("nan")}, "finite"),
    ({"distillation_gate_minimum_improvement_db": 0}, "gate requires"),
])
def test_training_rejects_incomplete_distillation_configs(tmp_path, settings, reason):
    config = TrainConfig("unused-train", "unused-val", str(tmp_path / "output"), **settings)
    with pytest.raises(ValueError, match=reason):
        train(config)
    assert not (tmp_path / "output").exists()


def _student_config(paths, **settings):
    return TrainConfig(str(paths["train"]), str(paths["val"]), str(paths["checkpoint"].parent / "student"),
                       epochs=1, batch_size=1, eval_batch_size=1, crop_seconds=.048,
                       learning_rate=1e-4, max_steps_per_epoch=1, width=4, dilations=(1, 2),
                       workers=0, amp=False, max_hours=.01, teacher_checkpoint=str(paths["checkpoint"]),
                       distillation_weight=.1, **settings)


@pytest.mark.parametrize("gate_margin", [None, 1000.0])
def test_training_uses_frozen_teacher_only_for_training_batches(teacher_files, monkeypatch, gate_margin):
    from esp32_denoiser import distillation
    paths, _ = teacher_files
    captured = []

    class TrackedTeacher(FrozenTeacher):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.original_weights = {key: value.clone() for key, value in self.model.state_dict().items()}
            self.calls = 0
            captured.append(self)

        def predict(self, noisy):
            self.calls += 1
            prediction = super().predict(noisy)
            assert not prediction.requires_grad and prediction.dtype == torch.float32
            return prediction

    monkeypatch.setattr(distillation, "FrozenTeacher", TrackedTeacher)
    config = _student_config(paths, distillation_gate_minimum_improvement_db=gate_margin)
    train(config)
    assert len(captured) == 1 and captured[0].calls == 1  # Initial/final validation call only the student.
    teacher = captured[0]
    for key, value in teacher.model.state_dict().items():
        torch.testing.assert_close(value, teacher.original_weights[key], rtol=0, atol=0)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.model.parameters())
    checkpoint = torch.load(paths["checkpoint"].parent / "student/last.pt", weights_only=False)
    assert checkpoint["provenance"]["distillation"]["teacher"]["checkpoint_sha256"] == teacher.provenance["checkpoint_sha256"]
    history = json.loads((paths["checkpoint"].parent / "student/history.jsonl").read_text())
    assert np.isfinite(history["loss"]) and np.isfinite(history["distillation_loss"])
    assert history["distillation_total_utterances"] == 1
    assert history["distillation_selected_utterances"] == (1 if gate_margin is None else 0)
    assert history["weighted_distillation_loss"] == pytest.approx(.1 * history["distillation_loss"])
    if gate_margin is not None:
        assert history["distillation_loss"] == 0
    # Explicit no-KD control does not instantiate or invoke another teacher.
    train(replace(config, output_dir=str(paths["checkpoint"].parent / "control"),
                  teacher_checkpoint=None, distillation_weight=0, distillation_gate_minimum_improvement_db=None))
    assert len(captured) == 1 and captured[0].calls == 1


def test_distillation_resume_enforces_settings_and_teacher_hash(teacher_files):
    paths, teacher_checkpoint = teacher_files
    config = _student_config(paths)
    train(config)
    student_path = paths["checkpoint"].parent / "student/last.pt"
    resumed = replace(config, epochs=2, resume=str(student_path))
    with pytest.raises(ValueError, match="resume_optimizer=False"):
        train(replace(resumed, distillation_weight=.2))
    result = train(resumed)
    assert result["epoch"] == 2
    teacher_checkpoint["model"]["input_proj.weight"] += .001
    torch.save(teacher_checkpoint, paths["checkpoint"])
    with pytest.raises(ValueError, match="Teacher lineage changed"):
        train(replace(resumed, epochs=3))


def test_training_rejects_test_selected_teacher_before_optimization(teacher_files):
    paths, checkpoint = teacher_files
    checkpoint["provenance"]["test_used_for_selection"] = True
    torch.save(checkpoint, paths["checkpoint"])
    config = _student_config(paths)
    with pytest.raises(ValueError, match="test-based"):
        train(config)
    assert not (paths["checkpoint"].parent / "student/history.jsonl").exists()


class ScaleWaveform(torch.nn.Module):
    def __init__(self, gain=.5):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(gain))
        self.fixed_dropout = torch.nn.Dropout(.5).eval()

    def forward(self, noisy):
        return self.fixed_dropout(noisy) * self.gain


def _calibration_batch():
    noisy = torch.randn(2, 1100, generator=torch.Generator().manual_seed(211)) * .1
    return {"noisy": noisy, "clean": noisy * .9, "length": torch.tensor([1100, 1100]),
            "id": ["p225_001", "p225_002"], "speaker": ["p225", "p225"]}


def test_kd_calibration_balances_gradients_and_preserves_modes_and_grad_buffers(teacher_files, monkeypatch):
    paths, _ = teacher_files
    teacher = FrozenTeacher(paths["checkpoint"], paths["val"], "cpu")
    monkeypatch.setattr(teacher.model, "forward", lambda noisy: noisy * .8)
    teacher_parameter = next(teacher.model.parameters())
    teacher_parameter.grad = torch.ones_like(teacher_parameter)
    student = ScaleWaveform()
    student.gain.grad = torch.tensor(3.)
    batch = _calibration_batch()
    consumed = []

    def batches():
        for index in range(5):
            consumed.append(index)
            yield batch

    def primary(estimate, clean, lengths):
        return (estimate - clean).square().mean()

    result = calibrate_distillation_weight(student, teacher, batches(), primary, target_ratio=.1, max_batches=2)
    assert consumed == [0, 1] and result["accepted_batches"] == result["attempted_batches"] == 2
    assert result["distillation_weight"] > 0 and np.isfinite(result["distillation_weight"])
    for measurement in result["batches"]:
        assert result["distillation_weight"] * measurement["distillation_norm"] / measurement["primary_norm"] == pytest.approx(.1)
        assert measurement["gradient_cosine"] == pytest.approx(1)
    assert student.training and not student.fixed_dropout.training
    assert student.gain.grad.item() == 3
    assert torch.equal(teacher_parameter.grad, torch.ones_like(teacher_parameter))
    assert all(not parameter.requires_grad for parameter in teacher.model.parameters())
    scaled = calibrate_distillation_weight(student, teacher, [batch],
                                           lambda estimate, clean, lengths: 10 * primary(estimate, clean, lengths))
    assert scaled["distillation_weight"] == pytest.approx(10 * result["distillation_weight"])
    assert scaled["teacher_checkpoint_sha256"] == teacher.provenance["checkpoint_sha256"]


def test_kd_calibration_rejects_validation_batches_and_restores_mode_on_failure(teacher_files, monkeypatch):
    paths, _ = teacher_files
    teacher = FrozenTeacher(paths["checkpoint"], paths["val"], "cpu")
    monkeypatch.setattr(teacher.model, "forward", lambda noisy: noisy * .8)
    student = ScaleWaveform()
    batch = _calibration_batch()
    batch["id"][0] = "p226_001"
    with pytest.raises(ValueError, match="current validation"):
        calibrate_distillation_weight(student, teacher, [batch], lambda estimate, clean, lengths: estimate.sum())
    assert student.training and not student.fixed_dropout.training
    assert student.gain.grad is None
    batch = _calibration_batch()
    monkeypatch.setattr(teacher.model, "forward", lambda noisy: torch.zeros_like(noisy))
    with torch.no_grad():
        student.gain.zero_()  # Zero student/teacher spectra have no KD gradient.
    with pytest.raises(ValueError, match="No finite nonzero"):
        calibrate_distillation_weight(student, teacher, [batch],
                                       lambda estimate, clean, lengths: (estimate - clean).square().mean())
    assert student.training and not student.fixed_dropout.training
    assert student.gain.grad is None
