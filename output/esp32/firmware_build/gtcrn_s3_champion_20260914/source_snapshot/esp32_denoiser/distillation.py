"""Strict fresh-teacher lineage and detached response distillation.

Controlled KD accepts a newly trained paired-only teacher or the repository's
verified paired-plus-LibriSpeech/MUSAN recipe. No improvement is assumed.
Teacher quality gating is an optional training-only ablation, disabled unless
the caller explicitly supplies its mask to response_distillation_loss.
"""

from __future__ import annotations

import hashlib
from itertools import islice
import json
import math
from pathlib import Path
from statistics import median

import torch

from .data import read_manifest
from .evaluate import load_checkpoint
from .losses import compressed_spectral_loss
from .metrics import si_sdr


KNOWN_KINDS = {"spectral_tcn", "frequency_unet"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_origin(path, label):
    records = read_manifest(path)
    if any(record.get("source_split") != "train" for record in records):
        raise ValueError(f"{label} must contain only official training-origin utterances")
    if any(not isinstance(record.get("speaker"), str) or not record["speaker"] or
           record.get("sample_rate") != 16000 or record.get("samples", 0) < 1 for record in records):
        raise ValueError(f"Invalid 16 kHz paired speech metadata in {label}")
    return records


def _disjoint(training, validation, label):
    for key in ("speaker", "id"):
        if {row[key] for row in training} & {row[key] for row in validation}:
            raise ValueError(f"Teacher training overlaps {label} {key}")
    train_paths = {row[role] for row in training for role in ("clean", "noisy")}
    val_paths = {row[role] for row in validation for role in ("clean", "noisy")}
    if train_paths & val_paths:
        raise ValueError(f"Teacher training overlaps {label} audio paths")


def checkpoint_clean_identity_probability(config, provenance):
    """Read the saved training recipe, including the historical 3% default.

    New checkpoints explicitly record the value in configuration and training
    provenance. Require agreement before reproducing calibration mixtures or
    describing a teacher. An old checkpoint omitting the field retains .03.
    """
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError("Unknown checkpoint clean-identity recipe metadata")
    value = config.get("clean_identity_probability", .03)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Invalid checkpoint clean_identity_probability")
    if "clean_identity_probability" in config and "clean_identity_probability" not in provenance:
        raise ValueError("Checkpoint clean_identity_probability lacks matching provenance")
    recorded = provenance.get("clean_identity_probability", .03)
    if (isinstance(recorded, bool) or not isinstance(recorded, (int, float)) or
            not math.isfinite(recorded) or recorded != value):
        raise ValueError("Checkpoint clean_identity_probability differs from provenance")
    return float(value)


def _audit_synthetic_sources(config, provenance, paired_count, selection_sets):
    """Verify the fixed hybrid recipe and preparation-time source partitions.

    This verifies manifest hashes, not every multi-GB source waveform again.
    Extra validation hashes were not embedded by the original trainer: they
    are checked against adjacent preparation provenance and recorded now.
    """
    from dataclasses import fields
    from .extra_data import VERSION, SAMPLE_RATE, read_extra_manifest
    from .train import TrainConfig

    if set(config) - {field.name for field in fields(TrainConfig)}:
        raise ValueError("Unknown synthetic training recipe fields")
    clean_identity_probability = checkpoint_clean_identity_probability(config, provenance)
    source_hashes = provenance.get("source_sha256")
    recorded_source_hashes = {}
    for filename in ("extra_data.py", "mixtures.py", "train.py"):
        digest = source_hashes.get(filename) if isinstance(source_hashes, dict) else None
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"Broad teacher lacks recorded source SHA256: {filename}")
        if filename != "train.py" and digest != _sha256(Path(__file__).parent / filename):
            raise ValueError(f"Unknown synthetic mixer source hash: {filename}")
        recorded_source_hashes[filename] = digest
    added = provenance.get("added_training_sources")
    expected_keys = {"speech_manifest_sha256", "noise_manifest_sha256", "speech_recordings",
                     "noise_recordings", "synthetic_probability", "samples_per_epoch"}
    if not isinstance(added, dict) or set(added) != expected_keys:
        raise ValueError("Broad teacher needs complete known added_training_sources provenance")
    probability, crop = config.get("synthetic_probability"), config.get("crop_seconds")
    if (isinstance(probability, bool) or not isinstance(probability, (int, float)) or
            not math.isfinite(probability) or not 0 < probability <= 1 or
            isinstance(crop, bool) or not isinstance(crop, (int, float)) or not math.isfinite(crop) or crop <= 0):
        raise ValueError("Unknown or invalid synthetic probability/crop recipe")
    epoch_samples = config.get("epoch_samples")
    epoch_samples = paired_count if epoch_samples is None else epoch_samples
    if isinstance(epoch_samples, bool) or not isinstance(epoch_samples, int) or epoch_samples < 1:
        raise ValueError("Invalid synthetic epoch_samples recipe")
    if added["synthetic_probability"] != probability or added["samples_per_epoch"] != epoch_samples:
        raise ValueError("Synthetic sampling recipe differs from teacher provenance")
    partitions, partition_provenance = {}, {}
    for kind in ("speech", "noise"):
        train_path = Path(config[f"synthetic_{kind}_manifest"]).resolve()
        paths = {"train": train_path, "val": train_path.with_name(f"{kind}_val.jsonl")}
        preparation_path = train_path.parent / "provenance.json"
        if not preparation_path.is_file() or not paths["val"].is_file():
            raise ValueError(f"Missing adjacent {kind} validation/preparation provenance")
        preparation = json.loads(preparation_path.read_text())
        if preparation.get("version") != VERSION or preparation.get("sample_rate") != SAMPLE_RATE:
            raise ValueError("Unknown synthetic preparation version or sample rate")
        partitions[kind] = {}
        partition_provenance[kind] = {"preparation_provenance": str(preparation_path),
                                       "preparation_provenance_sha256": _sha256(preparation_path)}
        for split, path in paths.items():
            rows = read_extra_manifest(path, kind)
            if {row["split"] for row in rows} != {split}:
                raise ValueError(f"Incorrect teacher synthetic {kind} {split} partition")
            if any(not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64 or
                   any(character not in "0123456789abcdef" for character in row["sha256"]) for row in rows):
                raise ValueError("Synthetic source content hashes must be valid SHA256 values")
            digest = _sha256(path)
            if split == "train" and (added[f"{kind}_manifest_sha256"] != digest or
                                     added[f"{kind}_recordings"] != len(rows)):
                raise ValueError(f"Teacher synthetic {kind} manifest hash/count mismatch")
            details = preparation.get("splits", {}).get(f"{kind}_{split}")
            groups = sorted({row["group"] for row in rows})
            if (not isinstance(details, dict) or details.get("manifest_sha256") != digest or
                    details.get("records") != len(rows) or details.get("groups") != groups):
                raise ValueError(f"Synthetic {kind} {split} preparation provenance mismatch")
            partitions[kind][split] = rows
            partition_provenance[kind][split] = {"manifest": str(path), "manifest_sha256": digest,
                                                 "recordings": len(rows), "groups": groups}
    training = [row for kind in partitions.values() for row in kind["train"]]
    validation = [row for kind in partitions.values() for row in kind["val"]]
    for key in ("id", "group", "path", "sha256"):
        if {row[key] for row in training} & {row[key] for row in validation}:
            raise ValueError(f"Teacher synthetic train/validation {key} overlap")
    for label, selection in selection_sets:
        if ({row["speaker"] for row in training if row["speaker"] is not None} &
                {row["speaker"] for row in selection}):
            raise ValueError(f"Teacher synthetic training overlaps {label} speakers")
        if {row["id"] for row in training} & {row["id"] for row in selection}:
            raise ValueError(f"Teacher synthetic training overlaps {label} IDs")
        if {row["path"] for row in training} & {row[role] for row in selection for role in ("clean", "noisy")}:
            raise ValueError(f"Teacher synthetic training overlaps {label} audio paths")
    return {"recipe": "repository DynamicMixtureDataset + HybridTrainingDataset",
            "source_manifests": {kind: config[f"synthetic_{kind}_manifest"] for kind in ("speech", "noise")},
            "synthetic_probability": probability, "samples_per_epoch": epoch_samples,
            "crop_seconds": crop, "snr_db": [-5, 20], "gain_db": [-6, 6],
            "clean_identity_probability": clean_identity_probability, "peak_limit": 0.99,
            "validation_speakers": sorted({row["speaker"] for row in validation if row["speaker"] is not None}),
            "partitions": partition_provenance,
            "source_code": {"recorded_sha256": recorded_source_hashes,
                            "verified_against_current_files": ["extra_data.py", "mixtures.py"],
                            "train_py_note": "Recorded for immutable-source-bundle review; not required to equal current train.py because unrelated loss/guard edits change its hash."},
            "verification_scope": "Training manifests match checkpoint hashes/counts; both partitions match adjacent preparation provenance. Source audio bytes are not rehashed here; extra validation hashes are audited now, not embedded training-time commitments."}


class FrozenTeacher:
    """Load a fresh teacher, audit paired and admitted synthetic lineage, freeze.

``validation_manifest`` is the student's training-origin selection set. Teacher
training must be disjoint from both teacher selection and student selection;
the two selection sets may coincide. Recorded manifest hashes are checked when
present. The current strict track requires explicit train_config.resume=None
and rejects resumed or upstream-teacher lineage. Synthetic training is admitted
only for the repository's fixed, source-audited LibriSpeech/MUSAN hybrid recipe.

Modern checkpoints require an explicit recognized model_kind and embedded
provenance. A legacy SpectralTCN lacking both may use adjacent provenance.json,
whose train/selection counts and speaker lists must match the actual manifests.
Neither format may omit the explicit declaration that test was not selected on.
Paths in train_config must remain readable in this environment. Use only trusted
local checkpoint files: the repository's loader uses PyTorch pickle checkpoints.
"""

    def __init__(self, checkpoint: str | Path, validation_manifest: str | Path, device="cpu", *,
                 gain_calibration: str | Path | None = None):
        checkpoint_path = Path(checkpoint).resolve()
        checkpoint_hash = _sha256(checkpoint_path)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or not isinstance(saved.get("train_config"), dict):
            raise ValueError("Teacher checkpoint has unknown training lineage")
        config = saved["train_config"]
        if "resume" not in config or config["resume"] is not None:
            raise ValueError("Strict teacher lineage requires an explicit fresh train_config.resume=None")
        unsupported = ("teacher_checkpoint", "teacher_gain_calibration", "distillation_teacher", "teacher_model", "pretrained_checkpoint")
        if any(config.get(key) for key in unsupported):
            raise ValueError("Strict teacher track rejects upstream teacher/pretrained lineage")
        if config.get("distillation_weight", 0) != 0 or config.get("distillation_gate_minimum_improvement_db") is not None:
            raise ValueError("Strict teacher track rejects upstream distillation configuration")
        synthetic_paths = [config.get(f"synthetic_{kind}_manifest") for kind in ("speech", "noise")]
        broad = any(synthetic_paths)
        if broad and any(not isinstance(path, str) or not path for path in synthetic_paths):
            raise ValueError("Broad teacher requires both configured synthetic speech and noise manifests")
        if saved.get("phase") != "float":
            raise ValueError("Strict response distillation requires a floating-point teacher checkpoint")
        sidecar_path = None
        kind = saved.get("model_kind")
        if kind is None:
            if saved.get("provenance") is not None:
                raise ValueError("Teacher model kind is missing from a nonlegacy checkpoint")
            kind = "spectral_tcn"
            sidecar_path = checkpoint_path.parent / "provenance.json"
            if not sidecar_path.is_file():
                raise ValueError("Legacy teacher requires adjacent provenance.json")
            provenance = json.loads(sidecar_path.read_text())
            required_config = {"width", "dilations", "sample_rate", "n_fft", "hop_length"}
            if not isinstance(saved.get("model_config"), dict) or not required_config <= saved["model_config"].keys():
                raise ValueError("Unrecognized legacy SpectralTCN configuration")
        else:
            if kind not in KNOWN_KINDS:
                raise ValueError(f"Unknown teacher model_kind: {kind}")
            provenance = saved.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("test_used_for_selection") is not False:
            raise ValueError("Teacher provenance must explicitly exclude test-based selection")
        clean_identity_probability = checkpoint_clean_identity_probability(config, provenance)
        if provenance.get("model_kind", kind) != kind:
            raise ValueError("Teacher architecture conflicts with its provenance")
        if provenance.get("resume_checkpoint_sha256") or provenance.get("distillation") or saved.get("initialization_only"):
            raise ValueError("Teacher provenance includes resumed, upstream, or initialization-only weights")
        if not broad and provenance.get("added_training_sources"):
            raise ValueError("Teacher provenance includes unconfigured added sources")
        if broad and sidecar_path is not None:
            raise ValueError("Broad teacher requires modern embedded training provenance")
        for key in ("train_manifest", "val_manifest"):
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError(f"Missing teacher lineage field: {key}")
        teacher_train_path, teacher_val_path = Path(config["train_manifest"]).resolve(), Path(config["val_manifest"]).resolve()
        current_val_path = Path(validation_manifest).resolve()
        teacher_train = _training_origin(teacher_train_path, "teacher training")
        teacher_val = _training_origin(teacher_val_path, "teacher selection")
        current_val = _training_origin(current_val_path, "student selection")
        _disjoint(teacher_train, teacher_val, "teacher selection")
        _disjoint(teacher_train, current_val, "student selection")
        hashes = {"train": _sha256(teacher_train_path), "val": _sha256(teacher_val_path)}
        recorded_hashes = provenance.get("manifest_sha256")
        if broad and (not isinstance(recorded_hashes, dict) or not hashes.keys() <= recorded_hashes.keys()):
            raise ValueError("Broad teacher requires complete paired manifest hash provenance")
        if recorded_hashes is not None:
            if not isinstance(recorded_hashes, dict):
                raise ValueError("Invalid teacher manifest hash provenance")
            for key, observed in hashes.items():
                if key in recorded_hashes and recorded_hashes[key] != observed:
                    raise ValueError(f"Teacher {key} manifest hash mismatch")
        observed_counts = {"train_utterances": len(teacher_train), "validation_utterances": len(teacher_val),
                           "train_speakers": sorted({row["speaker"] for row in teacher_train}),
                           "validation_speakers": sorted({row["speaker"] for row in teacher_val})}
        for key, value in observed_counts.items():
            if sidecar_path is not None and key not in provenance:
                raise ValueError(f"Legacy teacher sidecar lacks {key}")
            if key in provenance and provenance[key] != value:
                raise ValueError(f"Teacher provenance {key} does not match actual manifests")
        synthetic_provenance = _audit_synthetic_sources(
            config, provenance, len(teacher_train),
            (("teacher selection", teacher_val), ("student selection", current_val))) if broad else None
        self.model, model_metadata = load_checkpoint(checkpoint_path, device=device)
        # The loader performs strict state/config matching. Avoid trusting the
        # fallback name alone when accepting the narrowly defined legacy format.
        if model_metadata["model_kind"] != kind:
            raise ValueError("Loaded teacher architecture differs from audited lineage")
        if _sha256(checkpoint_path) != checkpoint_hash:
            raise ValueError("Teacher checkpoint changed while it was being audited")
        self.model.float().requires_grad_(False).eval()
        self.device = next(self.model.parameters()).device
        self.validation_ids = frozenset(row["id"] for row in current_val)
        self.validation_speakers = frozenset(row["speaker"] for row in current_val) | frozenset(
            synthetic_provenance["validation_speakers"] if synthetic_provenance else ())
        self.provenance = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
                           "lineage": ("fresh paired + audited LibriSpeech/MUSAN teacher" if broad else "fresh paired-only teacher") + "; train-origin selection; no test selection",
                           "added_training_sources": synthetic_provenance,
                           "clean_identity_probability": clean_identity_probability,
                           "legacy_sidecar": str(sidecar_path) if sidecar_path else None,
                           "legacy_sidecar_sha256": _sha256(sidecar_path) if sidecar_path else None,
                           "teacher_manifests": {"train": str(teacher_train_path), "val": str(teacher_val_path)},
                           "teacher_manifest_sha256": hashes,
                           "student_validation_manifest_sha256": _sha256(current_val_path),
                           "recorded_manifest_hashes_verified": isinstance(recorded_hashes, dict) and hashes.keys() <= recorded_hashes.keys(),
                           **observed_counts, **model_metadata}
        self.output_gain = 1.0
        self.provenance["output_gain"] = 1.0
        self.provenance["output_gain_calibration"] = None
        if gain_calibration is not None:
            from .teacher_gain import load_teacher_gain_calibration
            self.output_gain, correction = load_teacher_gain_calibration(self, gain_calibration)
            self.provenance.update(output_gain=self.output_gain, output_gain_calibration=correction)

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        """Return detached float32 teacher audio with the input's shape/device.

Known repository waveform models reset their local streaming state per forward.
Evaluation mode is reasserted each call, and outer AMP cannot alter this target.
The caller must put its input on the configured teacher device explicitly.
"""
        if not isinstance(noisy, torch.Tensor) or noisy.ndim != 2 or min(noisy.shape) < 1:
            raise ValueError("Teacher expects nonempty waveform tensors [batch,samples]")
        if not noisy.is_floating_point() or noisy.device != self.device or not torch.isfinite(noisy).all():
            raise ValueError("Teacher input must be finite floating audio on its configured device")
        self.model.eval()
        with torch.autocast(device_type=self.device.type, enabled=False):
            result = self.model(noisy.float())
        if not isinstance(result, torch.Tensor) or result.shape != noisy.shape or result.device != noisy.device:
            raise ValueError("Teacher changed waveform shape or device")
        if self.output_gain != 1.0:
            result = result * self.output_gain
        if not torch.isfinite(result).all():
            raise FloatingPointError("Teacher returned nonfinite audio")
        return result.detach()


def _aligned(student, teacher, lengths):
    if (not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor) or
            student.ndim != 2 or min(student.shape) < 1 or student.shape != teacher.shape or
            student.device != teacher.device or not student.is_floating_point() or not teacher.is_floating_point()):
        raise ValueError("Distillation expects matching floating [batch,samples] tensors on one device")
    lengths = torch.as_tensor(lengths, device=student.device)
    if lengths.shape != (len(student),) or lengths.is_floating_point() or lengths.dtype == torch.bool or bool(((lengths < 1) | (lengths > student.shape[-1])).any()):
        raise ValueError("Distillation requires one valid integer length per waveform")
    valid = torch.arange(student.shape[-1], device=student.device)[None] < lengths[:, None]
    student = torch.where(valid, student.float(), 0)
    teacher = torch.where(valid, teacher.detach().float(), 0)
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise FloatingPointError("Nonfinite valid waveform in distillation loss")
    return student, teacher, lengths


def response_distillation_loss(student_output: torch.Tensor, teacher_output: torch.Tensor,
                               lengths: torch.Tensor, *, example_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compressed spectral response loss, averaged equally over selected clips.

Teacher targets are always detached. No gating is applied by default. Optional
example_mask selects a training-only ablation cohort; an empty cohort returns a
differentiable zero without updating the teacher. External loss weight belongs
to the experiment configuration, keeping a zero-weight/no-KD control explicit.
"""
    student, teacher, lengths = _aligned(student_output, teacher_output, lengths)
    if example_mask is not None:
        example_mask = torch.as_tensor(example_mask, device=student.device)
        if example_mask.shape != (len(student),) or example_mask.dtype != torch.bool:
            raise ValueError("example_mask must contain one boolean per waveform")
        if not example_mask.any():
            return student.sum() * 0
        student, teacher, lengths = student[example_mask], teacher[example_mask], lengths[example_mask]
    return compressed_spectral_loss(student, teacher, lengths)


def calibrate_distillation_weight(model, teacher: FrozenTeacher, batches, primary_loss, *,
                                  target_ratio: float = 0.1, max_batches: int = 8) -> dict:
    """Choose a fixed KD coefficient from training-only parameter gradients.

For each training batch, measure the L2 norm of the student's primary-loss
gradient and its ungated response-KD gradient. The suggested coefficient is
the median of target_ratio * primary_norm / distillation_norm. This balances
local optimization scales; it does not establish an optimal weight or quality
gain. Diagnostics include the gradient cosine and achieved ratios for this
coefficient so conflicting objectives and batch variation remain visible.

Use the same primary objective and training augmentation as the planned run.
``batches`` uses pad_collate's noisy/clean/length dictionaries. Supplied IDs or
speakers are checked against the teacher-audited current validation set; the
caller must otherwise ensure these are training batches. Teacher targets are
detached FP32. autograd.grad preserves all parameter .grad buffers; the student
temporarily enters eval mode and each submodule's prior mode is restored.
"""
    if not math.isfinite(target_ratio) or target_ratio <= 0:
        raise ValueError("target_ratio must be positive and finite")
    if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1:
        raise ValueError("max_batches must be a positive integer")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("Calibration requires trainable student parameters")
    device = parameters[0].device
    if teacher.device != device:
        raise ValueError("Teacher and student must share the calibration device")
    prior_modes = [(module, module.training) for module in model.modules()]
    measurements = []
    try:
        model.eval()
        with torch.enable_grad():
            for index, batch in enumerate(islice(batches, max_batches)):
                if (set(batch.get("id", ())) & teacher.validation_ids or
                        set(batch.get("speaker", ())) & teacher.validation_speakers):
                    raise ValueError("KD calibration batch contains current validation utterances or speakers")
                noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
                lengths = batch["length"].to(device)
                # Match full-precision loss calculations, regardless of an
                # outer AMP context used by a caller's training script.
                with torch.autocast(device_type=device.type, enabled=False):
                    estimate = model(noisy.float())
                    primary = primary_loss(estimate, clean.float(), lengths)
                    target = teacher.predict(noisy)
                    auxiliary = response_distillation_loss(estimate, target, lengths)
                if primary.ndim != 0 or not torch.isfinite(primary) or not torch.isfinite(auxiliary):
                    raise FloatingPointError("KD calibration requires finite scalar losses")
                if not primary.requires_grad:
                    raise ValueError("The primary calibration loss has no student gradient")
                primary_grads = torch.autograd.grad(primary, parameters, retain_graph=True, allow_unused=True)
                auxiliary_grads = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
                norms = [math.sqrt(sum(float(gradient.detach().double().square().sum())
                                       for gradient in gradients if gradient is not None))
                         for gradients in (primary_grads, auxiliary_grads)]
                if not all(math.isfinite(norm) for norm in norms):
                    raise FloatingPointError("Nonfinite gradient norm during KD calibration")
                primary_norm, auxiliary_norm = norms
                measurement = {"batch": index, "utterances": len(noisy), "primary_norm": primary_norm,
                               "distillation_norm": auxiliary_norm, "accepted": min(norms) > 0}
                if measurement["accepted"]:
                    inner = sum(float(left.detach().double().mul(right.detach().double()).sum())
                                for left, right in zip(primary_grads, auxiliary_grads)
                                if left is not None and right is not None)
                    measurement.update(unweighted_gradient_ratio=auxiliary_norm / primary_norm,
                                       suggested_weight=target_ratio * primary_norm / auxiliary_norm,
                                       gradient_cosine=max(-1.0, min(1.0, inner / (primary_norm * auxiliary_norm))))
                else:
                    measurement["reason"] = "zero primary or distillation gradient norm"
                measurements.append(measurement)
    finally:
        for module, previous in prior_modes:
            module.training = previous
    accepted = [measurement for measurement in measurements if measurement["accepted"]]
    if not accepted:
        raise ValueError("No finite nonzero primary and distillation gradient norms for calibration")
    coefficient = median(measurement["suggested_weight"] for measurement in accepted)
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("KD calibration produced a nonfinite or nonpositive coefficient")
    for measurement in accepted:
        measurement["weighted_gradient_ratio"] = coefficient * measurement["unweighted_gradient_ratio"]
    return {"distillation_weight": coefficient, "target_gradient_ratio": target_ratio,
            "observed_weighted_gradient_ratio_median": median(m["weighted_gradient_ratio"] for m in accepted),
            "source": "training batches only", "method": "median of per-batch L2 gradient-norm ratios",
            "teacher_checkpoint_sha256": teacher.provenance["checkpoint_sha256"],
            "teacher_output_gain": teacher.output_gain,
            "teacher_gain_calibration_sha256": (teacher.provenance["output_gain_calibration"] or {}).get("sha256"),
            "attempted_batches": len(measurements), "accepted_batches": len(accepted),
            "batches": measurements}


@torch.no_grad()
def teacher_quality_gate(teacher_output: torch.Tensor, clean: torch.Tensor, noisy: torch.Tensor,
                         lengths: torch.Tensor, *, minimum_improvement_db: float = 0.0) -> torch.Tensor:
    """Optional training-only gate: teacher improves SI-SDR over noisy by margin.

This gate uses training clean targets only and must not be tuned on final test.
It does not assert that the teacher is better than the student. Plain response
KD remains the default; use the returned mask explicitly for a separate ablation.
Silent/DC-only references are excluded; nonfinite valid samples raise. Cohort
counts remain available from the returned mask.
"""
    if not math.isfinite(minimum_improvement_db):
        raise ValueError("minimum_improvement_db must be finite")
    teacher, reference, lengths = _aligned(teacher_output, clean, lengths)
    noisy, _, _ = _aligned(noisy, clean, lengths)
    improvement = si_sdr(teacher, reference, lengths) - si_sdr(noisy, reference, lengths)
    return torch.isfinite(improvement) & (improvement >= minimum_improvement_db)
