"""Budgeted, reproducible training; select checkpoints on training-speaker holdout only."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairedAudioDataset, pad_collate
from .metrics import si_sdr
from .models import build_model, checkpoint_kind, configure_model_qat, calibrate_model_hidden_exponent


@dataclass
class TrainConfig:
    train_manifest: str
    val_manifest: str
    output_dir: str
    epochs: int = 60
    batch_size: int = 32
    crop_seconds: float = 3.0
    learning_rate: float = 0.001
    min_learning_rate: float = 0.00002
    weight_decay: float = 0.00001
    seed: int = 2026
    workers: int = 2
    max_hours: float = 6.0
    patience: int = 15
    eval_batch_size: int = 8
    max_steps_per_epoch: int | None = None
    max_val_batches: int | None = None
    resume: str | None = None
    phase: str = "float"
    width: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    amp: bool = True
    zero_depthwise_bias: bool = False
    feature_layout: str = "erb_complex"
    model_kind: str = "spectral_tcn"
    model_options: dict = field(default_factory=dict)
    resume_optimizer: bool = True
    waveform_loss_weight: float = 0.1
    spectral_loss_weight: float = 0.0
    clean_identity_probability: float = 0.03
    synthetic_speech_manifest: str | None = None
    synthetic_noise_manifest: str | None = None
    synthetic_probability: float = 0.5
    epoch_samples: int | None = None
    teacher_checkpoint: str | None = None
    teacher_gain_calibration: str | None = None
    distillation_weight: float = 0.0
    # None keeps plain response KD; a margin enables a training-only ablation.
    distillation_gate_minimum_improvement_db: float | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def speech_loss(estimate: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor, *,
                waveform_loss_weight: float = 0.1) -> torch.Tensor:
    """Optimize SI-SDR plus a configurable, level-sensitive normalized L1 term."""
    if not math.isfinite(waveform_loss_weight) or waveform_loss_weight < 0:
        raise ValueError("waveform_loss_weight must be finite and nonnegative")
    estimate, clean = estimate.float(), clean.float()
    mask = torch.arange(clean.shape[-1], device=clean.device)[None, :] < lengths[:, None]
    scale = ((clean.square() * mask).sum(-1) / lengths).sqrt().clamp_min(1e-4)
    normalized_l1 = ((estimate - clean).abs() * mask).sum(-1) / lengths / scale
    scores = si_sdr(estimate, clean, lengths=lengths)
    # Silent references have no SI-SDR; retain their level-preserving term.
    valid = torch.isfinite(scores)
    sdr_loss = -scores[valid].mean() if valid.any() else estimate.sum() * 0
    return sdr_loss + waveform_loss_weight * normalized_l1.mean()


def check_training_split(train_data, val_data) -> dict:
    """Require train-origin audio and disjoint speakers, IDs and audio files."""
    for name, dataset in (("train", train_data), ("validation", val_data)):
        if any(record.get("source_split") != "train" for record in dataset.records):
            raise ValueError(f"{name} must contain only official training-origin utterances")
    for field in ("speaker", "id"):
        overlap = {r[field] for r in train_data.records} & {r[field] for r in val_data.records}
        if overlap:
            raise ValueError(f"Training/validation {field} overlap: {sorted(overlap)[:5]}")
    train_paths = {str(Path(r[role]).resolve()) for r in train_data.records for role in ("clean", "noisy")}
    val_paths = {str(Path(r[role]).resolve()) for r in val_data.records for role in ("clean", "noisy")}
    if train_paths & val_paths:
        raise ValueError("Training/validation audio paths overlap")
    return {"train_utterances": len(train_data), "validation_utterances": len(val_data),
            "train_speakers": sorted({r["speaker"] for r in train_data.records}),
            "validation_speakers": sorted({r["speaker"] for r in val_data.records})}


@torch.inference_mode()
def validate(model, loader, device, max_batches=None) -> dict:
    model.eval()
    results = []
    total = 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
        lengths = batch["length"].to(device)
        estimate = model(noisy).float()
        if not torch.isfinite(estimate).all():
            raise FloatingPointError("Non-finite validation output")
        baseline = si_sdr(noisy, clean, lengths=lengths)
        enhanced = si_sdr(estimate, clean, lengths=lengths)
        for i, item_id in enumerate(batch["id"]):
            total += 1
            b, e = float(baseline[i]), float(enhanced[i])
            if math.isfinite(b) and math.isfinite(e):
                results.append({"id": item_id, "noisy_si_sdr": b, "si_sdr": e, "si_sdri": e-b})
    if not results:
        raise RuntimeError("Validation contains no finite, non-silent utterances")
    return {"utterances": len(results), "total_utterances": total,
            "invalid_utterances": total - len(results), **{
        key: float(np.mean([r[key] for r in results]))
        for key in ("noisy_si_sdr", "si_sdr", "si_sdri")
    }, "per_utterance": results}


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config: TrainConfig) -> dict:
    if config.phase not in {"float", "qat"}:
        raise ValueError("phase must be float or qat")
    if config.epochs < 1 or config.max_hours <= 0:
        raise ValueError("epochs and max_hours must be positive")
    if config.batch_size < 1 or config.eval_batch_size < 1 or config.workers < 0 or config.patience < 1:
        raise ValueError("Batch sizes and patience must be positive; workers must be nonnegative")
    if any(value is not None and value < 1 for value in (config.max_steps_per_epoch, config.max_val_batches)):
        raise ValueError("Optional step/batch limits must be positive")
    if not math.isfinite(config.spectral_loss_weight) or config.spectral_loss_weight < 0:
        raise ValueError("spectral_loss_weight must be finite and nonnegative")
    if not math.isfinite(config.waveform_loss_weight) or config.waveform_loss_weight < 0:
        raise ValueError("waveform_loss_weight must be finite and nonnegative")
    if (isinstance(config.clean_identity_probability, bool) or
            not isinstance(config.clean_identity_probability, (int, float)) or
            not math.isfinite(config.clean_identity_probability) or
            not 0 <= config.clean_identity_probability <= 1):
        raise ValueError("clean_identity_probability must be a finite number in [0,1]")
    if not math.isfinite(config.distillation_weight) or config.distillation_weight < 0:
        raise ValueError("distillation_weight must be finite and nonnegative")
    if bool(config.teacher_checkpoint) != (config.distillation_weight > 0):
        raise ValueError("A teacher_checkpoint and positive distillation_weight must be configured together")
    if config.teacher_gain_calibration is not None and (not isinstance(config.teacher_gain_calibration, str) or
                                                       not config.teacher_gain_calibration or not config.teacher_checkpoint):
        raise ValueError("teacher_gain_calibration requires an active teacher and a nonempty file path")
    gate_margin = config.distillation_gate_minimum_improvement_db
    if gate_margin is not None and (not math.isfinite(gate_margin) or gate_margin < 0 or not config.teacher_checkpoint):
        raise ValueError("A distillation gate requires an active teacher and a finite nonnegative margin")
    if bool(config.synthetic_speech_manifest) != bool(config.synthetic_noise_manifest):
        raise ValueError("Both synthetic speech and noise manifests are required")
    if config.epoch_samples is not None and not config.synthetic_speech_manifest:
        raise ValueError("epoch_samples currently requires synthetic training sources")
    resumed = torch.load(config.resume, map_location="cpu", weights_only=False) if config.resume else None
    if resumed is not None and resumed.get("phase") == config.phase and config.resume_optimizer:
        from .distillation import checkpoint_clean_identity_probability
        previous_probability = checkpoint_clean_identity_probability(
            resumed.get("train_config", {}), resumed.get("provenance", {}))
        if previous_probability != config.clean_identity_probability:
            # Check before writing config.json: a rejected resume must not
            # relabel the existing run's recorded training recipe.
            raise ValueError("Changing clean_identity_probability requires resume_optimizer=False")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    seed_everything(config.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.phase == "float"
        torch.backends.cudnn.allow_tf32 = config.phase == "float"
    options = dict(config.model_options)
    if config.model_kind == "spectral_tcn":
        if options:
            raise ValueError("Use width/dilations/feature_layout for spectral_tcn, not model_options")
        options = dict(width=config.width, dilations=config.dilations, feature_layout=config.feature_layout)
    model = build_model(config.model_kind, options)
    model_config = model.config
    if config.zero_depthwise_bias:
        # Default Conv1d biases can silence whole ReLU depthwise channels when
        # their small normalized inputs start near zero. This changes only
        # initialization; checkpoint loading and the deployed graph are intact.
        if config.model_kind != "spectral_tcn":
            raise ValueError("zero_depthwise_bias is a spectral_tcn initialization experiment")
        for block in model.blocks:
            torch.nn.init.zeros_(block.depthwise.bias)
    if resumed is not None:
        if resumed.get("initialization_only") and config.resume_optimizer:
            raise ValueError("An initialization checkpoint requires resume_optimizer=False")
        if resumed.get("phase") == "qat" and config.phase == "float":
            raise ValueError("A QAT checkpoint cannot be resumed as a floating-point phase")
        expected = asdict(model_config)
        actual_kind = checkpoint_kind(resumed)
        actual = asdict(build_model(actual_kind, resumed["model_config"], checkpoint=True).config)
        if actual_kind != config.model_kind or actual != expected:
            raise ValueError(f"Checkpoint architecture differs: {actual} vs {expected}")
        if resumed.get("phase") == "qat":
            configure_model_qat(model, config.model_kind)
        model.load_state_dict(resumed["model"])
    model.to(device)
    train_data = PairedAudioDataset(config.train_manifest, crop_seconds=config.crop_seconds,
                                   random_crop=True, gain_db=(-6, 6),
                                   noise_scale_db=(-6, 6), clean_identity_prob=config.clean_identity_probability)
    val_data = PairedAudioDataset(config.val_manifest, crop_seconds=None,
                                 random_crop=False, gain_db=(0, 0))
    split_info = check_training_split(train_data, val_data)
    added_sources = None
    if config.synthetic_speech_manifest:
        from .extra_data import DynamicMixtureDataset
        from .mixtures import HybridTrainingDataset
        synthetic = DynamicMixtureDataset(config.synthetic_speech_manifest, config.synthetic_noise_manifest,
                                          crop_seconds=config.crop_seconds, snr_db=(-5, 20),
                                          gain_db=(-6, 6), clean_identity_prob=config.clean_identity_probability)
        if synthetic.split != "train":
            raise ValueError("Synthetic training sources must use the training partition")
        # The source loader enforces its training partition. Also protect the
        # existing validation speakers and audio paths at the integration seam.
        validation_speakers = {r["speaker"] for r in val_data.records}
        if validation_speakers & {r["speaker"] for r in synthetic.speech_records}:
            raise ValueError("Synthetic speech overlaps validation speakers")
        validation_paths = {str(Path(r[role]).resolve()) for r in val_data.records for role in ("clean", "noisy")}
        if validation_paths & {r["path"] for r in (*synthetic.speech_records, *synthetic.noise_records)}:
            raise ValueError("Synthetic source audio overlaps validation paths")
        added_sources = {
            "speech_manifest_sha256": hashlib.sha256(Path(config.synthetic_speech_manifest).read_bytes()).hexdigest(),
            "noise_manifest_sha256": hashlib.sha256(Path(config.synthetic_noise_manifest).read_bytes()).hexdigest(),
            "speech_recordings": len(synthetic.speech_records),
            "noise_recordings": len(synthetic.noise_records),
            "synthetic_probability": config.synthetic_probability,
        }
        train_data = HybridTrainingDataset(train_data, synthetic,
                                           synthetic_probability=config.synthetic_probability,
                                           epoch_samples=config.epoch_samples)
        added_sources["samples_per_epoch"] = len(train_data)
    # Worker RNGs are initialized by PyTorch from the seeded generator.
    common = dict(num_workers=config.workers, collate_fn=pad_collate,
                  pin_memory=device.type == "cuda", persistent_workers=False)
    train_generator = torch.Generator()
    val_generator = torch.Generator().manual_seed(config.seed + 1_000_000)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, generator=train_generator, **common)
    val_loader = DataLoader(val_data, batch_size=config.eval_batch_size, shuffle=False, generator=val_generator, **common)
    calibration = resumed.get("calibration") if resumed is not None else None
    if config.phase == "qat" and (resumed is None or resumed.get("phase") != "qat"):
        train_generator.manual_seed(config.seed + 2_000_000)
        hidden_exponent = calibrate_model_hidden_exponent(
            model, config.model_kind, (batch["noisy"].to(device) for batch in train_loader), max_batches=32)
        configure_model_qat(model, config.model_kind, hidden_exponent=hidden_exponent)
        # configure_qat adds scalar buffers; ensure these follow model tensors.
        model.to(device)
        calibration = {"source": "training manifest only", "max_batches": 32,
                       "hidden_exponent": hidden_exponent}
    teacher = None
    distillation = None
    if config.teacher_checkpoint:
        from .distillation import FrozenTeacher, response_distillation_loss, teacher_quality_gate
        teacher = FrozenTeacher(config.teacher_checkpoint, config.val_manifest, device=device,
                                gain_calibration=config.teacher_gain_calibration)
        distillation = {"teacher": teacher.provenance, "weight": config.distillation_weight,
                        "loss": "detached compressed spectral response, equal per selected training utterance",
                        "gate_minimum_improvement_db": gate_margin,
                        "target_source": "current training batches only; teacher is never called during validation",
                        "coefficient_policy": "fixed per run; this entry point does not calibrate KD gradient ratios"}
    if resumed is not None and resumed.get("phase") == config.phase and config.resume_optimizer:
        previous_config = resumed.get("train_config", {})
        for key, default in (("teacher_checkpoint", None), ("distillation_weight", 0.0),
                             ("teacher_gain_calibration", None),
                             ("distillation_gate_minimum_improvement_db", None)):
            if previous_config.get(key, default) != getattr(config, key):
                raise ValueError("Changing distillation settings requires resume_optimizer=False")
        if teacher is not None:
            previous_distillation = resumed.get("provenance", {}).get("distillation")
            if (not isinstance(previous_distillation, dict) or
                    previous_distillation.get("teacher", {}).get("checkpoint_sha256") != teacher.provenance["checkpoint_sha256"]):
                raise ValueError("Teacher lineage changed or is missing on optimizer resume")
            previous_gain = previous_distillation["teacher"].get("output_gain_calibration")
            current_gain = teacher.provenance["output_gain_calibration"]
            if ((previous_gain or {}).get("sha256") != (current_gain or {}).get("sha256") or
                    previous_distillation["teacher"].get("output_gain", 1.0) != teacher.output_gain):
                raise ValueError("Teacher gain calibration changed on optimizer resume")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    use_amp = config.amp and device.type == "cuda" and config.phase == "float"
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=config.min_learning_rate)
    start_epoch, best, stale = 0, -float("inf"), 0
    if resumed is not None and resumed.get("phase") == config.phase and config.resume_optimizer:
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed.get("scaler", {}))
        start_epoch = int(resumed["epoch"])
        best, stale = float(resumed["best_si_sdri"]), int(resumed.get("stale", 0))
    started = time.monotonic()
    deadline = started + config.max_hours * 3600
    provenance = {"device": str(device), "torch": torch.__version__, "model": model.model_stats(),
                  "model_kind": config.model_kind,
                  "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                  "test_used_for_selection": False, "selection_metric": "mean utterance SI-SDR improvement",
                  "validation_is_partial": config.max_val_batches is not None,
                  "calibration": calibration,
                  "clean_identity_probability": config.clean_identity_probability,
                  "added_training_sources": added_sources,
                  "distillation": distillation,
                  "resume_checkpoint_sha256": hashlib.sha256(Path(config.resume).read_bytes()).hexdigest()
                  if config.resume else None,
                  "manifest_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                      for name, path in (("train", config.train_manifest), ("val", config.val_manifest))},
                  "source_sha256": {str(path.relative_to(Path(__file__).parent)):
                                    hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(Path(__file__).parent.rglob("*.py"))},
                  **split_info}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"event": "start", **provenance, "phase": config.phase}), flush=True)
    initial = validate(model, val_loader, device, config.max_val_batches)
    (output / "initial_validation.json").write_text(json.dumps(initial, indent=2))
    print(json.dumps({"event": "initial_validation", **{k:v for k,v in initial.items() if k != "per_utterance"}}), flush=True)
    # Retain the starting model if training degrades it, including float->QAT.
    # For a new output directory on resume, initialize best from the actual
    # loaded weights instead of claiming an unavailable historical checkpoint.
    if initial["si_sdri"] >= best or not (output / "best.pt").exists():
        best = initial["si_sdri"]
        stale = 0
        payload = {"model": model.state_dict(), "model_config": asdict(model_config), "model_kind": config.model_kind,
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "scaler": scaler.state_dict(), "epoch": start_epoch, "phase": config.phase,
                   "best_si_sdri": best, "stale": stale, "train_config": asdict(config),
                   "calibration": calibration, "provenance": provenance}
        save_checkpoint(output / "best.pt", payload)
        save_checkpoint(output / "last.pt", payload)
        (output / "best_validation.json").write_text(json.dumps(initial, indent=2))
    history = output / "history.jsonl"
    final = {"status": "no_epochs", "best_si_sdri": best}
    for epoch in range(start_epoch + 1, config.epochs + 1):
        if time.monotonic() >= deadline:
            break
        epoch_start = time.monotonic()
        # Epoch-scoped seeds reproduce shuffle and crop RNGs after resume;
        # workers are recreated so their augmentation RNGs cannot drift.
        seed_everything(config.seed + epoch)
        train_generator.manual_seed(config.seed + epoch)
        model.train()
        losses = []
        distillation_losses = []
        distillation_selected, distillation_total = 0, 0
        for step, batch in enumerate(train_loader):
            if config.max_steps_per_epoch is not None and step >= config.max_steps_per_epoch:
                break
            if time.monotonic() >= deadline:
                break
            noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
            lengths = batch["length"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                estimate = model(noisy)
            loss = speech_loss(estimate, clean, lengths, waveform_loss_weight=config.waveform_loss_weight)
            if config.spectral_loss_weight:
                from .losses import compressed_spectral_loss
                loss = loss + config.spectral_loss_weight * compressed_spectral_loss(estimate, clean, lengths)
            if teacher is not None:
                teacher_output = teacher.predict(noisy)
                example_mask = (teacher_quality_gate(teacher_output, clean, noisy, lengths,
                                                    minimum_improvement_db=gate_margin)
                                if gate_margin is not None else None)
                response_loss = response_distillation_loss(estimate, teacher_output, lengths,
                                                           example_mask=example_mask)
                loss = loss + config.distillation_weight * response_loss
                distillation_losses.append(float(response_loss.detach()))
                distillation_total += len(noisy)
                distillation_selected += int(example_mask.sum()) if example_mask is not None else len(noisy)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            if step % 50 == 0:
                print(json.dumps({"event": "progress", "epoch": epoch, "step": step,
                                  "loss": losses[-1], "seconds": round(time.monotonic()-started)}), flush=True)
        if not losses:
            break
        metrics = validate(model, val_loader, device, config.max_val_batches)
        score = metrics["si_sdri"]
        scheduler.step(score)
        improved = score > best
        best = max(best, score)
        stale = 0 if improved else stale + 1
        row = {"epoch": epoch, "phase": config.phase, "loss": float(np.mean(losses)),
               "seconds": time.monotonic()-epoch_start, "elapsed_seconds": time.monotonic()-started,
               "learning_rate": optimizer.param_groups[0]["lr"],
               **{k:v for k,v in metrics.items() if k != "per_utterance"}}
        if teacher is not None:
            row.update(distillation_loss=float(np.mean(distillation_losses)),
                       weighted_distillation_loss=config.distillation_weight * float(np.mean(distillation_losses)),
                       distillation_selected_utterances=distillation_selected,
                       distillation_total_utterances=distillation_total)
        with history.open("a") as handle:
            handle.write(json.dumps(row)+"\n")
        payload = {"model": model.state_dict(), "model_config": asdict(model_config), "model_kind": config.model_kind,
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "scaler": scaler.state_dict(), "epoch": epoch, "phase": config.phase,
                   "best_si_sdri": best, "stale": stale, "train_config": asdict(config), "provenance": provenance}
        payload["calibration"] = calibration
        save_checkpoint(output / "last.pt", payload)
        if improved:
            save_checkpoint(output / "best.pt", payload)
            (output / "best_validation.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps({"event": "epoch", **row, "best_si_sdri": best}), flush=True)
        final = {"status": "complete", "best_si_sdri": best, "epoch": epoch,
                 "elapsed_hours": (time.monotonic()-started)/3600}
        if stale >= config.patience:
            final["status"] = "early_stopped"
            break
    if time.monotonic() >= deadline:
        final["status"] = "time_budget_exhausted"
    (output / "summary.json").write_text(json.dumps(final, indent=2))
    print(json.dumps({"event": "finished", **final}), flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    if "dilations" in values:
        values["dilations"] = tuple(values["dilations"])
    train(TrainConfig(**values))


if __name__ == "__main__":
    main()
