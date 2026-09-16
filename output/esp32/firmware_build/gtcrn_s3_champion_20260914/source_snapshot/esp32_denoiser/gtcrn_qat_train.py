"""Fixed-grid GTCRN QAT selected by actual C PCM16 development inference.

The initial float checkpoint supplies the audited training recipe. This entry
point never calibrates ranges, reads final-test audio, or purchases compute.
Model-only candidate checkpoints bind evaluated binaries; optimizer resumes
are separate files so validation bookkeeping cannot relabel a scored binary.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import time
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairedAudioDataset, pad_collate, read_manifest
from .distillation import _audit_synthetic_sources, checkpoint_clean_identity_probability
from .evaluate import evaluate_manifest, _json_finite, load_checkpoint
from .extra_data import DynamicMixtureDataset
from .gtcrn_embedded import EmbeddedGTCRN
from .gtcrn_integer_export import load_gtcrn_integer, pack_gtcrn_integer, _json_bytes
from .gtcrn_recurrent_probe import _audit_calibration, _development_records
from .mixtures import HybridTrainingDataset
from .train import save_checkpoint, seed_everything, speech_loss


@dataclass(frozen=True)
class QATTrainConfig:
    source_checkpoint: str
    integer_model: str
    calibration: str
    development_manifest: str
    output_dir: str
    epochs: int = 40
    batch_size: int = 16
    workers: int = 2
    learning_rate: float = 1e-4
    min_learning_rate: float = 2e-5
    weight_decay: float = 1e-5
    waveform_loss_weight: float = 0.1
    max_hours: float = 6.0
    patience: int = 12
    seed: int = 20260913
    device: str = "cuda"
    resume: str | None = None
    max_steps_per_epoch: int | None = None
    max_validation_utterances: int | None = None

    def __post_init__(self):
        for name in ("epochs", "batch_size", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 0:
            raise ValueError("workers must be a nonnegative integer")
        for name in ("max_steps_per_epoch", "max_validation_utterances"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be positive when provided")
        for name in ("learning_rate", "min_learning_rate", "max_hours"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "waveform_loss_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("Minimum learning rate exceeds initial learning rate")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("QAT device must be cpu or cuda")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_dataset(parent, development_manifest):
    """Reconstruct the recorded recipe and audit both development cohorts."""
    config, provenance = parent.get("train_config"), parent.get("provenance")
    if (parent.get("model_kind") != "gtcrn" or parent.get("phase", "float") != "float"
            or not isinstance(config, dict) or not isinstance(provenance, dict)
            or provenance.get("test_used_for_selection") is not False):
        raise ValueError("QAT needs a float GTCRN parent with known training provenance")
    root = Path(__file__).parent
    if provenance.get("source_sha256", {}).get("data.py") != _sha(root / "data.py"):
        raise ValueError("Paired augmentation source differs from parent training")
    paths = {key: Path(config[name]) for key, name in (("train", "train_manifest"), ("val", "val_manifest"))}
    for key, path in paths.items():
        if provenance.get("manifest_sha256", {}).get(key) != _sha(path):
            raise ValueError(f"Parent {key} manifest hash mismatch")
    paired_records = read_manifest(paths["train"])
    validation_records = _development_records(paths["val"])
    if any(row.get("source_split") != "train" for row in validation_records):
        raise ValueError("Parent validation must have training-origin audio")
    development = _development_records(development_manifest)
    _audit_calibration(paired_records, validation_records)
    _audit_calibration(paired_records, development)
    probability = checkpoint_clean_identity_probability(config, provenance)
    crop = config.get("crop_seconds")
    if isinstance(crop, bool) or not isinstance(crop, (int, float)) or not math.isfinite(crop) or crop <= 0:
        raise ValueError("Parent training crop must be finite and positive")
    paired = PairedAudioDataset(paired_records, crop_seconds=crop, random_crop=True,
                               gain_db=(-6, 6), noise_scale_db=(-6, 6), clean_identity_prob=probability)
    synthetic_paths = [config.get(f"synthetic_{kind}_manifest") for kind in ("speech", "noise")]
    extra_audit = None
    dataset = paired
    if any(synthetic_paths):
        if not all(isinstance(path, str) and path for path in synthetic_paths):
            raise ValueError("Both synthetic training manifests are required")
        extra_audit = _audit_synthetic_sources(config, provenance, len(paired),
            [("parent validation", validation_records), ("QAT development", development)])
        synthetic = DynamicMixtureDataset(*synthetic_paths, crop_seconds=crop, snr_db=(-5, 20),
                                         gain_db=(-6, 6), clean_identity_prob=probability)
        if synthetic.split != "train":
            raise ValueError("Synthetic QAT data must use its training partition")
        sources = synthetic.speech_records + synthetic.noise_records
        for fields, source_key in ((("speech_id", "noise_id"), "id"),
                                  (("speech_source_sha256", "noise_source_sha256"), "sha256")):
            used = {row.get(field) for row in development for field in fields} - {None}
            if used & {row[source_key] for row in sources}:
                raise ValueError(f"QAT data overlaps underlying development {source_key}")
        dataset = HybridTrainingDataset(paired, synthetic, synthetic_probability=config["synthetic_probability"],
                                         epoch_samples=config.get("epoch_samples"))
    elif provenance.get("added_training_sources") is not None:
        raise ValueError("Parent synthetic config and provenance disagree")
    recipe_paths = {**paths, "development": Path(development_manifest)}
    if any(synthetic_paths):
        recipe_paths.update(synthetic_speech=Path(synthetic_paths[0]), synthetic_noise=Path(synthetic_paths[1]))
    return dataset, dict(crop_seconds=crop, clean_identity_probability=probability,
        samples_per_epoch=len(dataset), paired_utterances=len(paired), extra_source_audit=extra_audit,
        manifests={name: dict(path=str(path.resolve()), sha256=_sha(path)) for name, path in recipe_paths.items()},
        source_sha256={name: _sha(root/name) for name in ("data.py", "extra_data.py", "mixtures.py")},
        scope="Recorded parent training augmentation; training and both development sources audited; no final test")


def _resume_contract(config, provenance):
    ignored = {"resume", "output_dir", "epochs", "max_hours", "workers"}
    return dict(settings={k: v for k, v in asdict(config).items() if k not in ignored}, provenance=provenance)


def _check_resume(saved, contract):
    if saved.get("phase") != "qat" or saved.get("resume_contract") != contract:
        raise ValueError("QAT optimizer resume configuration, source files or data recipe changed")
    if not all(key in saved for key in ("optimizer", "scheduler", "epoch", "best_score", "stale", "best_candidate")):
        raise ValueError("Resume requires a complete optimizer checkpoint, not a model-only candidate")


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_finite(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def train(config: QATTrainConfig):
    from .gtcrn_qat import GTCRNQAT
    source_paths = [Path(p) for p in (config.source_checkpoint, config.integer_model, config.calibration, config.development_manifest)]
    snapshots = {str(path.resolve()): path.read_bytes() for path in source_paths}
    parent = torch.load(io.BytesIO(snapshots[str(source_paths[0].resolve())]), map_location="cpu", weights_only=False)
    dataset, recipe = training_dataset(parent, config.development_manifest)
    source, _ = load_checkpoint(config.source_checkpoint)
    audit = json.loads(snapshots[str(source_paths[2].resolve())])
    packed_bytes = snapshots[str(source_paths[1].resolve())]
    packed = load_gtcrn_integer(packed_bytes, calibration=audit)
    if packed.source_sha256 != hashlib.sha256(snapshots[str(source_paths[0].resolve())]).hexdigest():
        raise ValueError("Packed model does not derive from the supplied float checkpoint")
    provenance = dict(parent_files={path: hashlib.sha256(data).hexdigest() for path, data in snapshots.items()},
        training_recipe=recipe, test_used_for_selection=False,
        selection="Actual full C PCM16 mean-utterance development SI-SDRi",
        partial_validation=config.max_validation_utterances is not None,
        trainer_sha256=_sha(__file__), fixed_grids=True, amp=False, tf32=False)
    contract = _resume_contract(config, provenance)
    resumed = torch.load(config.resume, map_location="cpu", weights_only=False) if config.resume else None
    if resumed is not None:
        _check_resume(resumed, contract)
        model = GTCRNQAT.from_checkpoint(resumed)
    else:
        model = GTCRNQAT.from_float(source, packed_bytes, calibration=audit)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA runtime is available")
    seed_everything(config.seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=4,
                                                         min_lr=config.min_learning_rate)
    epoch, best, stale, best_candidate = 0, -math.inf, 0, None
    if resumed is not None:
        optimizer.load_state_dict(resumed["optimizer"]); scheduler.load_state_dict(resumed["scheduler"])
        epoch, best, stale, best_candidate = (resumed[key] for key in ("epoch", "best_score", "stale", "best_candidate"))
        if config.epochs < epoch:
            raise ValueError("Requested epoch limit is below the completed resume epoch")
        best_directory = Path(best_candidate["directory"])
        if (_sha(best_directory/"model.bin") != best_candidate["packed_sha256"] or
                _sha(best_directory/"model.pt") != best_candidate["checkpoint_sha256"]):
            raise ValueError("Restore the unchanged best candidate artifacts before resuming")
    output = Path(config.output_dir)
    if output.exists() and (output/"last.pt").exists() and resumed is None:
        raise ValueError("Existing QAT run requires explicit optimizer resume or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output/"config.json", asdict(config)); _atomic_json(output/"provenance.json", provenance)
    generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator,
                        num_workers=config.workers, collate_fn=pad_collate, pin_memory=device.type == "cuda")
    started = time.monotonic(); deadline = started+config.max_hours*3600
    def verify_sources():
        for path, expected in snapshots.items():
            if Path(path).read_bytes() != expected:
                raise ValueError("A frozen QAT source changed during the run")
        for row in recipe["manifests"].values():
            if _sha(row["path"]) != row["sha256"]:
                raise ValueError("A QAT data manifest changed during the run")
    def evaluate_candidate(number):
        verify_sources()
        candidates = output/"candidates"
        candidates.mkdir(exist_ok=True)
        # A failed prior attempt is evidence, not an output to overwrite.
        directory = Path(tempfile.mkdtemp(prefix=f"epoch{number:04d}-", dir=candidates))
        payload = model.checkpoint_payload()
        payload.update(phase="qat", epoch=number, train_config=asdict(config), provenance=provenance)
        checkpoint = directory/"model.pt"; save_checkpoint(checkpoint, payload)
        integer = model.integer_snapshot(checkpoint_sha256=_sha(checkpoint), training_metadata=provenance)
        binary = pack_gtcrn_integer(integer)
        if len(binary) > 99000:
            raise ValueError("QAT model exceeds the 99,000-byte deployment limit")
        (directory/"model.bin").write_bytes(binary)
        (directory/"model.calibration.json").write_bytes(_json_bytes(integer.calibration)+b"\n")
        deployed = EmbeddedGTCRN(binary, calibration=integer.calibration, io_format="pcm16")
        try:
            report = evaluate_manifest(deployed, config.development_manifest,
                                       max_utterances=config.max_validation_utterances)
            report.update(model_stats=deployed.model_stats(), source_checkpoint_sha256=_sha(checkpoint),
                          packed_sha256=hashlib.sha256(binary).hexdigest(), epoch=number,
                          scope="Actual C PCM16 development selection; no final test or MCU timing")
        finally:
            deployed.close()
        verify_sources(); _atomic_json(directory/"evaluation.json", report)
        return report["summary"]["si_sdri"], dict(epoch=number, directory=str(directory.resolve()),
            checkpoint_sha256=_sha(checkpoint), packed_sha256=hashlib.sha256(binary).hexdigest(),
            si_sdri=report["summary"]["si_sdri"])
    def save_resume():
        payload=model.checkpoint_payload()
        payload.update(phase="qat", epoch=epoch, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            best_score=best, stale=stale, best_candidate=best_candidate, resume_contract=contract,
            train_config=asdict(config), provenance=provenance)
        save_checkpoint(output/"last.pt", payload)
        _atomic_json(output/"best.json", best_candidate)
    if resumed is None:
        best, best_candidate = evaluate_candidate(0); save_resume()
    status = "complete"
    for next_epoch in range(epoch+1, config.epochs+1):
        if time.monotonic() >= deadline:
            status="time_budget_exhausted"; break
        seed_everything(config.seed+next_epoch); generator.manual_seed(config.seed+next_epoch)
        model.train(); losses=[]; completed=True; epoch_started=time.monotonic()
        for step, batch in enumerate(loader):
            if config.max_steps_per_epoch is not None and step >= config.max_steps_per_epoch:
                break
            if time.monotonic() >= deadline:
                completed=False; break
            optimizer.zero_grad(set_to_none=True)
            noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
            estimate=model(noisy)
            loss=speech_loss(estimate, clean, batch["length"].to(device), waveform_loss_weight=config.waveform_loss_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite QAT loss at epoch {next_epoch}, step {step}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach()))
            if step % 25 == 0:
                print(json.dumps(dict(event="qat_progress", epoch=next_epoch, step=step, loss=losses[-1],
                                     elapsed_seconds=time.monotonic()-started)), flush=True)
        if not completed:
            # Epoch-scoped RNG resume is exact only at complete epoch boundaries.
            # Keep the last complete optimizer checkpoint; partial work is not
            # mislabeled as a completed epoch or used for deployment selection.
            status="time_budget_exhausted_partial_epoch_discarded"; break
        if not losses:
            raise RuntimeError("QAT epoch produced no optimizer updates")
        model.eval(); score, candidate=evaluate_candidate(next_epoch)
        scheduler.step(score); stale=0 if score > best else stale+1
        if score > best:
            best, best_candidate=score, candidate
        epoch=next_epoch; save_resume()
        row=dict(epoch=epoch, loss=float(np.mean(losses)), si_sdri=score, best_si_sdri=best,
            learning_rate=optimizer.param_groups[0]["lr"], seconds=time.monotonic()-epoch_started,
            optimizer_steps=len(losses), selection_runtime="full C PCM16", phase="qat")
        with (output/"history.jsonl").open("a") as handle:
            handle.write(json.dumps(row)+"\n")
        print(json.dumps(dict(event="qat_epoch", **row)), flush=True)
        if stale >= config.patience:
            status="early_stopped"; break
    result=dict(status=status, completed_epoch=epoch, best=best_candidate,
                elapsed_hours=(time.monotonic()-started)/3600, official_test_used=False)
    _atomic_json(output/"summary.json", result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args=parser.parse_args()
    print(json.dumps(train(QATTrainConfig(**json.loads(args.config.read_text())))), flush=True)


if __name__ == "__main__":
    main()
