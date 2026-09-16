"""Fit a frozen teacher's response gain using audited clean training inputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .data import read_manifest
from .extra_data import read_extra_manifest


METHOD = "equal-utterance reference-RMS-normalized least squares on clean training inputs"


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_sources(teacher):
    provenance = teacher.provenance
    path = Path(provenance["teacher_manifests"]["train"])
    digest = provenance["teacher_manifest_sha256"]["train"]
    if _hash(path) != digest:
        raise ValueError("Teacher gain training manifest changed since lineage audit")
    paired = read_manifest(path)
    if any(row.get("source_split") != "train" for row in paired):
        raise ValueError("Teacher gain fitting requires training-origin clean speech")
    sources = {"paired_train": {"manifest": str(path), "manifest_sha256": digest,
                                 "rows": [{**row, "path": row["clean"]} for row in paired]}}
    added = provenance.get("added_training_sources")
    if added:
        entry = added["partitions"]["speech"]["train"]
        if _hash(entry["manifest"]) != entry["manifest_sha256"]:
            raise ValueError("Teacher gain speech training manifest changed since lineage audit")
        rows = read_extra_manifest(entry["manifest"], "speech")
        if any(row["split"] != "train" for row in rows):
            raise ValueError("Teacher gain fitting cannot use an extra validation partition")
        sources["speech_train"] = {**entry, "rows": rows}
    for source in sources.values():
        if any(row["id"] in teacher.validation_ids or row["speaker"] in teacher.validation_speakers
               for row in source["rows"]):
            raise ValueError("Teacher gain source contains held-out IDs or speakers")
    return sources


@torch.no_grad()
def calibrate_teacher_output_gain(teacher, *, examples_per_source=64, crop_seconds=3.0,
                                   batch_size=8, seed=2026):
    """Fit one scalar; never accept an arbitrary caller-supplied calibration set.

    Draw up to examples_per_source distinct recordings from each audited clean
    training source, then one bounded crop per recording. Broad teachers use
    paired clean speech and LibriSpeech training speech equally when available.
    Reference RMS normalization makes this an equal-example squared-error fit.
    No noise, common-gain augmentation, clipping, or validation audio is used.
    The teacher must be uncorrected; its parameters and global RNG are untouched.
    """
    for label, value in (("examples_per_source", examples_per_source), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
            raise ValueError(f"{label} must be an integer in [1,4096]")
    if not math.isfinite(crop_seconds) or not 0 < crop_seconds <= 30:
        raise ValueError("crop_seconds must be finite and in (0,30]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if teacher.output_gain != 1.0 or teacher.provenance.get("output_gain_calibration") is not None:
        raise ValueError("Fit gain using an uncorrected frozen teacher")
    sources = _training_sources(teacher)
    random = np.random.default_rng(seed)
    pending, records = [], []
    fit = {"cross_sum": 0.0, "estimate_energy_sum": 0.0, "reference_energy_sum": 0.0,
           "valid_examples": 0, "silent_examples": 0}
    projection_gains = []

    def process_batch():
        if not pending:
            return
        lengths = [len(clean) for clean, _ in pending]
        clean = torch.zeros(len(pending), max(lengths), dtype=torch.float32, device=teacher.device)
        for index, (waveform, _) in enumerate(pending):
            clean[index, :len(waveform)] = torch.from_numpy(waveform).to(teacher.device)
        prediction = teacher.predict(clean)
        for index, (waveform, record) in enumerate(pending):
            target = waveform.astype(np.float64)
            estimate = prediction[index, :len(target)].detach().cpu().double().numpy()
            reference_energy = float(np.mean(target**2))
            if reference_energy <= 1e-10:
                fit["silent_examples"] += 1
                record["included"] = False
            else:
                scale = max(reference_energy, 1e-8)
                cross = float(np.mean(estimate * target))
                fit["cross_sum"] += cross / scale
                fit["estimate_energy_sum"] += float(np.mean(estimate**2)) / scale
                fit["reference_energy_sum"] += reference_energy / scale
                fit["valid_examples"] += 1
                projection_gains.append(cross / reference_energy)
                record["included"] = True
            records.append(record)
        pending.clear()

    for name, source in sources.items():
        rows = source["rows"]
        indices = random.choice(len(rows), size=min(examples_per_source, len(rows)), replace=False)
        for index in indices:
            row = rows[int(index)]
            length = min(row["samples"], max(1, round(crop_seconds * 16000)))
            offset = int(random.integers(row["samples"] - length + 1))
            path = Path(row["path"])
            digest = _hash(path)
            if name == "speech_train" and digest != row["sha256"]:
                raise ValueError("Selected teacher-gain source audio differs from its audited content hash")
            waveform, rate = sf.read(path, start=offset, frames=length, dtype="float32")
            if rate != 16000 or waveform.ndim != 1 or len(waveform) != length or not np.isfinite(waveform).all():
                raise ValueError(f"Invalid clean training crop for teacher gain: {row['id']}")
            record = {"source": name, "id": row["id"], "speaker": row["speaker"],
                      "path": str(path.resolve()), "audio_sha256": digest,
                      "offset": offset, "samples": length}
            pending.append((waveform, record))
            if len(pending) == batch_size:
                process_batch()
    process_batch()
    denominator, numerator = fit["estimate_energy_sum"], fit["cross_sum"]
    if fit["valid_examples"] < 1 or denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("No valid nonzero teacher responses for gain calibration")
    gain = numerator / denominator
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Teacher gain fit is not finite and positive")
    count = fit["valid_examples"]
    return {"version": 1, "method": METHOD, "output_gain": gain,
            "teacher_checkpoint_sha256": teacher.provenance["checkpoint_sha256"],
            "teacher_manifest_sha256": dict(teacher.provenance["teacher_manifest_sha256"]),
            "training_sources": {name: {"manifest": source["manifest"], "manifest_sha256": source["manifest_sha256"]}
                                 for name, source in sources.items()},
            "sampling": {"examples_per_source": examples_per_source, "crop_seconds": crop_seconds,
                         "batch_size": batch_size, "seed": seed, "sample_rate": 16000,
                         "selection": "distinct training recordings; one random clean crop each; no augmentation"},
            "fit": fit, "selected_examples": records,
            "diagnostics": {"raw_mean_projection_gain": float(np.mean(projection_gains)),
                            "corrected_mean_projection_gain": gain * float(np.mean(projection_gains)),
                            "normalized_mse_before": (denominator - 2*numerator + fit["reference_energy_sum"]) / count,
                            "normalized_mse_after": (gain*gain*denominator - 2*gain*numerator + fit["reference_energy_sum"]) / count},
            "scope": "Teacher training response correction only; no teacher weight change, student inference operation, or development-set fit"}


def load_teacher_gain_calibration(teacher, path):
    """Verify model/source identity and selected training audio before reuse."""
    path = Path(path).resolve()
    contents = path.read_bytes()
    calibration = json.loads(contents)
    if not isinstance(calibration, dict) or calibration.get("version") != 1 or calibration.get("method") != METHOD:
        raise ValueError("Unknown teacher gain calibration format")
    if (calibration.get("teacher_checkpoint_sha256") != teacher.provenance["checkpoint_sha256"] or
            calibration.get("teacher_manifest_sha256") != teacher.provenance["teacher_manifest_sha256"]):
        raise ValueError("Teacher gain calibration checkpoint/manifest identity mismatch")
    sources = _training_sources(teacher)
    expected = {name: {"manifest": source["manifest"], "manifest_sha256": source["manifest_sha256"]}
                for name, source in sources.items()}
    if calibration.get("training_sources") != expected:
        raise ValueError("Teacher gain calibration training sources mismatch")
    fit, gain = calibration.get("fit", {}), calibration.get("output_gain")
    numerator, denominator = fit.get("cross_sum"), fit.get("estimate_energy_sum")
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (gain, numerator, denominator)) or gain <= 0 or denominator <= 0 or
            not math.isclose(gain, numerator / denominator, rel_tol=1e-12, abs_tol=0)):
        raise ValueError("Invalid teacher gain least-squares fit")
    rows = calibration.get("selected_examples")
    if (not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows) or
            not isinstance(fit.get("valid_examples"), int) or fit["valid_examples"] < 1 or
            sum(row.get("included") is True for row in rows) != fit["valid_examples"]):
        raise ValueError("Invalid teacher gain calibration cohort")
    by_source = {name: {row["id"]: row for row in source["rows"]} for name, source in sources.items()}
    seen = set()
    for row in rows:
        key = (row.get("source"), row.get("id"))
        if key in seen:
            raise ValueError("Duplicate teacher gain calibration recording")
        seen.add(key)
        source = by_source.get(key[0], {}).get(key[1])
        if (source is None or row.get("speaker") != source["speaker"] or
                row.get("path") != str(Path(source["path"]).resolve()) or
                isinstance(row.get("offset"), bool) or not isinstance(row.get("offset"), int) or row["offset"] < 0 or
                isinstance(row.get("samples"), bool) or not isinstance(row.get("samples"), int) or
                not 1 <= row["samples"] <= source["samples"] - row["offset"]):
            raise ValueError("Teacher gain calibration includes an unaudited training crop")
        if _hash(source["path"]) != row.get("audio_sha256"):
            raise ValueError("Teacher gain calibration source audio changed")
    return float(gain), {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest(),
                         "calibration": calibration}


def main():
    from .distillation import FrozenTeacher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples-per-source", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(2)
    teacher = FrozenTeacher(args.teacher, args.validation_manifest, args.device)
    result = calibrate_teacher_output_gain(teacher, examples_per_source=args.examples_per_source,
                                            batch_size=args.batch_size, crop_seconds=args.crop_seconds, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_gain": result["output_gain"], "fit": result["fit"],
                      "diagnostics": result["diagnostics"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
