"""Fit one deployed DSP output gain from audited clean training crops only.

This writes a calibration report, never a model or an evaluation result. The
source EDNSI8 binary must have unity output gain. C float-output inference uses
PCM16-rounded inputs, so the fit sees the actual integer neural/frontend path
before output saturation. No development waveform is used in the objective.
"""
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
from .embedded import EmbeddedWaveformEnhancer
from .export import IntegerDenoiser, MAGIC
from .extra_data import read_extra_manifest


METHOD = "equal-example reference-energy-normalized positive least squares on clean training crops"


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _snapshot(path, extra=False):
    path = Path(path).resolve()
    contents = path.read_bytes()
    rows = read_extra_manifest(path, "speech") if extra else read_manifest(path)
    if path.read_bytes() != contents:
        raise ValueError("Gain manifest changed while loading")
    return {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest(),
            "records": len(rows)}, rows


def _verify_audio(path, samples, declared_hash, cache):
    path = str(Path(path).resolve())
    if path not in cache:
        signature = _signature(path)
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.frames < 1:
            raise ValueError(f"Gain sources must be nonempty 16kHz mono audio: {path}")
        digest = _hash(path)
        if _signature(path) != signature:
            raise ValueError("Gain source audio changed while hashing")
        cache[path] = {"sha256": digest, "samples": info.frames, "signature": signature}
    value = cache[path]
    if value["samples"] != samples:
        raise ValueError(f"Gain source whole-file length mismatch: {path}")
    if declared_hash is not None and declared_hash != value["sha256"]:
        raise ValueError(f"Gain source SHA256 mismatch: {path}")
    return value["sha256"]


def _audit_sources(paths):
    manifests, rows, files = {}, {}, {}
    if len({str(Path(path).resolve()) for path in paths.values()}) != len(paths):
        raise ValueError("Gain fitting requires distinct training and development manifests")
    for name, path in paths.items():
        manifests[name], rows[name] = _snapshot(path, extra=name.startswith("speech_"))
    for name, expected in (("speech_train", "train"), ("speech_development", "val")):
        if {row["split"] for row in rows[name]} != {expected}:
            raise ValueError(f"Gain source has the wrong partition: {name}")
    if any(row.get("source_split") != "train" for row in rows["paired_train"]):
        raise ValueError("Gain calibration requires training-origin paired clean speech")
    if any(row.get("source_split") not in {"train", "development"} for row in rows["primary_development"]):
        raise ValueError("Primary gain exclusion manifest must be development, never official test")
    for name, suite in (("external_mixtures", "mixtures"), ("external_clean", "clean")):
        if any(row.get("source_split") != "development" or row.get("suite") != suite for row in rows[name]):
            raise ValueError(f"Gain exclusion manifest must be the external development {suite} suite")

    excluded = {key: set() for key in ("id", "speaker", "group", "path", "sha256")}
    for name in ("primary_development", "speech_development", "external_mixtures", "external_clean"):
        for row in rows[name]:
            for key in ("id", "speech_id", "noise_id"):
                if row.get(key) is not None:
                    excluded["id"].add(row[key])
            if row.get("speaker") is None:
                raise ValueError("Held-out gain exclusions need explicit speaker identity")
            excluded["speaker"].add(row["speaker"])
            excluded["group"].add(row.get("group", row["speaker"]))
            for key in ("speech_group", "noise_group"):
                if row.get(key) is not None:
                    excluded["group"].add(row[key])
            for key in ("speech_source_sha256", "noise_source_sha256"):
                if row.get(key) is not None:
                    excluded["sha256"].add(row[key])
            source_paths = [(row["path"], row.get("sha256"))] if name == "speech_development" else [
                (row[role], row.get(role + "_sha256")) for role in ("clean", "noisy")]
            for path, digest in source_paths:
                excluded["path"].add(str(Path(path).resolve()))
                excluded["sha256"].add(_verify_audio(path, row["samples"], digest, files))
    heldout_files = len(files)
    for name in ("paired_train", "speech_train"):
        for row in rows[name]:
            row["path"] = row["clean"] if name == "paired_train" else row["path"]
            identity = {"id": row["id"], "speaker": row["speaker"],
                        "group": row.get("group", row["speaker"]),
                        "path": str(Path(row["path"]).resolve())}
            for key, value in identity.items():
                if value in excluded[key]:
                    raise ValueError(f"Gain training source overlaps development {key}: {row['id']}")
            digest = row.get("sha256") if name == "speech_train" else row.get("clean_sha256")
            if digest is not None and digest in excluded["sha256"]:
                raise ValueError(f"Gain training source overlaps development content hash: {row['id']}")
    return manifests, rows, excluded, files, heldout_files


@torch.inference_mode()
def calibrate_integer_output_gain(binary, *, paired_train_manifest, speech_train_manifest,
                                  primary_development_manifest, speech_development_manifest,
                                  external_mixtures_manifest, external_clean_manifest,
                                  examples_per_source=64, crop_seconds=3.0, seed=2026):
    """Return a fit/provenance report for a frozen, uncorrected EDNSI8 binary.

    Select exactly the requested number of distinct recordings from EACH
    training source. Near-silent crops are recorded and skipped; insufficient
    eligible recordings fail. The clean target retains its original level.
    Normalization is by each target's mean square, with RMS>1e-5 eligibility,
    making every included crop's reference energy exactly one in the fit.
    """
    if isinstance(examples_per_source, bool) or not isinstance(examples_per_source, int) or not 1 <= examples_per_source <= 4096:
        raise ValueError("examples_per_source must be an integer in [1,4096]")
    if isinstance(crop_seconds, bool) or not math.isfinite(crop_seconds) or not 0 < crop_seconds <= 30:
        raise ValueError("crop_seconds must be finite and in (0,30]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    binary = Path(binary).resolve()
    blob = binary.read_bytes()
    if blob[:8] != MAGIC:
        raise ValueError("Output-gain calibration currently supports EDNSI8 only")
    metadata = IntegerDenoiser(blob)
    if metadata.output_gain != 1.0:
        raise ValueError("Fit only an uncorrected binary with output_gain=1.0")
    if len(blob) > 99_000:
        raise ValueError("Source integer model exceeds the 99,000-byte deployment budget")
    paths = {"paired_train": paired_train_manifest, "speech_train": speech_train_manifest,
             "primary_development": primary_development_manifest, "speech_development": speech_development_manifest,
             "external_mixtures": external_mixtures_manifest, "external_clean": external_clean_manifest}
    manifests, sources, excluded, files, heldout_files = _audit_sources(paths)
    if any(len(sources[name]) < examples_per_source for name in ("paired_train", "speech_train")):
        raise ValueError("Insufficient distinct clean training recordings for the requested balanced fit")
    random = np.random.default_rng(seed)
    selected, skipped, gains = [], [], []
    fit = {"cross_sum": 0.0, "estimate_energy_sum": 0.0, "reference_energy_sum": 0.0,
           "valid_examples": 0, "input_clipped_samples": 0, "input_samples": 0}
    enhancer = EmbeddedWaveformEnhancer(blob, "float32")
    seen_paths, seen_hashes = set(), set()
    try:
        for name in ("paired_train", "speech_train"):
            accepted = 0
            rows = sources[name]
            for index in random.permutation(len(rows)):
                row = rows[int(index)]
                path = str(Path(row["path"]).resolve())
                declared = row.get("sha256") if name == "speech_train" else row.get("clean_sha256")
                digest = _verify_audio(path, row["samples"], declared, files)
                if digest in excluded["sha256"]:
                    raise ValueError(f"Selected gain training audio overlaps development content hash: {row['id']}")
                if path in seen_paths or digest in seen_hashes:
                    raise ValueError("Gain calibration must use distinct training paths and content")
                seen_paths.add(path)
                seen_hashes.add(digest)
                length = min(row["samples"], max(1, round(crop_seconds * 16000)))
                offset = int(random.integers(row["samples"] - length + 1))
                waveform, rate = sf.read(path, start=offset, frames=length, dtype="float32")
                if rate != 16000 or waveform.ndim != 1 or len(waveform) != length or not np.isfinite(waveform).all():
                    raise ValueError("Invalid gain training crop")
                record = {"source": name, "id": row["id"], "speaker": row["speaker"],
                          "group": row.get("group", row["speaker"]), "path": path,
                          "audio_sha256": digest, "offset": offset, "samples": length}
                target = waveform.astype(np.float64)
                reference_energy = float(np.mean(target * target))
                if reference_energy <= 1e-10:
                    skipped.append({**record, "reason": "reference RMS <= 1e-5"})
                    continue
                scaled = np.rint(waveform * 32768.0)
                clipped = int(np.count_nonzero((scaled < -32768) | (scaled > 32767)))
                encoded = np.clip(scaled, -32768, 32767).astype(np.float32) / 32768.0
                estimate = enhancer(torch.from_numpy(encoded)[None])[0].double().numpy()
                if estimate.shape != target.shape or not np.isfinite(estimate).all():
                    raise ValueError("Invalid full-C response during output gain calibration")
                cross = float(np.mean(estimate * target)) / reference_energy
                energy = float(np.mean(estimate * estimate)) / reference_energy
                fit["cross_sum"] += cross
                fit["estimate_energy_sum"] += energy
                fit["reference_energy_sum"] += 1.0
                fit["valid_examples"] += 1
                fit["input_clipped_samples"] += clipped
                fit["input_samples"] += length
                gains.append(cross)
                selected.append({**record, "reference_rms": math.sqrt(reference_energy),
                                 "raw_projection_gain": cross, "normalized_estimate_energy": energy,
                                 "input_clipped_samples": clipped})
                accepted += 1
                if accepted == examples_per_source:
                    break
            if accepted != examples_per_source:
                raise ValueError(f"Insufficient nonsilent clean training recordings: {name}")
    finally:
        enhancer.close()
    numerator, denominator = fit["cross_sum"], fit["estimate_energy_sum"]
    if not math.isfinite(denominator) or denominator <= 0 or not math.isfinite(numerator) or numerator <= 0:
        raise ValueError("No finite positive least-squares output gain from C responses")
    gain = numerator / denominator
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Output gain must be finite and positive")
    if binary.read_bytes() != blob or any(_hash(value["path"]) != value["sha256"] for value in manifests.values()):
        raise ValueError("Frozen gain binary or manifest changed during calibration")
    if any(_signature(path) != value["signature"] for path, value in files.items()):
        raise ValueError("Audited gain audio changed during calibration")
    count = fit["valid_examples"]
    return {"version": 1, "method": METHOD, "output_gain": gain,
            "source_model": {"path": str(binary), "sha256": hashlib.sha256(blob).hexdigest(),
                             "bytes": len(blob), "output_gain": metadata.output_gain},
            "manifests": manifests, "selected_examples": selected, "skipped_examples": skipped,
            "sampling": {"examples_per_source": examples_per_source, "crop_seconds": crop_seconds,
                         "seed": seed, "sample_rate": 16000, "augmentation": "none"},
            "inference": {"backend": "actual INT8 C neural core and float32 C DSP, portable host FFT",
                          "input": "nearest-even signed PCM16 rounding/clipping then decode at 1/32768",
                          "output": "unclipped C float32 output before final PCM16 conversion",
                          "target": "original clean training crop; no independent level normalization"},
            "exclusions": {"source": "all supplied primary/external development identities and speech-source inventory",
                           "heldout_audio_files_hashed": heldout_files,
                           "identity_counts": {key: len(values) for key, values in excluded.items()}},
            "fit": fit,
            "diagnostics": {"raw_mean_projection_gain": float(np.mean(gains)),
                            "corrected_mean_projection_gain": gain * float(np.mean(gains)),
                            "normalized_mse_before": (denominator - 2*numerator + count) / count,
                            "normalized_mse_after": (gain*gain*denominator - 2*gain*numerator + count) / count},
            "implementation_sha256": {"output_gain.py": _hash(__file__)},
            "scope": "Training-only calibration. No development fit, model write, or final inference. Float64 fitted gain must be rounded to the serialized float32 value and evaluated independently."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integer-model", required=True, type=Path)
    for name in ("paired-train", "speech-train", "primary-development", "speech-development", "external-mixtures", "external-clean"):
        parser.add_argument("--" + name + "-manifest", required=True, type=Path)
    parser.add_argument("--examples-per-source", type=int, default=64)
    parser.add_argument("--crop-seconds", type=float, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    values = vars(args).copy()
    output, binary = values.pop("output"), values.pop("integer_model")
    if output.exists():
        parser.error("Use a new calibration report path; existing reports are not overwritten")
    result = calibrate_integer_output_gain(binary, **values)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_gain": result["output_gain"], "diagnostics": result["diagnostics"], "output": str(output)}))


if __name__ == "__main__":
    main()
