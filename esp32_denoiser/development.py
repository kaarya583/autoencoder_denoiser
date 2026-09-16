"""Freeze external development mixtures without accessing the official test set.

Default CLI: ``python -m esp32_denoiser.development --root /content/extra_audio``.
Reads extra_data's held-out LibriSpeech/MUSAN manifests, renders 100 shared
speech/noise crops at -5/0/5/10/20 dB active-frame SNR, and adds 100 separate
clean-preservation examples. Original training groups are audited as disjoint.
Output FLOAT WAVs retain exact mixed samples; the embedded evaluator applies
its normal PCM16 conversion. These are development crops, not a new benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch

from .data import read_manifest
from .extra_data import DynamicMixtureDataset, SAMPLE_RATE, _atomic_json, _hash_file, read_extra_manifest
from .metrics import summarize_utterances


VERSION = 1
DEFAULT_SNRS = (-5, 0, 5, 10, 20)


def _item_seed(seed, suite, index):
    digest = hashlib.sha256(f"esp32-development-v{VERSION}:{seed}:{suite}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _audit_holdouts(manifest_dir):
    records, provenance = {}, {}
    for kind in ("speech", "noise"):
        train_path, val_path = [manifest_dir / f"{kind}_{split}.jsonl" for split in ("train", "val")]
        train, val = read_extra_manifest(train_path, kind), read_extra_manifest(val_path, kind)
        if {r["split"] for r in train} != {"train"} or {r["split"] for r in val} != {"val"}:
            raise ValueError(f"Incorrect {kind} train/validation manifest partition")
        for key in ("id", "group", "path", "sha256"):
            if {r[key] for r in train} & {r[key] for r in val}:
                raise ValueError(f"Development source overlaps training {kind} {key}")
        records[kind] = val
        provenance[kind] = {"train_manifest_sha256": _hash_file(train_path)["sha256"],
                            "val_manifest_sha256": _hash_file(val_path)["sha256"],
                            "val_groups": sorted({r["group"] for r in val}),
                            "val_records": len(val), "source": val[0]["source"],
                            "license": val[0]["license"]}
    return records, provenance


def _write_audio(path, audio):
    """Write deterministic IEEE-float WAV bytes without timestamped PEAK chunks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    wavfile.write(temporary, SAMPLE_RATE, np.asarray(audio, dtype="<f4"))
    temporary.replace(path)
    return _hash_file(path)["sha256"]


def prepare_development(root: str | Path, *, output_dir: str | Path | None = None,
                        num_mixtures: int = 500, snrs=DEFAULT_SNRS, clean_examples: int = 100,
                        crop_seconds: float = 3.0, seed: int = 2026) -> dict[str, Path]:
    """Freeze matched SNR conditions and a separate clean-preservation suite.

``root/manifests`` must contain all four audited speech/noise train/val files
from extra_data. Only val audio is sampled. The mixture count must be divisible
by the number of SNRs: each base crop is repeated at every SNR. A separate seed
namespace selects clean-suite crops. Sampling preserves the caller's Torch RNG
state and is independent of item order. Changing parameters or source manifests
requires a new output directory; existing experiment definitions are not replaced.

Returns ``mixtures``, ``clean``, ``provenance`` Paths. WAVs exclude padding, and
both JSONL manifests are directly compatible with PairedAudioDataset/evaluate.
"""
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else root / "development"
    snrs = tuple(float(value) for value in snrs)
    if not snrs or not all(math.isfinite(value) for value in snrs) or len(set(snrs)) != len(snrs):
        raise ValueError("snrs must contain distinct finite levels")
    if isinstance(num_mixtures, bool) or not isinstance(num_mixtures, int) or num_mixtures < 1 or num_mixtures % len(snrs):
        raise ValueError("num_mixtures must be positive and divisible by the number of SNRs")
    if isinstance(clean_examples, bool) or not isinstance(clean_examples, int) or clean_examples < 1:
        raise ValueError("clean_examples must be a positive integer")
    if not math.isfinite(crop_seconds) or crop_seconds <= 0:
        raise ValueError("crop_seconds must be positive")
    manifest_dir = root / "manifests"
    source_records, source_provenance = _audit_holdouts(manifest_dir)
    specification = {"version": VERSION, "sample_rate": SAMPLE_RATE, "seed": seed,
                     "num_mixtures": num_mixtures, "snr_db": list(snrs),
                     "base_crops": num_mixtures // len(snrs), "clean_examples": clean_examples,
                     "crop_seconds": crop_seconds, "sources": source_provenance,
                     "sample_format": "WAV IEEE FLOAT; no clipping or quantization at rendering",
                     "gain_db": [0, 0], "shared_pair_peak_limit": .99,
                     "snr_definition": "reference-frame power within 20 dB of maximum, nonoverlapping 20 ms frames",
                     "official_test_accessed": False}
    provenance_path = output_dir / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()).get("specification") != specification:
        raise ValueError("Development specification changed; choose a new output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    speech_path, noise_path = manifest_dir / "speech_val.jsonl", manifest_dir / "noise_val.jsonl"
    datasets = [DynamicMixtureDataset(speech_path, noise_path, crop_seconds=crop_seconds,
                                     snr_db=(snr, snr), gain_db=(0, 0), clean_identity_prob=0) for snr in snrs]
    clean_dataset = DynamicMixtureDataset(speech_path, noise_path, crop_seconds=crop_seconds,
                                         gain_db=(0, 0), clean_identity_prob=1)
    by_id = {row["id"]: row for records in source_records.values() for row in records}
    verified = set()
    rows = {"mixtures": [], "clean": []}

    def render(dataset, suite, base_index, condition_index=None):
        item_seed = _item_seed(seed, suite, base_index)
        with torch.random.fork_rng(devices=[]):
            # Seed CPU only: CUDA state is neither touched nor initialized.
            torch.random.default_generator.manual_seed(item_seed)
            speech_index = int(torch.randint(len(dataset.speech_records), ()).item())
            item = dataset[speech_index]
        mixture = item["mixture"]
        for identifier in (mixture["speech_id"], mixture["noise_id"]):
            if identifier is not None and identifier not in verified:
                if _hash_file(by_id[identifier]["path"])["sha256"] != by_id[identifier]["sha256"]:
                    raise ValueError(f"Source waveform changed after manifest preparation: {identifier}")
                verified.add(identifier)
        suffix = "" if condition_index is None else f"_condition{condition_index:02d}"
        identifier = f"extra_development_{suite}_{base_index:05d}{suffix}"
        length = int(item["length"])
        record = {"id": identifier, "speaker": item["speaker"], "samples": length,
                  "sample_rate": SAMPLE_RATE, "source_split": "development",
                  "suite": suite, "base_crop": base_index, "item_seed": item_seed,
                  "speech_source_sha256": by_id[mixture["speech_id"]]["sha256"],
                  "noise_source_sha256": by_id[mixture["noise_id"]]["sha256"] if mixture["noise_id"] else None,
                  **mixture}
        for role in ("clean", "noisy"):
            destination = output_dir / "audio" / suite / role / f"{identifier}.wav"
            record[f"{role}_sha256"] = _write_audio(destination, item[role][:length].numpy())
            record[role] = os.path.relpath(destination, output_dir)
        rows[suite].append(record)

    for base in range(num_mixtures // len(snrs)):
        for condition, dataset in enumerate(datasets):
            render(dataset, "mixtures", base, condition)
    for base in range(clean_examples):
        render(clean_dataset, "clean", base)
    outputs, summaries = {}, {}
    for suite, records in rows.items():
        path = output_dir / f"{suite}.jsonl"
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
        temporary.replace(path)
        outputs[suite] = path
        summaries[suite] = {"manifest_sha256": _hash_file(path)["sha256"], "utterances": len(records),
                            "speakers": sorted({r["speaker"] for r in records}),
                            "speech_recordings": len({r["speech_id"] for r in records}),
                            "noise_recordings": len({r["noise_id"] for r in records} - {None}),
                            "duration_seconds": sum(r["samples"] for r in records) / SAMPLE_RATE}
    _atomic_json(provenance_path, {"specification": specification, "suites": summaries,
                                  "verified_source_files": len(verified),
                                  "interpretation": "Fixed development crops for model selection; no official-test claims"})
    outputs["provenance"] = provenance_path
    return outputs


def _score_rows(records):
    if isinstance(records, dict):
        records = records["utterances"]
    records = list(records)
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate scored utterance IDs")
    return [{**row, **{key: float(row[key]) if row[key] is not None else float("nan")
                      for key in ("si_sdr_noisy", "si_sdr_enhanced")}} for row in records]


def common_valid_ids(*evaluations, metric_keys=("si_sdr_noisy", "si_sdr_enhanced")) -> set[str]:
    """Return IDs having every requested finite metric in every evaluation.

Pass this set to summarize_development for like-for-like model comparisons.
For PESQ/STOI comparisons, request their paired noisy/enhanced keys explicitly;
an unavailable perceptual metric must not silently change the SI-SDR cohort.
"""
    if not evaluations or not metric_keys:
        raise ValueError("Provide at least one evaluation and metric key")
    valid_sets = []
    for evaluation in evaluations:
        rows = _score_rows(evaluation)
        valid_sets.append({row["id"] for row in rows if all(row.get(key) is not None and math.isfinite(row[key]) for key in metric_keys)})
    return set.intersection(*valid_sets)


def preservation_metrics(noisy, clean, estimate, *, length: int | None = None,
                         silence_rms: float = 1e-5) -> dict:
    """Measure amplitude and waveform changes that SI-SDR deliberately ignores.

Inputs are aligned, matching one-dimensional NumPy arrays or Torch tensors.
Optional ``length`` excludes trailing padding; all reported counts use only
valid samples. Projection gain is the signed, uncentered least-squares gain
onto clean speech; RMS ratio and L1 likewise preserve the original DC/level.
Normalized waveform L1 divides mean absolute error by reference RMS.

For near-silent clean references, gain/normalized ratios are None; waveform
error, output peak, and PCM16 counts remain defined. Counts distinguish values
outside [-1, 32767/32768] from touching/exceeding its rails. Rail contact alone
does not prove clipping occurred. Nonfinite valid audio fails explicitly.
"""
    arrays = []
    for value in (noisy, clean, estimate):
        if isinstance(value, torch.Tensor):
            value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
        arrays.append(np.asarray(value, dtype=np.float64))
    if any(value.ndim != 1 or value.shape != arrays[0].shape for value in arrays) or not arrays[0].size:
        raise ValueError("Preservation metrics require matching nonempty one-dimensional waveforms")
    if length is None:
        length = arrays[0].size
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or not 1 <= length <= arrays[0].size:
        raise ValueError("length must be a valid integer sample count")
    if not math.isfinite(silence_rms) or silence_rms < 0:
        raise ValueError("silence_rms must be finite and nonnegative")
    noisy, clean, estimate = [value[:length] for value in arrays]
    if not all(np.isfinite(value).all() for value in (noisy, clean, estimate)):
        raise ValueError("Nonfinite waveform in preservation metrics")
    reference_energy = float(np.dot(clean, clean))
    reference_rms = math.sqrt(reference_energy / length)
    valid_reference = reference_rms > silence_rms

    def score(value):
        waveform_l1 = float(np.mean(np.abs(value - clean)))
        return {"projection_gain": float(np.dot(value, clean) / reference_energy) if valid_reference else None,
                "rms_ratio": float(np.sqrt(np.mean(value**2)) / reference_rms) if valid_reference else None,
                "normalized_waveform_l1": waveform_l1 / reference_rms if valid_reference else None,
                "waveform_l1": waveform_l1, "peak_abs": float(np.max(np.abs(value))),
                "pcm16_out_of_range_samples": int(np.count_nonzero((value < -1) | (value > 32767 / 32768))),
                "pcm16_rail_or_exceeds_samples": int(np.count_nonzero((value <= -1) | (value >= 32767 / 32768)))}

    return {"samples": int(length), "reference_rms": reference_rms, "valid_reference": valid_reference,
            "noisy": score(noisy), "enhanced": score(estimate)}


def summarize_development(records, manifest, *, common_ids=None) -> dict:
    """Equal-utterance SI-SDR summaries overall and by SNR and speaker.

Unknown/duplicate evaluation IDs fail. Missing evaluations, invalid metrics,
and explicit common-cohort exclusions are counted separately. ``common_ids``
must be available and SI-SDR-valid in this model, including when it is empty.
Clean-suite SI-SDRi uses the capped perfect-input baseline; interpret absolute
enhanced SI-SDR for preservation rather than comparing its SI-SDRi to mixtures.
"""
    metadata = read_manifest(manifest) if isinstance(manifest, (str, Path)) else list(manifest)
    if len({row["id"] for row in metadata}) != len(metadata):
        raise ValueError("Duplicate development manifest IDs")
    if not metadata or any(row.get("source_split") != "development" for row in metadata):
        raise ValueError("Expected a development-only manifest")
    scored = _score_rows(records)
    by_id = {row["id"]: row for row in metadata}
    if set(row["id"] for row in scored) - set(by_id):
        raise ValueError("Scored IDs are absent from the development manifest")
    included = scored
    if common_ids is not None:
        common_ids = set(common_ids)
        if not common_ids <= common_valid_ids(scored):
            raise ValueError("Common cohort includes missing or invalid SI-SDR records")
        included = [row for row in scored if row["id"] in common_ids]

    def grouped(key):
        groups = {}
        for row in included:
            value = by_id[row["id"]].get(key)
            label = "clean" if key == "snr_db" and value is None else str(value)
            groups.setdefault(label, []).append(row)
        return {label: summarize_utterances(group) for label, group in sorted(groups.items())}

    return {"manifest_utterances": len(metadata), "evaluated_utterances": len(scored),
            "missing_utterances": len(metadata) - len(scored),
            "excluded_by_common_cohort": len(scored) - len(included),
            "common_cohort_used": common_ids is not None,
            "summary": summarize_utterances(included),
            "by_snr_db": grouped("snr_db"), "by_speaker": grouped("speaker"),
            "clean_suite_note": "Clean-input SI-SDR is capped at 80 dB; use enhanced SI-SDR to assess preservation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-mixtures", type=int, default=500)
    parser.add_argument("--clean-examples", type=int, default=100)
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--crop-seconds", type=float, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    outputs = prepare_development(args.root, output_dir=args.output_dir, num_mixtures=args.num_mixtures,
                                  clean_examples=args.clean_examples, snrs=args.snrs,
                                  crop_seconds=args.crop_seconds, seed=args.seed)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
