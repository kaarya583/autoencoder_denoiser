"""Paired development comparisons with explicit cohorts and bootstrap units.

Accepts training ``per_utterance`` records (noisy_si_sdr / si_sdr) or
evaluation ``utterances`` records (si_sdr_noisy / si_sdr_enhanced). Neither
summary averages nor positional row matching are used. Positive differences
always favor the candidate. These intervals describe the observed development
population, not a new-speaker or new-noise population.
"""
from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


def _number(value, field):
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be numeric or null")
    return float(value)


def _index(rows, name):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} must contain a nonempty record list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"{name} needs nonempty string IDs")
        if row["id"] in result:
            raise ValueError(f"Duplicate ID in {name}: {row['id']}")
        result[row["id"]] = row
    return result


def _evaluation(value):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if not isinstance(value, dict):
        raise ValueError("Evaluation must be a JSON object or its path")
    training = isinstance(value.get("per_utterance"), list)
    if training and value.get("invalid_utterances", 0):
        raise ValueError("Training export omits invalid IDs; export every scored row before comparing")
    rows = _index(value.get("per_utterance") if training else value.get("utterances"), "evaluation")
    normalized = {}
    for item_id, original in rows.items():
        row = dict(original)
        for target, source in (("si_sdr_noisy", "noisy_si_sdr"), ("si_sdr_enhanced", "si_sdr")):
            key = source if training else target
            if key not in row:
                raise ValueError(f"Missing {key} for {item_id}")
            row[target] = _number(row[key], key)
        improvement = row["si_sdr_enhanced"] - row["si_sdr_noisy"]
        if "si_sdri" in row and math.isfinite(improvement):
            recorded = _number(row["si_sdri"], "si_sdri")
            if not math.isfinite(recorded) or abs(recorded - improvement) > 1e-6:
                raise ValueError(f"Inconsistent SI-SDR improvement for {item_id}")
        row["si_sdri"] = improvement
        normalized[item_id] = row
    return value, normalized


def _same_baseline(candidate, reference, key, item_id, tolerance, *, allow_missing=False):
    first, second = _number(candidate.get(key), key), _number(reference.get(key), key)
    if allow_missing and not (math.isfinite(first) and math.isfinite(second)):
        return
    if math.isfinite(first) != math.isfinite(second):
        raise ValueError(f"Noisy baseline validity differs for {item_id}: {key}")
    if math.isfinite(first) and abs(first - second) > tolerance:
        raise ValueError(f"Noisy baseline differs for {item_id}: {key}")


def _audio_hashes(row, item_id):
    if "audio_sha256" not in row:
        return None
    hashes = row["audio_sha256"]
    if (not isinstance(hashes, dict) or set(hashes) != {"clean", "noisy"} or
            any(not isinstance(value, str) or len(value) != 64 or
                set(value.lower()) - set("0123456789abcdef") for value in hashes.values())):
        raise ValueError(f"Invalid audio_sha256 for {item_id}; expected clean/noisy SHA256 values")
    return {key: value.lower() for key, value in hashes.items()}


def _interval(differences, units, samples, seed):
    # Sorted units and IDs make results independent of input record order.
    unique = sorted(set(units))
    positions = {unit: index for index, unit in enumerate(unique)}
    unit_indices = np.asarray([positions[unit] for unit in units])
    sums = np.bincount(unit_indices, weights=differences, minlength=len(unique))
    counts = np.bincount(unit_indices, minlength=len(unique))
    random = np.random.default_rng(seed)
    means = np.empty(samples)
    batch = max(1, min(256, 1_000_000 // len(unique)))
    for offset in range(0, samples, batch):
        selected = random.integers(len(unique), size=(min(batch, samples-offset), len(unique)))
        means[offset:offset+len(selected)] = sums[selected].sum(1) / counts[selected].sum(1)
    return np.quantile(means, (0.025, 0.975)).tolist(), len(unique)


def compare_evaluations(candidate, reference, *, manifest=None, cluster_key=None,
                        bootstrap_samples=10000, seed=2026, baseline_tolerance=1e-6):
    """Compare strictly paired records; use ``cluster_key='base_crop'`` for SNR sweeps.

    Manifest IDs must exactly equal both evaluations. A cluster is the pair
    (suite, cluster_key), avoiding collisions between clean and mixture suites.
    Cluster resampling preserves all finite conditions of each selected crop;
    point estimates and bootstrap ratios still weight each utterance equally.
    PESQ/STOI are included when present in both inputs, with their own common
    finite cohort and matched noisy baselines. Missing scores remain counted.
    Per-utterance audio hashes must match when both reports supply them. Older
    reports remain comparable, with the unverified audio cohort made explicit.
    """
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or not 100 <= bootstrap_samples <= 100000:
        raise ValueError("bootstrap_samples must be an integer in [100,100000]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(baseline_tolerance) or baseline_tolerance < 0:
        raise ValueError("baseline_tolerance must be finite and nonnegative")
    candidate_document, candidate = _evaluation(candidate)
    reference_document, reference = _evaluation(reference)
    if candidate.keys() != reference.keys():
        raise ValueError("Candidate and reference must have exactly the same ID sets")
    for key in ("manifest_sha256",):
        if candidate_document.get(key) and reference_document.get(key) and candidate_document[key] != reference_document[key]:
            raise ValueError(f"Evaluation {key} differs")
    ids = sorted(candidate)
    metadata = None
    if manifest is not None:
        if isinstance(manifest, (str, Path)):
            manifest = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        metadata = _index(manifest, "manifest")
        if metadata.keys() != candidate.keys():
            raise ValueError("Manifest and evaluations must have exactly the same ID sets")
    if cluster_key is not None and (not isinstance(cluster_key, str) or not cluster_key or metadata is None):
        raise ValueError("A nonempty cluster_key requires a manifest")
    units = {}
    audio_verified, samples_verified = 0, 0
    for item_id in ids:
        first, second = candidate[item_id], reference[item_id]
        _same_baseline(first, second, "si_sdr_noisy", item_id, baseline_tolerance)
        if "samples" in first and "samples" in second:
            if first["samples"] != second["samples"]:
                raise ValueError(f"Scored sample counts differ for {item_id}")
            samples_verified += 1
        first_hashes, second_hashes = _audio_hashes(first, item_id), _audio_hashes(second, item_id)
        if first_hashes is not None and second_hashes is not None:
            if first_hashes != second_hashes:
                raise ValueError(f"Scored audio_sha256 differs for {item_id}")
            audio_verified += 1
        units[item_id] = item_id
        if cluster_key is not None:
            row = metadata[item_id]
            value = row.get(cluster_key)
            if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
                raise ValueError(f"Missing or invalid cluster {cluster_key} for {item_id}")
            suite = row.get("suite", "")
            if not isinstance(suite, str):
                raise ValueError(f"Invalid manifest suite for {item_id}")
            units[item_id] = json.dumps([suite, value])

    metric_baselines = {"si_sdri": "si_sdr_noisy"}
    for metric, noisy in (("pesq_wb_enhanced", "pesq_wb_noisy"), ("stoi_enhanced", "stoi_noisy")):
        if any(metric in row for row in candidate.values()) and any(metric in row for row in reference.values()):
            metric_baselines[metric] = noisy
    results = {}
    for metric, noisy_key in metric_baselines.items():
        valid_ids, first_values, second_values = [], [], []
        for item_id in ids:
            first, second = candidate[item_id], reference[item_id]
            _same_baseline(first, second, noisy_key, item_id, baseline_tolerance,
                           allow_missing=metric != "si_sdri")
            values = (_number(first.get(metric), metric), _number(second.get(metric), metric),
                      _number(first.get(noisy_key), noisy_key), _number(second.get(noisy_key), noisy_key))
            if all(math.isfinite(value) for value in values):
                valid_ids.append(item_id)
                first_values.append(values[0])
                second_values.append(values[1])
        differences = np.asarray(first_values) - np.asarray(second_values)
        count = len(differences)
        interval, unit_count = _interval(differences, [units[item_id] for item_id in valid_ids], bootstrap_samples, seed) if count else (None, 0)
        results[metric] = {
            "common_finite_utterances": count, "excluded_utterances": len(ids)-count,
            "bootstrap_units": unit_count,
            "candidate_mean": float(np.mean(first_values)) if count else None,
            "reference_mean": float(np.mean(second_values)) if count else None,
            "mean_difference": float(np.mean(differences)) if count else None,
            "confidence_interval_95": interval,
            "wins": int(np.count_nonzero(differences > 0)),
            "losses": int(np.count_nonzero(differences < 0)),
            "ties": int(np.count_nonzero(differences == 0)),
            "win_fraction": float(np.mean(differences > 0)) if count else None,
        }
    speakers = sorted({row["speaker"] for row in metadata.values() if isinstance(row.get("speaker"), str)}) if metadata else []
    if not speakers and all(item_id.startswith(("p226_", "p287_")) for item_id in ids):
        speakers = sorted({item_id.split("_", 1)[0] for item_id in ids})
    scope = "Conditional on the evaluated speakers and recordings; not an independent-speaker population confidence interval."
    if set(speakers) == {"p226", "p287"}:
        scope = "VoiceBank utterance comparison conditional on the two held-out speakers p226/p287; not an independent-speaker population confidence interval."
    if cluster_key:
        scope += " Crop clustering accounts for repeated SNR conditions, not dependence between distinct crops sharing a speaker or recording."
    return {"direction": "candidate minus reference; positive favors candidate", "paired_utterances": len(ids),
            "weighting": "equal per common finite utterance", "metrics": results,
            "verification": {
                "manifest_sha256_matched": bool(candidate_document.get("manifest_sha256") and reference_document.get("manifest_sha256")),
                "sample_counts_matched_utterances": samples_verified,
                "audio_sha256_matched_utterances": audio_verified,
                "audio_sha256_unverified_utterances": len(ids) - audio_verified,
                "scope": "IDs and noisy baselines match. Audio bytes are bound only for utterances with matching clean/noisy hashes in both reports; historical omissions are not byte verification."},
            "bootstrap": {"method": "paired percentile", "samples": bootstrap_samples, "seed": seed,
                          "unit": f"(suite, {cluster_key})" if cluster_key else "utterance",
                          "scope": scope, "speakers": speakers},
            "baseline_tolerance": baseline_tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cluster-key")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_evaluations(args.candidate, args.reference, manifest=args.manifest,
                                 cluster_key=args.cluster_key, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
