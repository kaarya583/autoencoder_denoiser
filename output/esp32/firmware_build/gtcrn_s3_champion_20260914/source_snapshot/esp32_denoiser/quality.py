"""Optional CPU perceptual metrics; reporting only, never model selection.

Install with ``python -m pip install pesq==0.0.4 pystoi==0.4.1``.
PESQ WB reports P.862.2 MOS-LQO from the ludlows wrapper; this is not a claim
of conformance to later PESQ corrigenda. STOI is the original (not extended)
metric. Primary implementation references:
https://github.com/ludlows/PESQ
https://github.com/mpariente/pystoi
"""

from __future__ import annotations

from collections import Counter
from importlib import import_module, metadata
import math
import warnings

import numpy as np


class PerceptualMetrics:
    """Score aligned 16 kHz mono arrays without clipping or independent gains.

One common attenuation is applied to clean/noisy/enhanced if any peak exceeds
one. PESQ additionally applies its documented common reference/degraded peak
normalization internally. STOI handles its own standard resampling to 10 kHz.
Undefined speech metrics return None with a reason; unexpected library errors
propagate instead of silently disappearing from the average.
"""

    def __init__(self, sample_rate: int = 16000):
        if sample_rate != 16000:
            raise ValueError("Perceptual reporting requires 16000 Hz audio for PESQ WB")
        try:
            pesq = import_module("pesq")
            stoi = import_module("pystoi")
        except ImportError as error:
            raise RuntimeError(
                "--perceptual requires optional packages: "
                "python -m pip install pesq==0.0.4 pystoi==0.4.1"
            ) from error
        self.pesq = pesq.pesq
        self.stoi = stoi.stoi
        self.pesq_invalid = (pesq.BufferTooShortError, pesq.NoUtterancesError)
        self.sample_rate = sample_rate
        versions = {}
        for package in ("pesq", "pystoi"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = "unknown"
        self.provenance = {
            "packages": versions, "sample_rate": sample_rate,
            "pesq": "ludlows PESQ WB P.862.2 MOS-LQO; no later-corrigenda conformance claim",
            "stoi": "pystoi, extended=False; internal 10 kHz resampling",
            "amplitude_policy": "Common triplet attenuation if peak > 1; no clipping; PESQ additionally normalizes each reference/degraded pair internally",
            "selection_metric": False,
            "sources": ["https://github.com/ludlows/PESQ", "https://github.com/mpariente/pystoi"],
        }

    def __call__(self, clean, noisy, enhanced) -> dict:
        arrays = [np.asarray(value, dtype=np.float64) for value in (clean, noisy, enhanced)]
        if any(value.ndim != 1 or value.size == 0 for value in arrays) or any(
            value.shape != arrays[0].shape for value in arrays[1:]
        ):
            raise ValueError("Perceptual inputs must be aligned, nonempty mono arrays")
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("Perceptual inputs must be finite")
        peaks = [float(np.abs(value).max()) for value in arrays]
        gain = 1.0 / max(1.0, *peaks)
        result = {f"{metric}_{role}": None for metric in ("pesq_wb", "stoi")
                  for role in ("noisy", "enhanced")}
        errors = {}
        result["perceptual_common_gain"] = gain
        result["perceptual_errors"] = errors
        reference = arrays[0]
        if np.sqrt(np.mean((reference - reference.mean()) ** 2)) <= 1e-5:
            errors.update({key: "silent_reference" for key in result if key.startswith(("pesq_wb_", "stoi_"))})
            return result
        reference, noisy, enhanced = [np.ascontiguousarray(value * gain) for value in arrays]
        for role, degraded in (("noisy", noisy), ("enhanced", enhanced)):
            key = f"pesq_wb_{role}"
            try:
                value = float(self.pesq(self.sample_rate, reference, degraded, "wb"))
            except self.pesq_invalid as error:
                errors[key] = type(error).__name__
            else:
                if math.isfinite(value) and value >= 0:
                    result[key] = value
                else:
                    errors[key] = "invalid_metric_result"
            key = f"stoi_{role}"
            # pystoi otherwise returns the numeric sentinel 1e-5 when too
            # little active speech remains, which must not enter the mean.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                value = float(self.stoi(reference, degraded, self.sample_rate, extended=False))
            if any(issubclass(warning.category, RuntimeWarning) for warning in caught):
                errors[key] = "; ".join(str(warning.message) for warning in caught)
            elif math.isfinite(value):
                result[key] = value
            else:
                errors[key] = "invalid_metric_result"
        return result


def summarize_perceptual(records) -> dict:
    """Equal-utterance means on matching valid baseline/enhanced pairs.

Each metric reports its exact denominator and omitted-utterance count. A
failure on either waveform excludes that pair from both compared means.
"""
    records = list(records)
    result = {"utterances": len(records), "weighting": "equal per utterance, paired valid subset"}
    for metric in ("pesq_wb", "stoi"):
        keys = (f"{metric}_noisy", f"{metric}_enhanced")
        valid = [record for record in records if all(
            record.get(key) is not None and math.isfinite(record[key]) for key in keys)]
        total = len(valid)
        result[metric] = {
            "valid_utterances": total, "invalid_utterances": len(records) - total,
            "noisy": math.fsum(record[keys[0]] for record in valid) / total if total else None,
            "enhanced": math.fsum(record[keys[1]] for record in valid) / total if total else None,
            "improvement": math.fsum(record[keys[1]] - record[keys[0]] for record in valid) / total if total else None,
        }
    result["error_counts"] = dict(Counter(
        f"{key}: {reason}" for record in records
        for key, reason in record.get("perceptual_errors", {}).items()))
    return result
