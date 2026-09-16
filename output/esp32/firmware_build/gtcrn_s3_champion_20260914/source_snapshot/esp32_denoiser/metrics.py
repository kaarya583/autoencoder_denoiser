"""Zero-mean SI-SDR and honest, equally weighted full-utterance evaluation."""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch


def si_sdr(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    lengths: torch.Tensor | None = None,
    *,
    silence_rms: float = 1e-5,
    max_db: float = 80.0,
) -> torch.Tensor:
    """Return zero-mean SI-SDR per signal, with time on the final dimension.

    Inputs are matching ``[T]`` or ``[B,T]`` floating tensors. ``lengths`` masks
    trailing padding before computing means and energies. Computation uses
    float64 for metric stability; the result is not reduced across utterances.

    Silent/DC-only references (centered RMS <= silence_rms) and nonfinite valid
    samples produce NaN, making invalid examples explicit. Zero estimates on
    valid speech receive ``-max_db`` rather than being silently discarded.
    Finite scores are capped to +/-max_db; numerical floors are relative to
    estimate energy so ordinary scaling of the estimate does not change them.
    """
    if estimate.shape != reference.shape or estimate.ndim not in (1, 2) or estimate.shape[-1] == 0:
        raise ValueError("estimate and reference must have matching nonempty [T] or [B,T] shapes")
    if not estimate.is_floating_point() or not reference.is_floating_point():
        raise TypeError("SI-SDR inputs must be floating-point tensors")
    if estimate.device != reference.device:
        raise ValueError("estimate and reference must be on the same device")
    if not math.isfinite(silence_rms) or silence_rms < 0 or not math.isfinite(max_db) or max_db <= 0:
        raise ValueError("silence_rms must be nonnegative and max_db positive")
    single = estimate.ndim == 1
    estimate = estimate.reshape(-1, estimate.shape[-1]).double()
    reference = reference.reshape_as(estimate).double()
    batch, samples = estimate.shape
    if lengths is None:
        lengths = torch.full((batch,), samples, device=estimate.device, dtype=torch.long)
    else:
        lengths = torch.as_tensor(lengths, device=estimate.device)
        if lengths.ndim == 0 and single:
            lengths = lengths.unsqueeze(0)
        if lengths.shape != (batch,) or lengths.is_floating_point() or (lengths < 1).any() or (lengths > samples).any():
            raise ValueError("lengths must contain one valid integer sample count per signal")
    mask = torch.arange(samples, device=estimate.device).unsqueeze(0) < lengths.unsqueeze(1)
    estimate = torch.where(mask, estimate, 0.0)
    reference = torch.where(mask, reference, 0.0)
    finite = torch.isfinite(estimate).all(-1) & torch.isfinite(reference).all(-1)
    count = lengths.unsqueeze(1)
    estimate = torch.where(mask, estimate - estimate.sum(-1, keepdim=True) / count, 0.0)
    reference = torch.where(mask, reference - reference.sum(-1, keepdim=True) / count, 0.0)
    reference_energy = reference.square().sum(-1, keepdim=True)
    estimate_energy = estimate.square().sum(-1, keepdim=True)
    tiny = torch.finfo(torch.float64).tiny
    projection = (estimate * reference).sum(-1, keepdim=True) * reference / reference_energy.clamp_min(tiny)
    residual = estimate - projection
    projected_fraction = projection.square().sum(-1) / estimate_energy.squeeze(-1).clamp_min(tiny)
    residual_fraction = residual.square().sum(-1) / estimate_energy.squeeze(-1).clamp_min(tiny)
    floor = 10.0 ** (-max_db / 10.0)
    score = (10 * torch.log10(projected_fraction.clamp_min(floor) / residual_fraction.clamp_min(floor))).clamp(-max_db, max_db)
    score = torch.where(estimate_energy.squeeze(-1) <= tiny, -max_db, score)
    valid = finite & (reference_energy.squeeze(-1) / lengths > silence_rms**2)
    score = torch.where(valid, score, torch.nan)
    return score[0] if single else score


def si_sdri(estimate: torch.Tensor, noisy: torch.Tensor, reference: torch.Tensor,
            lengths: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
    """SI-SDR improvement over the paired noisy signal (dB per utterance)."""
    return si_sdr(estimate, reference, lengths, **kwargs) - si_sdr(noisy, reference, lengths, **kwargs)


def summarize_utterances(records: Iterable[dict]) -> dict:
    """Summarize scored records with one equal vote per valid utterance.

    Records contain ``id``, ``si_sdr_noisy``, ``si_sdr_enhanced``. Counts include
    invalid/silent references; all-undefined results return None, never 0 dB.
    Duplicate IDs fail rather than accidentally weighting an utterance twice.
    """
    records = list(records)
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate utterance IDs in evaluation")
    valid = [r for r in records if math.isfinite(r["si_sdr_noisy"]) and math.isfinite(r["si_sdr_enhanced"])]
    count = len(valid)
    return {
        "utterances": len(records), "valid_utterances": count, "invalid_utterances": len(records) - count,
        "si_sdr_noisy": math.fsum(r["si_sdr_noisy"] for r in valid) / count if count else None,
        "si_sdr_enhanced": math.fsum(r["si_sdr_enhanced"] for r in valid) / count if count else None,
        "si_sdri": math.fsum(r["si_sdr_enhanced"] - r["si_sdr_noisy"] for r in valid) / count if count else None,
        "weighting": "equal per utterance", "metric": "zero-mean SI-SDR, capped +/-80 dB",
    }


@torch.inference_mode()
def evaluate_utterances(
    enhance: Callable[[torch.Tensor], torch.Tensor],
    dataset,
    *,
    device: str | torch.device = "cpu",
    max_utterances: int | None = None,
) -> dict:
    """Score full clips separately, preserving equal utterance weighting.

    ``enhance`` receives ``[1,T]`` and must return the same shape after its own
    latency compensation; it must reset streaming state for every call. Pass
    a dataset created with ``crop_seconds=None, random_crop=False, gain_db=(0,0)``.
    A callable wrapper can adapt a streaming model. For torch modules, this
    helper temporarily enters eval mode and restores the previous mode.
    Nonfinite model output raises, preventing model failures from disappearing
    through NaN-skipping. Silent references are counted as invalid.
    """
    if (getattr(dataset, "crop_range", None) is not None
            or getattr(dataset, "gain_db", (0, 0)) != (0, 0)
            or getattr(dataset, "noise_scale_db", (0, 0)) != (0, 0)
            or getattr(dataset, "clean_identity_prob", 0) != 0):
        raise ValueError("Evaluation requires full utterances and no augmentation")
    if max_utterances is not None and max_utterances < 1:
        raise ValueError("max_utterances must be positive or None")
    was_training = enhance.training if isinstance(enhance, torch.nn.Module) else None
    if was_training is not None:
        enhance.eval()
    records = []
    try:
        count = len(dataset) if max_utterances is None else min(len(dataset), max_utterances)
        for index in range(count):
            item = dataset[index]
            length = int(item["length"])
            noisy = item["noisy"][:length].unsqueeze(0).to(device)
            clean = item["clean"][:length].unsqueeze(0).to(device)
            enhanced = enhance(noisy)
            if not isinstance(enhanced, torch.Tensor) or enhanced.shape != noisy.shape:
                raise ValueError(f"Enhancer changed waveform shape for {item['id']}")
            if not torch.isfinite(enhanced).all():
                raise ValueError(f"Nonfinite enhanced audio for {item['id']}")
            baseline = float(si_sdr(noisy, clean).item())
            result = float(si_sdr(enhanced, clean).item())
            records.append({"id": item["id"], "samples": length, "si_sdr_noisy": baseline,
                            "si_sdr_enhanced": result, "si_sdri": result - baseline})
    finally:
        if was_training is not None:
            enhance.train(was_training)
    return {"summary": summarize_utterances(records), "utterances": records}
