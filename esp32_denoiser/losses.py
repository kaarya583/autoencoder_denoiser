"""Training-only spectral supervision; inference needs none of these transforms."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def compressed_spectral_loss(estimate: torch.Tensor, target: torch.Tensor,
                             lengths: torch.Tensor, *, fft_sizes=(256, 512, 1024),
                             compression: float = 0.3) -> torch.Tensor:
    """Average utterance-level magnitude/complex distances on several frame grids.

    Both signals use the same target RMS. Padding is zeroed before analysis,
    and only frames whose centers lie in the original clip count in the loss.
    This gives short utterances the same weight as long ones and prevents a
    batch's padding or amplitude from changing another utterance's loss.
    """
    if estimate.shape != target.shape or estimate.ndim != 2:
        raise ValueError("Spectral loss expects matching [batch,samples] tensors")
    if not fft_sizes or any(n < 4 or n % 4 for n in fft_sizes):
        raise ValueError("FFT sizes must be positive multiples of four")
    if not 0 < compression <= 1:
        raise ValueError("compression must be in (0,1]")
    estimate, target = estimate.float(), target.float()
    if lengths.shape != (len(target),) or bool(((lengths < 1) | (lengths > target.shape[-1])).any()):
        raise ValueError("Every target length must be positive and within its tensor")
    valid = torch.arange(target.shape[-1], device=target.device)[None] < lengths[:, None]
    target = target * valid
    estimate = estimate * valid
    rms = (target.square().sum(-1) / lengths).sqrt().clamp_min(1e-4)
    pair = torch.cat((estimate / rms[:, None], target / rms[:, None]), dim=0)
    losses = []
    for size in fft_sizes:
        hop = size // 4
        window = torch.hann_window(size, device=pair.device)
        spectrum = torch.stft(F.pad(pair, (size // 2, size // 2)), size,
                              hop_length=hop, window=window, center=False,
                              return_complex=True) / size
        magnitude = spectrum.abs().clamp_min(1e-8)
        compressed = magnitude.pow(compression)
        complex_values = spectrum * magnitude.pow(compression - 1)
        em, tm = compressed.chunk(2)
        ec, tc = complex_values.chunk(2)
        distance = 0.7 * (em - tm).abs() + 0.3 * (ec - tc).abs()
        frame_mask = torch.arange(distance.shape[-1], device=pair.device)[None] * hop < lengths[:, None]
        per_frame = distance.mean(1)
        losses.append(((per_frame * frame_mask).sum(-1) / frame_mask.sum(-1)).mean())
    return torch.stack(losses).mean()


def calibrate_spectral_weight(model, batches, primary_loss, *, target_ratio=0.1, max_batches=8) -> dict:
    """Set an auxiliary coefficient from training-only parameter gradient norms.

    This measures a local optimization scale, not the best loss weight. The
    returned coefficient stays fixed for its ablation and must be calibrated
    on training data, never on validation or final test utterances.
    """
    if not math.isfinite(target_ratio) or target_ratio <= 0 or max_batches < 1:
        raise ValueError("Positive finite target_ratio and max_batches are required")
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    device = parameters[0].device
    was_training = model.training
    measurements = []
    try:
        model.eval()
        for index, batch in enumerate(batches):
            if index >= max_batches:
                break
            noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
            lengths = batch["length"].to(device)
            estimate = model(noisy)
            primary = primary_loss(estimate, clean, lengths)
            auxiliary = compressed_spectral_loss(estimate, clean, lengths)
            norms = []
            for loss, retain in ((primary, True), (auxiliary, False)):
                gradients = torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True)
                norms.append(math.sqrt(sum(float(g.detach().double().square().sum())
                                           for g in gradients if g is not None)))
            if all(math.isfinite(n) and n > 0 for n in norms):
                measurements.append({"primary_norm": norms[0], "spectral_norm": norms[1],
                                     "suggested_weight": target_ratio * norms[0] / norms[1]})
    finally:
        model.train(was_training)
    if not measurements:
        raise ValueError("No finite nonzero paired gradient norms for loss calibration")
    weights = sorted(m["suggested_weight"] for m in measurements)
    middle = len(weights) // 2
    median = weights[middle] if len(weights) % 2 else (weights[middle-1] + weights[middle]) / 2
    return {"spectral_loss_weight": median, "target_gradient_ratio": target_ratio,
            "source": "training batches only", "batches": measurements}
