"""Speech denoising demo: waveform log-FFT router + MoE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from moe_baseline.audio import frame_audio, load_wav
from moe_baseline.config import CHANNEL_NAMES, FRAME_SIZE, HOP_SIZE, NUM_FAMILIES, TARGET_SR
from moe_baseline.noise import apply_family_noise
from moe_baseline.routers import predict_waveform_route_batch


def channel_segments(n_frames: int, n_channels: int = NUM_FAMILIES) -> dict[str, np.ndarray]:
    """Split frame indices into contiguous segments (one per noise family)."""
    splits = np.array_split(np.arange(n_frames), n_channels)
    return {CHANNEL_NAMES[i]: splits[i] for i in range(n_channels)}


def build_mixed_noisy_clip(
    audio: np.ndarray,
    split: str = "val",
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
  Frame clean speech and inject a different noise family per segment.

  Returns:
    clean_frames, noisy_frames, true_family_per_frame, segments
    """
    dev = device or torch.device("cpu")
    clean_frames = frame_audio(audio, FRAME_SIZE, HOP_SIZE)
    if clean_frames.shape[0] == 0:
        raise ValueError("Audio too short for one frame.")

    noisy_frames = clean_frames.copy()
    true_labels = np.zeros(len(clean_frames), dtype=np.int64)
    segments = channel_segments(len(clean_frames))

    for fam_id, name in enumerate(CHANNEL_NAMES):
        for i in segments[name]:
            clean_t = torch.from_numpy(clean_frames[i])
            noisy_frames[i] = apply_family_noise(clean_t, fam_id, split, device=dev).cpu().numpy()
            true_labels[i] = fam_id

    return clean_frames, noisy_frames, true_labels, segments


def frames_to_waveform(frames: np.ndarray) -> np.ndarray:
    """Non-overlapping frames (hop == frame_size) → 1-D waveform."""
    return frames.reshape(-1).astype(np.float32)


@torch.no_grad()
def denoise_frames(
    noisy_frames: np.ndarray,
    model: torch.nn.Module,
    waveform_router,
    device: torch.device,
    route_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
  Denoise framed audio with MoE + external waveform RF router.

  If route_indices is None, predict routes with waveform_router.
  """
    model.eval()
    n = len(noisy_frames)
    out = np.zeros_like(noisy_frames)

    if route_indices is None:
        route_indices = predict_waveform_route_batch(waveform_router, noisy_frames)

    for i in range(n):
        x = torch.from_numpy(noisy_frames[i].astype(np.float32)).unsqueeze(0).to(device)
        r = torch.tensor([int(route_indices[i])], dtype=torch.long, device=device)
        y = model.forward_with_route(x, r)
        out[i] = y.cpu().numpy()[0]

    return out, route_indices


def routing_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    return float((true_labels == pred_labels).mean())


def smooth_route_predictions(pred: np.ndarray, window: int = 5) -> np.ndarray:
    """Majority vote over a sliding window to reduce per-frame routing flicker."""
    pred = np.asarray(pred, dtype=np.int64)
    if window <= 1 or len(pred) == 0:
        return pred.copy()
    window = max(3, int(window) | 1)  # odd window
    half = window // 2
    out = pred.copy()
    for i in range(len(pred)):
        chunk = pred[max(0, i - half) : min(len(pred), i + half + 1)]
        counts = np.bincount(chunk, minlength=NUM_FAMILIES)
        out[i] = int(np.argmax(counts))
    return out


def predict_routes_with_smoothing(waveform_router, noisy_frames: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Raw RF predictions and majority-smoothed routes."""
    raw = predict_waveform_route_batch(waveform_router, noisy_frames)
    smooth = smooth_route_predictions(raw, window=window)
    return raw, smooth


def noise_reduction_db(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> float:
    """How much frame energy vs clean improved (dB); higher is better."""
    err_in = float(np.mean((noisy - clean) ** 2))
    err_out = float(np.mean((denoised - clean) ** 2))
    return float(10.0 * np.log10((err_in + 1e-12) / (err_out + 1e-12)))


def clip_noise_reduction_db(clean_frames: np.ndarray, noisy_frames: np.ndarray, denoised_frames: np.ndarray) -> float:
    """Mean per-frame noise reduction (dB) across the clip."""
    vals = [
        noise_reduction_db(clean_frames[i], noisy_frames[i], denoised_frames[i])
        for i in range(len(clean_frames))
    ]
    return float(np.mean(vals))


def save_wav(path: Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr)


def choose_demo_clip(file_paths: list, seconds: float = 6.0) -> tuple[np.ndarray, Path]:
    """Pick first file long enough for a short demo."""
    min_samples = int(seconds * TARGET_SR)
    for path in file_paths:
        try:
            audio = load_wav(path)
            if len(audio) >= min_samples:
                return audio[:min_samples], Path(path)
        except Exception:
            continue
    raise FileNotFoundError("No suitable demo clip found in file list.")
