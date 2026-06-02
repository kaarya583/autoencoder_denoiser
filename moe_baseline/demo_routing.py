"""Fast routing bundle for MoE demo notebook (no MoE forward pass)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from moe_baseline.config import CHANNEL_NAMES, NUM_FAMILIES, SEED
from moe_baseline.demo import build_mixed_noisy_clip, predict_routes_with_smoothing


def compute_routing_bundle(
    clean_audio: np.ndarray,
    demo_router,
    split: str = "val",
    vote_window: int = 5,
    device=None,
) -> dict:
    """Rebuild mixed-noise frames and RF routes (seconds on CPU)."""
    import torch

    dev = device or torch.device("cpu")
    clean_frames, noisy_frames, true_labels, segments = build_mixed_noisy_clip(
        clean_audio, split=split, device=dev
    )
    pred_raw, pred_smooth = predict_routes_with_smoothing(
        demo_router, noisy_frames, window=vote_window
    )
    seg_starts = np.array([int(segments[name][0]) if len(segments[name]) else -1 for name in CHANNEL_NAMES])
    return {
        "true_labels": true_labels.astype(np.int64),
        "pred_raw": pred_raw.astype(np.int64),
        "pred_smooth": pred_smooth.astype(np.int64),
        "noisy_frames": noisy_frames.astype(np.float32),
        "clean_frames": clean_frames.astype(np.float32),
        "segment_starts": seg_starts,
        "channel_names": np.array(CHANNEL_NAMES),
        "vote_window": np.int64(vote_window),
    }


def save_routing_bundle(path: Path, bundle: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle)


def load_routing_bundle(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_or_compute_routing(
    out_dir: Path,
    ckpt_dir: Path,
    clean_audio: np.ndarray,
    vote_window: int = 5,
) -> dict:
    """Load cached routing or compute from demo router checkpoint."""
    cache = out_dir / "routing_demo.npz"
    if cache.exists():
        return load_routing_bundle(cache)

    router_path = ckpt_dir / "demo_router.joblib"
    if not router_path.exists():
        raise FileNotFoundError(
            f"No {cache} and no {router_path}. Run: python scripts/run_moe_presentation_demo.py"
        )
    demo_router = joblib.load(router_path)
    bundle = compute_routing_bundle(clean_audio, demo_router, vote_window=vote_window)
    save_routing_bundle(cache, bundle)
    return bundle
