"""Train MoE + routers for presentation demo checkpoints."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from moe_baseline.audio import FrameDataset
from moe_baseline.config import (
    FRAME_SIZE,
    MAX_TEST_FILES,
    MAX_TRAIN_FILES,
    NOISE_CLF_N_PER_FAMILY,
    NUM_FAMILIES,
    SEED,
    data_root,
)
from moe_baseline.datasets import NoisyFrameDataset
from moe_baseline.librispeech import get_train_test_files
from moe_baseline.model import MoEHardDenoiser
from moe_baseline.routers import train_demo_router_speech, train_waveform_router
from moe_baseline.train import eval_denoising_oracle, train_moe_speech_with_denoising

TRAIN_PROFILES = {
    "fast": {
        "train_cap": 200,
        "test_cap": 80,
        "n_per_family": 1500,
        "demo_router_samples": 10000,
        "joint_epochs": 3,
        "denoise_epochs": 18,
        "router_only_epochs": 2,
        "mse_weight": 0.08,
        "batches_per_family": 120,
        "use_si_sdr": False,
        "denoise_lr": 2e-4,
    },
    "quality": {
        "train_cap": MAX_TRAIN_FILES,
        "test_cap": MAX_TEST_FILES,
        "n_per_family": NOISE_CLF_N_PER_FAMILY,
        "demo_router_samples": 12000,
        "joint_epochs": 6,
        "denoise_epochs": 22,
        "router_only_epochs": 3,
        "mse_weight": 0.1,
        "batches_per_family": 150,
        "use_si_sdr": False,
        "denoise_lr": 2e-4,
    },
}


def train_presentation_checkpoints(
    ckpt_dir: Path,
    device: torch.device,
    profile: str = "fast",
) -> dict[str, float]:
    """Train waveform + demo routers and MoE denoiser; save to ckpt_dir."""
    if profile not in TRAIN_PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {list(TRAIN_PROFILES)}")
    cfg = TRAIN_PROFILES[profile]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_files, test_files = get_train_test_files(data_root())
    train_files = train_files[: cfg["train_cap"]]
    test_files = test_files[: cfg["test_cap"]]

    print(f"Training profile={profile!r} | train files={len(train_files)} | test={len(test_files)}")
    train_frame_ds = FrameDataset(train_files)
    train_ds = NoisyFrameDataset(train_frame_ds, "train", np.random.default_rng(SEED + 2))
    test_frame_ds = FrameDataset(test_files)
    test_ds = NoisyFrameDataset(test_frame_ds, "val", np.random.default_rng(SEED + 3))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    waveform_router, _, _ = train_waveform_router(device=device, n_per_family=cfg["n_per_family"])
    demo_router, _ = train_demo_router_speech(train_ds, max_n=cfg["demo_router_samples"])
    joblib.dump(waveform_router, ckpt_dir / "waveform_router.joblib")
    joblib.dump(demo_router, ckpt_dir / "demo_router.joblib")

    model = MoEHardDenoiser(FRAME_SIZE, NUM_FAMILIES, use_shared_expert=True).to(device)
    denoise_metrics = train_moe_speech_with_denoising(
        model,
        train_frame_ds,
        train_loader,
        test_loader,
        device,
        joint_epochs=cfg["joint_epochs"],
        denoise_epochs=cfg["denoise_epochs"],
        router_only_epochs=cfg["router_only_epochs"],
        mse_weight=cfg["mse_weight"],
        batches_per_family=cfg["batches_per_family"],
        use_si_sdr=cfg["use_si_sdr"],
        denoise_lr=cfg["denoise_lr"],
    )
    torch.save(model.state_dict(), ckpt_dir / "moe_denoiser.pt")
    meta = {
        "profile": profile,
        "val_nr_db": float(denoise_metrics["noise_reduction_db"]),
        "val_mse": float(denoise_metrics["mse"]),
    }
    joblib.dump(meta, ckpt_dir / "train_meta.joblib")
    print(f"Saved checkpoints to {ckpt_dir} | val NR {meta['val_nr_db']:.2f} dB")
    return meta


def checkpoint_val_nr(ckpt_dir: Path, device: torch.device) -> float | None:
    """Oracle val NR for saved MoE (quick check)."""
    pt = ckpt_dir / "moe_denoiser.pt"
    if not pt.exists():
        return None
    meta_path = ckpt_dir / "train_meta.joblib"
    if meta_path.exists():
        return float(joblib.load(meta_path).get("val_nr_db", 0))

    from moe_baseline.config import FRAME_SIZE, NUM_FAMILIES

    train_files, test_files = get_train_test_files(data_root())
    test_loader = DataLoader(
        NoisyFrameDataset(
            FrameDataset(test_files[:60]), "val", np.random.default_rng(SEED + 3)
        ),
        batch_size=256,
        shuffle=False,
    )
    model = MoEHardDenoiser(FRAME_SIZE, NUM_FAMILIES, use_shared_expert=True).to(device)
    model.load_state_dict(torch.load(pt, map_location=device))
    return float(eval_denoising_oracle(model, test_loader, device, max_batches=30)["noise_reduction_db"])
