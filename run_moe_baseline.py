#!/usr/bin/env python3
"""
Full MoE pipeline: LibriSpeech denoising + multiple routers.

For **pure-noise routing only (no speech)** with plots and confusion matrices,
open and run:  MoE_Denoiser_Baseline.ipynb

Same LibriSpeech file subsets as Adaptive_Autoencoders_Project.ipynb (cell 12).

Usage:
  python run_moe_baseline.py
  python run_moe_baseline.py --epochs 10 --router-only-epochs 3
  python run_moe_baseline.py --routers-only   # latent + waveform + speech RF only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from moe_baseline.config import data_root
from moe_baseline.train import run_full_pipeline


def main():
    p = argparse.ArgumentParser(description="MoE denoiser baseline training")
    p.add_argument("--data-root", type=Path, default=None, help="LibriSpeech root")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--router-only-epochs", type=int, default=5)
    p.add_argument(
        "--denoise-epochs",
        type=int,
        default=15,
        help="Oracle-route denoising epochs per family (0 to skip)",
    )
    p.add_argument(
        "--routers-only",
        action="store_true",
        help="Train fair/demo RF only (no MoE neural training)",
    )
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.routers_only:
        import numpy as np
        from moe_baseline.audio import FrameDataset
        from moe_baseline.datasets import NoisyFrameDataset
        from moe_baseline.config import SEED
        from moe_baseline.librispeech import get_train_test_files
        from moe_baseline.routers import (
            train_demo_router_speech,
            train_fair_router,
            train_fair_router_waveform,
        )

        root = args.data_root or data_root()
        train_files, _ = get_train_test_files(root)
        train_frame_ds = FrameDataset(train_files)
        train_ds = NoisyFrameDataset(train_frame_ds, "train", np.random.default_rng(SEED + 2))
        train_fair_router(device=device)
        train_fair_router_waveform(device=device)
        train_demo_router_speech(train_ds)
        return

    run_full_pipeline(
        args.data_root,
        device=device,
        epochs=args.epochs,
        router_only_epochs=args.router_only_epochs,
        denoise_epochs=args.denoise_epochs,
    )


if __name__ == "__main__":
    main()
