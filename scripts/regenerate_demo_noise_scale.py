#!/usr/bin/env python3
"""Regenerate presentation WAVs at a given noise scale (default 0.5). Needs moe_denoiser.pt."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import joblib
import numpy as np
import torch

from moe_baseline.config import FRAME_SIZE, NUM_FAMILIES, TARGET_SR
from moe_baseline.demo import (
    build_mixed_noisy_clip,
    choose_demo_clip,
    denoise_frames,
    frames_to_waveform,
    predict_routes_with_smoothing,
    save_wav,
)
from moe_baseline.demo_routing import save_routing_bundle
from moe_baseline.config import CHANNEL_NAMES
from moe_baseline.config import data_root
from moe_baseline.librispeech import get_train_test_files
from moe_baseline.model import MoEHardDenoiser

OUT = REPO / "outputs" / "moe_presentation_demo"
CKPT = REPO / "checkpoints" / "moe_speech_demo"
NOISE_SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
VOTE_WINDOW = 5


def main():
    pt = CKPT / "moe_denoiser.pt"
    if not pt.exists():
        sys.exit(f"Missing {pt}. Run: python scripts/run_moe_presentation_demo.py --fast-train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_router = joblib.load(CKPT / "demo_router.joblib")
    model = MoEHardDenoiser(FRAME_SIZE, NUM_FAMILIES, use_shared_expert=True).to(device)
    model.load_state_dict(torch.load(pt, map_location=device))
    model.eval()

    _, test_files = get_train_test_files(data_root())
    demo_audio, _ = choose_demo_clip(test_files, seconds=6.0)
    clean_f, noisy_f, true_labels, segments = build_mixed_noisy_clip(
        demo_audio, split="val", device=device, noise_scale=NOISE_SCALE
    )
    pred_raw, pred_smooth = predict_routes_with_smoothing(
        demo_router, noisy_f, window=VOTE_WINDOW
    )
    den_smooth, _ = denoise_frames(noisy_f, model, demo_router, device, pred_smooth)
    den_oracle, _ = denoise_frames(noisy_f, model, demo_router, device, true_labels)

    wav = OUT / "wav"
    save_wav(wav / "01_original.wav", frames_to_waveform(clean_f))
    save_wav(wav / "02_mixed_noisy.wav", frames_to_waveform(noisy_f))
    save_wav(wav / "03_moe_oracle.wav", frames_to_waveform(den_oracle))
    save_wav(wav / "05_moe_classifier_smoothed.wav", frames_to_waveform(den_smooth))

    save_routing_bundle(
        OUT / f"routing_demo_ns{NOISE_SCALE:.2f}.npz",
        {
            "true_labels": true_labels.astype(np.int64),
            "pred_raw": pred_raw.astype(np.int64),
            "pred_smooth": pred_smooth.astype(np.int64),
            "segment_starts": np.array(
                [int(segments[n][0]) if len(segments[n]) else -1 for n in CHANNEL_NAMES],
                dtype=np.int64,
            ),
            "vote_window": np.int64(VOTE_WINDOW),
            "noise_scale": np.float32(NOISE_SCALE),
        },
    )
    print(f"Wrote WAVs + routing cache (noise_scale={NOISE_SCALE}) to {OUT}")


if __name__ == "__main__":
    main()
