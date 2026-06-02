#!/usr/bin/env python3
"""Run speech demo pipeline and write outputs (uses checkpoints if present)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from moe_baseline.config import CHANNEL_NAMES, FRAME_SIZE, NUM_FAMILIES, SEED, data_root
from moe_baseline.librispeech import get_train_test_files
from moe_baseline.model import MoEHardDenoiser
from moe_baseline.demo import (
    build_mixed_noisy_clip,
    choose_demo_clip,
    clip_noise_reduction_db,
    denoise_frames,
    frames_to_waveform,
    predict_routes_with_smoothing,
    routing_accuracy,
    save_wav,
)
from moe_baseline.train import set_seed

DEMO_SECONDS = 6.0
ROUTE_VOTE_WINDOW = 5
OUT_DIR = REPO / "outputs" / "moe_speech_demo"
CKPT_DIR = REPO / "checkpoints" / "moe_speech_demo"


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints...")
    demo_router = joblib.load(CKPT_DIR / "demo_router.joblib")
    waveform_router = joblib.load(CKPT_DIR / "waveform_router.joblib")
    model = MoEHardDenoiser(FRAME_SIZE, NUM_FAMILIES, use_shared_expert=True).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "moe_denoiser.pt", map_location=device))

    _, test_files = get_train_test_files(data_root())
    demo_audio, demo_path = choose_demo_clip(test_files, seconds=DEMO_SECONDS)
    print("Demo:", demo_path, f"({len(demo_audio)/16000:.1f}s)")

    clean_frames, noisy_frames, true_labels, segments = build_mixed_noisy_clip(
        demo_audio, split="val", device=device
    )
    clean_audio = frames_to_waveform(clean_frames)
    noisy_audio = frames_to_waveform(noisy_frames)

    save_wav(OUT_DIR / "01_clean_speech.wav", clean_audio)
    save_wav(OUT_DIR / "02_mixed_noisy.wav", noisy_audio)

    pred_raw, pred_smooth = predict_routes_with_smoothing(
        demo_router, noisy_frames, window=ROUTE_VOTE_WINDOW
    )
    pred_fair, _ = predict_routes_with_smoothing(waveform_router, noisy_frames, window=1)

    den_oracle, _ = denoise_frames(noisy_frames, model, demo_router, device, true_labels)
    den_raw, _ = denoise_frames(noisy_frames, model, demo_router, device, pred_raw)
    den_smooth, _ = denoise_frames(noisy_frames, model, demo_router, device, pred_smooth)
    den_fair, _ = denoise_frames(noisy_frames, model, waveform_router, device, pred_fair)

    metrics = {
        "demo_rf_raw_acc": routing_accuracy(true_labels, pred_raw),
        "demo_rf_smooth_acc": routing_accuracy(true_labels, pred_smooth),
        "fair_rf_acc": routing_accuracy(true_labels, pred_fair),
        "nr_db_oracle": clip_noise_reduction_db(clean_frames, noisy_frames, den_oracle),
        "nr_db_demo_raw": clip_noise_reduction_db(clean_frames, noisy_frames, den_raw),
        "nr_db_demo_smooth": clip_noise_reduction_db(clean_frames, noisy_frames, den_smooth),
    }

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if "acc" in k else f"  {k}: {v:.2f} dB")

    save_wav(OUT_DIR / "03_denoised_oracle_route.wav", frames_to_waveform(den_oracle))
    save_wav(OUT_DIR / "04_denoised_demo_raw.wav", frames_to_waveform(den_raw))
    save_wav(OUT_DIR / "05_denoised_demo_smoothed.wav", frames_to_waveform(den_smooth))
    save_wav(OUT_DIR / "06_denoised_fair_router.wav", frames_to_waveform(den_fair))

    pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()]).to_csv(
        OUT_DIR / "demo_metrics.csv", index=False
    )

    sr = 16000
    n_show = int(min(4 * sr, len(clean_audio)))
    t = np.arange(n_show) / sr
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, clean_audio[:n_show], lw=0.7)
    axes[0].set_title("Clean")
    axes[1].plot(t, noisy_audio[:n_show], lw=0.7)
    axes[1].set_title("Mixed noisy")
    axes[2].plot(t, frames_to_waveform(den_raw)[:n_show], lw=0.7)
    axes[2].set_title(f"Denoised raw (acc {metrics['demo_rf_raw_acc']:.2f})")
    axes[3].plot(t, frames_to_waveform(den_smooth)[:n_show], lw=0.7)
    axes[3].set_title(f"Denoised smoothed (acc {metrics['demo_rf_smooth_acc']:.2f})")
    axes[3].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "waveforms_comparison.png", dpi=120)
    plt.close()

    print("\nWrote outputs to", OUT_DIR)
    print(classification_report(true_labels, pred_smooth, target_names=CHANNEL_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
