#!/usr/bin/env python3
"""
MoE presentation demo — mirrors adaptive mixed-noise clip demo.

Produces slide-ready WAVs, figures (150 dpi), and metrics under
outputs/moe_presentation_demo/.

Usage:
  python scripts/run_moe_presentation_demo.py
  python scripts/run_moe_presentation_demo.py --fast-train   # if checkpoints missing
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from moe_baseline.config import CHANNEL_NAMES, FRAME_SIZE, NUM_FAMILIES, SEED, TARGET_SR, data_root
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
from moe_baseline.librispeech import get_train_test_files
from moe_baseline.model import MoEHardDenoiser
from moe_baseline.train import set_seed

OUT_DIR = REPO / "outputs" / "moe_presentation_demo"
CKPT_DIR = REPO / "checkpoints" / "moe_speech_demo"
ADAPTIVE_DIR = REPO / "outputs"
DEMO_SECONDS = 6.0
ROUTE_VOTE_WINDOW = 5
DPI = 150


def _ensure_checkpoints(device: torch.device, fast_train: bool) -> None:
    need = not all(
        (CKPT_DIR / n).exists()
        for n in ("demo_router.joblib", "waveform_router.joblib", "moe_denoiser.pt")
    )
    if not need:
        return
    if not fast_train:
        raise FileNotFoundError(
            f"Missing checkpoints in {CKPT_DIR}. Run with --fast-train or train via MoE_Speech_Denoising_Demo.ipynb."
        )
    print("FAST_MODE training (checkpoints missing)...")
    from torch.utils.data import DataLoader

    from moe_baseline.audio import FrameDataset
    from moe_baseline.datasets import NoisyFrameDataset
    from moe_baseline.routers import train_demo_router_speech, train_waveform_router
    from moe_baseline.train import train_moe_speech_with_denoising

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_files, test_files = get_train_test_files(data_root())
    train_frame_ds = FrameDataset(train_files[:80])
    train_ds = NoisyFrameDataset(train_frame_ds, "train", np.random.default_rng(SEED + 2))
    test_loader = DataLoader(
        NoisyFrameDataset(
            FrameDataset(test_files[:40]), "val", np.random.default_rng(SEED + 3)
        ),
        batch_size=256,
        shuffle=False,
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)

    waveform_router, _, _ = train_waveform_router(device=device, n_per_family=500)
    demo_router, _ = train_demo_router_speech(train_ds, max_n=4000)
    joblib.dump(waveform_router, CKPT_DIR / "waveform_router.joblib")
    joblib.dump(demo_router, CKPT_DIR / "demo_router.joblib")

    model = MoEHardDenoiser(FRAME_SIZE, NUM_FAMILIES, use_shared_expert=True).to(device)
    train_moe_speech_with_denoising(
        model,
        train_frame_ds,
        train_loader,
        test_loader,
        device,
        joint_epochs=2,
        denoise_epochs=4,
        router_only_epochs=1,
        mse_weight=0.05,
        batches_per_family=40,
        use_si_sdr=False,
    )
    torch.save(model.state_dict(), CKPT_DIR / "moe_denoiser.pt")
    print("Saved checkpoints to", CKPT_DIR)


def _copy_adaptive_reference(dest: Path) -> list[str]:
    """Copy adaptive demo WAVs for side-by-side comparison in slides."""
    dest.mkdir(parents=True, exist_ok=True)
    mapping = {
        "original_demo.wav": "adaptive_original.wav",
        "oracle_adaptive_reconstruction.wav": "adaptive_oracle.wav",
        "classifier_adaptive_reconstruction.wav": "adaptive_classifier.wav",
    }
    copied = []
    for src_name, dst_name in mapping.items():
        src = ADAPTIVE_DIR / src_name
        if src.exists():
            shutil.copy2(src, dest / dst_name)
            copied.append(dst_name)
    return copied


def _plot_waveforms(
    clean: np.ndarray,
    noisy: np.ndarray,
    oracle: np.ndarray,
    classifier: np.ndarray,
    metrics: dict,
    path: Path,
    seconds_show: float = 4.0,
) -> None:
    n = int(min(seconds_show * TARGET_SR, len(clean)))
    t = np.arange(n) / TARGET_SR
    fig, axes = plt.subplots(4, 1, figsize=(11, 6.5), sharex=True)
    titles = [
        "Clean speech (reference)",
        "Mixed noise (6 families, 6 s clip)",
        f"MoE + oracle route (NR {metrics['nr_db_oracle']:.1f} dB)",
        f"MoE + RF route smoothed (acc {metrics['routing_acc_smoothed']:.0%}, NR {metrics['nr_db_classifier']:.1f} dB)",
    ]
    for ax, sig, title in zip(
        axes,
        [clean[:n], noisy[:n], oracle[:n], classifier[:n]],
        titles,
    ):
        ax.plot(t, sig, lw=0.65, color="C0")
        ax.set_ylabel("Amp")
        ax.set_title(title, fontsize=10)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("MoE denoiser — presentation demo (matches adaptive 6-segment protocol)", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_routing(true_labels, pred_raw, pred_smooth, segments, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.scatter(np.arange(len(true_labels)), true_labels, s=14, c="black", label="True family", zorder=3)
    ax.scatter(np.arange(len(pred_raw)), pred_raw, s=10, c="C1", alpha=0.45, label="RF raw")
    ax.scatter(np.arange(len(pred_smooth)), pred_smooth, s=12, c="C2", marker="x", label=f"RF smoothed (w={ROUTE_VOTE_WINDOW})")
    for name, idx in segments.items():
        if len(idx):
            ax.axvline(idx[0], color="gray", ls=":", lw=0.5, alpha=0.7)
    ax.set_yticks(range(NUM_FAMILIES))
    ax.set_yticklabels(CHANNEL_NAMES)
    ax.set_xlabel("Frame index")
    ax.set_title("Per-frame routing on mixed-noise clip")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics_bar(metrics: dict, path: Path) -> None:
    labels = [
        "Routing acc\n(raw RF)",
        "Routing acc\n(smoothed)",
        "Noise reduction\n(oracle MoE)",
        "Noise reduction\n(classifier MoE)",
    ]
    values = [
        metrics["routing_acc_raw"],
        metrics["routing_acc_smoothed"],
        metrics["nr_db_oracle"] / 10.0,
        metrics["nr_db_classifier"] / 10.0,
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.axhline(1 / NUM_FAMILIES, color="gray", ls="--", lw=1, label="Chance (routing)")
    ax.set_ylabel("Accuracy (routing) or NR/10 dB (denoise)")
    ax.set_title("MoE presentation metrics")
    ax.set_ylim(0, max(max(values) * 1.15, 0.25))
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    path: Path,
    demo_path: Path,
    metrics: dict,
    adaptive_copied: list[str],
) -> None:
    lines = [
        "# MoE presentation demo",
        "",
        f"- **Clip:** `{demo_path.name}` ({DEMO_SECONDS}s, 16 kHz)",
        f"- **Protocol:** 6 contiguous segments, one noise family each (same as adaptive notebook)",
        "",
        "## MoE metrics (this run)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Routing accuracy (RF raw) | {metrics['routing_acc_raw']:.1%} |",
        f"| Routing accuracy (RF smoothed, w={ROUTE_VOTE_WINDOW}) | {metrics['routing_acc_smoothed']:.1%} |",
        f"| Noise reduction — oracle route | {metrics['nr_db_oracle']:.2f} dB |",
        f"| Noise reduction — classifier route | {metrics['nr_db_classifier']:.2f} dB |",
        "",
        "## Audio files (for slides / live demo)",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `wav/01_original.wav` | Clean speech |",
        "| `wav/02_mixed_noisy.wav` | Six-family mixed noise |",
        "| `wav/03_moe_oracle.wav` | MoE + true family (upper bound) |",
        "| `wav/04_moe_classifier.wav` | MoE + log-FFT RF (smoothed) — **main result** |",
        "",
        "## Adaptive comparison (pre-generated)",
        "",
        "Run `Adaptive_Autoencoders_Project.ipynb` demo cells first. Reference copies:",
        "",
        "| Adaptive (repo root `outputs/`) | Copied to `adaptive_reference/` |",
        "|----------------------------------|--------------------------------|",
        "| `original_demo.wav` | `adaptive_original.wav` |",
        "| `oracle_adaptive_reconstruction.wav` | `adaptive_oracle.wav` |",
        "| `classifier_adaptive_reconstruction.wav` | `adaptive_classifier.wav` |",
        "",
    ]
    if adaptive_copied:
        lines.append(f"Copied: {', '.join(adaptive_copied)}")
    else:
        lines.append("_Adaptive WAVs not found — run adaptive notebook demo first._")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/waveforms_comparison.png` — 4-panel waveform slide",
            "- `figures/routing_timeline.png` — true vs predicted routes",
            "- `figures/metrics_bar.png` — single-slide metrics",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="MoE presentation demo")
    p.add_argument("--fast-train", action="store_true", help="Train checkpoints if missing (FAST_MODE)")
    args = p.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav_dir = OUT_DIR / "wav"
    fig_dir = OUT_DIR / "figures"
    ref_dir = OUT_DIR / "adaptive_reference"
    wav_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _ensure_checkpoints(device, args.fast_train)

    print("Loading MoE checkpoints...")
    demo_router = joblib.load(CKPT_DIR / "demo_router.joblib")
    model = MoEHardDenoiser(FRAME_SIZE, NUM_FAMILIES, use_shared_expert=True).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "moe_denoiser.pt", map_location=device))
    model.eval()

    _, test_files = get_train_test_files(data_root())
    demo_audio, demo_path = choose_demo_clip(test_files, seconds=DEMO_SECONDS)
    print(f"Demo clip: {demo_path} ({len(demo_audio) / TARGET_SR:.1f}s)")

    clean_frames, noisy_frames, true_labels, segments = build_mixed_noisy_clip(
        demo_audio, split="val", device=device
    )
    clean_audio = frames_to_waveform(clean_frames)
    noisy_audio = frames_to_waveform(noisy_frames)

    pred_raw, pred_smooth = predict_routes_with_smoothing(
        demo_router, noisy_frames, window=ROUTE_VOTE_WINDOW
    )
    den_oracle, _ = denoise_frames(noisy_frames, model, demo_router, device, true_labels)
    den_classifier, _ = denoise_frames(noisy_frames, model, demo_router, device, pred_smooth)

    metrics = {
        "routing_acc_raw": routing_accuracy(true_labels, pred_raw),
        "routing_acc_smoothed": routing_accuracy(true_labels, pred_smooth),
        "nr_db_oracle": clip_noise_reduction_db(clean_frames, noisy_frames, den_oracle),
        "nr_db_classifier": clip_noise_reduction_db(clean_frames, noisy_frames, den_classifier),
        "demo_seconds": DEMO_SECONDS,
        "demo_file": str(demo_path.name),
    }

    save_wav(wav_dir / "01_original.wav", clean_audio)
    save_wav(wav_dir / "02_mixed_noisy.wav", noisy_audio)
    save_wav(wav_dir / "03_moe_oracle.wav", frames_to_waveform(den_oracle))
    save_wav(wav_dir / "04_moe_classifier.wav", frames_to_waveform(den_classifier))

    oracle_audio = frames_to_waveform(den_oracle)
    classifier_audio = frames_to_waveform(den_classifier)

    _plot_waveforms(clean_audio, noisy_audio, oracle_audio, classifier_audio, metrics, fig_dir / "waveforms_comparison.png")
    _plot_routing(true_labels, pred_raw, pred_smooth, segments, fig_dir / "routing_timeline.png")
    _plot_metrics_bar(metrics, fig_dir / "metrics_bar.png")

    pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()]).to_csv(OUT_DIR / "demo_metrics.csv", index=False)
    adaptive_copied = _copy_adaptive_reference(ref_dir)
    _write_summary(OUT_DIR / "SUMMARY.md", demo_path, metrics, adaptive_copied)

    print("\n" + "=" * 56)
    print("MoE PRESENTATION DEMO — SUMMARY")
    print("=" * 56)
    print(f"  Routing (RF smoothed):  {metrics['routing_acc_smoothed']:.1%}")
    print(f"  Routing (RF raw):       {metrics['routing_acc_raw']:.1%}")
    print(f"  NR oracle MoE:           {metrics['nr_db_oracle']:.2f} dB")
    print(f"  NR classifier MoE:      {metrics['nr_db_classifier']:.2f} dB")
    print("=" * 56)
    print(classification_report(true_labels, pred_smooth, target_names=CHANNEL_NAMES, zero_division=0))
    print(f"\nOutputs: {OUT_DIR}")
    print("  SUMMARY.md, demo_metrics.csv, wav/*.wav, figures/*.png")
    if adaptive_copied:
        print(f"  adaptive_reference/: {', '.join(adaptive_copied)}")


if __name__ == "__main__":
    main()
