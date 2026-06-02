#!/usr/bin/env python3
"""
MoE presentation demo — mirrors adaptive mixed-noise clip demo.

Produces slide-ready WAVs, figures (150 dpi), and metrics under
outputs/moe_presentation_demo/.

Usage:
  python scripts/run_moe_presentation_demo.py
  python scripts/run_moe_presentation_demo.py --fast-train      # if checkpoints missing
  python scripts/run_moe_presentation_demo.py --quality-train # best denoising (~20–40 min CPU)
  python scripts/run_moe_presentation_demo.py --retrain --quality-train
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import joblib
import matplotlib

matplotlib.use("Agg")
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
    noise_reduction_db,
    predict_routes_with_smoothing,
    routing_accuracy,
    save_wav,
)
from moe_baseline.demo_routing import save_routing_bundle
from moe_baseline.demo_train import checkpoint_val_nr, train_presentation_checkpoints
from moe_baseline.demo_viz import (
    plot_adaptive_vs_moe,
    plot_comparison_systems,
    plot_confusion_pair,
    plot_metrics_bar_compact,
    plot_routing_timeline,
    plot_segment_boundaries,
    plot_segment_nr,
    plot_summary_bars,
    plot_waveforms_four,
    plot_waveforms_oracle_classifier,
)
from moe_baseline.librispeech import get_train_test_files
from moe_baseline.model import MoEHardDenoiser
from moe_baseline.train import set_seed

OUT_DIR = REPO / "outputs" / "moe_presentation_demo"
CKPT_DIR = REPO / "checkpoints" / "moe_speech_demo"
ADAPTIVE_DIR = REPO / "outputs"
DEMO_SECONDS = 6.0
ROUTE_VOTE_WINDOW = 5
MIN_VAL_NR_DB = 2.5  # retrain if saved model weaker than this


def _ensure_checkpoints(
    device: torch.device,
    profile: str | None,
    retrain: bool,
) -> None:
    names = ("demo_router.joblib", "waveform_router.joblib", "moe_denoiser.pt")
    if retrain:
        for n in names:
            p = CKPT_DIR / n
            if p.exists():
                p.unlink()
        meta = CKPT_DIR / "train_meta.joblib"
        if meta.exists():
            meta.unlink()

    need = not all((CKPT_DIR / n).exists() for n in names)
    weak = False
    if not need and profile is None:
        val_nr = checkpoint_val_nr(CKPT_DIR, device)
        if val_nr is not None and val_nr < MIN_VAL_NR_DB:
            print(f"Checkpoints weak (val NR {val_nr:.2f} dB < {MIN_VAL_NR_DB}) — use --quality-train --retrain")
            weak = True

    if not need and not weak:
        return
    if profile is None:
        raise FileNotFoundError(
            f"Missing or weak checkpoints in {CKPT_DIR}. "
            "Run: python scripts/run_moe_presentation_demo.py --fast-train\n"
            "Or:  python scripts/run_moe_presentation_demo.py --quality-train --retrain"
        )
    train_presentation_checkpoints(CKPT_DIR, device, profile=profile)


def _copy_adaptive_reference(dest: Path) -> list[str]:
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


def _align_wavs(*arrays: np.ndarray) -> list[np.ndarray]:
    n = min(len(a) for a in arrays)
    return [a[:n].astype(np.float32) for a in arrays]


def _adaptive_comparison_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    moe_classifier: np.ndarray,
    ref_dir: Path,
) -> dict[str, float]:
    out: dict[str, float] = {}
    c, n, m = _align_wavs(clean, noisy, moe_classifier)
    out["nr_db_moe_classifier"] = noise_reduction_db(c, n, m)
    for fname, key in {
        "adaptive_oracle.wav": "nr_db_adaptive_oracle",
        "adaptive_classifier.wav": "nr_db_adaptive_classifier",
    }.items():
        p = ref_dir / fname
        if not p.exists():
            continue
        import soundfile as sf

        den = sf.read(str(p), dtype="float32")[0]
        if den.ndim > 1:
            den = den.mean(axis=1)
        c2, n2, d2 = _align_wavs(clean, noisy, den)
        out[key] = noise_reduction_db(c2, n2, d2)
    return out


def _write_summary(path: Path, demo_path: Path, metrics: dict, adaptive_copied: list[str]) -> None:
    lines = [
        "# MoE presentation demo",
        "",
        f"- **Clip:** `{demo_path.name}` ({DEMO_SECONDS}s, 16 kHz)",
        "- **Protocol:** 6 contiguous segments, one noise family each (same as adaptive notebook)",
        "",
        "## MoE metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Routing accuracy (RF raw) | {metrics['routing_acc_raw']:.1%} |",
        f"| Routing accuracy (RF smoothed, w={ROUTE_VOTE_WINDOW}) | {metrics['routing_acc_smoothed']:.1%} |",
        f"| Fair RF routing accuracy | {metrics.get('fair_rf_acc', 0):.1%} |",
        f"| Noise reduction — oracle route | {metrics['nr_db_oracle']:.2f} dB |",
        f"| Noise reduction — demo RF raw | {metrics['nr_db_demo_raw']:.2f} dB |",
        f"| Noise reduction — demo RF smoothed | {metrics['nr_db_demo_smooth']:.2f} dB |",
        "",
    ]
    if metrics.get("nr_db_adaptive_classifier") is not None:
        lines.extend(
            [
                "## Adaptive vs MoE",
                "",
                "| System | NR (dB) |",
                "|--------|---------|",
                f"| MoE (smoothed RF) | {metrics.get('nr_db_moe_classifier', metrics['nr_db_demo_smooth']):.2f} |",
                f"| Adaptive (classifier) | {metrics['nr_db_adaptive_classifier']:.2f} |",
            ]
        )
        if metrics.get("nr_db_adaptive_oracle") is not None:
            lines.append(f"| Adaptive (oracle) | {metrics['nr_db_adaptive_oracle']:.2f} |")
        lines.append("")
    lines.extend(
        [
            "## Audio (`wav/`)",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `01_original.wav` | Clean speech |",
            "| `02_mixed_noisy.wav` | Six-family mixed noise |",
            "| `03_moe_oracle.wav` | MoE + true family (upper bound) |",
            "| `04_moe_classifier_raw.wav` | MoE + per-frame RF |",
            "| `05_moe_classifier_smoothed.wav` | **Main result** — RF + majority vote |",
            "| `06_moe_fair_router.wav` | MoE + pure-noise RF (comparison) |",
            "",
            "## Figures (`figures/`)",
            "",
            "- `segment_boundaries.png` — colored noise segments",
            "- `waveforms_comparison.png` — clean / noisy / raw / smoothed",
            "- `waveforms_oracle_classifier.png` — oracle vs classifier",
            "- `routing_timeline.png` — per-frame routes",
            "- `confusion_raw_vs_smoothed.png` — routing confusion matrices",
            "- `summary_metrics.png` — routing + NR bar chart",
            "- `segment_nr.png` — NR per noise family",
            "- `systems_comparison.png` — stacked systems for slides",
            "- `adaptive_vs_moe.png` — vs adaptive (if reference WAVs exist)",
            "",
        ]
    )
    if adaptive_copied:
        lines.append(f"Adaptive copies: {', '.join(adaptive_copied)}")
    else:
        lines.append("_Run adaptive notebook demo first for side-by-side WAVs._")
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="MoE presentation demo")
    p.add_argument("--fast-train", action="store_true", help="Train with fast profile if needed")
    p.add_argument("--quality-train", action="store_true", help="Train with quality profile (best denoising)")
    p.add_argument("--retrain", action="store_true", help="Delete checkpoints and retrain")
    args = p.parse_args()

    if args.quality_train:
        train_profile = "quality"
    elif args.fast_train or args.retrain:
        train_profile = "fast"
    else:
        train_profile = None

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    wav_dir = OUT_DIR / "wav"
    fig_dir = OUT_DIR / "figures"
    ref_dir = OUT_DIR / "adaptive_reference"
    wav_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _ensure_checkpoints(device, train_profile, retrain=args.retrain)

    print("Loading MoE checkpoints...")
    demo_router = joblib.load(CKPT_DIR / "demo_router.joblib")
    waveform_router = joblib.load(CKPT_DIR / "waveform_router.joblib")
    model = MoEHardDenoiser(FRAME_SIZE, NUM_FAMILIES, use_shared_expert=True).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "moe_denoiser.pt", map_location=device))
    model.eval()
    val_nr = checkpoint_val_nr(CKPT_DIR, device)
    if val_nr is not None:
        print(f"Checkpoint val oracle NR: {val_nr:.2f} dB")

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
    pred_fair = predict_routes_with_smoothing(waveform_router, noisy_frames, window=1)[0]

    den_oracle, _ = denoise_frames(noisy_frames, model, demo_router, device, true_labels)
    den_raw, _ = denoise_frames(noisy_frames, model, demo_router, device, pred_raw)
    den_smooth, _ = denoise_frames(noisy_frames, model, demo_router, device, pred_smooth)
    den_fair, _ = denoise_frames(noisy_frames, model, waveform_router, device, pred_fair)

    metrics = {
        "routing_acc_raw": routing_accuracy(true_labels, pred_raw),
        "routing_acc_smoothed": routing_accuracy(true_labels, pred_smooth),
        "fair_rf_acc": routing_accuracy(true_labels, pred_fair),
        "nr_db_oracle": clip_noise_reduction_db(clean_frames, noisy_frames, den_oracle),
        "nr_db_demo_raw": clip_noise_reduction_db(clean_frames, noisy_frames, den_raw),
        "nr_db_demo_smooth": clip_noise_reduction_db(clean_frames, noisy_frames, den_smooth),
        "nr_db_classifier": clip_noise_reduction_db(clean_frames, noisy_frames, den_smooth),
        "nr_db_fair": clip_noise_reduction_db(clean_frames, noisy_frames, den_fair),
        "demo_seconds": DEMO_SECONDS,
        "demo_file": str(demo_path.name),
        "val_checkpoint_nr_db": val_nr if val_nr is not None else float("nan"),
    }

    oracle_audio = frames_to_waveform(den_oracle)
    raw_audio = frames_to_waveform(den_raw)
    smooth_audio = frames_to_waveform(den_smooth)
    fair_audio = frames_to_waveform(den_fair)

    save_wav(wav_dir / "01_original.wav", clean_audio)
    save_wav(wav_dir / "02_mixed_noisy.wav", noisy_audio)
    save_wav(wav_dir / "03_moe_oracle.wav", oracle_audio)
    save_wav(wav_dir / "04_moe_classifier_raw.wav", raw_audio)
    save_wav(wav_dir / "05_moe_classifier_smoothed.wav", smooth_audio)
    save_wav(wav_dir / "06_moe_fair_router.wav", fair_audio)

    plot_segment_boundaries(clean_audio, noisy_audio, true_labels, segments, fig_dir / "segment_boundaries.png")
    plot_waveforms_four(clean_audio, noisy_audio, raw_audio, smooth_audio, metrics, fig_dir / "waveforms_comparison.png")
    plot_waveforms_oracle_classifier(
        clean_audio, noisy_audio, oracle_audio, smooth_audio, metrics, fig_dir / "waveforms_oracle_classifier.png"
    )
    plot_routing_timeline(true_labels, pred_raw, pred_smooth, segments, fig_dir / "routing_timeline.png", ROUTE_VOTE_WINDOW)
    plot_confusion_pair(true_labels, pred_raw, pred_smooth, fig_dir / "confusion_raw_vs_smoothed.png", ROUTE_VOTE_WINDOW)
    plot_summary_bars(metrics, fig_dir / "summary_metrics.png")
    plot_metrics_bar_compact(metrics, fig_dir / "metrics_bar.png")
    plot_segment_nr(clean_frames, noisy_frames, den_smooth, true_labels, fig_dir / "segment_nr.png", "smoothed RF")

    save_routing_bundle(
        OUT_DIR / "routing_demo.npz",
        {
            "true_labels": true_labels.astype(np.int64),
            "pred_raw": pred_raw.astype(np.int64),
            "pred_smooth": pred_smooth.astype(np.int64),
            "segment_starts": np.array(
                [int(segments[name][0]) if len(segments[name]) else -1 for name in CHANNEL_NAMES],
                dtype=np.int64,
            ),
            "vote_window": np.int64(ROUTE_VOTE_WINDOW),
        },
    )

    adaptive_copied = _copy_adaptive_reference(ref_dir)
    compare = _adaptive_comparison_metrics(clean_audio, noisy_audio, smooth_audio, ref_dir)
    metrics.update(compare)

    adaptive_audio = None
    adaptive_cls = ref_dir / "adaptive_classifier.wav"
    if adaptive_cls.exists():
        import soundfile as sf

        adaptive_audio = sf.read(str(adaptive_cls), dtype="float32")[0]
        if adaptive_audio.ndim > 1:
            adaptive_audio = adaptive_audio.mean(axis=1)
        plot_adaptive_vs_moe(clean_audio, noisy_audio, smooth_audio, adaptive_audio, fig_dir / "adaptive_vs_moe.png")

    systems = [
        ("Clean", clean_audio),
        ("Mixed noisy", noisy_audio),
        ("MoE — smoothed RF", smooth_audio),
        ("MoE — oracle", oracle_audio),
    ]
    if adaptive_audio is not None:
        systems.append(("Adaptive — classifier", adaptive_audio))
    plot_comparison_systems(systems, fig_dir / "systems_comparison.png")

    pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()]).to_csv(OUT_DIR / "demo_metrics.csv", index=False)
    if compare:
        pd.DataFrame(
            [
                {"system": "MoE (smoothed RF)", "noise_reduction_db": compare.get("nr_db_moe_classifier")},
                {"system": "MoE (oracle)", "noise_reduction_db": metrics["nr_db_oracle"]},
                {"system": "Adaptive (classifier)", "noise_reduction_db": compare.get("nr_db_adaptive_classifier")},
                {"system": "Adaptive (oracle)", "noise_reduction_db": compare.get("nr_db_adaptive_oracle")},
            ]
        ).dropna(subset=["noise_reduction_db"]).to_csv(OUT_DIR / "comparison_adaptive_vs_moe.csv", index=False)

    _write_summary(OUT_DIR / "SUMMARY.md", demo_path, metrics, adaptive_copied)

    print("\n" + "=" * 60)
    print("MoE PRESENTATION DEMO")
    print("=" * 60)
    print(f"  Routing (smoothed):  {metrics['routing_acc_smoothed']:.1%}")
    print(f"  Routing (raw):       {metrics['routing_acc_raw']:.1%}")
    print(f"  NR oracle:           {metrics['nr_db_oracle']:.2f} dB")
    print(f"  NR smoothed (main):  {metrics['nr_db_demo_smooth']:.2f} dB")
    print(f"  NR raw:              {metrics['nr_db_demo_raw']:.2f} dB")
    if "nr_db_adaptive_classifier" in metrics:
        print(f"  NR adaptive clf:     {metrics['nr_db_adaptive_classifier']:.2f} dB")
    print("=" * 60)
    print(classification_report(true_labels, pred_smooth, target_names=CHANNEL_NAMES, zero_division=0))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
