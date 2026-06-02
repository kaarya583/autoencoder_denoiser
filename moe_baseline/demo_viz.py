"""Presentation figures for MoE speech denoising demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from moe_baseline.config import CHANNEL_NAMES, NUM_FAMILIES, TARGET_SR

DPI = 150
SEGMENT_COLORS = plt.cm.tab10(np.linspace(0, 1, NUM_FAMILIES))


def _time_axis(n_samples: int, sr: int = TARGET_SR) -> np.ndarray:
    return np.arange(n_samples) / sr


def plot_waveforms_four(
    clean: np.ndarray,
    noisy: np.ndarray,
    den_raw: np.ndarray,
    den_smooth: np.ndarray,
    metrics: dict,
    path: Path,
    seconds: float = 4.0,
) -> None:
    n = int(min(seconds * TARGET_SR, len(clean)))
    t = _time_axis(n)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    specs = [
        (clean, "Clean speech (reference)", "C0"),
        (noisy, "Mixed noise — 6 families", "C1"),
        (
            den_raw,
            f"Denoised — RF raw (route acc {metrics.get('routing_acc_raw', 0):.0%}, "
            f"NR {metrics.get('nr_db_demo_raw', metrics.get('nr_db_classifier', 0)):.1f} dB)",
            "C2",
        ),
        (
            den_smooth,
            f"Denoised — RF smoothed (acc {metrics.get('routing_acc_smoothed', 0):.0%}, "
            f"NR {metrics.get('nr_db_demo_smooth', metrics.get('nr_db_classifier', 0)):.1f} dB)",
            "C3",
        ),
    ]
    for ax, (sig, title, color) in zip(axes, specs):
        ax.plot(t, sig[:n], lw=0.7, color=color)
        ax.set_ylabel("Amp")
        ax.set_title(title, fontsize=10)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("MoE denoising on mixed-noise speech clip", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_waveforms_oracle_classifier(
    clean: np.ndarray,
    noisy: np.ndarray,
    oracle: np.ndarray,
    classifier: np.ndarray,
    metrics: dict,
    path: Path,
    seconds: float = 4.0,
) -> None:
    n = int(min(seconds * TARGET_SR, len(clean)))
    t = _time_axis(n)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax, sig, title in zip(
        axes,
        [clean, noisy, oracle, classifier],
        [
            "Clean speech",
            "Mixed noisy input",
            f"Oracle route (NR {metrics.get('nr_db_oracle', 0):.1f} dB)",
            f"Classifier route — smoothed (NR {metrics.get('nr_db_classifier', 0):.1f} dB)",
        ],
    ):
        ax.plot(t, sig[:n], lw=0.7)
        ax.set_ylabel("Amp")
        ax.set_title(title, fontsize=10)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_segment_boundaries(
    clean: np.ndarray,
    noisy: np.ndarray,
    true_labels: np.ndarray,
    segments: dict,
    path: Path,
    seconds: float = 6.0,
    hop: int = 512,
) -> None:
    """Waveform with colored spans per true noise family."""
    n = int(min(seconds * TARGET_SR, len(clean)))
    t = _time_axis(n)
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True)
    for ax, sig, title in zip(axes, [clean[:n], noisy[:n]], ["Clean", "Mixed noisy"]):
        ax.plot(t, sig, color="0.35", lw=0.5, zorder=1)
        for fam_id, name in enumerate(CHANNEL_NAMES):
            idx = segments.get(name, [])
            if len(idx) == 0:
                continue
            t0 = idx[0] * hop / TARGET_SR
            t1 = (idx[-1] + 1) * hop / TARGET_SR
            ax.axvspan(t0, min(t1, seconds), alpha=0.35, color=SEGMENT_COLORS[fam_id], label=name)
        ax.set_ylabel("Amp")
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), ncol=6, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Demo clip: one noise family per segment", fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_routing_timeline(
    true_labels: np.ndarray,
    pred_raw: np.ndarray,
    pred_smooth: np.ndarray,
    segments: dict,
    path: Path,
    vote_window: int = 5,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    x = np.arange(len(true_labels))
    ax.scatter(x, true_labels, s=18, c="black", label="True family", zorder=3)
    ax.scatter(x, pred_raw, s=12, c="C1", alpha=0.5, label="RF raw")
    ax.scatter(x, pred_smooth, s=14, c="C2", marker="x", label=f"RF smoothed (w={vote_window})")
    for name, idx in segments.items():
        if len(idx):
            ax.axvline(idx[0], color="gray", ls=":", lw=0.6, alpha=0.8)
    ax.set_yticks(range(NUM_FAMILIES))
    ax.set_yticklabels(CHANNEL_NAMES)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Expert")
    ax.set_title("Per-frame routing on mixed-noise clip (dotted lines = segment boundaries)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_pair(
    true_labels: np.ndarray,
    pred_raw: np.ndarray,
    pred_smooth: np.ndarray,
    path: Path,
    vote_window: int = 5,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pred, title in zip(
        axes,
        [pred_raw, pred_smooth],
        ["Demo RF — raw", f"Demo RF — smoothed (w={vote_window})"],
    ):
        cm = confusion_matrix(true_labels, pred, labels=list(range(NUM_FAMILIES)))
        im = ax.imshow(cm, cmap="Blues")
        acc = float((true_labels == pred).mean())
        ax.set_xticks(range(NUM_FAMILIES))
        ax.set_yticks(range(NUM_FAMILIES))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CHANNEL_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\naccuracy = {acc:.1%}")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(metrics: dict, path: Path) -> None:
    rows = [
        ("Routing acc — raw", metrics.get("routing_acc_raw", 0), "routing"),
        ("Routing acc — smoothed", metrics.get("routing_acc_smoothed", 0), "routing"),
        ("Routing acc — fair RF", metrics.get("fair_rf_acc", 0), "routing"),
        ("NR (dB) — oracle", metrics.get("nr_db_oracle", 0) / 20.0, "nr"),
        ("NR (dB) — demo raw", metrics.get("nr_db_demo_raw", 0) / 20.0, "nr"),
        ("NR (dB) — demo smooth", metrics.get("nr_db_demo_smooth", metrics.get("nr_db_classifier", 0)) / 20.0, "nr"),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2", "#937860"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(1 / NUM_FAMILIES, color="gray", ls="--", lw=1, label="Chance (routing)")
    ax.set_xlabel("Score (routing acc, or NR÷20 for scale)")
    ax.set_title("Mixed-noise demo — routing & denoising summary")
    for b, v in zip(bars, values):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar_compact(metrics: dict, path: Path) -> None:
    labels = [
        "Routing\n(raw)",
        "Routing\n(smooth)",
        "NR oracle\n(/10 dB)",
        "NR classifier\n(/10 dB)",
    ]
    values = [
        metrics["routing_acc_raw"],
        metrics["routing_acc_smoothed"],
        metrics["nr_db_oracle"] / 10.0,
        metrics["nr_db_classifier"] / 10.0,
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.axhline(1 / NUM_FAMILIES, color="gray", ls="--", lw=1, label="Chance")
    ax.set_ylabel("Accuracy or NR/10")
    ax.set_title("MoE presentation metrics")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_segment_nr(
    clean_frames: np.ndarray,
    noisy_frames: np.ndarray,
    denoised_frames: np.ndarray,
    true_labels: np.ndarray,
    path: Path,
    title_suffix: str = "smoothed route",
) -> None:
    from moe_baseline.demo import noise_reduction_db

    per_fam = []
    for fam_id, name in enumerate(CHANNEL_NAMES):
        mask = true_labels == fam_id
        if not mask.any():
            continue
        nrs = [
            noise_reduction_db(clean_frames[i], noisy_frames[i], denoised_frames[i])
            for i in np.where(mask)[0]
        ]
        per_fam.append((name, float(np.mean(nrs))))
    if not per_fam:
        return
    names, nrs = zip(*per_fam)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [SEGMENT_COLORS[CHANNEL_NAMES.index(n)] for n in names]
    bars = ax.bar(names, nrs, color=colors)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Noise reduction (dB)")
    ax.set_title(f"Per-segment denoising quality ({title_suffix})")
    for b, v in zip(bars, nrs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_adaptive_vs_moe(
    clean: np.ndarray,
    noisy: np.ndarray,
    moe: np.ndarray,
    adaptive: np.ndarray | None,
    path: Path,
    seconds: float = 4.0,
) -> None:
    if adaptive is None:
        return
    n = int(min(seconds * TARGET_SR, len(clean), len(noisy), len(moe), len(adaptive)))
    t = _time_axis(n)
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, clean[:n], lw=0.65, label="clean")
    axes[0].plot(t, noisy[:n], lw=0.4, alpha=0.55, color="C1", label="noisy")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Reference")
    axes[1].plot(t, adaptive[:n], lw=0.65, color="C2")
    axes[1].set_title("Adaptive AE — classifier route")
    axes[2].plot(t, moe[:n], lw=0.65, color="C3")
    axes[2].set_title("MoE — RF classifier (smoothed)")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_ylabel("Amp")
    fig.suptitle("Adaptive vs MoE denoising", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_systems(
    systems: list[tuple[str, np.ndarray]],
    path: Path,
    seconds: float = 3.5,
) -> None:
    """Stacked waveforms for slide comparing multiple systems."""
    n = int(min(seconds * TARGET_SR, min(len(s[1]) for s in systems)))
    t = _time_axis(n)
    fig, axes = plt.subplots(len(systems), 1, figsize=(12, 1.8 * len(systems)), sharex=True)
    if len(systems) == 1:
        axes = [axes]
    for ax, (label, sig) in zip(axes, systems):
        ax.plot(t, sig[:n], lw=0.65)
        ax.set_ylabel("Amp")
        ax.set_title(label, fontsize=10, loc="left")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
