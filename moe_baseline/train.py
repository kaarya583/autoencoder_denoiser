"""Train MoE + evaluate routing."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from moe_baseline.audio import FrameDataset
from moe_baseline.config import (
    CHANNEL_NAMES,
    FRAME_SIZE,
    LATENT_DIM_FAIR,
    NUM_FAMILIES,
    SEED,
)
from moe_baseline.datasets import NoisyFrameDataset, PureNoiseLatentDataset
from moe_baseline.model import MoEHardDenoiser
from moe_baseline.routers import (
    predict_demo_route,
    train_demo_router_speech,
    train_fair_router,
    train_fair_router_waveform,
)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


@torch.no_grad()
def routing_metrics_neural(model, loader, device, max_batches=None, max_samples=None):
    model.eval()
    ys, ps = [], []
    n = 0
    for bi, (noisy, _c, fam) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        noisy, fam = noisy.to(device), fam.to(device)
        _, logits = model(noisy)
        pred = logits.argmax(dim=-1)
        ys.append(fam.cpu().numpy())
        ps.append(pred.cpu().numpy())
        n += noisy.size(0)
        if max_samples is not None and n >= max_samples:
            break
    y, p = np.concatenate(ys), np.concatenate(ps)
    if max_samples is not None:
        y, p = y[:max_samples], p[:max_samples]
    return float((y == p).mean()), y, p


def train_router_pure_noise(model, loader, opt, device, n_epochs: int):
    model.train()
    for ep in range(n_epochs):
        tot, n = 0.0, 0
        for noisy, fam in loader:
            noisy, fam = noisy.to(device), fam.to(device)
            opt.zero_grad(set_to_none=True)
            _, logits = model(noisy)
            loss = F.cross_entropy(logits, fam)
            loss.backward()
            opt.step()
            tot += loss.item() * noisy.size(0)
            n += noisy.size(0)
        print(f"  pure-noise router epoch {ep + 1}/{n_epochs} CE {tot / n:.4f}")


def si_sdr_loss(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR loss (minimize negative SI-SDR)."""
    est = estimate - estimate.mean(dim=-1, keepdim=True)
    ref = reference - reference.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    ref_energy = (ref**2).sum(dim=-1, keepdim=True).clamp(min=eps)
    target = dot / ref_energy * ref
    noise = est - target
    si_sdr = (target**2).sum(dim=-1) / ((noise**2).sum(dim=-1) + eps)
    return -10.0 * torch.log10(si_sdr.clamp(min=eps)).mean()


@torch.no_grad()
def eval_denoising_oracle(
    model: MoEHardDenoiser,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Reconstruction quality with oracle (ground-truth) expert routes."""
    model.eval()
    mse_sum = nr_sum = n = 0
    for bi, (noisy, clean, fam) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        noisy, clean, fam = noisy.to(device), clean.to(device), fam.to(device)
        pred = model.forward_with_route(noisy, fam)
        mse_sum += float(F.mse_loss(pred, clean)) * noisy.size(0)
        err_in = (noisy - clean).pow(2).mean(dim=-1)
        err_out = (pred - clean).pow(2).mean(dim=-1)
        nr = 10.0 * torch.log10((err_in + 1e-8) / (err_out + 1e-8))
        nr_sum += float(nr.sum())
        n += noisy.size(0)
    return {"mse": mse_sum / max(n, 1), "noise_reduction_db": nr_sum / max(n, 1)}


def train_moe_denoising_oracle(
    model: MoEHardDenoiser,
    train_frame_ds,
    test_loader: DataLoader,
    device: torch.device,
    split: str = "train",
    epochs: int = 15,
    lr: float = 2e-4,
    batches_per_family: int = 150,
    batch_size: int = 256,
    use_si_sdr: bool = False,
    si_sdr_weight: float = 0.05,
    freeze_router: bool = True,
) -> dict[str, float]:
    """
    Train each expert to map (clean speech + family noise) -> clean speech.

    Uses ground-truth family labels with forward_with_route so only the matching
    specialist (+ shared encoder/expert) is trained for that noise type.
    """
    if freeze_router:
        for p in model.router.parameters():
            p.requires_grad = False

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-5,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    rng = np.random.default_rng(SEED + 42)

    print(
        f"Phase C: oracle-route denoising ({epochs} epochs, "
        f"{batches_per_family} batches/family/epoch, SI-SDR={use_si_sdr})..."
    )

    for ep in range(1, epochs + 1):
        model.train()
        fam_mse = {name: 0.0 for name in CHANNEL_NAMES}
        fam_n = {name: 0 for name in CHANNEL_NAMES}
        tot_loss = 0.0
        n_samples = 0

        for fam_id, fam_name in enumerate(CHANNEL_NAMES):
            fam_ds = NoisyFrameDataset(
                train_frame_ds,
                split,
                np.random.default_rng(int(rng.integers(0, 2**31)) + fam_id),
                family_id=fam_id,
            )
            # Override sampling: fix family for this expert pass
            loader = DataLoader(fam_ds, batch_size=batch_size, shuffle=True, drop_last=True)
            steps = min(batches_per_family, len(loader))
            for step, (noisy, clean, fam) in enumerate(loader):
                if step >= steps:
                    break
                noisy, clean, fam = noisy.to(device), clean.to(device), fam.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model.forward_with_route(noisy, fam)
                loss = F.mse_loss(pred, clean)
                if use_si_sdr:
                    loss = loss + si_sdr_weight * si_sdr_loss(pred, clean)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                bs = noisy.size(0)
                tot_loss += float(loss.item()) * bs
                n_samples += bs
                fam_mse[fam_name] += float(F.mse_loss(pred, clean).detach()) * bs
                fam_n[fam_name] += bs

        sched.step()
        val = eval_denoising_oracle(model, test_loader, device, max_batches=40)
        print(
            f"  denoise epoch {ep:02d}/{epochs} | train loss {tot_loss / max(n_samples, 1):.5f} | "
            f"val MSE {val['mse']:.6f} | val NR {val['noise_reduction_db']:.2f} dB"
        )
        per_fam = " | ".join(
            f"{k[:4]}:{fam_mse[k] / max(fam_n[k], 1):.5f}" for k in CHANNEL_NAMES
        )
        print(f"    per-family train MSE: {per_fam}")

    final = eval_denoising_oracle(model, test_loader, device)
    print(
        f"  Final oracle denoising — val MSE {final['mse']:.6f}, "
        f"noise reduction {final['noise_reduction_db']:.2f} dB"
    )
    return final


def train_moe_speech(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int = 20,
    lr: float = 3e-4,
    router_only_epochs: int = 5,
    mse_weight: float = 0.1,
):
    pure_loader = DataLoader(
        PureNoiseLatentDataset(8000, FRAME_SIZE, LATENT_DIM_FAIR, device),
        batch_size=256,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    print("Phase A: neural router on pure-noise inputs (no speech)...")
    train_router_pure_noise(model, pure_loader, opt, device, router_only_epochs)

    print("Phase B: joint training on noisy speech frames...")
    for ep in range(1, epochs + 1):
        model.train()
        n = 0
        for noisy, clean, fam in train_loader:
            noisy, clean, fam = noisy.to(device), clean.to(device), fam.to(device)
            opt.zero_grad(set_to_none=True)
            pred, logits = model(noisy)
            loss = F.cross_entropy(logits, fam) + mse_weight * F.mse_loss(pred, clean)
            loss.backward()
            opt.step()
            n += noisy.size(0)
        sched.step()
        acc, _, _ = routing_metrics_neural(model, test_loader, device)
        print(f"Epoch {ep:02d}/{epochs} | neural val acc (noisy speech) {acc:.4f}")

    return routing_metrics_neural(model, test_loader, device, max_samples=None)


def train_moe_speech_with_denoising(
    model: MoEHardDenoiser,
    train_frame_ds,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    joint_epochs: int = 5,
    denoise_epochs: int = 15,
    router_only_epochs: int = 2,
    mse_weight: float = 0.1,
    denoise_lr: float = 1e-4,
    batches_per_family: int = 150,
    use_si_sdr: bool = False,
) -> dict:
    """Joint routing pretrain, then oracle-route specialist denoising."""
    train_moe_speech(
        model,
        train_loader,
        test_loader,
        device,
        epochs=joint_epochs,
        router_only_epochs=router_only_epochs,
        mse_weight=mse_weight,
    )
    denoise_metrics = train_moe_denoising_oracle(
        model,
        train_frame_ds,
        test_loader,
        device,
        epochs=denoise_epochs,
        lr=denoise_lr,
        batches_per_family=batches_per_family,
        use_si_sdr=use_si_sdr,
    )
    return denoise_metrics


def eval_demo_router(demo_router, model, test_ds, device, max_n=4000):
    model.eval()
    max_n = min(max_n, len(test_ds))
    ys, ps = [], []
    for i in range(max_n):
        noisy, _c, fam = test_ds[i]
        ys.append(fam)
        with torch.no_grad():
            route = predict_demo_route(demo_router, noisy.unsqueeze(0).to(device))
            ps.append(int(route.item()))
    y, p = np.array(ys), np.array(ps)
    return float((y == p).mean()), y, p


def run_full_pipeline(
    data_root,
    device=None,
    epochs: int = 20,
    router_only_epochs: int = 5,
    denoise_epochs: int = 15,
):
    from moe_baseline.config import data_root as default_root
    from moe_baseline.librispeech import get_train_test_files

    set_seed()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = data_root or default_root()
    print("Device:", device)

    train_files, test_files = get_train_test_files(root)
    train_frame_ds = FrameDataset(train_files)
    test_frame_ds = FrameDataset(test_files)
    train_ds = NoisyFrameDataset(train_frame_ds, "train", np.random.default_rng(SEED + 2))
    test_ds = NoisyFrameDataset(test_frame_ds, "val", np.random.default_rng(SEED + 3))

    fair_router, acc_fair = train_fair_router(device=device)
    _, acc_fair_wav = train_fair_router_waveform(device=device)
    demo_router, acc_demo = train_demo_router_speech(train_ds)

    model = MoEHardDenoiser(FRAME_SIZE, NUM_FAMILIES, use_shared_expert=True).to(device)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    if denoise_epochs > 0:
        train_moe_speech(
            model,
            train_loader,
            test_loader,
            device,
            epochs=min(epochs, 8),
            router_only_epochs=router_only_epochs,
            mse_weight=0.1,
        )
        train_moe_denoising_oracle(
            model,
            train_frame_ds,
            test_loader,
            device,
            epochs=denoise_epochs,
            batches_per_family=150,
            use_si_sdr=False,
        )
        acc_neural, y_te, p_neural = routing_metrics_neural(model, test_loader, device)
    else:
        acc_neural, y_te, p_neural = train_moe_speech(
            model, train_loader, test_loader, device, epochs, router_only_epochs=router_only_epochs
        )
    acc_demo_te, _, p_demo = eval_demo_router(demo_router, model, test_ds, device)

    print("\n" + "=" * 72)
    print("ROUTING SUMMARY")
    print("=" * 72)
    print(f"Fair RF (latent, no speech)     : {acc_fair:.4f}  [compare to adaptive noise_clf]")
    print(f"Fair RF (waveform pure noise)   : {acc_fair_wav:.4f}")
    print(f"Deployment RF (noisy speech)    : {acc_demo:.4f} train holdout | {acc_demo_te:.4f} test")
    print(f"Neural MLP (noisy speech)       : {acc_neural:.4f}")
    print(f"Chance                          : {1 / NUM_FAMILIES:.4f}")
    print("=" * 72)
    print(classification_report(y_te, p_neural, target_names=CHANNEL_NAMES, zero_division=0))

    return {
        "model": model,
        "fair_router": fair_router,
        "fair_router_wav": None,
        "demo_router": demo_router,
        "acc_fair": acc_fair,
        "acc_fair_wav": acc_fair_wav,
        "acc_demo": acc_demo_te,
        "acc_neural": acc_neural,
    }
