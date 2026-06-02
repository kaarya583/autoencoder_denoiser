"""Random-forest routers: fair (no speech) and deployment (noisy speech)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from moe_baseline.config import CHANNEL_NAMES, NOISE_CLF_N_PER_FAMILY, SEED
from moe_baseline.features import (
    build_latent_noise_classifier_dataset,
    build_waveform_noise_classifier_dataset,
    build_waveform_spectral_classifier_dataset,
    frame_to_spectral_feature_vector,
    pure_waveform_noise_frame,
)


def train_fair_router(device=None, n_per_family: int = NOISE_CLF_N_PER_FAMILY):
    """Latent pure-noise RF — same task as adaptive noise_clf."""
    print("Building latent pure-noise dataset...")
    X, y = build_latent_noise_classifier_dataset(n_per_family=n_per_family, device=device)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, random_state=SEED, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = float((pred == y_te).mean())
    print(f"Fair router (latent, no speech) holdout acc: {acc:.4f}")
    print(classification_report(y_te, pred, target_names=CHANNEL_NAMES, zero_division=0))
    return clf, acc


def train_fair_router_waveform(device=None, n_per_family: int = NOISE_CLF_N_PER_FAMILY):
    print("Building waveform pure-noise dataset (adaptive-style features)...")
    X, y = build_waveform_noise_classifier_dataset(n_per_family=n_per_family, device=device)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, random_state=SEED, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    acc = float((clf.predict(X_te) == y_te).mean())
    print(f"Fair router (waveform pure noise) holdout acc: {acc:.4f}")
    return clf, acc


def train_waveform_router(device=None, n_per_family: int = NOISE_CLF_N_PER_FAMILY):
    """Primary MoE notebook router: spectral features on pure 512-sample noise (no speech)."""
    print("Building pure waveform figures (512 samples / family)...")
    X, y = build_waveform_spectral_classifier_dataset(n_per_family=n_per_family, device=device)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    acc = float((pred == y_te).mean())
    print(f"Waveform router (no speech) holdout acc: {acc:.4f}")
    print(classification_report(y_te, pred, target_names=CHANNEL_NAMES, zero_division=0))
    return pipe, acc, (X_te, y_te, pred)


def predict_waveform_route(waveform_router, frame: np.ndarray) -> int:
    feat = frame_to_spectral_feature_vector(frame).reshape(1, -1)
    return int(waveform_router.predict(feat)[0])


def predict_waveform_route_batch(waveform_router, frames: np.ndarray) -> np.ndarray:
    feat = np.stack([frame_to_spectral_feature_vector(frames[i]) for i in range(len(frames))])
    return waveform_router.predict(feat)


def train_demo_router_speech(train_ds, max_n: int = 12000):
    """RF on spectral features of noisy speech frames."""
    xs, ys = [], []
    n = min(max_n, len(train_ds))
    for i in range(n):
        noisy, _c, fam = train_ds[i]
        xs.append(frame_to_spectral_feature_vector(noisy.numpy()))
        ys.append(int(fam))
    X, y = np.vstack(xs), np.array(ys)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_tr, y_tr)
    acc = float((pipe.predict(X_te) == y_te).mean())
    print(f"Deployment router (noisy speech) holdout acc: {acc:.4f}")
    return pipe, acc


def predict_fair_route(fair_router, noise_1d: np.ndarray) -> int:
    from moe_baseline.features import extract_noise_features

    return int(fair_router.predict(extract_noise_features(noise_1d).reshape(1, -1))[0])


def predict_demo_route(demo_router, noisy: torch.Tensor) -> torch.Tensor:
    feat = np.stack([frame_to_spectral_feature_vector(noisy[i].cpu().numpy()) for i in range(noisy.size(0))])
    pred = demo_router.predict(feat)
    return torch.from_numpy(pred.astype(np.int64)).to(noisy.device)
