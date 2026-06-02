"""Routing features: adaptive latent-noise (fair) + spectral (deployment)."""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, skew

from moe_baseline.config import CHANNEL_NAMES, FRAME_SIZE, LATENT_DIM_FAIR
from moe_baseline.noise import TRAIN_CHANNELS

N_FFT_BINS = 64


def latent_noise_to_numpy(channel_fn, n_samples: int = 1, latent_dim: int = LATENT_DIM_FAIR, device=None):
    dev = device or torch.device("cpu")
    z = torch.zeros((n_samples, latent_dim), device=dev)
    with torch.no_grad():
        z_noisy = channel_fn(z)
    return (z_noisy - z).cpu().numpy()


def safe_skew(x):
    val = skew(x)
    return 0.0 if np.isnan(val) else float(val)


def safe_kurtosis(x):
    val = kurtosis(x, fisher=True)
    return 0.0 if np.isnan(val) else float(val)


def zero_crossing_rate(x):
    return float(np.mean(x[:-1] * x[1:] < 0))


def spectral_features(x):
    f, pxx = welch(x, fs=1.0, nperseg=min(32, len(x)), noverlap=None)
    pxx = pxx + 1e-12
    p = pxx / np.sum(pxx)
    centroid = np.sum(f * p)
    bandwidth = np.sqrt(np.sum(((f - centroid) ** 2) * p))
    flatness = np.exp(np.mean(np.log(pxx))) / np.mean(pxx)
    dom_freq = f[np.argmax(pxx)]
    peaks, _ = find_peaks(pxx, distance=1)
    return [centroid, bandwidth, flatness, dom_freq, len(peaks), np.max(pxx) / np.sum(pxx)]


def autocorr_at_lag(x, lag):
    x = x - np.mean(x)
    if lag >= len(x):
        return 0.0
    denom = np.dot(x, x) + 1e-12
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def extract_noise_features(x) -> np.ndarray:
    """Same feature vector as Adaptive_Autoencoders_Project noise_clf."""
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    std = np.std(x) + 1e-12
    z = (x - mu) / std
    abs_z = np.abs(z)
    feats = [
        safe_skew(z),
        safe_kurtosis(z),
        np.mean(abs_z),
        np.max(abs_z),
        np.mean(abs_z > 1),
        np.mean(abs_z > 2),
        np.mean(abs_z > 3),
        np.mean(abs_z > 4),
        zero_crossing_rate(z),
    ]
    feats.extend(spectral_features(z))
    for lag in [1, 2, 4, 8, 16, 32]:
        feats.append(autocorr_at_lag(z, lag))
    return np.array(feats, dtype=np.float32)


def build_latent_noise_classifier_dataset(n_per_family: int = 3000, device=None):
    X, y = [], []
    for label, name in enumerate(CHANNEL_NAMES):
        noise_batch = latent_noise_to_numpy(
            TRAIN_CHANNELS[name], n_samples=n_per_family, device=device
        )
        for seq in noise_batch:
            X.append(extract_noise_features(seq))
            y.append(label)
    return np.vstack(X), np.array(y, dtype=np.int64)


def pure_waveform_noise_frame(family_id: int, split: str, device=None) -> np.ndarray:
    from moe_baseline.noise import apply_family_noise

    clean = torch.zeros(FRAME_SIZE, device=device or torch.device("cpu"))
    noisy = apply_family_noise(clean, family_id, split, device=device)
    return (noisy - clean).cpu().numpy()


def build_waveform_noise_classifier_dataset(n_per_family: int = 3000, split: str = "train", device=None):
    """Adaptive-style stats on 512-sample pure noise waveforms."""
    X, y = [], []
    for label in range(len(CHANNEL_NAMES)):
        for _ in range(n_per_family):
            X.append(extract_noise_features(pure_waveform_noise_frame(label, split, device=device)))
            y.append(label)
    return np.vstack(X), np.array(y, dtype=np.int64)


def build_waveform_spectral_classifier_dataset(n_per_family: int = 3000, split: str = "train", device=None):
    """Log-FFT + time stats on 512-sample pure noise waveforms (no speech)."""
    X, y = [], []
    for label in range(len(CHANNEL_NAMES)):
        for _ in range(n_per_family):
            w = pure_waveform_noise_frame(label, split, device=device)
            X.append(frame_to_spectral_feature_vector(w))
            y.append(label)
    return np.vstack(X), np.array(y, dtype=np.int64)


def frame_to_spectral_feature_vector(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame, dtype=np.float32).ravel()
    mag = np.abs(np.fft.rfft(x))
    mag = mag[:N_FFT_BINS] if len(mag) >= N_FFT_BINS else np.pad(mag, (0, N_FFT_BINS - len(mag)))
    log_mag = np.log1p(mag)
    p = (mag + 1e-12) / (mag.sum() + 1e-12)
    freqs = np.linspace(0, 1, len(p))
    centroid = float(np.sum(freqs * p))
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))
    flatness = float(np.exp(np.mean(np.log(mag + 1e-12))) / (np.mean(mag) + 1e-12))
    stats = np.array(
        [
            np.sqrt(np.mean(x**2) + 1e-12),
            np.std(x),
            np.max(np.abs(x)),
            float(np.mean(x[:-1] * x[1:] < 0)),
            centroid,
            spread,
            flatness,
        ],
        dtype=np.float32,
    )
    return np.concatenate([stats, log_mag])
