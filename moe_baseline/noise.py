"""Synthetic tape noise (waveform / latent)."""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from moe_baseline.config import CHANNEL_NAMES, FRAME_SIZE, NUM_FAMILIES

def normalize_noise_rms_per_sample(noise: torch.Tensor, target_rms: float = 0.10):
    rms = torch.sqrt(torch.mean(noise**2, dim=1, keepdim=True) + 1e-12)
    return noise * (target_rms / rms)


def add_hiss_noise(z: torch.Tensor, rms: float = 0.10):
    noise = torch.randn_like(z)
    noise = normalize_noise_rms_per_sample(noise, rms)
    return z + noise


def add_pulse_noise(
    z: torch.Tensor,
    pulse_prob: float = 0.08,
    pulse_amp: float = 3.0,
    pulse_width: int = 2,
    rms: float = 0.10,
):
    noise = torch.zeros_like(z)
    base_mask = torch.rand_like(z) < pulse_prob
    for shift in range(pulse_width):
        shifted = torch.roll(base_mask, shifts=shift, dims=1)
        noise += shifted.float() * pulse_amp * torch.randn_like(z)
    noise = normalize_noise_rms_per_sample(noise, rms)
    return z + noise


def add_stepped_tone_noise(z: torch.Tensor, n_steps: int = 4, rms: float = 0.10):
    B, D = z.shape
    noise = torch.zeros_like(z)
    step_len = max(1, D // n_steps)
    for b in range(B):
        phase = random.uniform(0, 2 * math.pi)
        for s in range(n_steps):
            start = s * step_len
            end = D if s == n_steps - 1 else (s + 1) * step_len
            idx = torch.arange(end - start, device=z.device).float()
            freq = random.uniform(0.03, 0.35)
            tone = torch.sin(2 * math.pi * freq * idx + phase)
            noise[b, start:end] = tone
            phase += float(2 * math.pi * freq * len(idx))
    noise = normalize_noise_rms_per_sample(noise, rms)
    return z + noise


def add_warbler_noise(
    z: torch.Tensor,
    carrier=None,
    mod_freq=None,
    mod_depth=None,
    rms: float = 0.10,
):
    B, D = z.shape
    idx = torch.arange(D, device=z.device).float()
    noise = torch.zeros_like(z)
    for b in range(B):
        c = random.uniform(0.05, 0.30) if carrier is None else carrier
        mf = random.uniform(0.005, 0.04) if mod_freq is None else mod_freq
        md = random.uniform(0.03, 0.12) if mod_depth is None else mod_depth
        inst_phase = 2 * math.pi * (c * idx + md * torch.sin(2 * math.pi * mf * idx))
        noise[b] = torch.sin(inst_phase)
    noise = normalize_noise_rms_per_sample(noise, rms)
    return z + noise


def add_recorded_like_noise(z: torch.Tensor, n_components: int = 5, rms: float = 0.10):
    B, D = z.shape
    idx = torch.arange(D, device=z.device).float()
    noise = torch.zeros_like(z)
    for b in range(B):
        signal = torch.zeros(D, device=z.device)
        for _ in range(n_components):
            freq = random.uniform(0.02, 0.40)
            phase = random.uniform(0, 2 * math.pi)
            amp = random.uniform(0.2, 1.0)
            signal += amp * torch.sin(2 * math.pi * freq * idx + phase)
        env_freq = random.uniform(0.005, 0.04)
        env_phase = random.uniform(0, 2 * math.pi)
        envelope = 0.5 + 0.5 * torch.sin(2 * math.pi * env_freq * idx + env_phase)
        signal = envelope * signal
        texture = torch.randn(D, device=z.device)
        kernel = torch.ones(5, device=z.device) / 5.0
        texture = F.conv1d(texture.view(1, 1, -1), kernel.view(1, 1, -1), padding=2).view(-1)
        signal = signal + 0.35 * texture
        noise[b] = signal
    noise = normalize_noise_rms_per_sample(noise, rms)
    return z + noise


def add_spark_noise(
    z: torch.Tensor,
    spark_prob: float = 0.05,
    spark_amp: float = 4.0,
    rms: float = 0.10,
):
    B, D = z.shape
    noise = torch.zeros_like(z)
    for b in range(B):
        for i in range(D):
            if random.random() < spark_prob:
                decay_len = random.randint(2, 8)
                end = min(D, i + decay_len)
                k = torch.arange(end - i, device=z.device).float()
                decay = torch.exp(-k / random.uniform(1.5, 4.0))
                burst = spark_amp * decay * torch.randn(end - i, device=z.device)
                noise[b, i:end] += burst
    noise = normalize_noise_rms_per_sample(noise, rms)
    return z + noise


# --- Train / test wrappers: same RMS schedule as Adaptive notebook ---


def train_hiss_channel(z):
    return add_hiss_noise(z, rms=0.08)


def train_pulse_channel(z):
    return add_pulse_noise(z, rms=0.08)


def train_stepped_channel(z):
    return add_stepped_tone_noise(z, rms=0.08)


def train_warbler_channel(z):
    return add_warbler_noise(z, rms=0.08)


def train_recorded_channel(z):
    return add_recorded_like_noise(z, rms=0.08)


def train_spark_channel(z):
    return add_spark_noise(z, rms=0.08)


def test_hiss_channel(z):
    return add_hiss_noise(z, rms=0.12)


def test_pulse_channel(z):
    return add_pulse_noise(z, pulse_prob=0.10, pulse_amp=3.5, pulse_width=3, rms=0.12)


def test_stepped_channel(z):
    return add_stepped_tone_noise(z, n_steps=5, rms=0.12)


def test_warbler_channel(z):
    return add_warbler_noise(z, rms=0.12)


def test_recorded_channel(z):
    return add_recorded_like_noise(z, n_components=6, rms=0.12)


def test_spark_channel(z):
    return add_spark_noise(z, spark_prob=0.07, spark_amp=4.5, rms=0.12)


TRAIN_CHANNELS = {
    "hiss": train_hiss_channel,
    "pulse": train_pulse_channel,
    "stepped": train_stepped_channel,
    "warbler": train_warbler_channel,
    "recorded": train_recorded_channel,
    "spark": train_spark_channel,
}
TEST_CHANNELS = {
    "hiss": test_hiss_channel,
    "pulse": test_pulse_channel,
    "stepped": test_stepped_channel,
    "warbler": test_warbler_channel,
    "recorded": test_recorded_channel,
    "spark": test_spark_channel,
}


def apply_family_noise(
    clean_frame: torch.Tensor,
    family_id: int,
    split: str,
    device=None,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Apply family noise to a frame; split is 'train' or 'val' (test RMS)."""
    name = CHANNEL_NAMES[int(family_id)]
    dev = device or clean_frame.device
    z = clean_frame.unsqueeze(0).to(dev)
    fn = TRAIN_CHANNELS[name] if split == "train" else TEST_CHANNELS[name]
    noisy = fn(z).squeeze(0)
    if noise_scale != 1.0:
        noisy = clean_frame + float(noise_scale) * (noisy - clean_frame)
    return noisy
