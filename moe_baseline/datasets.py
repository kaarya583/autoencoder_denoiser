"""PyTorch datasets."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from moe_baseline.config import NUM_FAMILIES, SEED
from moe_baseline.noise import apply_family_noise


class NoisyFrameDataset(Dataset):
    """Clean frames from FrameDataset + family noise (random or fixed family)."""

    def __init__(
        self,
        frame_ds: Dataset,
        split: str,
        rng: np.random.Generator,
        family_id: int | None = None,
    ):
        self.frame_ds = frame_ds
        self.split = split
        self.rng = rng
        self.family_id = family_id

    def __len__(self) -> int:
        return len(self.frame_ds)

    def __getitem__(self, idx: int):
        clean = self.frame_ds[idx]
        fam = (
            int(self.family_id)
            if self.family_id is not None
            else int(self.rng.integers(0, NUM_FAMILIES))
        )
        noisy = apply_family_noise(clean, fam, self.split)
        return noisy.cpu(), clean.cpu(), fam


class PureNoiseLatentDataset(Dataset):
    """Pure latent noise padded into FRAME_SIZE for neural router pretraining."""

    def __init__(self, n_samples: int, frame_size: int, latent_dim: int, device: torch.device):
        from moe_baseline.config import CHANNEL_NAMES
        from moe_baseline.noise import TRAIN_CHANNELS

        self.n = n_samples
        self.frame_size = frame_size
        self.latent_dim = latent_dim
        self.device = device
        self.channels = TRAIN_CHANNELS
        self.names = CHANNEL_NAMES
        self.rng = np.random.default_rng(SEED + 99)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        from moe_baseline.features import latent_noise_to_numpy

        fam = int(self.rng.integers(0, NUM_FAMILIES))
        noise = latent_noise_to_numpy(self.channels[self.names[fam]], 1, self.latent_dim, self.device)[0]
        x = np.zeros(self.frame_size, dtype=np.float32)
        x[: self.latent_dim] = noise.astype(np.float32)
        return torch.from_numpy(x), fam
