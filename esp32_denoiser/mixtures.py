"""Mix paired and freshly synthesized training examples at a declared ratio."""
import math

import torch
from torch.utils.data import Dataset


class HybridTrainingDataset(Dataset):
    def __init__(self, paired, synthetic, *, synthetic_probability=0.5, epoch_samples=None):
        if not math.isfinite(synthetic_probability) or not 0 < synthetic_probability <= 1:
            raise ValueError("synthetic_probability must be in (0,1]")
        if len(paired) < 1 or len(synthetic) < 1 or paired.sample_rate != synthetic.sample_rate:
            raise ValueError("Both nonempty datasets must have the same sample rate")
        self.paired, self.synthetic = paired, synthetic
        self.synthetic_probability = synthetic_probability
        self.sample_rate = paired.sample_rate
        self.epoch_samples = len(paired) if epoch_samples is None else epoch_samples
        if not isinstance(self.epoch_samples, int) or self.epoch_samples < 1:
            raise ValueError("epoch_samples must be a positive integer")

    def __len__(self):
        return self.epoch_samples

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        # PyTorch seeds each loader worker; the training loop also resets its
        # generator per epoch so resumed runs reproduce this sampling policy.
        if float(torch.rand(())) < self.synthetic_probability:
            return self.synthetic[int(torch.randint(len(self.synthetic), ()).item())]
        return self.paired[int(torch.randint(len(self.paired), ()).item())]
