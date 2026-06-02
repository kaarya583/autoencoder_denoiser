"""Waveform loading and framing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from moe_baseline.config import FRAME_SIZE, HOP_SIZE, TARGET_SR


def load_wav(path, sr: int = TARGET_SR) -> np.ndarray:
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if file_sr != sr:
        raise ValueError(f"Unexpected sample rate {file_sr} for {path}")
    return np.asarray(x, dtype=np.float32)


def frame_audio(
    audio: np.ndarray,
    frame_size: int = FRAME_SIZE,
    hop_size: int = HOP_SIZE,
) -> np.ndarray:
    n = len(audio)
    if n < frame_size:
        return np.zeros((0, frame_size), dtype=np.float32)
    frames = [audio[start : start + frame_size] for start in range(0, n - frame_size + 1, hop_size)]
    return np.stack(frames, axis=0).astype(np.float32)


class FrameDataset(Dataset):
    """Preload frames from file list (adaptive notebook)."""

    def __init__(self, file_paths, frame_size: int = FRAME_SIZE, hop_size: int = HOP_SIZE):
        frames = []
        for path in file_paths:
            audio = load_wav(Path(path), TARGET_SR)
            x = frame_audio(audio, frame_size, hop_size)
            if len(x) > 0:
                frames.append(x)
        if not frames:
            raise ValueError("No frames found.")
        self.frames = np.concatenate(frames, axis=0).astype(np.float32)
        print(f"Built dataset with {len(self.frames)} frames from {len(file_paths)} files.")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.frames[idx].copy())
