"""LibriSpeech download and file selection (same logic as adaptive notebook)."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

from moe_baseline.config import LIBRISPEECH_ARCHIVES, MAX_TEST_FILES, MAX_TRAIN_FILES, TARGET_SR


def _is_valid_gzip(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} from OpenSLR...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def ensure_librispeech(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(data_root.rglob("*.flac"))
    if any("dev-clean" in str(p) for p in existing) and any("test-clean" in str(p) for p in existing):
        print("LibriSpeech audio already present under", data_root)
        return
    for name, url in LIBRISPEECH_ARCHIVES.items():
        dest = data_root / name
        if dest.exists() and not _is_valid_gzip(dest):
            dest.unlink(missing_ok=True)
        if not dest.exists():
            _download(url, dest)
        print("Extracting", dest)
        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(path=data_root)


def pick_training_subset(files, max_files: int = 80, min_seconds: float = 3.0) -> list:
    chosen = []
    for path in files:
        try:
            x, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=-1)
            if file_sr != TARGET_SR:
                continue
            if len(x) / TARGET_SR >= min_seconds:
                chosen.append(path)
        except Exception:
            continue
        if len(chosen) >= max_files:
            break
    return chosen


def list_flac_splits(data_root: Path) -> tuple[list, list]:
    all_flac = sorted(data_root.rglob("*.flac"))
    dev = [p for p in all_flac if "dev-clean" in str(p)]
    test = [p for p in all_flac if "test-clean" in str(p)]
    if not dev or not test:
        raise FileNotFoundError("No LibriSpeech .flac files found under " + str(data_root))
    return dev, test


def get_train_test_files(data_root: Path) -> tuple[list, list]:
    ensure_librispeech(data_root)
    dev, test = list_flac_splits(data_root)
    train_files = pick_training_subset(dev, max_files=MAX_TRAIN_FILES)
    test_files = pick_training_subset(test, max_files=MAX_TEST_FILES)
    print(f"Train files selected: {len(train_files)}")
    print(f"Test files selected:  {len(test_files)}")
    if not train_files or not test_files:
        raise ValueError("Empty train/test file list.")
    return train_files, test_files
