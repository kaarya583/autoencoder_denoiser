"""Paired VoiceBank-DEMAND preparation, without touching the official test split.

``prepare_voicebank(root, download=True)`` downloads only the two official
28-speaker training ZIPs (~5 GB). With ``download=False`` (the default), put
those ZIPs in ``root/archives`` or extracted directories in ``root/raw``.
Validation holds out entire training speakers p226 and p287. The official test
set is prepared only with ``include_test=True``; never use it for selection.

Audio is resampled once with scipy's polyphase FIR and cached as FLOAT WAV,
preserving scale and avoiding additional PCM quantization/clipping. JSONL
manifests use paths relative to their containing directory and carry source
split, speaker, sample rate, and length. No import or Dataset call downloads.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000
VALIDATION_SPEAKERS = ("p226", "p287")
SOURCE_URL = "https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757"
DATASET_DOI = "https://doi.org/10.7488/ds/2117"
CACHE_VERSION = 1
HF_REPO = "JacobLinCool/VoiceBank-DEMAND-16k"
HF_REVISION = "4497db342d7312978c45690591fda86117831940"


@dataclass(frozen=True)
class Archive:
    filename: str
    url: str
    approximate_bytes: int
    md5: str | None


# URLs and checksums are published by University of Edinburgh DataShare.
# The noisy training MD5 was not exposed on the first metadata page; ZIP CRCs
# are still checked while reading each member. Do not invent a checksum.
ARCHIVES = {
    "train_clean": Archive(
        "clean_trainset_28spk_wav.zip",
        "https://datashare.ed.ac.uk/bitstreams/245452b6-6235-44b6-a6f9-e7eb19797769/download",
        2_320_000_000,
        "d2d5a45ec32f8fcbf201bde0447e20ba",
    ),
    "train_noisy": Archive(
        "noisy_trainset_28spk_wav.zip",
        "https://datashare.ed.ac.uk/bitstreams/ecb5a102-bb00-46d3-8af5-40c79823b837/download",
        2_640_000_000,
        None,
    ),
    "test_clean": Archive(
        "clean_testset_wav.zip",
        "https://datashare.ed.ac.uk/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/download",
        147_180_000,
        "34eb1c0ba7ef667e9b966866c542fc16",
    ),
    "test_noisy": Archive(
        "noisy_testset_wav.zip",
        "https://datashare.ed.ac.uk/bitstreams/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/download",
        162_680_000,
        "fb1b86caa31e8ba5b506c0c64da9aab5",
    ),
}


def _index_names(names: Iterable[str]) -> dict[str, str]:
    result = {}
    for name in sorted(names):
        path = Path(name)
        if path.suffix.lower() != ".wav" or "__MACOSX" in path.parts or path.name.startswith("."):
            continue
        if not re.fullmatch(r"p\d+_\d+", path.stem):
            raise ValueError(f"Expected a VoiceBank speaker/utterance filename, got {name}")
        if path.stem in result:
            raise ValueError(f"Duplicate utterance: {path.stem}")
        result[path.stem] = name
    if not result:
        raise ValueError("No VoiceBank WAV files found")
    return result


def _check_pairs(clean: dict, noisy: dict) -> None:
    if clean.keys() != noisy.keys():
        missing_noisy = sorted(clean.keys() - noisy.keys())[:5]
        missing_clean = sorted(noisy.keys() - clean.keys())[:5]
        raise ValueError(f"Unpaired utterances: missing noisy={missing_noisy}, missing clean={missing_clean}")


def discover_pairs(clean_dir: str | Path, noisy_dir: str | Path) -> list[dict]:
    """Match WAVs by utterance id, failing on duplicates or missing partners."""
    clean = _index_names(str(p.resolve()) for p in Path(clean_dir).rglob("*.wav"))
    noisy = _index_names(str(p.resolve()) for p in Path(noisy_dir).rglob("*.wav"))
    _check_pairs(clean, noisy)
    return [dict(id=k, speaker=k.split("_")[0], clean=clean[k], noisy=noisy[k]) for k in sorted(clean)]


def split_training_pairs(pairs: list[dict], val_speakers=VALIDATION_SPEAKERS) -> tuple[list[dict], list[dict]]:
    """Hold out whole speakers from training; missing requested speakers fail."""
    held_out = set(val_speakers)
    speakers = {p["speaker"] for p in pairs}
    if not held_out or held_out - speakers:
        raise ValueError(f"Validation speakers must be present in training: missing={sorted(held_out - speakers)}")
    train = [p for p in pairs if p["speaker"] not in held_out]
    val = [p for p in pairs if p["speaker"] in held_out]
    if not train:
        raise ValueError("Validation split leaves no training speakers")
    return train, val


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_archive(path: Path, archive: Archive) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a complete ZIP archive: {path}")
    if archive.md5:
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != archive.md5:
            raise ValueError(f"Official MD5 mismatch: {path}; remove or replace this archive")


def download_archive(archive: Archive, directory: str | Path) -> Path:
    """Explicitly download one official archive; validate before atomic rename.

    Existing ZIPs are reused after validation. Failed downloads remain as
    ``.part`` files for diagnosis, and a later call restarts the transfer.
    Approximate sizes are metadata only, not exact integrity checks.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / archive.filename
    if destination.exists():
        _verify_archive(destination, archive)
        return destination
    temporary = destination.with_suffix(".zip.part")
    request = urllib.request.Request(archive.url, headers={"User-Agent": "ESP32-Denoiser-Research/1.0"})
    print(f"Downloading {archive.filename} (~{archive.approximate_bytes / 1e9:.2f} GB)", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
    _verify_archive(temporary, archive)
    temporary.replace(destination)
    return destination


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(suffix=".wav", dir=path.parent)
    os.close(descriptor)
    temporary = Path(filename)
    try:
        sf.write(temporary, audio, sample_rate, subtype="FLOAT")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resample_pair(clean_source, noisy_source, clean_path: Path, noisy_path: Path, sample_rate: int) -> int:
    clean, clean_sr = sf.read(clean_source, dtype="float32")
    noisy, noisy_sr = sf.read(noisy_source, dtype="float32")
    if clean.ndim != 1 or noisy.ndim != 1:
        raise ValueError("Paired speech must be mono; implicit downmixing is not permitted")
    if clean_sr != noisy_sr or clean.shape != noisy.shape or clean.size == 0:
        raise ValueError("Clean/noisy sample rates and nonempty lengths must match exactly")
    if not np.isfinite(clean).all() or not np.isfinite(noisy).all():
        raise ValueError("Audio contains NaN or infinity")
    if clean_sr != sample_rate:
        divisor = math.gcd(clean_sr, sample_rate)
        # A shared filter on both signals preserves their sample alignment.
        pair = resample_poly(np.stack((clean, noisy)), sample_rate // divisor, clean_sr // divisor, axis=-1)
        clean, noisy = pair.astype(np.float32, copy=False)
    _write_audio(clean_path, clean, sample_rate)
    _write_audio(noisy_path, noisy, sample_rate)
    return len(clean)


def _prepare_split(root: Path, split: str, download: bool, sample_rate: int, workers: int) -> list[dict]:
    with ExitStack() as stack:
        sources = {}
        signatures = {}
        for role in ("clean", "noisy"):
            archive = ARCHIVES[f"{split}_{role}"]
            extracted = root / "raw" / archive.filename.removesuffix(".zip")
            if extracted.is_dir():
                index = _index_names(str(p.resolve()) for p in extracted.rglob("*.wav"))
                sources[role] = (index, None)
                signatures[role] = "files"
            else:
                path = root / "archives" / archive.filename
                if not path.is_file():
                    if not download:
                        raise FileNotFoundError(f"Missing {path}. Supply it, extract into {extracted}, or pass download=True.")
                    download_archive(archive, path.parent)
                else:
                    # Full MD5 verification of multi-GB files is done once per
                    # unchanged local archive, rather than once per training run.
                    stat = path.stat()
                    marker = path.with_suffix(".verified.json")
                    stamp = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "md5": archive.md5}
                    if not marker.exists() or json.loads(marker.read_text()) != stamp:
                        _verify_archive(path, archive)
                        _atomic_json(marker, stamp)
                stat = path.stat()
                signatures[role] = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
                reader = stack.enter_context(zipfile.ZipFile(path))
                sources[role] = (_index_names(reader.namelist()), reader)
        _check_pairs(sources["clean"][0], sources["noisy"][0])

        def prepare_one(utterance: str) -> dict:
            identity = [CACHE_VERSION, sample_rate, utterance, signatures]
            for role in ("clean", "noisy"):
                index, reader = sources[role]
                if reader is None:
                    source_path = Path(index[utterance])
                    stat = source_path.stat()
                    identity.append([str(source_path), stat.st_size, stat.st_mtime_ns])
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
            paths = {role: root / f"audio_{sample_rate}" / role / f"{utterance}_{digest}.wav" for role in ("clean", "noisy")}
            if all(p.is_file() for p in paths.values()):
                infos = [sf.info(paths[role]) for role in ("clean", "noisy")]
                if any(info.samplerate != sample_rate or info.channels != 1 for info in infos) or infos[0].frames != infos[1].frames:
                    raise ValueError(f"Invalid cached pair: {utterance}")
                samples = infos[0].frames
            else:
                audio_sources = {}
                for role in ("clean", "noisy"):
                    index, reader = sources[role]
                    audio_sources[role] = io.BytesIO(reader.read(index[utterance])) if reader else index[utterance]
                samples = _resample_pair(audio_sources["clean"], audio_sources["noisy"], paths["clean"], paths["noisy"], sample_rate)
            return dict(id=utterance, speaker=utterance.split("_")[0], source_split=split,
                        samples=samples, sample_rate=sample_rate,
                        clean=os.path.relpath(paths["clean"], root / "manifests"),
                        noisy=os.path.relpath(paths["noisy"], root / "manifests"))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(prepare_one, sorted(sources["clean"][0])))


def prepare_voicebank(
    root: str | Path,
    *,
    download: bool = False,
    include_test: bool = False,
    val_speakers=VALIDATION_SPEAKERS,
    sample_rate: int = SAMPLE_RATE,
    workers: int = 4,
) -> dict[str, Path]:
    """Create train/val JSONL manifests and provenance; return their paths.

    Set ``include_test=True`` only for a frozen-model final evaluation. The
    default does not read, download, or score official test data. Changing
    sources or resampling rate produces a fresh, fingerprinted audio cache.
    """
    if sample_rate <= 0 or workers < 1:
        raise ValueError("sample_rate and workers must be positive")
    root = Path(root).resolve()
    training = _prepare_split(root, "train", download, sample_rate, workers)
    train, val = split_training_pairs(training, val_speakers)
    splits = {"train": train, "val": val}
    if include_test:
        test = _prepare_split(root, "test", download, sample_rate, workers)
        if {p["speaker"] for p in training} & {p["speaker"] for p in test}:
            raise ValueError("Official train/test speakers overlap")
        splits["test"] = test
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, records in splits.items():
        destination = manifest_dir / f"{split}.jsonl"
        with tempfile.NamedTemporaryFile("w", dir=manifest_dir, delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(destination)
        result[split] = destination
    _atomic_json(manifest_dir / "provenance.json", {
        "dataset": "VoiceBank-DEMAND 28-speaker", "doi": DATASET_DOI, "source": SOURCE_URL,
        "author": "Cassia Valentini-Botinhao", "date_available": "2017-08-21", "license": "CC-BY-4.0",
        "cache_version": CACHE_VERSION, "resampling": "scipy.signal.resample_poly, default Kaiser FIR",
        "cache_format": "WAV FLOAT", "sample_rate": sample_rate, "validation_speakers": sorted(val_speakers),
        "official_test_prepared": include_test,
        "archives": {k: asdict(v) for k, v in ARCHIVES.items() if include_test or k.startswith("train_")},
        "splits": {k: {"utterances": len(v), "samples": sum(p["samples"] for p in v),
                       "speakers": sorted({p["speaker"] for p in v})} for k, v in splits.items()},
    })
    return result


def prepare_voicebank_parquet(
    root: str | Path,
    *,
    download: bool = False,
    include_test: bool = False,
    val_speakers=VALIDATION_SPEAKERS,
    workers: int = 4,
) -> dict[str, Path]:
    """Prepare the pinned third-party 16 kHz mirror if Edinburgh is unavailable.

    Requires pyarrow; explicit downloading additionally requires huggingface_hub.
    Reuses ``root/archives/hf16k/data/train-00000-of-00005.parquet`` etc.
    Only requested split shards are downloaded. Parquet is read in small
    batches, not loaded into pandas or decoded into RAM as an entire dataset.
    The mirror's original resampler and byte identity to official archives are
    unverified; this limitation is recorded in provenance, not concealed.
    """
    import pyarrow.parquet as pq

    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(root).resolve()
    mirror_root = root / "archives" / "hf16k"
    split_records = {}
    file_metadata = []
    for split in ("train", "test") if include_test else ("train",):
        shards = 5 if split == "train" else 1
        records = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for shard in range(shards):
                name = f"data/{split}-{shard:05d}-of-{shards:05d}.parquet"
                path = mirror_root / name
                if not path.is_file():
                    if not download:
                        raise FileNotFoundError(f"Missing {path}; populate the pinned mirror or pass download=True")
                    from huggingface_hub import hf_hub_download

                    hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=name,
                                    revision=HF_REVISION, local_dir=mirror_root)
                stat = path.stat()
                source_identity = [HF_REPO, HF_REVISION, name, stat.st_size, stat.st_mtime_ns, CACHE_VERSION]
                file_metadata.append({"path": name, "bytes": stat.st_size})
                digest = hashlib.sha256(json.dumps(source_identity).encode()).hexdigest()[:20]

                def prepare_row(row: dict) -> dict:
                    utterance = row["id"]
                    if not isinstance(utterance, str) or not re.fullmatch(r"p\d+_\d+", utterance):
                        raise ValueError(f"Invalid mirror utterance ID: {utterance}")
                    paths = {role: root / "audio_16000_hf" / role / f"{utterance}_{digest}.wav" for role in ("clean", "noisy")}
                    if all(p.is_file() for p in paths.values()):
                        infos = [sf.info(paths[role]) for role in ("clean", "noisy")]
                        if any(i.samplerate != SAMPLE_RATE or i.channels != 1 for i in infos) or infos[0].frames != infos[1].frames:
                            raise ValueError(f"Invalid cached mirror pair: {utterance}")
                        samples = infos[0].frames
                    else:
                        streams = {}
                        for role in ("clean", "noisy"):
                            blob = row[role].get("bytes")
                            if not isinstance(blob, bytes):
                                raise ValueError(f"Missing embedded {role} audio bytes for {utterance}")
                            streams[role] = io.BytesIO(blob)
                            if sf.info(streams[role]).samplerate != SAMPLE_RATE:
                                raise ValueError(f"Mirror is not 16 kHz for {utterance}")
                            streams[role].seek(0)
                        samples = _resample_pair(streams["clean"], streams["noisy"], paths["clean"], paths["noisy"], SAMPLE_RATE)
                    return dict(id=utterance, speaker=utterance.split("_")[0], source_split=split,
                                samples=samples, sample_rate=SAMPLE_RATE,
                                clean=os.path.relpath(paths["clean"], root / "manifests"),
                                noisy=os.path.relpath(paths["noisy"], root / "manifests"))

                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(batch_size=max(8, workers * 2), columns=["id", "clean", "noisy"]):
                    records.extend(pool.map(prepare_row, batch.to_pylist()))
        if len({p["id"] for p in records}) != len(records):
            raise ValueError(f"Duplicate utterance IDs in {split} mirror shards")
        split_records[split] = sorted(records, key=lambda p: p["id"])
    training = split_records.pop("train")
    train, val = split_training_pairs(training, val_speakers)
    splits = {"train": train, "val": val, **split_records}
    if include_test and {p["speaker"] for p in training} & {p["speaker"] for p in splits["test"]}:
        raise ValueError("Mirror train/test speakers overlap")
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, records in splits.items():
        destination = manifest_dir / f"{split}.jsonl"
        with tempfile.NamedTemporaryFile("w", dir=manifest_dir, delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(destination)
        result[split] = destination
    _atomic_json(manifest_dir / "provenance.json", {
        "dataset": "VoiceBank-DEMAND 28-speaker, third-party 16 kHz mirror", "doi": DATASET_DOI,
        "original_source": SOURCE_URL, "mirror": f"https://huggingface.co/datasets/{HF_REPO}",
        "mirror_revision": HF_REVISION, "mirror_files": file_metadata,
        "author": "Cassia Valentini-Botinhao", "license": "CC-BY-4.0 (original and mirror declaration)",
        "resampling": "Already 16 kHz; mirror resampling method not documented; byte identity to official archives unverified",
        "cache_format": "WAV FLOAT; decoded samples preserved without additional resampling",
        "cache_version": CACHE_VERSION, "sample_rate": SAMPLE_RATE,
        "validation_speakers": sorted(val_speakers), "official_test_prepared": include_test,
        "splits": {k: {"utterances": len(v), "samples": sum(p["samples"] for p in v),
                       "speakers": sorted({p["speaker"] for p in v})} for k, v in splits.items()},
    })
    return result


def read_manifest(path: str | Path) -> list[dict]:
    """Read records, resolving audio paths relative to this JSONL file."""
    path = Path(path).resolve()
    records = []
    seen = set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in seen:
                raise ValueError(f"Duplicate manifest utterance: {record['id']}")
            seen.add(record["id"])
            for role in ("clean", "noisy"):
                record[role] = str((path.parent / record[role]).resolve())
            records.append(record)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


class PairedAudioDataset(Dataset):
    """Aligned crops with shared gain, or unmodified full utterances for eval.

    Returns ``{noisy, clean, length, id, speaker}``; tensors are float32 ``[T]``.
    ``crop_seconds`` may be one duration, a ``(min,max)`` range, or None.
    Short clips are zero padded; ``length`` excludes padding. Use ``pad_collate``
    for batches and pass lengths to losses/metrics. ``random_crop=False`` uses
    a deterministic center crop and midpoint duration. Gain defaults to off.
    Torch's worker-local RNG supplies random crops and gain (DataLoader seeds
    workers automatically); no shared NumPy generator is copied into workers.
    """

    def __init__(self, manifest: str | Path | list[dict], *, crop_seconds: float | tuple[float, float] | None = 3.0,
                 random_crop: bool = True, gain_db: tuple[float, float] = (0.0, 0.0), sample_rate: int = SAMPLE_RATE,
                 noise_scale_db: tuple[float, float] = (0.0, 0.0), clean_identity_prob: float = 0.0):
        self.records = read_manifest(manifest) if isinstance(manifest, (str, Path)) else list(manifest)
        if not self.records:
            raise ValueError("Dataset is empty")
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.gain_db = tuple(gain_db)
        if len(self.gain_db) != 2 or not all(math.isfinite(x) for x in self.gain_db) or self.gain_db[0] > self.gain_db[1]:
            raise ValueError("gain_db must be a finite ordered pair")
        self.noise_scale_db = tuple(noise_scale_db)
        if len(self.noise_scale_db) != 2 or not all(math.isfinite(x) for x in self.noise_scale_db) or self.noise_scale_db[0] > self.noise_scale_db[1]:
            raise ValueError("noise_scale_db must be a finite ordered pair")
        if not math.isfinite(clean_identity_prob) or not 0 <= clean_identity_prob <= 1:
            raise ValueError("clean_identity_prob must be between 0 and 1")
        self.clean_identity_prob = clean_identity_prob
        self.crop_range = None
        if crop_seconds is not None:
            durations = (crop_seconds, crop_seconds) if isinstance(crop_seconds, (int, float)) else crop_seconds
            if len(durations) != 2 or not all(math.isfinite(x) and x > 0 for x in durations) or durations[0] > durations[1]:
                raise ValueError("crop_seconds must be positive and ordered")
            self.crop_range = tuple(max(1, round(x * sample_rate)) for x in durations)
        for record in self.records:
            if record["sample_rate"] != sample_rate or record["samples"] <= 0:
                raise ValueError("Manifest contains wrong sample rate or empty audio")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        total = record["samples"]
        size = total
        if self.crop_range:
            low, high = self.crop_range
            size = int(torch.randint(low, high + 1, ()).item()) if self.random_crop and low != high else (low + high) // 2
        available = max(0, total - size)
        start = int(torch.randint(available + 1, ()).item()) if self.random_crop and available else available // 2
        length = min(total, size)
        result = {"id": record["id"], "speaker": record["speaker"], "length": length}
        gain = 10 ** ((self.gain_db[0] + torch.rand(()).item() * (self.gain_db[1] - self.gain_db[0])) / 20) if self.gain_db[0] != self.gain_db[1] else 10 ** (self.gain_db[0] / 20)
        for role in ("clean", "noisy"):
            audio, rate = sf.read(record[role], start=start, frames=length, dtype="float32")
            if rate != self.sample_rate or audio.ndim != 1 or len(audio) != length or not np.isfinite(audio).all():
                raise ValueError(f"Invalid audio or stale manifest for {record['id']}: {role}")
            # Shared gain preserves the mixture and SNR; never independently
            # normalize or clip clean/noisy channels.
            if gain != 1:
                audio *= gain
            if size > length:
                audio = np.pad(audio, (0, size - length))
            result[role] = torch.from_numpy(audio)
        if self.noise_scale_db != (0.0, 0.0):
            low, high = self.noise_scale_db
            scale = 10 ** ((low + torch.rand(()).item() * (high - low)) / 20)
            result["noisy"] = result["clean"] + (result["noisy"] - result["clean"]) * scale
        if self.clean_identity_prob > 0 and torch.rand(()).item() < self.clean_identity_prob:
            result["noisy"] = result["clean"].clone()
        return result


def pad_collate(items: list[dict]) -> dict:
    """Batch variable-length items, retaining valid lengths for masked losses."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    return {
        "noisy": torch.nn.utils.rnn.pad_sequence([item["noisy"] for item in items], batch_first=True),
        "clean": torch.nn.utils.rnn.pad_sequence([item["clean"] for item in items], batch_first=True),
        "length": torch.tensor([item["length"] for item in items], dtype=torch.long),
        "id": [item["id"] for item in items], "speaker": [item["speaker"] for item in items],
    }
