"""Audited optional speech/noise sources and aligned dynamic mixtures.

No imports or Dataset calls download data. ``prepare_extra_data(download=True)``
downloads LibriSpeech train-clean-100 (~6.3 GB) and MUSAN (~11 GB), but extracts
only LibriSpeech FLAC, MUSAN noise WAV, and attribution text. Optional OpenSLR28
preparation extracts simulated RIRs only, holding out entire simulated rooms.
DNS, DEMAND, VCTK, MUSAN speech/music and OpenSLR28 noise are never admitted.

Speech/recording/room holdouts are assigned before cropping. Sources stay at
their published 16 kHz; training reads bounded crops without a float WAV copy.
Official reference: https://www.openslr.org/{12,17,28}/ . Only LibriSpeech has a
publisher checksum verified here; all archives additionally get a local SHA256.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import http.client
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


SAMPLE_RATE = 16000
VERSION = 1


@dataclass(frozen=True)
class SourceArchive:
    filename: str
    urls: tuple[str, ...]
    source: str
    license: str
    expected_md5: str | None = None


def _urls(resource, filename):
    return tuple(f"https://{host}/resources/{resource}/{filename}" for host in
                 ("www.openslr.org", "openslr.trmal.net", "openslr.elda.org"))


SOURCES = {
    "speech": SourceArchive("train-clean-100.tar.gz", _urls(12, "train-clean-100.tar.gz"),
                            "https://www.openslr.org/12/", "CC-BY-4.0",
                            "2a93770f6d5c6c964bc36631d331a522"),
    "noise": SourceArchive("musan.tar.gz", _urls(17, "musan.tar.gz"),
                           "https://www.openslr.org/17/", "CC-BY-4.0"),
    "rir": SourceArchive("rirs_noises.zip", _urls(28, "rirs_noises.zip"),
                         "https://www.openslr.org/28/", "Apache-2.0"),
}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hash_file(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def _check_archive(path, source):
    with Path(path).open("rb") as handle:
        magic = handle.read(4)
    if not (magic.startswith(b"\x1f\x8b") if source.filename.endswith(".gz") else magic == b"PK\x03\x04"):
        raise ValueError(f"Not the expected archive format (possibly HTML): {path}")
    hashes = _hash_file(path)
    if source.expected_md5 is not None and hashes["md5"] != source.expected_md5:
        raise ValueError(f"Publisher MD5 mismatch: {path}")
    return hashes


def download_source(source: SourceArchive, directory: str | Path) -> Path:
    """Stream a download, resume only a matching byte range, verify, then rename.

Partial files stay on interrupted transfers. A server ignoring Range causes a
fresh write, never concatenation of two complete responses. Mirror fallback
starts afresh because the partial file belongs to its recorded URL.
"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.filename
    if destination.exists():
        _check_archive(destination, source)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    url_marker = partial.with_suffix(partial.suffix + ".url")
    failures = []
    for url in source.urls:
        for attempt in range(2):
            try:
                same_url = url_marker.exists() and url_marker.read_text() == url
                offset = partial.stat().st_size if partial.exists() and same_url else 0
                headers = {"User-Agent": "ESP32-Denoiser-Research/1.0", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError("Server returned the wrong resume byte range")
                    elif response.status == 200:
                        offset = 0
                    else:
                        raise ValueError(f"Unexpected download status {response.status}")
                    if "text/html" in response.headers.get("Content-Type", "").lower():
                        raise ValueError("Archive server returned HTML")
                    url_marker.write_text(url)
                    received, last_print = 0, time.monotonic()
                    with partial.open("ab" if offset else "wb") as handle:
                        while block := response.read(4 * 1024 * 1024):
                            handle.write(block)
                            received += len(block)
                            if time.monotonic() - last_print > 20:
                                print(f"{source.filename}: {(offset + received) / 1e9:.2f} GB downloaded", flush=True)
                                last_print = time.monotonic()
                    expected = response.headers.get("Content-Length")
                    if expected is not None and received != int(expected):
                        raise IOError(f"Incomplete archive body: {received} != {expected}")
                _check_archive(partial, source)
                partial.replace(destination)
                url_marker.unlink(missing_ok=True)
                return destination
            except (OSError, ValueError, urllib.error.URLError, http.client.IncompleteRead) as error:
                failures.append(f"{url} attempt {attempt + 1}: {error}")
                # Integrity failures cannot be repaired by appending a range.
                if isinstance(error, ValueError):
                    partial.unlink(missing_ok=True)
                    url_marker.unlink(missing_ok=True)
    raise RuntimeError("Could not download source archive:\n" + "\n".join(failures))


def _safe_parts(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    return path.parts


def _asset_identity(kind, name):
    """Return corpus ID/group for an admitted audio member, otherwise None."""
    parts = _safe_parts(name)
    if kind == "speech" and len(parts) == 5 and parts[:2] == ("LibriSpeech", "train-clean-100") and parts[-1].endswith(".flac"):
        speaker, chapter = parts[2:4]
        stem = PurePosixPath(name).stem
        if not speaker.isdigit() or not chapter.isdigit() or not stem.startswith(f"{speaker}-{chapter}-"):
            raise ValueError(f"Invalid LibriSpeech recording path: {name}")
        return f"librispeech:{stem}", f"librispeech:speaker:{speaker}", speaker
    if kind == "noise" and len(parts) >= 4 and parts[:2] == ("musan", "noise") and parts[-1].endswith(".wav"):
        recording = "/".join(parts[2:])
        return f"musan:{recording}", f"musan:recording:{recording}", None
    if kind == "rir" and len(parts) >= 5 and parts[:2] == ("RIRS_NOISES", "simulated_rirs") and parts[-1].endswith(".wav"):
        # Original layout: simulated_rirs/{small,medium,large}room/RoomNNN/...
        if parts[2] not in {"smallroom", "mediumroom", "largeroom"} or not parts[3].lower().startswith("room"):
            raise ValueError(f"Cannot determine original simulated room: {name}")
        recording = "/".join(parts[2:])
        return f"openslr28:{recording}", f"openslr28:room:{parts[2]}/{parts[3]}", None
    return None


def _extract_source(archive, source, kind, destination):
    destination.mkdir(parents=True, exist_ok=True)
    hashes = _check_archive(archive, source)
    stamp = {"version": VERSION, "archive_sha256": hashes["sha256"], "kind": kind}
    inventory_path = destination / "EXTRACTION.json"
    if inventory_path.exists():
        previous = json.loads(inventory_path.read_text())
        if previous.get("stamp") == stamp:
            for item in previous["audio"]:
                path = destination / item["member"]
                if (not path.is_file() or path.is_symlink() or
                        not path.resolve().is_relative_to(destination.resolve()) or
                        path.stat().st_size != item["bytes"]):
                    raise ValueError(f"Missing or changed extracted source: {path}")
                if path.stat().st_mtime_ns != item.get("mtime_ns") and _hash_file(path)["sha256"] != item["sha256"]:
                    raise ValueError(f"Changed extracted source checksum: {path}")
            return previous, hashes
    inventory, seen = [], set()

    def copy_member(name, size, stream, regular):
        parts = _safe_parts(name)
        identity = _asset_identity(kind, name)
        attribution = PurePosixPath(name).suffix.lower() in {".txt", ".md"} or parts[-1].upper() in {"LICENSE", "COPYING", "README", "AUTHORS", "INFO"}
        if not regular or (identity is None and not attribution):
            return
        if size < 0 or size > 1024**3:
            raise ValueError(f"Unexpectedly large audio/metadata archive member: {name}")
        if name in seen:
            raise ValueError(f"Duplicate archive member: {name}")
        seen.add(name)
        path = destination.joinpath(*parts)
        # Never follow a link left in an extracted directory by another tool.
        if not path.resolve().is_relative_to(destination.resolve()) or path.is_symlink():
            raise ValueError(f"Extracted member escapes destination: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        digest, count = hashlib.sha256(), 0
        temporary = path.with_suffix(path.suffix + ".part")
        if temporary.is_symlink():
            raise ValueError(f"Temporary extraction path is a link: {temporary}")
        with temporary.open("wb") as output:
            while block := stream.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                count += len(block)
        if count != size:
            raise ValueError(f"Truncated archive member: {name}")
        temporary.replace(path)
        if identity is not None:
            info = sf.info(path)
            if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames < 1:
                raise ValueError(f"Expected nonempty 16 kHz mono source: {name}")
            identifier, group, speaker = identity
            inventory.append({"id": identifier, "group": group, "speaker": speaker,
                              "member": name, "samples": info.frames, "sample_rate": SAMPLE_RATE,
                              "bytes": count, "sha256": digest.hexdigest(), "mtime_ns": path.stat().st_mtime_ns})

    if source.filename.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                regular = not item.is_dir() and (item.external_attr >> 16) & 0o170000 != 0o120000
                with bundle.open(item) as stream:
                    copy_member(item.filename, item.file_size, stream, regular)
    else:
        # Sequential tar mode does not allocate a multi-GB archive in memory.
        with tarfile.open(archive, "r|gz") as bundle:
            for item in bundle:
                _safe_parts(item.name)
                if item.isfile():
                    with bundle.extractfile(item) as stream:
                        copy_member(item.name, item.size, stream, True)
    if not inventory:
        raise ValueError(f"No admitted {kind} audio in {archive}")
    result = {"stamp": stamp, "audio": inventory}
    _atomic_json(inventory_path, result)
    return result, hashes


def _group_split(records, fraction, seed):
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate extra-data recording identity")
    groups = sorted({record["group"] for record in records},
                    key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    if len(groups) < 2:
        raise ValueError("At least two original speaker/recording/room groups are required")
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    held_out = set(groups[:count])
    splits = {"train": [], "val": []}
    for record in sorted(records, key=lambda record: record["id"]):
        split = "val" if record["group"] in held_out else "train"
        splits[split].append({**record, "split": split})
    # Exact duplicate source files cannot appear on opposite sides even when
    # filenames claim different identities. Fail instead of hiding the overlap.
    for key in ("id", "group", "path", "sha256"):
        if {r[key] for r in splits["train"]} & {r[key] for r in splits["val"]}:
            raise ValueError(f"Extra source train/validation {key} overlap")
    return splits


def prepare_extra_data(root: str | Path, *, download: bool = False, include_rir: bool = False,
                       validation_fraction: float = 0.1, seed: int = 2026) -> dict[str, Path]:
    """Prepare original-source manifests and return their paths plus provenance.

Keys: speech_train, speech_val, noise_train, noise_val, optional rir_train /
rir_val, and provenance. Audio paths are relative to the manifest directory.
Changing split parameters requires a new experiment: provenance records them.
"""
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    root = Path(root).resolve()
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    outputs, provenance = {}, {"version": VERSION, "sample_rate": SAMPLE_RATE, "seed": seed,
                              "validation_fraction": validation_fraction,
                              "holdout": "whole speakers / original noise recordings / simulated rooms before cropping",
                              "excluded": ["DNS", "DEMAND", "VCTK", "MUSAN speech/music", "OpenSLR28 real RIRs and noise"],
                              "resampling": "none; preserve original 16 kHz source files", "sources": {}, "splits": {}}
    for kind in ("speech", "noise", "rir") if include_rir else ("speech", "noise"):
        source = SOURCES[kind]
        archive = root / "archives" / source.filename
        if not archive.is_file():
            if not download:
                raise FileNotFoundError(f"Missing {archive}; supply the archive or pass download=True")
            download_source(source, archive.parent)
        destination = root / "sources" / kind
        inventory, hashes = _extract_source(archive, source, kind, destination)
        records = [{**{key: value for key, value in item.items() if key != "mtime_ns"},
                    "kind": kind, "license": source.license, "source": source.source,
                    "path": os.path.relpath(destination / item["member"], manifest_dir)} for item in inventory["audio"]]
        for split, subset in _group_split(records, validation_fraction, seed).items():
            key = f"{kind}_{split}"
            path = manifest_dir / f"{key}.jsonl"
            temporary = path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in subset))
            temporary.replace(path)
            outputs[key] = path
            provenance["splits"][key] = {"records": len(subset), "groups": sorted({r["group"] for r in subset}),
                                          "hours": sum(r["samples"] for r in subset) / SAMPLE_RATE / 3600,
                                          "manifest_sha256": _hash_file(path)["sha256"]}
        provenance["sources"][kind] = {**asdict(source), **hashes, "archive_bytes": archive.stat().st_size,
                                       "publisher_checksum_verified": source.expected_md5 is not None}
    outputs["provenance"] = manifest_dir / "provenance.json"
    _atomic_json(outputs["provenance"], provenance)
    return outputs


def read_extra_manifest(path, kind):
    path = Path(path).resolve()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Empty or duplicate extra-data manifest")
    expected_source = SOURCES[kind].source
    for record in records:
        identity = _asset_identity(kind, record.get("member", ""))
        if (record.get("kind") != kind or record.get("source") != expected_source or
                identity != (record.get("id"), record.get("group"), record.get("speaker")) or
                record.get("license") != SOURCES[kind].license or
                record.get("sample_rate") != SAMPLE_RATE or record.get("samples", 0) < 1 or
                record.get("split") not in {"train", "val"}):
            raise ValueError(f"Invalid or unapproved {kind} record: {record.get('id')}")
        record["path"] = str((path.parent / record["path"]).resolve())
    if len({r["split"] for r in records}) != 1:
        raise ValueError("A manifest must contain exactly one train/validation partition")
    return records


def _read_crop(record, size, *, tile=False):
    total = record["samples"]
    offset = int(torch.randint(max(0, total - size) + 1, ()).item())
    length = min(size, total)
    audio, rate = sf.read(record["path"], start=offset, frames=length, dtype="float32")
    if rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) != length or not np.isfinite(audio).all():
        raise ValueError(f"Invalid source crop: {record['id']}")
    if tile and length < size:
        audio = np.tile(audio, math.ceil(size / length))[:size]
    return audio, offset


def _active_mask(speech):
    # Zero-pad only for framing, correcting the final frame's denominator.
    frame_count = math.ceil(len(speech) / 320)
    framed = np.pad(speech.astype(np.float64), (0, frame_count * 320 - len(speech))).reshape(-1, 320)
    powers = np.sum(framed**2, axis=1) / 320
    powers[-1] *= 320 / (len(speech) - 320 * (frame_count - 1))
    threshold = max(1e-10, float(powers.max()) * 0.01)
    return np.repeat(powers > threshold, 320)[:len(speech)]


class DynamicMixtureDataset(Dataset):
    """Generate aligned 16 kHz mixtures using PyTorch's worker-local RNG.

Speech is cropped by utterance; short noise is repeated, and short speech is
zero padded after mixing. SNR uses nonoverlapping 20 ms reference frames within
20 dB of maximum frame power, excluding near-silence. One shared gain and any
peak attenuation apply to the entire pair. ``length`` excludes speech padding.
Validation assets remain reserved; synthetic validation should be rendered with
a fixed seed once rather than regenerated while selecting checkpoints.
"""
    def __init__(self, speech_manifest, noise_manifest, crop_seconds=3.0, snr_db=(-5, 20),
                 gain_db=(-6, 6), clean_identity_prob=0.03, samples_per_epoch=None,
                 peak_limit=0.99):
        self.speech_records = read_extra_manifest(speech_manifest, "speech")
        self.noise_records = read_extra_manifest(noise_manifest, "noise")
        self.split = self.speech_records[0]["split"]
        if self.noise_records[0]["split"] != self.split:
            raise ValueError("Speech and noise must use the same train/validation partition")
        for name, values in (("snr_db", snr_db), ("gain_db", gain_db)):
            if len(values) != 2 or not all(math.isfinite(x) for x in values) or values[0] > values[1]:
                raise ValueError(f"{name} must be a finite ordered pair")
        if not math.isfinite(crop_seconds) or crop_seconds <= 0:
            raise ValueError("crop_seconds must be positive")
        if not math.isfinite(clean_identity_prob) or not 0 <= clean_identity_prob <= 1:
            raise ValueError("clean_identity_prob must be in [0,1]")
        if not math.isfinite(peak_limit) or not 0 < peak_limit <= 1:
            raise ValueError("peak_limit must be in (0,1]")
        if samples_per_epoch is not None and (not isinstance(samples_per_epoch, int) or samples_per_epoch < 1):
            raise ValueError("samples_per_epoch must be a positive integer")
        self.size = max(1, round(crop_seconds * SAMPLE_RATE))
        self.snr_db, self.gain_db = tuple(snr_db), tuple(gain_db)
        self.clean_identity_prob, self.peak_limit = clean_identity_prob, peak_limit
        self.sample_rate = SAMPLE_RATE
        self.samples_per_epoch = samples_per_epoch or len(self.speech_records)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        for attempt in range(8):
            speech_index = index % len(self.speech_records) if attempt == 0 else int(torch.randint(len(self.speech_records), ()).item())
            speech_record = self.speech_records[speech_index]
            clean, speech_offset = _read_crop(speech_record, self.size)
            active = _active_mask(clean)
            if active.any():
                break
        else:
            raise ValueError("Unable to sample active speech after eight attempts")
        identity = torch.rand(()).item() < self.clean_identity_prob
        length = len(clean)
        snr = self.snr_db[0] + torch.rand(()).item() * (self.snr_db[1] - self.snr_db[0])
        noise_record, noise_offset, noise_scale = None, None, 0.0
        if identity:
            noisy = clean.copy()
        else:
            for _ in range(8):
                noise_record = self.noise_records[int(torch.randint(len(self.noise_records), ()).item())]
                noise, noise_offset = _read_crop(noise_record, length, tile=True)
                noise_rms = math.sqrt(float(np.mean(noise[active].astype(np.float64)**2)))
                if noise_rms >= 1e-8:
                    break
            else:
                raise ValueError("Unable to sample active noise after eight attempts")
            speech_rms = math.sqrt(float(np.mean(clean[active].astype(np.float64)**2)))
            noise_scale = speech_rms / noise_rms * 10**(-snr / 20)
            noisy = clean + noise * noise_scale
        gain = 10**((self.gain_db[0] + torch.rand(()).item() * (self.gain_db[1] - self.gain_db[0])) / 20)
        gain = min(gain, self.peak_limit / max(float(np.abs(clean).max()), float(np.abs(noisy).max()), 1e-8))
        clean = np.pad(clean * gain, (0, self.size - length)).astype(np.float32)
        noisy = np.pad(noisy * gain, (0, self.size - length)).astype(np.float32)
        return {"id": f"synthetic_{self.split}_{index:09d}", "speaker": speech_record["speaker"],
                "clean": torch.from_numpy(clean), "noisy": torch.from_numpy(noisy), "length": length,
                "mixture": {"speech_id": speech_record["id"], "speech_offset": speech_offset,
                            "noise_id": noise_record["id"] if noise_record else None, "noise_offset": noise_offset,
                            "snr_db": None if identity else snr, "noise_scale": noise_scale,
                            "common_gain": gain, "clean_identity": identity, "split": self.split}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--include-rir", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = prepare_extra_data(args.root, download=args.download, include_rir=args.include_rir,
                                validation_fraction=args.validation_fraction, seed=args.seed)
    print(json.dumps({name: str(path) for name, path in result.items()}, indent=2))


if __name__ == "__main__":
    main()
