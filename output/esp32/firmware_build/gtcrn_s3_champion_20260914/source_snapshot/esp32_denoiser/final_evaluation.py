"""Prepare the sealed external corpus only after verifying a model freeze.

No download, model loading, training, inference or metric computation occurs.
The public preparer locks the 200x5 +100 recipe from the sealed source plan.
It publishes a new output directory only after source/model re-verification.
CLI defaults to a read-only input audit; --prepare explicitly renders audio.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

import numpy as np
import soundfile as sf
import torch

from .data import read_manifest
from . import development, extra_data
from .development import _write_audio
from .extra_data import SAMPLE_RATE, _active_mask, _asset_identity, _hash_file, _read_crop, read_extra_manifest


TEST_CLEAN_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
SOURCE_ROLES = frozenset({"paired_train", "paired_development", "speech_train", "speech_val",
                          "noise_train", "noise_val", "development_mixtures", "development_clean"})
PLAN_BINDINGS = {"manifests/noise_val.jsonl": "noise_val", "development/mixtures.jsonl": "development_mixtures",
                 "development/clean.jsonl": "development_clean"}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _json_snapshot(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, hashlib.sha256(data).hexdigest(), data


def _verified_file(entry, parent):
    if not isinstance(entry, dict) or not _sha(entry.get("sha256")) or not isinstance(entry.get("path"), str):
        raise ValueError("Every frozen file requires a path and valid SHA256")
    path = (parent / entry["path"]).resolve()
    if not path.is_file() or path.stat().st_size < 1:
        raise ValueError(f"Missing or empty frozen file: {path}")
    if _hash_file(path)["sha256"] != entry["sha256"]:
        raise ValueError(f"Frozen file hash mismatch: {path}")
    return path


def verify_model_freeze(path):
    """Verify all models/baselines before touching any final audio archive.

    The freeze is an explicit operator declaration, not proof of provenance
    against a malicious author. Artifact roles must include model and baseline;
    architecture/grids/DSP/gain/implementation are bound through its nonempty
    inference_specification and any referenced, hashed artifact files.
    """
    path = Path(path).resolve()
    value, digest, snapshot = _json_snapshot(path)
    if (value.get("version") != 1 or value.get("selection_frozen") is not True or
            value.get("test_used_for_selection") is not False or not _sha(value.get("plan_sha256"))):
        raise ValueError("A complete explicit pre-test model freeze is required")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("The model/baseline artifact list must be nonempty")
    if any(not isinstance(row, dict) or not isinstance(row.get("role"), str) for row in artifacts):
        raise ValueError("Invalid frozen artifact role")
    if not {"model", "baseline"} <= {row["role"] for row in artifacts}:
        raise ValueError("Freeze at least one model and one comparison baseline")
    if not isinstance(value.get("inference_specification"), dict) or not value["inference_specification"]:
        raise ValueError("Record frozen architecture, grids, DSP, gain and inference specification")
    paths = [_verified_file(row, path.parent) for row in artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError("Frozen artifacts must identify distinct files")
    sources = value.get("source_manifests")
    if not isinstance(sources, dict) or set(sources) != SOURCE_ROLES:
        raise ValueError("Freeze every required training/development source manifest")
    source_paths = {role: _verified_file(row, path.parent) for role, row in sources.items()}
    return {"path": path, "sha256": digest, "snapshot": snapshot, "declaration": value,
            "artifact_paths": paths, "source_paths": source_paths}


@dataclass(frozen=True)
class FinalRecipe:
    base_crops: int = 200
    snr_db: tuple = (-5, 0, 5, 10, 20)
    crop_seconds: float = 3.0
    clean_examples: int = 100
    seed: int = 20260913


def _plan(path, expected_hash):
    plan, digest, snapshot = _json_snapshot(path)
    if digest != expected_hash:
        raise ValueError("Sealed plan hash differs from the model freeze")
    speech = plan.get("speech_source", {})
    if speech.get("dataset") != "LibriSpeech test-clean" or speech.get("official_md5") != TEST_CLEAN_MD5:
        raise ValueError("Only official checksum-verified LibriSpeech test-clean is admitted")
    if plan.get("mixture_plan") != {"base_crops": 200, "snr_db": [-5, 0, 5, 10, 20],
                                    "crop_seconds": 3, "clean_examples": 100, "seed": 20260913}:
        raise ValueError("The sealed final recipe must remain 200x5 plus 100, three seconds, seed 20260913")
    rows = plan.get("reserved_noise_records")
    if plan.get("reserved_noise_count") != 29 or not isinstance(rows, list) or len(rows) != 29:
        raise ValueError("The sealed plan must reserve exactly 29 MUSAN recordings")
    for row in rows:
        if not isinstance(row, dict) or not _sha(row.get("sha256")):
            raise ValueError("Invalid reserved noise content hash")
        identity = _asset_identity("noise", row.get("member", ""))
        if (identity is None or identity[:2] != (row.get("id"), row.get("group")) or
                row.get("sample_rate") != SAMPLE_RATE or not isinstance(row.get("samples"), int) or row["samples"] < 1 or
                row.get("source") != "https://www.openslr.org/17/" or row.get("license") != "CC-BY-4.0"):
            raise ValueError("Invalid reserved MUSAN source identity")
    if any(len({row[key] for row in rows}) != len(rows) for key in ("id", "group", "member", "sha256")):
        raise ValueError("Reserved noise contains duplicate identities/content")
    return plan, digest, snapshot


def _empty_exclusions():
    return {key: set() for key in ("id", "group", "speaker", "sha256", "path")}


def _add_source(excluded, row):
    for key in excluded:
        if row.get(key) is not None:
            excluded[key].add(row[key])


def _disjoint(row, excluded, label):
    for key, values in excluded.items():
        if row.get(key) is not None and row[key] in values:
            raise ValueError(f"Final {label} overlaps training/development {key}: {row[key]}")


def _source_audit(freeze, plan):
    paths = freeze["source_paths"]
    bindings = plan.get("manifest_sha256")
    if not isinstance(bindings, dict) or set(bindings) != set(PLAN_BINDINGS):
        raise ValueError("Incomplete sealed reservation manifest hashes")
    for plan_name, role in PLAN_BINDINGS.items():
        if bindings[plan_name] != freeze["declaration"]["source_manifests"][role]["sha256"]:
            raise ValueError(f"Reservation manifest changed: {plan_name}")
    excluded, sources = _empty_exclusions(), {}
    for kind in ("speech", "noise"):
        for split in ("train", "val"):
            role = f"{kind}_{split}"
            rows = read_extra_manifest(paths[role], kind)
            if {row["split"] for row in rows} != {split} or any(not _sha(row.get("sha256")) for row in rows):
                raise ValueError(f"Invalid source partition/content hashes: {role}")
            sources[role] = rows
            if role != "noise_val":
                for row in rows:
                    _add_source(excluded, row)
        for key in ("id", "group", "sha256", "path"):
            if {r[key] for r in sources[f"{kind}_train"]} & {r[key] for r in sources[f"{kind}_val"]}:
                raise ValueError(f"Existing extra source train/val {key} overlap")
    for role in ("paired_train", "paired_development"):
        for row in read_manifest(paths[role]):
            if row.get("source_split") != "train":
                raise ValueError("Paired exclusion manifests must be training-origin")
            _add_source(excluded, row)
            for audio_role in ("clean", "noisy"):
                path = Path(row[audio_role])
                digest = _hash_file(path)["sha256"]
                recorded = row.get(audio_role + "_sha256")
                if recorded is not None and recorded != digest:
                    raise ValueError("Paired exclusion waveform differs from its recorded hash")
                excluded["sha256"].add(digest)
                excluded["path"].add(str(path.resolve()))
    inventory = {row["id"]: row for kind in sources.values() for row in kind}
    for role in ("development_mixtures", "development_clean"):
        for row in read_manifest(paths[role]):
            if row.get("source_split") != "development":
                raise ValueError("Existing development manifest has an incorrect source split")
            _add_source(excluded, row)
            for kind in ("speech", "noise"):
                identifier = row.get(f"{kind}_id")
                if identifier is None and role == "development_clean" and kind == "noise":
                    continue
                if identifier not in inventory:
                    raise ValueError("Development references an unknown original source")
                original = inventory[identifier]
                if row.get(f"{kind}_source_sha256") != original["sha256"]:
                    raise ValueError("Development original-source content hash mismatch")
                if original["split"] != "val":
                    raise ValueError("Development contains a training source")
                _add_source(excluded, original)
    heldout = {row["id"]: row for row in sources["noise_val"]}
    reserved = []
    for planned in plan["reserved_noise_records"]:
        row = heldout.get(planned["id"])
        if row is None or any(row.get(key) != value for key, value in planned.items()):
            raise ValueError("Reserved noise differs from its exact held-out source record")
        _disjoint(row, excluded, "noise")
        if _hash_file(row["path"])["sha256"] != row["sha256"]:
            raise ValueError("Reserved noise waveform hash changed")
        info = sf.info(row["path"])
        if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames != row["samples"]:
            raise ValueError("Reserved noise is not the declared 16k mono recording")
        reserved.append(row)
    return excluded, sorted(reserved, key=lambda row: row["id"])


def audit_final_inputs(*, plan_path, freeze_path, speech_archive):
    """Read-only verification; does not extract, render or inspect model outputs."""
    freeze = verify_model_freeze(freeze_path)  # Must precede all final audio work.
    plan, plan_hash, snapshot = _plan(plan_path, freeze["declaration"]["plan_sha256"])
    excluded, reserved = _source_audit(freeze, plan)
    archive = Path(speech_archive).resolve()
    archive_hashes = _hash_file(archive)
    if archive_hashes["md5"] != TEST_CLEAN_MD5:
        raise ValueError("Official LibriSpeech test-clean archive MD5 mismatch")
    return {"freeze": freeze, "plan": plan, "plan_sha256": plan_hash, "plan_snapshot": snapshot,
            "archive": archive, "archive_hashes": archive_hashes, "excluded": excluded,
            "noise": reserved, "recipe": FinalRecipe()}


def _safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError("Unsafe final speech archive member")
    return path.parts


def _extract_speech(archive, destination, excluded):
    rows, seen = [], set()
    with tarfile.open(archive, "r|gz") as bundle:
        for item in bundle:
            parts = _safe_member(item.name)
            normalized = "/".join(parts)
            if normalized in seen:
                raise ValueError("Duplicate final speech archive member")
            seen.add(normalized)
            if not item.isdir() and not item.isfile():
                raise ValueError("Links and special files are forbidden in the final speech archive")
            if item.isdir() or not item.name.endswith(".flac"):
                continue
            stem = PurePosixPath(item.name).stem
            if (len(parts) != 5 or parts[:2] != ("LibriSpeech", "test-clean") or
                    not parts[2].isdigit() or not parts[3].isdigit() or
                    len(stem.split("-")) != 3 or not stem.startswith(f"{parts[2]}-{parts[3]}-") or
                    not stem.split("-")[-1].isdigit() or
                    item.size < 1 or item.size > 512 * 1024 * 1024):
                raise ValueError("Unexpected final speech source member")
            row = {"id": f"librispeech:{stem}", "speaker": parts[2], "group": f"librispeech:speaker:{parts[2]}",
                   "member": normalized, "kind": "speech", "split": "test", "source_split": "test",
                   "source": "https://www.openslr.org/12/", "license": "CC-BY-4.0", "sample_rate": SAMPLE_RATE}
            _disjoint(row, excluded, "speech")
            path = destination.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            digest, count = hashlib.sha256(), 0
            with bundle.extractfile(item) as incoming, path.open("xb") as output:
                for block in iter(lambda: incoming.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    count += len(block)
            if count != item.size:
                raise ValueError("Truncated final speech archive member")
            info = sf.info(path)
            if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.frames < 1:
                raise ValueError("Final speech must be nonempty 16k mono audio")
            row.update(path=str(path), sha256=digest.hexdigest(), samples=info.frames)
            _disjoint(row, excluded, "speech")
            rows.append(row)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate final speech inventory")
    return sorted(rows, key=lambda row: row["id"])


def _item_seed(seed, suite, index):
    digest = hashlib.sha256(f"esp32-final-v1:{seed}:{suite}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _render_suites(speech, noise, output, recipe):
    size = round(recipe.crop_seconds * SAMPLE_RATE)
    speech = sorted((row for row in speech if row["samples"] >= size), key=lambda row: row["id"])
    noise = sorted(noise, key=lambda row: row["id"])
    if not speech or not noise:
        raise ValueError("Need full-length final speech crops and reserved noise")
    generator = torch.Generator().manual_seed(_item_seed(recipe.seed, "noise_schedule", 0))
    noise_order = torch.randperm(len(noise), generator=generator).tolist()
    used, rows = set(), {"mixtures": [], "clean": []}
    for suite, count in (("mixtures", recipe.base_crops), ("clean", recipe.clean_examples)):
        for base in range(count):
            seed = _item_seed(recipe.seed, suite, base)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                for _ in range(32):
                    speech_row = speech[int(torch.randint(len(speech), ()).item())]
                    clean, offset = _read_crop(speech_row, size)
                    identity = (speech_row["id"], offset, len(clean))
                    active = _active_mask(clean)
                    if identity not in used and active.any():
                        used.add(identity)
                        break
                else:
                    raise ValueError("Could not choose a distinct active final speech crop")
                noise_row = noise[noise_order[base % len(noise)]] if suite == "mixtures" else None
                noise_crop, noise_offset, noise_rms = None, None, None
                if noise_row is not None:
                    for _ in range(8):
                        noise_crop, noise_offset = _read_crop(noise_row, size, tile=True)
                        noise_rms = math.sqrt(float(np.mean(noise_crop[active].astype(np.float64) ** 2)))
                        if noise_rms >= 1e-8:
                            break
                    else:
                        raise ValueError("Reserved final noise has no usable active energy")
            levels = recipe.snr_db if suite == "mixtures" else (None,)
            for condition, snr in enumerate(levels):
                scale = 0.0 if snr is None else math.sqrt(float(np.mean(clean[active].astype(np.float64) ** 2))) / noise_rms * 10 ** (-snr / 20)
                noisy = clean.copy() if snr is None else clean + noise_crop * scale
                gain = min(1.0, .99 / max(float(np.abs(clean).max()), float(np.abs(noisy).max()), 1e-8))
                identifier = f"extra_final_{suite}_{base:05d}_condition{condition:02d}"
                row = {"id": identifier, "speaker": speech_row["speaker"], "samples": size, "sample_rate": SAMPLE_RATE,
                       "source_split": "test", "split": "test", "suite": suite, "base_crop": base, "item_seed": seed,
                       "speech_id": speech_row["id"], "speech_group": speech_row["group"], "speech_offset": offset,
                       "speech_source_sha256": speech_row["sha256"], "noise_id": noise_row["id"] if noise_row else None,
                       "noise_group": noise_row["group"] if noise_row else None, "noise_offset": noise_offset,
                       "noise_source_sha256": noise_row["sha256"] if noise_row else None,
                       "snr_db": snr, "noise_scale": scale, "common_gain": gain, "clean_identity": snr is None}
                for role, audio in (("clean", clean * gain), ("noisy", noisy * gain)):
                    destination = output / "audio" / suite / role / f"{identifier}.wav"
                    row[f"{role}_sha256"] = _write_audio(destination, audio)
                    row[role] = os.path.relpath(destination, output)
                rows[suite].append(row)
    return rows


def prepare_final_evaluation(*, plan_path, freeze_path, speech_archive, output_dir):
    """Publish only after freeze, official archive and source-disjointness checks.

    Requires an already downloaded official archive. Existing output directories
    are never reused. Real final preparation should run only after final model
    selection; importing this module or running its tests does not access it.
    """
    audited = audit_final_inputs(plan_path=plan_path, freeze_path=freeze_path, speech_archive=speech_archive)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Final output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".final-evaluation-", dir=output.parent) as temporary:
        staging = Path(temporary) / "prepared"
        staging.mkdir()
        speech = _extract_speech(audited["archive"], staging / "sources/speech", audited["excluded"])
        rows = _render_suites(speech, audited["noise"], staging, audited["recipe"])
        summaries = {}
        for suite, records in rows.items():
            path = staging / f"{suite}.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
            summaries[suite] = {"records": len(records), "manifest_sha256": _hash_file(path)["sha256"],
                                "speakers": sorted({r["speaker"] for r in records}),
                                "noise_groups": sorted({r["noise_group"] for r in records if r["noise_group"]})}
        (staging / "model_freeze.json").write_bytes(audited["freeze"]["snapshot"])
        (staging / "sealed_plan.json").write_bytes(audited["plan_snapshot"])
        inventory = [{**row, "path": os.path.relpath(row["path"], staging)} for row in speech]
        (staging / "speech_sources.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in inventory))
        fresh = verify_model_freeze(freeze_path)
        if fresh["sha256"] != audited["freeze"]["sha256"] or _hash_file(speech_archive) != audited["archive_hashes"]:
            raise ValueError("Frozen artifacts or final archive changed during preparation")
        if _hash_file(plan_path)["sha256"] != audited["plan_sha256"]:
            raise ValueError("Sealed plan changed during preparation")
        for row in audited["noise"]:
            if _hash_file(row["path"])["sha256"] != row["sha256"]:
                raise ValueError("Reserved noise changed during final rendering")
        provenance = {"version": 1, "source_split": "test", "model_freeze_sha256": fresh["sha256"],
                      "plan_sha256": audited["plan_sha256"], "speech_archive": audited["archive_hashes"],
                      "speech_recordings": len(speech), "reserved_noise_recordings": len(audited["noise"]),
                      "suites": summaries, "sample_format": "deterministic IEEE FLOAT WAV, final I/O quantization deferred",
                      "mixer": {"snr": "20 ms reference frames within 20dB of maximum; same as extra_data",
                                "gain_db": [0, 0], "shared_peak_limit": .99, "short_speech": "excluded; all crops are3s",
                                "noise_sampling": "seeded permutation cycled across base crops; short recordings tiled",
                                "speech_sampling": "uniform eligible recordings with distinct active crop retries"},
                      "exclusion_counts": {key: len(value) for key, value in audited["excluded"].items()},
                      "content_hash_scope": "SHA256 of source file bytes; not cross-codec perceptual fingerprinting",
                      "preparation_source_sha256": {Path(path).name: _hash_file(path)["sha256"]
                                                    for path in (__file__, extra_data.__file__, development.__file__)},
                      "inference_performed": False, "test_used_for_selection": False}
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        if output.exists():
            raise FileExistsError("Final output appeared during preparation")
        staging.rename(output)
    return {key: output / name for key, name in (("mixtures", "mixtures.jsonl"), ("clean", "clean.jsonl"),
                                                ("provenance", "provenance.json"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--speech-archive", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    arguments = dict(plan_path=args.plan, freeze_path=args.freeze, speech_archive=args.speech_archive)
    if args.prepare:
        if args.output_dir is None:
            parser.error("--prepare requires a new --output-dir")
        result = prepare_final_evaluation(**arguments, output_dir=args.output_dir)
        print(json.dumps({key: str(path) for key, path in result.items()}, indent=2))
    else:
        audited = audit_final_inputs(**arguments)
        print(json.dumps({"status": "inputs audited; no final audio extracted or rendered",
                          "freeze_sha256": audited["freeze"]["sha256"], "plan_sha256": audited["plan_sha256"],
                          "archive": audited["archive_hashes"], "reserved_noise_count": len(audited["noise"])}, indent=2))


if __name__ == "__main__":
    main()
