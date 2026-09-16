"""Score periodic training checkpoints on a separate development-only corpus.

This finite training companion never changes training or the primary validation
winner. It preserves the checkpoint with highest secondary development SI-SDRi
so a generalization gain is not lost when the primary score decreases slightly.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader

from .data import PairedAudioDataset, pad_collate
from .evaluate import load_checkpoint
from .train import validate


def _validated_dataset(manifest):
    data = PairedAudioDataset(manifest, crop_seconds=None, random_crop=False)
    if any(row.get("source_split") != "development" for row in data.records):
        raise ValueError("Secondary selection requires explicitly development-only source records")
    return data


@contextmanager
def _selection_lock(output):
    """Keep the lock inode in place; OS process exit releases the writer lock."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "selector.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"A development selector is already writing {output}") from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid()}) + "\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def score_checkpoint(source: Path, manifest: Path, output: Path, *, device="cpu") -> dict:
    """Score one immutable snapshot, failing if another selector owns output."""
    data = _validated_dataset(manifest)
    with _selection_lock(output):
        return _score_checkpoint(source, manifest, output, data, device=device)


def _score_checkpoint(source, manifest, output, data, *, device):
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    best_path = output / "best.json"
    previous = json.loads(best_path.read_text()) if best_path.exists() else None
    if previous and previous["manifest_sha256"] != manifest_hash:
        raise ValueError("Development manifest changed during checkpoint selection")
    # Training replaces checkpoints atomically. Open/copy one complete inode
    # before loading so the selected parameters cannot change during scoring.
    snapshot = output / "evaluating.pt"
    shutil.copyfile(source, snapshot)
    checkpoint_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    model, metadata = load_checkpoint(snapshot, device)
    try:
        if previous:
            keys = ("model_kind", "model_config")
            previous_architecture = {key: previous["model"].get(key) for key in keys}
            current_architecture = {key: metadata.get(key) for key in keys}
            if json.dumps(previous_architecture, sort_keys=True) != json.dumps(current_architecture, sort_keys=True):
                raise ValueError("Model architecture changed during checkpoint selection; use a separate output directory")
        loader = DataLoader(data, batch_size=8, shuffle=False, collate_fn=pad_collate, num_workers=0)
        metrics = validate(model, loader, torch.device(device))
    finally:
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if metrics["total_utterances"] != len(data) or metrics["invalid_utterances"]:
        raise ValueError("Secondary selection requires a complete finite cohort")
    result = {"selection_split": "external development; not final test",
              "manifest": str(manifest.resolve()), "manifest_sha256": manifest_hash,
              "source_checkpoint": str(source.resolve()), "checkpoint_sha256": checkpoint_hash,
              "model": metadata, "validation": metrics}
    history = output / "history.jsonl"
    with history.open("a") as handle:
        handle.write(json.dumps({key: value for key, value in result.items() if key != "validation"}
                               | {"si_sdri": metrics["si_sdri"]}) + "\n")
    if previous is None or metrics["si_sdri"] > previous["validation"]["si_sdri"]:
        # Keep improved checkpoints immutable; the pointer identifies exact
        # weights even if a backup happens while the next update is written.
        selected = output / f"selected_{checkpoint_hash}.pt"
        snapshot.replace(selected)
        result["selected_checkpoint"] = str(selected.resolve())
        temporary = output / "best.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(best_path)
    else:
        snapshot.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--every-epochs", type=int, default=5)
    parser.add_argument("--max-hours", type=float, default=7)
    args = parser.parse_args()
    if args.every_epochs < 1 or not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error("Epoch interval and time limit must be positive and finite")
    # Validate before creating output; final-test input never starts a selector.
    _validated_dataset(args.manifest)
    output = args.run / "external_development"
    with _selection_lock(output):
        _watch(args, output)


def _watch(args, output):
    torch.set_num_threads(2)
    end = time.monotonic() + 3600 * args.max_hours
    observed_epoch = -1
    while time.monotonic() < end:
        history = args.run / "history.jsonl"
        complete = (args.run / "summary.json").exists()
        text = history.read_text() if history.exists() else ""
        # Ignore a writer's unfinished final line; completed lines are durable
        # epoch reports, while parameters are copied from one atomic checkpoint.
        lines = text.splitlines() if text.endswith("\n") else text.splitlines()[:-1]
        records = [json.loads(line) for line in lines]
        epoch = records[-1]["epoch"] if records else 0
        checkpoint = args.run / "last.pt"
        if checkpoint.exists() and (observed_epoch < 0 or epoch >= observed_epoch + args.every_epochs
                                    or complete and epoch != observed_epoch):
            result = _score_checkpoint(checkpoint, args.manifest, output,
                                       _validated_dataset(args.manifest), device=args.device)
            observed_epoch = result["model"]["checkpoint_epoch"]
            print(json.dumps({"event": "secondary_development", "run": str(args.run),
                              "epoch": observed_epoch, "si_sdri": result["validation"]["si_sdri"]}), flush=True)
        if complete:
            return
        time.sleep(15)
    raise TimeoutError("Development companion reached its finite monitoring limit")


if __name__ == "__main__":
    main()
