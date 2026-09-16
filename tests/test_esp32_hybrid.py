"""Hybrid sampling policy, resumed RNGs, and training-source boundaries."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from esp32_denoiser.extra_data import SOURCES, _asset_identity
from esp32_denoiser.mixtures import HybridTrainingDataset
from esp32_denoiser.train import TrainConfig, train


class MarkedDataset(Dataset):
    sample_rate = 16000

    def __init__(self, count, offset):
        self.count, self.offset = count, offset

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return torch.tensor([self.offset + index, torch.rand(()).item()])


def python_rows(items):
    # Exercise worker RNGs without requiring shared-memory tensor allocation,
    # which is unavailable in the local macOS test sandbox.
    return [item.tolist() for item in items]


def test_paired_sampling_is_not_tied_to_epoch_length():
    hybrid = HybridTrainingDataset(MarkedDataset(10, 0), MarkedDataset(10, 100),
                                   synthetic_probability=.5, epoch_samples=1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(37)
        observed = [int(hybrid[0][0]) for _ in range(1000)]
    assert set(value for value in observed if value < 100) == set(range(10))
    assert .43 < np.mean(np.array(observed) >= 100) < .57


@pytest.mark.parametrize("workers", [0, 2])
def test_epoch_seeds_reproduce_hybrid_worker_sampling(workers):
    dataset = HybridTrainingDataset(MarkedDataset(7, 0), MarkedDataset(11, 100),
                                    synthetic_probability=.5, epoch_samples=32)

    def epoch(seed):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=generator,
                                num_workers=workers, persistent_workers=False, collate_fn=python_rows)
            return torch.tensor([row for batch in loader for row in batch])

    expected = epoch(2028)
    epoch(2027)  # Intervening work must not alter a reconstructed epoch.
    assert torch.equal(expected, epoch(2028))
    assert not torch.equal(expected, epoch(2029))


@pytest.fixture
def hybrid_config(tmp_path):
    paths = {}
    for split, speaker in (("train", "p225"), ("val", "p226")):
        clean = (.15 * np.sin(np.arange(1024) * .17)).astype(np.float32)
        row = {"id": speaker + "_001", "speaker": speaker, "source_split": "train",
               "sample_rate": 16000, "samples": len(clean)}
        for role, audio in (("clean", clean), ("noisy", clean + .03 * np.cos(np.arange(1024) * .49))):
            path = tmp_path / f"{speaker}_{role}.wav"
            sf.write(path, audio, 16000, subtype="FLOAT")
            row[role] = str(path)
        path = tmp_path / f"paired_{split}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        paths[split] = str(path)
    for kind in ("speech", "noise"):
        rows = []
        for index in (1, 2):
            member = (f"LibriSpeech/train-clean-100/{index}/10/{index}-10-0001.flac" if kind == "speech"
                      else f"musan/noise/free-sound/noise-{index}.wav")
            path = tmp_path / member
            path.parent.mkdir(parents=True, exist_ok=True)
            samples = .1 * np.sin(np.arange(1600) * (.17 + index * .017))
            sf.write(path, samples, 16000, subtype="PCM_16")
            identifier, group, speaker = _asset_identity(kind, member)
            rows.append({"id": identifier, "group": group, "speaker": speaker, "member": member,
                         "path": str(path), "kind": kind, "split": "train", "source": SOURCES[kind].source,
                         "license": SOURCES[kind].license, "samples": len(samples), "sample_rate": 16000,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        path = tmp_path / f"{kind}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths[kind] = str(path)
    return TrainConfig(paths["train"], paths["val"], str(tmp_path / "run"),
                       synthetic_speech_manifest=paths["speech"], synthetic_noise_manifest=paths["noise"],
                       synthetic_probability=.5, epoch_samples=6, epochs=1, batch_size=2,
                       eval_batch_size=1, crop_seconds=.048, max_steps_per_epoch=2,
                       width=4, dilations=(1, 2), workers=0, amp=False, max_hours=.01)


def test_hybrid_training_resume_and_source_provenance(hybrid_config):
    config = hybrid_config
    full_dir = Path(config.output_dir).parent / "full"
    train(replace(config, epochs=2, output_dir=str(full_dir)))
    train(config)
    train(replace(config, epochs=2, resume=str(Path(config.output_dir) / "last.pt")))
    expected = torch.load(full_dir / "last.pt", weights_only=False)
    resumed = torch.load(Path(config.output_dir) / "last.pt", weights_only=False)
    for name, value in expected["model"].items():
        tolerance = {"rtol": 1e-6, "atol": 1e-7} if value.is_cuda else {"rtol": 0, "atol": 0}
        torch.testing.assert_close(resumed["model"][name], value, **tolerance)
    added = resumed["provenance"]["added_training_sources"]
    assert added["samples_per_epoch"] == 6 and added["synthetic_probability"] == .5
    assert added["speech_recordings"] == added["noise_recordings"] == 2
    assert added["speech_manifest_sha256"] == hashlib.sha256(Path(config.synthetic_speech_manifest).read_bytes()).hexdigest()
    history = [json.loads(line) for line in (Path(config.output_dir) / "history.jsonl").read_text().splitlines()]
    assert all(np.isfinite(row["loss"]) for row in history)


def test_hybrid_rejects_heldout_sources_and_validation_audio_alias(hybrid_config):
    config = hybrid_config
    originals = {}
    for path_string in (config.synthetic_speech_manifest, config.synthetic_noise_manifest):
        path = Path(path_string)
        originals[path] = path.read_text()
        rows = [json.loads(line) for line in originals[path].splitlines()]
        path.write_text("".join(json.dumps({**row, "split": "val"}) + "\n" for row in rows))
    with pytest.raises(ValueError, match="training partition"):
        train(config)
    for path, content in originals.items():
        path.write_text(content)
    path = Path(config.synthetic_speech_manifest)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["path"] = json.loads(Path(config.val_manifest).read_text())["clean"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="validation paths"):
        train(config)
