"""Keep secondary selection auditable and separate from final-test data."""
from dataclasses import asdict
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from esp32_denoiser.development_checkpoints import score_checkpoint
from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig


def fixture_files(root, source_split="development"):
    model = SpectralTCN(SpectralTCNConfig(width=4, dilations=(1,)))
    checkpoint = root / "last.pt"
    torch.save(dict(model=model.state_dict(), model_config=asdict(model.config),
                    model_kind="spectral_tcn", phase="float", epoch=5), checkpoint)
    wave = np.arange(1024) / 16000
    clean = (0.15 * np.sin(2 * np.pi * 400 * wave)).astype(np.float32)
    noisy = clean + (0.03 * np.cos(2 * np.pi * 1100 * wave)).astype(np.float32)
    record = dict(id="development_0", speaker="heldout", samples=1024,
                  sample_rate=16000, source_split=source_split)
    for role, audio in (("clean", clean), ("noisy", noisy)):
        path = root / (role + ".wav")
        sf.write(path, audio, 16000, subtype="FLOAT")
        record[role] = str(path)
    manifest = root / "development.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    return checkpoint, manifest


def test_secondary_selection_keeps_exact_immutable_weights_and_primary_source(tmp_path):
    torch.set_num_threads(2)
    source, manifest = fixture_files(tmp_path)
    source_bytes = source.read_bytes()
    output = tmp_path / "secondary"
    result = score_checkpoint(source, manifest, output)
    best = json.loads((output / "best.json").read_text())
    assert result["validation"]["total_utterances"] == 1
    assert best["checkpoint_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert best["model"]["checkpoint_epoch"] == 5
    assert len(list(output.glob("selected_*.pt"))) == 1
    assert next(output.glob("selected_*.pt")).read_bytes() == source_bytes
    score_checkpoint(source, manifest, output)
    assert len(list(output.glob("selected_*.pt"))) == 1
    assert len((output / "history.jsonl").read_text().splitlines()) == 2
    assert source.read_bytes() == source_bytes
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest changed"):
        score_checkpoint(source, manifest, output)


def test_final_test_cannot_be_used_for_secondary_selection(tmp_path):
    source, manifest = fixture_files(tmp_path, source_split="test")
    output = tmp_path / "secondary"
    with pytest.raises(ValueError, match="development-only"):
        score_checkpoint(source, manifest, output)
    assert not output.exists()


def test_duplicate_process_cannot_race_selection_and_exit_releases_lock(tmp_path):
    source, manifest = fixture_files(tmp_path)
    output = tmp_path / "secondary"
    code = """from pathlib import Path
import sys, time
from esp32_denoiser.development_checkpoints import _selection_lock
with _selection_lock(Path(sys.argv[1])):
    print('locked', flush=True)
    time.sleep(30)
"""
    process = subprocess.Popen([sys.executable, "-c", code, str(output)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="already writing"):
            score_checkpoint(source, manifest, output)
        assert not (output / "evaluating.pt").exists()
        assert not (output / "best.json").exists()
    finally:
        process.terminate()
        process.communicate(timeout=10)
    # Stale PID text is harmless: ownership is the OS lock, not file existence.
    score_checkpoint(source, manifest, output)
    assert (output / "best.json").exists()


def test_existing_selection_cannot_mix_architectures(tmp_path):
    source, manifest = fixture_files(tmp_path)
    output = tmp_path / "secondary"
    score_checkpoint(source, manifest, output)
    best_before = (output / "best.json").read_bytes()
    history_before = (output / "history.jsonl").read_bytes()
    model = SpectralTCN(SpectralTCNConfig(width=5, dilations=(1,)))
    torch.save(dict(model=model.state_dict(), model_config=asdict(model.config),
                    model_kind="spectral_tcn", phase="float", epoch=6), source)
    with pytest.raises(ValueError, match="architecture changed"):
        score_checkpoint(source, manifest, output)
    assert (output / "best.json").read_bytes() == best_before
    assert (output / "history.jsonl").read_bytes() == history_before
