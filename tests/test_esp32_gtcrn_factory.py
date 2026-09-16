"""Factory/optimizer/checkpoint integration for the float-only GTCRN reference."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest
import soundfile as sf
import torch

from esp32_denoiser.evaluate import load_checkpoint
from esp32_denoiser.export import export_model
from esp32_denoiser.gtcrn_model import GTCRNDenoiser
from esp32_denoiser.models import build_model, calibrate_model_hidden_exponent, configure_model_qat
from esp32_denoiser.train import TrainConfig, train


def test_gtcrn_factory_rejects_integer_paths(tmp_path):
    model = build_model("gtcrn", {})
    assert isinstance(model, GTCRNDenoiser)
    with pytest.raises(ValueError, match="float-only"):
        configure_model_qat(model, "gtcrn")
    with pytest.raises(ValueError, match="float-only"):
        calibrate_model_hidden_exponent(model, "gtcrn", [])
    with pytest.raises((ValueError, TypeError), match="SpectralTCN"):
        export_model(model, tmp_path / "unsupported.bin")
    assert not (tmp_path / "unsupported.bin").exists()


@pytest.mark.parametrize("normalize_input", [False, True])
def test_gtcrn_real_cpu_training_resume_and_loader(tmp_path, monkeypatch, normalize_input):
    # This integration check deliberately runs on CPU even in the CUDA suite.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)
    manifests = {}
    for split, speakers in (("train", ("p225", "p227")), ("val", ("p226", "p287"))):
        rows = []
        for index, speaker in enumerate(speakers):
            length = 768 + index * 256
            time = np.arange(length) / 16000
            clean = (.13 * np.sin(2 * np.pi * 430 * time)).astype(np.float32)
            noisy = (clean + .035 * np.cos(2 * np.pi * 1600 * time)).astype(np.float32)
            row = dict(id=f"{speaker}_001", speaker=speaker, samples=length,
                       sample_rate=16000, source_split="train")
            for role, audio in (("clean", clean), ("noisy", noisy)):
                path = tmp_path / f"{speaker}_{role}.wav"
                sf.write(path, audio, 16000, subtype="FLOAT")
                row[role] = str(path)
            rows.append(row)
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifests[split] = str(path)
    output = tmp_path / "float"
    config = TrainConfig(train_manifest=manifests["train"], val_manifest=manifests["val"],
                         output_dir=str(output), model_kind="gtcrn", model_options={"normalize_input": True} if normalize_input else {},
                         epochs=1, batch_size=2, crop_seconds=.048, workers=0,
                         eval_batch_size=2, max_steps_per_epoch=1, max_hours=.02, amp=False)
    first = train(config)
    assert first["epoch"] == 1 and math.isfinite(first["best_si_sdri"])
    checkpoint1 = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint1["model_kind"] == "gtcrn"
    assert checkpoint1["train_config"]["resume"] is None
    assert checkpoint1["model_config"]["normalize_input"] is normalize_input
    resumed = train(replace(config, epochs=2, resume=str(output / "last.pt")))
    assert resumed["epoch"] == 2 and math.isfinite(resumed["best_si_sdri"])
    checkpoint2 = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint2["optimizer"]["state"]
    assert all(bool(torch.isfinite(value).all()) for value in checkpoint2["model"].values())
    weight = "core.encoder.en_convs.0.conv.weight"
    assert not torch.equal(checkpoint1["model"][weight], checkpoint2["model"][weight])
    history = [json.loads(row) for row in (output / "history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in history] == [1, 2]
    assert all(math.isfinite(row["loss"]) for row in history)
    loaded, _ = load_checkpoint(output / "last.pt")
    assert isinstance(loaded, GTCRNDenoiser) and not loaded.training
    assert loaded.config.normalize_input is normalize_input
    assert loaded.model_stats()["learned_parameters"] == 23_669
    audio = torch.randn(1, 913) * .04
    with torch.no_grad():
        expected = loaded(audio)
    torch.testing.assert_close(loaded.make_streaming().denoise(audio), expected, rtol=3e-4, atol=2e-6)
    with pytest.raises(ValueError, match="float-only"):
        train(replace(config, phase="qat", output_dir=str(tmp_path / "qat"), resume=str(output / "best.pt")))
