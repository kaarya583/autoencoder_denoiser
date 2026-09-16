"""Float frequency ablations: factory, real training, and INT8 rejection."""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from esp32_denoiser.evaluate import load_checkpoint
from esp32_denoiser.frequency_export import export_frequency_model
from esp32_denoiser.frequency_deep_filter import FrequencyDeepFilter
from esp32_denoiser.frequency_gru import FrequencyGRU
from esp32_denoiser.models import build_model, calibrate_model_hidden_exponent, configure_model_qat
from esp32_denoiser.train import TrainConfig, train


CASES = (("frequency_gru", False, FrequencyGRU),
         ("frequency_gru", True, FrequencyGRU),
         ("frequency_deep_filter", True, FrequencyDeepFilter))


@pytest.mark.parametrize("kind,batch_norm,model_class", CASES)
def test_training_configs_match_frequency_controls_and_checkpoint_factory(kind, batch_norm, model_class, tmp_path):
    configs = Path(__file__).resolve().parents[1] / "configs"
    suffix = "_bn" if batch_norm else ""
    control = json.loads((configs / f"esp32_frequency{suffix}_float.json").read_text())
    candidate_suffix = suffix if kind == "frequency_gru" else ""
    candidate = json.loads((configs / f"esp32_{kind}{candidate_suffix}_float.json").read_text())
    assert candidate["model_kind"] == kind
    assert candidate["output_dir"].endswith(f"float_{kind}{candidate_suffix}")
    for values in (control, candidate):
        values.pop("model_kind")
        values.pop("output_dir")
    assert candidate == control
    model = build_model(kind, candidate["model_options"])
    assert isinstance(model, model_class)
    restored = build_model(kind, asdict(model.config), checkpoint=True)
    restored.load_state_dict(model.state_dict())
    assert restored.config.encoder_batch_norm == batch_norm
    for function, args in ((configure_model_qat, ()), (calibrate_model_hidden_exponent, ([],))):
        with pytest.raises(ValueError, match="float-only"):
            function(model, kind, *args)
    destination = tmp_path / "unsupported.bin"
    with pytest.raises(ValueError, match=model_class.__name__):
        export_frequency_model(model, destination)
    assert not destination.exists()


def _pilot_data(root):
    paths = {}
    for split, speakers in (("train", ("p225", "p227")), ("val", ("p226", "p287"))):
        records = []
        for index, speaker in enumerate(speakers):
            length = 768 + 256 * index
            timeline = np.arange(length) / 16000
            clean = (0.15 * np.sin(2 * np.pi * 420 * timeline)).astype(np.float32)
            noisy = clean + (0.06 * np.cos(2 * np.pi * 1200 * timeline)).astype(np.float32)
            record = dict(id=f"{speaker}_001", speaker=speaker, samples=length,
                          sample_rate=16000, source_split="train")
            for role, audio in (("clean", clean), ("noisy", noisy)):
                path = root / f"{speaker}_{role}.wav"
                sf.write(path, audio, 16000, subtype="FLOAT")
                record[role] = str(path)
            records.append(record)
        manifest = root / f"{split}.jsonl"
        manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
        paths[split] = str(manifest)
    return paths


@pytest.mark.parametrize("kind,batch_norm,model_class", CASES)
def test_cpu_training_pilot_checkpoint_loading_and_resume(tmp_path, monkeypatch, kind, batch_norm, model_class):
    # This tests the real training loop; the remote CUDA pilot is separate.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    paths = _pilot_data(tmp_path)
    options = dict(encoder_channels=[2, 3, 4], global_width=4, global_dilations=[1, 2],
                   encoder_batch_norm=batch_norm)
    if kind == "frequency_gru":
        options["recurrent_hidden"] = 2
        new_parameter = "local_gru.weight_ih_l0"
    else:
        options.update(local_dilations=[1, 2], filter_order=3, filter_bins=17)
        new_parameter = "filter_head.weight"
    config = TrainConfig(
        train_manifest=paths["train"], val_manifest=paths["val"], output_dir=str(tmp_path / "run"),
        model_kind=kind, model_options=options,
        epochs=1, batch_size=2, crop_seconds=0.048, workers=0, max_steps_per_epoch=1,
        eval_batch_size=2, amp=False, max_hours=0.01,
    )
    result = train(config)
    assert result["epoch"] == 1 and math.isfinite(result["best_si_sdri"])
    path = Path(config.output_dir) / "last.pt"
    model, metadata = load_checkpoint(path)
    assert isinstance(model, model_class)
    assert metadata["model_kind"] == kind
    assert model.config.encoder_batch_norm == batch_norm
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["model_kind"] == kind
    assert new_parameter in checkpoint["model"]
    assert checkpoint["provenance"]["device"] == "cpu"
    assert checkpoint["provenance"]["test_used_for_selection"] is False
    resumed = train(replace(config, epochs=2, resume=str(path)))
    assert resumed["epoch"] == 2 and math.isfinite(resumed["best_si_sdri"])
    checkpoint = torch.load(path, weights_only=False)
    assert all(torch.isfinite(tensor).all() for tensor in checkpoint["model"].values())
