"""The QAT calibration data path must include the actual hybrid samples."""
from dataclasses import replace
import importlib
import json
from pathlib import Path

import torch

from esp32_denoiser.data import PairedAudioDataset
from esp32_denoiser.evaluate import load_checkpoint
from esp32_denoiser.extra_data import DynamicMixtureDataset
from esp32_denoiser.quantization import QuantConv1d
from test_esp32_hybrid import hybrid_config


def test_float_to_qat_calibrates_actual_hybrid_audio_and_resume_preserves_grids(hybrid_config, monkeypatch):
    training = importlib.import_module("esp32_denoiser.train")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = replace(hybrid_config, epoch_samples=64, max_steps_per_epoch=1)
    training.train(config)
    source = Path(config.output_dir) / "last.pt"
    observed, pending, in_calibration = [], [], False
    counts = {"paired": 0, "synthetic": 0}
    original_paired = PairedAudioDataset.__getitem__
    original_synthetic = DynamicMixtureDataset.__getitem__

    def paired(dataset, index):
        item = original_paired(dataset, index)
        if in_calibration:
            assert dataset.random_crop, "Validation data reached QAT calibration"
            counts["paired"] += 1
            pending.append(item["noisy"].clone())
        return item

    def synthetic(dataset, index):
        item = original_synthetic(dataset, index)
        if in_calibration:
            assert dataset.split == "train"
            counts["synthetic"] += 1
            pending.append(item["noisy"].clone())
        return item

    original_calibrate = training.calibrate_model_hidden_exponent
    calls = []

    def calibrate(model, kind, waveforms, *, max_batches):
        nonlocal in_calibration
        assert not calls, "QAT resume must use its saved activation grid"
        assert max_batches == 32
        in_calibration = True

        def record_input(_module, arguments):
            # This hook observes the actual floating model consumed by the
            # real calibrator, not merely the supplied generator's contents.
            expected = torch.stack(pending)
            torch.testing.assert_close(arguments[0].cpu(), expected, rtol=0, atol=0)
            observed.append(arguments[0].detach().cpu().clone())
            pending.clear()

        handle = model.register_forward_pre_hook(record_input)
        try:
            result = original_calibrate(model, kind, waveforms, max_batches=max_batches)
            calls.append(result)
            return result
        finally:
            handle.remove()
            in_calibration = False

    monkeypatch.setattr(PairedAudioDataset, "__getitem__", paired)
    monkeypatch.setattr(DynamicMixtureDataset, "__getitem__", synthetic)
    monkeypatch.setattr(training, "calibrate_model_hidden_exponent", calibrate)
    qat = replace(config, phase="qat", resume=str(source), resume_optimizer=False,
                  output_dir=str(Path(config.output_dir).parent / "qat"))
    training.train(qat)
    assert len(observed) == 32 and sum(map(len, observed)) == 64
    assert sum(counts.values()) == 64 and min(counts.values()) > 0
    assert not pending
    checkpoint_path = Path(qat.output_dir) / "last.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["calibration"]["hidden_exponent"] == calls[0]
    sources = checkpoint["provenance"]["added_training_sources"]
    assert sources["synthetic_probability"] == .5 and sources["samples_per_epoch"] == 64
    restored, _ = load_checkpoint(checkpoint_path)
    layers = [module for module in restored.modules() if isinstance(module, QuantConv1d)]
    assert layers and all(int(layer.output_exponent) == calls[0] for layer in layers[:-1])
    training.train(replace(qat, resume=str(checkpoint_path), resume_optimizer=True, epochs=2))
    resumed = torch.load(checkpoint_path, weights_only=False)
    assert len(calls) == 1 and resumed["calibration"] == checkpoint["calibration"]
    for name, tensor in checkpoint["model"].items():
        if "exponent" in name:
            torch.testing.assert_close(resumed["model"][name], tensor, rtol=0, atol=0)


def test_broad_qat_config_uses_frozen_input_fresh_optimizer_and_same_hybrid_recipe():
    config_path = Path(__file__).resolve().parents[1] / "configs/esp32_zero_bias_broad_qat.json"
    config = json.loads(config_path.read_text())
    assert config["phase"] == "qat" and config["resume_optimizer"] is False
    assert config["resume"] == "/content/esp32_runs/broad_qat_inputs/student.pt"
    assert config["synthetic_probability"] == .5 and config["epoch_samples"] == 10802
    assert config["teacher_checkpoint"] is None and config["distillation_weight"] == 0
    assert config["distillation_gate_minimum_improvement_db"] is None
    assert config["amp"] is False and config["epochs"] == config["patience"] == 40
