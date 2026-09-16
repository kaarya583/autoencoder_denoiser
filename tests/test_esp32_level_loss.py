"""Level control changes optimization while preserving the original default."""
import json
from pathlib import Path

import pytest
import torch

from esp32_denoiser.metrics import si_sdr
from esp32_denoiser.train import TrainConfig, speech_loss, train


def test_gain_error_has_ten_times_stronger_restoring_gradient_without_si_sdr_reward():
    clean = torch.sin(torch.arange(512, dtype=torch.float32)[None] * 0.1) * 0.2
    lengths = torch.tensor([400])
    gain = torch.tensor(1.56, requires_grad=True)
    estimate = clean * gain
    default = speech_loss(estimate, clean, lengths)
    explicit = speech_loss(estimate, clean, lengths, waveform_loss_weight=0.1)
    stronger = speech_loss(estimate, clean, lengths, waveform_loss_weight=1.0)
    unweighted = speech_loss(estimate, clean, lengths, waveform_loss_weight=0)
    torch.testing.assert_close(default, explicit, rtol=0, atol=0)
    torch.testing.assert_close(stronger-unweighted, 10*(default-unweighted), rtol=1e-6, atol=1e-7)
    identity_score = si_sdr(clean, clean, lengths)
    amplified_score = si_sdr(estimate, clean, lengths)
    torch.testing.assert_close(identity_score, amplified_score, atol=1e-5, rtol=0)
    small_gradient = torch.autograd.grad(default, gain, retain_graph=True)[0]
    large_gradient = torch.autograd.grad(stronger, gain)[0]
    assert small_gradient > 0  # Gradient descent reduces amplification.
    torch.testing.assert_close(large_gradient, 10*small_gradient, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_weight_rejected_before_training_io(weight, tmp_path):
    config = TrainConfig("missing_train", "missing_val", str(tmp_path / "run"), waveform_loss_weight=weight)
    with pytest.raises(ValueError, match="waveform_loss_weight"):
        train(config)
    assert not (tmp_path / "run").exists()
    with pytest.raises(ValueError, match="waveform_loss_weight"):
        speech_loss(torch.ones(1, 10), torch.ones(1, 10), torch.tensor([10]), waveform_loss_weight=weight)


def test_level_ablation_is_matched_to_finetune_control():
    root = Path(__file__).resolve().parents[1] / "configs"
    control = json.loads((root / "esp32_zero_bias_finetune_float.json").read_text())
    level = json.loads((root / "esp32_zero_bias_level_float.json").read_text())
    assert level.pop("waveform_loss_weight") == 1.0
    assert TrainConfig(**control).waveform_loss_weight == 0.1
    assert level.pop("output_dir") != control.pop("output_dir")
    assert level == control
