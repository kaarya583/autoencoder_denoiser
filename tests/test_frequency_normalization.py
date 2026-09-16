"""Folded encoder normalization preserves the deployed causal graph."""

import copy
from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from esp32_denoiser.frequency_export import export_frequency_model
from esp32_denoiser.frequency_model import FrequencyUNet, FrequencyUNetConfig
from esp32_denoiser.frequency_normalization import EncoderBatchNormConv2d, fold_encoder_batch_norm
from esp32_denoiser.frequency_quantization import (
    calibrate_frequency_hidden_exponent, configure_frequency_qat,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model(batch_norm=True):
    return FrequencyUNet(FrequencyUNetConfig(encoder_channels=(4, 6, 8), global_width=8,
                                             local_dilations=(1, 2), global_dilations=(1, 2),
                                             encoder_batch_norm=batch_norm))


def _trained():
    torch.manual_seed(51)
    model = _model().train()
    with torch.no_grad():
        model.head.weight.normal_(std=0.25)
        model.head.bias.normal_(std=0.05)
        for _ in range(5):
            model(torch.randn(3, 1792) * 0.1)
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.uniform_(0.5, 1.5)
                layer.bias.uniform_(-0.1, 0.1)
    return model.eval()


def test_normalization_is_opt_in_and_preserves_matched_initialization():
    torch.manual_seed(8)
    ordinary = _model(False)
    torch.manual_seed(8)
    normalized = _model(True)
    wrappers = [name for name, layer in normalized.named_modules()
                if isinstance(layer, EncoderBatchNormConv2d)]
    assert wrappers == ["stem", "down1.depthwise", "down1.pointwise",
                        "down2.depthwise", "down2.pointwise"]
    for name, parameter in ordinary.named_parameters():
        torch.testing.assert_close(parameter, normalized.get_parameter(name), atol=0, rtol=0)
    old_config = asdict(ordinary.config)
    old_config.pop("encoder_batch_norm")
    restored = FrequencyUNet(FrequencyUNetConfig.from_checkpoint(old_config))
    restored.load_state_dict(ordinary.state_dict())
    assert restored.config.encoder_batch_norm is False
    stats = normalized.model_stats()
    assert stats["training_normalization_parameters"] == 2 * (4 + 4 + 6 + 6 + 8)
    assert stats["deployed_parameters"] == ordinary.model_stats()["learned_parameters"]
    assert stats["macs_per_second"] == ordinary.model_stats()["macs_per_second"]


def test_eval_folding_preserves_waveform_streaming_and_causality():
    model = _trained()
    audio = torch.randn(2, 2057) * 0.1
    features = torch.randn(2, 3, 15, 257) * 0.1
    with torch.inference_mode():
        before = model(audio)
        before_features = model.forward_features(features)
        assert (before - audio).abs().max() > 0.01
        folded = fold_encoder_batch_norm(copy.deepcopy(model))
        assert not any(isinstance(layer, nn.BatchNorm2d) for layer in folded.modules())
        assert not folded.training
        torch.testing.assert_close(folded(audio), before, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(folded.forward_features(features), before_features, atol=3e-6, rtol=3e-5)
        changed = features.clone()
        changed[:, :, 8:] = torch.randn_like(changed[:, :, 8:])
        torch.testing.assert_close(folded.forward_features(changed)[:, :, :8],
                                   folded.forward_features(features)[:, :, :8], atol=0, rtol=0)
        state = folded.init_stream_state(batch_size=2)
        chunks = F.pad(audio, (0, (-audio.shape[-1]) % 256 + 256)).split(256, dim=-1)
        outputs = []
        for chunk in chunks:
            output, state = folded.stream_step(chunk, state)
            outputs.append(output)
        streamed = torch.cat(outputs, -1)[:, 256:256 + audio.shape[-1]]
        torch.testing.assert_close(streamed, before, atol=2e-6, rtol=2e-5)
        weights = {name: p.clone() for name, p in folded.named_parameters()}
        assert fold_encoder_batch_norm(folded) is folded
        for name, p in folded.named_parameters():
            torch.testing.assert_close(p, weights[name], atol=0, rtol=0)


def test_calibration_folds_before_qat_and_preserves_training_mode(tmp_path):
    model = _trained().train()
    exponent = calibrate_frequency_hidden_exponent(model, [torch.randn(2, 1025) * 0.1])
    assert model.training
    assert not any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
    assert model.model_stats()["training_normalization_parameters"] == 0
    qat = configure_frequency_qat(model, hidden_exponent=exponent).eval()
    checkpoint = {"config": asdict(qat.config), "state_dict": qat.state_dict()}
    reloaded = FrequencyUNet(FrequencyUNetConfig.from_checkpoint(checkpoint["config"]))
    assert any(isinstance(layer, nn.BatchNorm2d) for layer in reloaded.modules())
    configure_frequency_qat(reloaded)
    reloaded.load_state_dict(checkpoint["state_dict"])
    reloaded.eval()
    assert not any(isinstance(layer, nn.BatchNorm2d) for layer in reloaded.modules())
    audio = torch.randn(1, 1281) * 0.1
    with torch.inference_mode():
        torch.testing.assert_close(qat(audio), reloaded(audio), atol=0, rtol=0)
    first, second = tmp_path / "first.bin", tmp_path / "second.bin"
    export_frequency_model(qat, first)
    export_frequency_model(reloaded, second)
    assert first.read_bytes() == second.read_bytes()
    # Calibration runs under inference_mode, but folded Parameters must remain
    # ordinary tensors that can participate in the subsequent QAT backward pass.
    assert not qat.stem.weight.is_inference()
    qat.train()
    (qat(audio) - audio * 0.7).square().mean().backward()
    assert torch.isfinite(qat.stem.weight.grad).all()


def test_direct_qat_configuration_folds_copy_without_mutating_float_model():
    model = _trained()
    prepared = configure_frequency_qat(model, enabled=False, inplace=False).eval()
    assert sum(isinstance(m, nn.BatchNorm2d) for m in model.modules()) == 5
    assert not any(isinstance(m, nn.BatchNorm2d) for m in prepared.modules())
    audio = torch.randn(1, 1111) * 0.1
    with torch.inference_mode():
        torch.testing.assert_close(prepared(audio), model(audio), atol=2e-6, rtol=2e-5)
    prepared.train()
    loss = (prepared(audio) - audio * 0.7).square().mean()
    loss.backward()
    assert torch.isfinite(prepared.stem.weight.grad).all()
    assert prepared.stem.weight.grad.abs().sum() > 0
