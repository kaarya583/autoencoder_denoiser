"""Controlled frame-RMS input ablation, independent of upstream operators."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from esp32_denoiser.gtcrn_model import (
    GTCRNConfig, GTCRNDenoiser, _network_spectrum, _spectrum,
)


@pytest.fixture
def model():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(91)
        result = GTCRNDenoiser(GTCRNConfig(normalize_input=True)).eval()
    return result


def test_raw_spectrum_is_unchanged_and_normalized_bins_are_bounded(model):
    # Include a bin-bound equality case (samples proportional to the window),
    # impulses and silence, at amplitudes spanning the normalization floor.
    frames = torch.randn(2, 6, 512) * .05
    frames[:, 0] = model.window
    frames[:, 1] = 0
    frames[:, 2] = 0
    frames[:, 2, 100] = .3
    frames[:, 3] *= 1e-8
    raw, raw_scale = _network_spectrum(frames, model.window, GTCRNConfig())
    assert raw_scale is None
    assert torch.equal(raw, _spectrum(frames, model.window))
    normalized, scale = _network_spectrum(frames, model.window, model.config)
    torch.testing.assert_close(normalized * scale, raw, rtol=2e-7, atol=1e-7)
    magnitude = torch.linalg.vector_norm(normalized, dim=-1)
    bound = model.window.square().mean().sqrt()
    assert float(magnitude.max()) <= float(bound) + 2e-7
    torch.testing.assert_close(magnitude[:, 0, 0], bound.expand(2), rtol=1e-6, atol=1e-7)
    assert float(scale.min()) >= 512 * model.config.rms_floor - 1e-8
    assert model.model_stats()["learned_parameters"] == 23_669
    assert model.model_stats()["input_normalization"] == "frame_rms"


@pytest.mark.parametrize("length", [1, 257, 2048, 2317])
def test_normalized_offline_streaming_reset_parity(model, length):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-.15, .15)
                module.running_var.uniform_(.8, 1.3)
    audio = torch.randn(2, length) * .03
    with torch.no_grad():
        offline = model(audio)
    stream = model.make_streaming()
    streamed = stream.denoise(audio)
    assert offline.shape == audio.shape
    torch.testing.assert_close(streamed, offline, rtol=4e-4, atol=2e-6)
    torch.testing.assert_close(streamed, stream.denoise(audio), rtol=0, atol=0)


def test_normalized_prefix_causality_and_positive_gain_equivariance(model):
    prefix = torch.randn(2, 2048) * .05
    audio = torch.cat((prefix, torch.randn(2, 913) * .1), -1)
    changed_suffix = torch.cat((prefix, torch.randn(2, 1341) * .3), -1)
    with torch.no_grad():
        expected = model(audio)
        changed = model(changed_suffix)
        # Last hop of this prefix still participates in the next analysis frame.
        torch.testing.assert_close(expected[:, :1792], changed[:, :1792], rtol=3e-4, atol=2e-6)
        for gain in (.25, 3.0):
            # Assert the premise: all frames stay above the floor under the gain.
            frames = F.pad(audio * gain, (256, (-audio.shape[-1]) % 256 + 256)).unfold(-1, 512, 256)
            assert float(frames.square().mean(-1).sqrt().min()) > model.config.rms_floor
            torch.testing.assert_close(model(audio * gain), expected * gain, rtol=4e-4, atol=3e-6)
            torch.testing.assert_close(model.make_streaming().denoise(audio * gain), expected * gain,
                                       rtol=4e-4, atol=3e-6)


def test_floor_silence_reconstruction_and_finite_gradients(model):
    noisy = torch.cat((torch.zeros(1, 1024), torch.randn(1, 1024) * 1e-9,
                       torch.randn(1, 1024) * .04), dim=0).requires_grad_()
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(noisy)
        loss = (output - .7 * noisy).square().mean()
    loss.backward()
    assert bool(torch.isfinite(output).all())
    assert bool(torch.isfinite(noisy.grad).all())
    assert torch.count_nonzero(output[0]) == 0
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
               for p in model.parameters() if p.requires_grad)
    assert any(float(p.grad.abs().max()) > 0 for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        expected = model(noisy.detach())
    torch.testing.assert_close(model.make_streaming().denoise(noisy.detach()), expected, rtol=4e-4, atol=2e-6)
    # Rescaling is a complete inverse even when the floor dominates.
    model.core = torch.nn.Identity()
    torch.testing.assert_close(model(noisy.detach()), noisy.detach(), rtol=2e-5, atol=1e-7)


def test_config_roundtrip_validation_and_matched_experiment(model):
    assert GTCRNConfig.from_checkpoint(asdict(model.config)) == model.config
    for value in (0, -1, float("inf"), float("nan"), 1e-30, 1e30, True):
        with pytest.raises(ValueError, match="rms_floor"):
            GTCRNConfig(normalize_input=True, rms_floor=value)
    with pytest.raises(ValueError, match="boolean"):
        GTCRNConfig(normalize_input="true")
    config_root = Path(__file__).parents[1] / "configs"
    baseline = json.loads((config_root / "esp32_gtcrn_broad_float.json").read_text())
    ablation = json.loads((config_root / "esp32_gtcrn_broad_normalized_float.json").read_text())
    assert ablation.pop("output_dir") != baseline.pop("output_dir")
    assert ablation.pop("model_options") == {"normalize_input": True}
    assert baseline.pop("model_options") == {}
    assert ablation == baseline
