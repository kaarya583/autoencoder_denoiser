"""Frequency sharing must preserve waveform alignment and causal execution."""

from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from esp32_denoiser.frequency_model import FrequencyUNet, FrequencyUNetConfig


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _small():
    return FrequencyUNet(FrequencyUNetConfig(encoder_channels=(4, 6, 8), global_width=8,
                                             local_dilations=(1, 2), global_dilations=(1, 2, 4)))


def test_counted_default_and_smaller_compute_budget():
    default = FrequencyUNet().model_stats()
    assert default["learned_parameters"] == 83_170
    assert default["convolution_weights"] == 81_296
    assert default["convolution_biases"] == 1_874
    assert default["macs_per_second"] == 27_561_000
    assert default["neural_state_bytes_int8"] == 18_816
    assert default["estimated_packed_bytes_upper"] == 94_511
    small = FrequencyUNet(FrequencyUNetConfig(encoder_channels=(12, 16, 24), global_width=24,
                                              local_dilations=(1, 2))).model_stats()
    assert small["macs_per_second"] == 14_584_000
    assert small["learned_parameters"] == 46_534
    assert small["estimated_packed_bytes_upper"] == 55_685


def test_analytic_macs_match_all_executed_convolution_shapes():
    model = FrequencyUNet().eval()
    counted = []
    hooks = []

    def count(layer, _inputs, output):
        counted.append(layer.weight.numel() * output.numel() // layer.out_channels)

    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            hooks.append(layer.register_forward_hook(count))
    with torch.inference_mode():
        model.forward_features(torch.randn(1, 3, 9, 257))
    for hook in hooks:
        hook.remove()
    assert sum(counted) == model.model_stats()["macs_per_frame"] * 9


def test_identity_reconstruction_and_silence():
    model = _small().eval()
    with torch.inference_mode():
        for length in (1, 255, 256, 257, 1031):
            audio = torch.randn(2, length) * 0.1
            torch.testing.assert_close(model(audio), audio, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(model(torch.zeros(1, 771)), torch.zeros(1, 771), atol=0, rtol=0)


def test_streaming_equals_nontrivial_offline_waveform_and_state_is_bounded():
    torch.manual_seed(24)
    model = _small().eval()
    with torch.no_grad():
        model.head.weight.normal_(std=0.3)
        model.head.bias.normal_(std=0.1)
        audio = torch.randn(2, 1549) * 0.1
        expected = model(audio)
        padded = F.pad(audio, (0, (-audio.shape[-1]) % 256 + 256))
        state = model.init_stream_state(2)
        shapes = [h.shape for h in (*state.local, *state.global_temporal)]
        outputs = []
        for chunk in padded.split(256, dim=-1):
            output, state = model.stream_step(chunk, state)
            outputs.append(output)
            assert [h.shape for h in (*state.local, *state.global_temporal)] == shapes
        actual = torch.cat(outputs, dim=-1)[:, 256:256 + audio.shape[-1]]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        assert (actual - audio).abs().max() > 0.01
        assert sum(h.numel() for h in (*state.local, *state.global_temporal)) == (
            2 * model.model_stats()["neural_state_bytes_int8"])


def test_neural_and_waveform_features_do_not_use_future_frames():
    torch.manual_seed(19)
    model = _small().eval()
    with torch.no_grad():
        model.head.weight.normal_(std=0.4)
        frames = torch.randn(1, 11, 512) * 0.1
        spectrum, features = model.frame_features(frames)
        assert features.shape == (1, 3, 11, 257)
        assert features.abs().max() <= 0.841
        _, individual = model.frame_features(frames[:, 2:3])
        torch.testing.assert_close(features[:, :, 2:3], individual)
        modified = features.clone()
        modified[:, :, 6:] = torch.randn_like(modified[:, :, 6:])
        torch.testing.assert_close(model.forward_features(features)[..., :6],
                                   model.forward_features(modified)[..., :6], atol=0, rtol=0)
        assert spectrum.shape == (1, 11, 257)


def test_bfloat16_training_gradients_reach_local_and_global_paths():
    model = _small()
    with torch.no_grad():
        model.head.weight.normal_(std=0.2)
    noisy = torch.randn(2, 2049) * 0.1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        enhanced = model(noisy)
        loss = (enhanced - noisy * 0.7).square().mean()
    loss.backward()
    assert enhanced.dtype == torch.float32
    for layer in (model.stem, model.local_blocks[0].pointwise, model.global_in, model.global_out, model.head):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0


def test_checkpoint_config_and_larger_teacher_widths():
    original = FrequencyUNetConfig()
    saved = asdict(original)
    for key in ("encoder_channels", "local_dilations", "global_dilations"):
        saved[key] = list(saved[key])
    assert FrequencyUNetConfig.from_checkpoint(saved) == original
    teacher = FrequencyUNet(FrequencyUNetConfig(encoder_channels=(24, 32, 48), global_width=64))
    assert teacher.model_stats()["learned_parameters"] > original.global_width * 1000
    with torch.inference_mode():
        assert teacher(torch.randn(1, 513)).shape == (1, 513)
