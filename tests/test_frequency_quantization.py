"""Quantization grid and streaming checks for the new frequency graph."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from esp32_denoiser.frequency_model import FrequencyUNet, FrequencyUNetConfig
from esp32_denoiser.frequency_quantization import (
    QuantConv2d, calibrate_frequency_hidden_exponent, configure_frequency_qat,
)
from esp32_denoiser.quantization import QuantActivation, QuantConv1d, round_away


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model():
    return FrequencyUNet(FrequencyUNetConfig(encoder_channels=(4, 6, 8), global_width=8,
                                             local_dilations=(1, 2), global_dilations=(1, 2)))


@pytest.mark.parametrize("groups, kernel, stride, dilation", [
    (1, (1, 5), (1, 2), (1, 1)), (4, (3, 3), (1, 1), (2, 1)),
])
def test_quantized_conv2d_matches_exact_integer_accumulators(groups, kernel, stride, dilation):
    torch.manual_seed(12)
    layer = QuantConv2d.from_float(nn.Conv2d(4, 4, kernel, stride=stride, dilation=dilation,
                                            groups=groups, padding=(0, 2)), -7, -5)
    inputs = torch.randint(-127, 128, (1, 4, 9, 15)).float() / 128
    exponents = layer.weight_exponents()
    scales = 2.0 ** exponents.double()
    weights = round_away(layer.weight.double() / scales[:, None, None, None]).clamp(-127, 127)
    bias = round_away(layer.bias.double() / (scales / 128))
    # Integer-valued doubles represent these bounded accumulators exactly.
    accumulator = F.conv2d(inputs.double() * 128, weights, bias,
                            stride, layer.padding, dilation, groups)
    shift = layer.input_exponent.double() + exponents.double() - layer.output_exponent.double()
    expected = round_away(accumulator * (2.0 ** shift)[None, :, None, None]).clamp(-128, 127) / 32
    torch.testing.assert_close(layer(inputs).double(), expected, atol=0, rtol=0)


def test_frequency_qat_covers_all_convolutions_additions_and_streaming():
    torch.manual_seed(18)
    floating = _model().eval()
    with torch.no_grad():
        floating.head.weight.normal_(std=0.3)
        floating.head.bias.normal_(std=0.1)
    quantized = configure_frequency_qat(floating, hidden_exponent=-6, inplace=False)
    assert not any(isinstance(m, (QuantConv1d, QuantConv2d)) for m in floating.modules())
    for layer in quantized.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            assert isinstance(layer, (QuantConv1d, QuantConv2d))
    for boundary in (quantized.input_quant, quantized.head_quant, quantized.global_residual_quant,
                     quantized.up1.skip_quant, quantized.up2.skip_quant,
                     *(b.residual_quant for b in (*quantized.local_blocks, *quantized.global_blocks))):
        assert isinstance(boundary, QuantActivation)
    audio = torch.randn(1, 1031) * 0.1
    with torch.inference_mode():
        expected = quantized(audio)
        state = quantized.init_stream_state()
        chunks = F.pad(audio, (0, (-audio.shape[-1]) % 256 + 256)).split(256, dim=-1)
        outputs = []
        for chunk in chunks:
            output, state = quantized.stream_step(chunk, state)
            outputs.append(output)
        actual = torch.cat(outputs, dim=-1)[:, 256:256 + audio.shape[-1]]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        assert (actual - audio).abs().max() > 0.01
    loss = (quantized(audio) - audio * 0.8).square().mean()
    loss.backward()
    assert torch.isfinite(quantized.head.weight.grad).all()
    assert quantized.head.weight.grad.abs().sum() > 0
    configure_frequency_qat(quantized, enabled=False, hidden_exponent=-4)
    assert int(quantized.global_in.input_exponent) == -6  # Saved grids survive toggles.
    with torch.inference_mode():
        torch.testing.assert_close(quantized(audio), floating(audio), atol=0, rtol=0)


@pytest.mark.parametrize("branch_bias, expected_exponent", [(0.0, -4), (-9.0, -3)])
def test_frequency_calibration_ignores_clipped_extrema_and_keeps_skip_branches(branch_bias, expected_exponent):
    model = _model().train()
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
                layer.weight.zero_()
                layer.bias.zero_()
        model.stem.bias.fill_(100)
        model.down1.depthwise.bias.fill_(-100)
        model.down2.depthwise.bias.fill_(100)
        model.global_in.bias.fill_(100)
        model.head_dw.bias.fill_(100)
        model.up2.depthwise.bias.fill_(branch_bias)
    assert calibrate_frequency_hidden_exponent(model, [torch.zeros(1, 513)]) == expected_exponent
    assert model.training
    assert not any(layer._forward_hooks for layer in model.modules())
    configure_frequency_qat(model)
    with pytest.raises(ValueError, match="before configure_frequency_qat"):
        calibrate_frequency_hidden_exponent(model, [torch.zeros(1, 513)])
