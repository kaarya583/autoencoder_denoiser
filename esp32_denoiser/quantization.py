"""Quantization used by both training and the portable integer runtime.

Values are represented as ``q * 2**exponent``. Activations and FIFO state are
signed INT8; weights use one power-of-two exponent per output channel. Dot
products and biases use INT32. Rounding is nearest, with ties away from zero.
This is a deliberately small, explicit format, not a claim of ESP-DL support.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def round_away(x: Tensor) -> Tensor:
    return x.sign() * torch.floor(x.abs() + 0.5)


def fake_quantize(x: Tensor, exponent: Tensor | int, *, limit: int = 128) -> Tensor:
    """Straight-through quantizer; forward values lie exactly on the grid."""
    scale = torch.pow(x.new_tensor(2.0), exponent)
    quantized = round_away(x / scale).clamp(-limit, 127) * scale
    return x + (quantized - x).detach()


class QuantActivation(nn.Module):
    def __init__(self, exponent: int, enabled: bool = True) -> None:
        super().__init__()
        self.register_buffer("exponent", torch.tensor(exponent, dtype=torch.int32))
        self.enabled = enabled

    def forward(self, x: Tensor) -> Tensor:
        return fake_quantize(x, self.exponent) if self.enabled else x


class QuantConv1d(nn.Conv1d):
    """Conv1d with the same grids, bias rounding and clipping as the C kernel."""

    @classmethod
    def from_float(cls, layer: nn.Conv1d, input_exponent: int,
                   output_exponent: int) -> "QuantConv1d":
        if layer.padding_mode != "zeros" or layer.stride != (1,):
            raise ValueError("Only stride-one, zero-padded convolutions are supported")
        result = cls(layer.in_channels, layer.out_channels, layer.kernel_size,
                     stride=layer.stride, padding=layer.padding,
                     dilation=layer.dilation, groups=layer.groups,
                     bias=layer.bias is not None, device=layer.weight.device,
                     dtype=layer.weight.dtype)
        result.weight = layer.weight
        result.bias = layer.bias
        result.register_buffer("input_exponent", torch.tensor(input_exponent, dtype=torch.int32, device=layer.weight.device))
        result.register_buffer("output_exponent", torch.tensor(output_exponent, dtype=torch.int32, device=layer.weight.device))
        result.enabled = True
        return result

    def weight_exponents(self) -> Tensor:
        # A per-channel grid avoids sacrificing quiet channels to a large outlier.
        peak = self.weight.detach().abs().flatten(1).amax(1)
        return torch.ceil(torch.log2((peak / 127).clamp_min(2.0 ** -24))).to(torch.int32)

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return super().forward(x)
        weight_exp = self.weight_exponents()
        x = fake_quantize(x, self.input_exponent)
        weight = fake_quantize(self.weight, weight_exp[:, None, None], limit=127)
        bias = self.bias
        if bias is not None:
            scale = torch.pow(bias.new_tensor(2.0), self.input_exponent + weight_exp)
            rounded = round_away(bias / scale).clamp(-(2**31), 2**31 - 1) * scale
            bias = bias + (rounded - bias).detach()
        output = F.conv1d(x, weight, bias, self.stride, self.padding,
                          self.dilation, self.groups)
        return fake_quantize(output, self.output_exponent)


def configure_qat(model: nn.Module, *, enabled: bool = True,
                  input_exponent: int = -7, hidden_exponent: int = -4,
                  output_exponent: int = -7, inplace: bool = True) -> nn.Module:
    """Prepare the spectral TCN for QAT without modifying its float checkpoint.

    Configure before creating an optimizer. Repeated calls toggle existing
    quantizers while preserving their saved grids. Scales stay constant over a
    stream, so cached INT8 values cannot silently change interpretation.
    """
    result = model if inplace else copy.deepcopy(model)
    for name, layer in list(result.named_modules()):
        if not isinstance(layer, nn.Conv1d) or isinstance(layer, QuantConv1d):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = result.get_submodule(parent_name) if parent_name else result
        input_exp = input_exponent if name == "input_proj" else hidden_exponent
        output_exp = output_exponent if name == "head" else hidden_exponent
        setattr(parent, child_name, QuantConv1d.from_float(layer, input_exp, output_exp))
    if not isinstance(result.input_quant, QuantActivation):
        result.input_quant = QuantActivation(input_exponent)
    for block in result.blocks:
        if not isinstance(block.residual_quant, QuantActivation):
            block.residual_quant = QuantActivation(hidden_exponent)
    if hasattr(result, "head_quant") and not isinstance(result.head_quant, QuantActivation):
        result.head_quant = QuantActivation(output_exponent)
    set_quantization(result, enabled)
    return result


def set_quantization(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, (QuantActivation, QuantConv1d)):
            module.enabled = bool(enabled)


@torch.inference_mode()
def calibrate_hidden_exponent(model: nn.Module, waveforms: Iterable[Tensor],
                              *, max_batches: int = 32) -> int:
    """Choose a shared hidden grid from representative training audio.

    Observe the states after their fixed nonlinearities, plus each pointwise
    residual branch before addition. Extrema discarded by ReLU6/Hardtanh do
    not require range, but a signed branch must retain its range before it
    cancels with the skip connection. Maximum magnitude calibration keeps
    the FIFO/residual invariant: every hidden tensor has one scale.
    Inputs must already be on the model's device. Use training audio, never the
    held-out final test set. Configure QAT after this pass, before an optimizer.
    """
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    if any(isinstance(layer, QuantConv1d) for layer in model.modules()):
        raise ValueError("Calibrate the floating-point model before configure_qat")
    was_training = model.training
    maximum = 0.0
    seen = 0
    handles = []

    def observe(_module, _inputs, output):
        nonlocal maximum
        value = float(output.detach().abs().amax())
        if not math.isfinite(value):
            raise ValueError("Non-finite activation encountered during calibration")
        maximum = max(maximum, value)

    try:
        model.eval()
        handles.append(model.input_activation.register_forward_hook(observe))
        for block in model.blocks:
            for layer in (block.depth_activation, block.pointwise, block.output_activation):
                handles.append(layer.register_forward_hook(observe))
        for batch in waveforms:
            model(batch)
            seen += 1
            if seen >= max_batches:
                break
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if not seen:
        raise ValueError("Calibration requires at least one audio batch")
    if maximum == 0:
        return -12
    exponent = math.ceil(math.log2(maximum / 127))
    if exponent > 0:
        raise ValueError("Hidden activation range exceeds the deployment format")
    return max(-12, exponent)
