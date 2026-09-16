"""Power-of-two QAT for the frequency-shared model, including Conv2d.

This simulates INT8 tensors and INT32 biases/accumulators using PyTorch. The
frequency exporter supplies the matching binary and integer reference.
"""

import copy
import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .quantization import QuantActivation, QuantConv1d, fake_quantize, round_away
from .frequency_normalization import fold_encoder_batch_norm


class QuantConv2d(nn.Conv2d):
    @classmethod
    def from_float(cls, layer: nn.Conv2d, input_exponent: int, output_exponent: int) -> "QuantConv2d":
        if layer.padding_mode != "zeros" or layer.stride[0] != 1:
            raise ValueError("Frequency QAT requires zero padding and temporal stride one")
        result = cls(layer.in_channels, layer.out_channels, layer.kernel_size,
                     stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                     groups=layer.groups, bias=layer.bias is not None,
                     device=layer.weight.device, dtype=layer.weight.dtype)
        result.weight = layer.weight
        result.bias = layer.bias
        for name, value in (("input_exponent", input_exponent), ("output_exponent", output_exponent)):
            result.register_buffer(name, torch.tensor(value, dtype=torch.int32, device=layer.weight.device))
        result.enabled = True
        return result

    def weight_exponents(self) -> Tensor:
        peak = self.weight.detach().abs().flatten(1).amax(1)
        return torch.ceil(torch.log2((peak / 127).clamp_min(2.0 ** -24))).to(torch.int32)

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return super().forward(x)
        exponent = self.weight_exponents()
        x = fake_quantize(x, self.input_exponent)
        weight = fake_quantize(self.weight, exponent[:, None, None, None], limit=127)
        bias = self.bias
        if bias is not None:
            scale = torch.pow(bias.new_tensor(2.0), self.input_exponent + exponent)
            rounded = round_away(bias / scale).clamp(-(2**31), 2**31 - 1) * scale
            bias = bias + (rounded - bias).detach()
        output = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return fake_quantize(output, self.output_exponent)


def configure_frequency_qat(model: nn.Module, *, enabled: bool = True,
                            input_exponent: int = -7, hidden_exponent: int = -4,
                            output_exponent: int = -7, inplace: bool = True) -> nn.Module:
    """Prepare every convolution and addition boundary; preserve saved grids.

    Call before constructing the optimizer. All hidden maps, temporal FIFOs,
    encoder skips and residual branches use the same scale. This permits
    integer additions without intermediate rescaling or floating fallbacks.
    """
    result = model if inplace else copy.deepcopy(model)
    fold_encoder_batch_norm(result)
    for name, layer in list(result.named_modules()):
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d)) or isinstance(layer, (QuantConv1d, QuantConv2d)):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = result.get_submodule(parent_name) if parent_name else result
        input_exp = input_exponent if name == "stem" else hidden_exponent
        output_exp = output_exponent if name == "head" else hidden_exponent
        quantized_type = QuantConv2d if isinstance(layer, nn.Conv2d) else QuantConv1d
        setattr(parent, child_name, quantized_type.from_float(layer, input_exp, output_exp))
    boundaries = [(result, "input_quant", input_exponent),
                  (result, "head_quant", output_exponent),
                  (result, "global_residual_quant", hidden_exponent),
                  (result.up1, "skip_quant", hidden_exponent),
                  (result.up2, "skip_quant", hidden_exponent)]
    boundaries.extend((block, "residual_quant", hidden_exponent)
                      for block in (*result.local_blocks, *result.global_blocks))
    device = result.stem.weight.device
    for parent, name, exponent in boundaries:
        if not isinstance(getattr(parent, name), QuantActivation):
            setattr(parent, name, QuantActivation(exponent).to(device))
    set_frequency_quantization(result, enabled)
    return result


def set_frequency_quantization(model: nn.Module, enabled: bool) -> None:
    for layer in model.modules():
        if isinstance(layer, (QuantConv1d, QuantConv2d, QuantActivation)):
            layer.enabled = bool(enabled)


@torch.inference_mode()
def calibrate_frequency_hidden_exponent(model: nn.Module, waveforms: Iterable[Tensor],
                                        *, max_batches: int = 32) -> int:
    """Observe used activation ranges, retaining every pre-add branch range."""
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    if any(isinstance(layer, (QuantConv1d, QuantConv2d)) for layer in model.modules()):
        raise ValueError("Calibrate the floating-point model before configure_frequency_qat")
    fold_encoder_batch_norm(model)
    maximum = 0.0
    seen = 0
    handles = []
    was_training = model.training

    def observe(_module, _inputs, output):
        nonlocal maximum
        value = float(output.detach().abs().amax())
        if not math.isfinite(value):
            raise ValueError("Non-finite activation encountered during calibration")
        maximum = max(maximum, value)

    observed = [model.stem_activation, model.global_in_activation,
                model.global_out, model.global_out_activation, model.head_dw_activation]
    for down in (model.down1, model.down2):
        observed.extend((down.depth_activation, down.output_activation))
    for block in (*model.local_blocks, *model.global_blocks):
        observed.extend((block.depth_activation, block.pointwise, block.output_activation))
    for up in (model.up1, model.up2):
        # The depthwise output is a signed branch added to an encoder skip.
        observed.extend((up.point_activation, up.depthwise, up.output_activation))
    try:
        model.eval()
        handles = [layer.register_forward_hook(observe) for layer in observed]
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
    exponent = -12 if maximum == 0 else max(-12, math.ceil(math.log2(maximum / 127)))
    if exponent > 0:
        raise ValueError("Hidden activation range exceeds the deployment format")
    return exponent
