"""Export the causal spectral TCN as a small, versioned integer model blob."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Real
from pathlib import Path
import struct

import numpy as np
import torch

from .quantization import QuantActivation, QuantConv1d, round_away
from .model import SpectralTCN

MAGIC = b"EDNSI8\0\0"
FORMAT_VERSION = 2  # Signed hardtanh residual states; v1 used ReLU6 states.
HEADER = struct.Struct("<8sIHHHHbbbBII")
LAYER = struct.Struct("<BbbBHHHHIII")
DSP_HEADER = struct.Struct("<4sHHHHH2xf")
MAX_MODEL_BYTES = 99_000


@dataclass(frozen=True)
class IntegerLayer:
    kind: int
    input_exponent: int
    output_exponent: int
    kernel: int
    input_channels: int
    output_channels: int
    dilation: int
    weights: np.ndarray
    bias: np.ndarray
    weight_exponents: np.ndarray


def _quantized_layer(module: QuantConv1d) -> IntegerLayer:
    if module.groups not in (1, module.in_channels):
        raise ValueError("Only dense pointwise and channel-multiplier-one depthwise layers are supported")
    depthwise = module.groups == module.in_channels and module.kernel_size != (1,)
    if (not depthwise and module.kernel_size != (1,)) or (depthwise and (module.out_channels != module.in_channels or module.kernel_size != (3,))):
        raise ValueError("Unsupported convolution shape")
    exponent = module.weight_exponents().cpu()
    if bool(((exponent < -24) | (exponent > 16)).any()):
        raise ValueError("Weight exponent outside supported range")
    weight = module.weight.detach().cpu()
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Cannot export non-finite weights")
    scale = torch.pow(torch.tensor(2.0), exponent)
    qweight = round_away(weight / scale[:, None, None]).clamp(-127, 127).to(torch.int8).numpy()
    input_exp = int(module.input_exponent)
    output_exp = int(module.output_exponent)
    bias = module.bias.detach().cpu() if module.bias is not None else torch.zeros(module.out_channels)
    qbias = round_away(bias.double() / torch.pow(torch.tensor(2.0, dtype=torch.float64), exponent + input_exp))
    terms = module.weight[0].numel()
    bound = terms * 128 * 127
    if not bool(torch.isfinite(qbias).all()) or bool((qbias.abs() > (2**31 - 1 - bound)).any()):
        raise ValueError("Bias/weights could overflow an INT32 accumulator")
    return IntegerLayer(int(depthwise), input_exp, output_exp, module.kernel_size[0],
                        module.in_channels, module.out_channels, module.dilation[0],
                        qweight, qbias.to(torch.int32).numpy(), exponent.to(torch.int8).numpy())


def export_model(model, destination: str | Path, *, max_bytes: int = MAX_MODEL_BYTES,
                 output_gain: float = 1.0) -> dict:
    """Write actual deployed bytes, not a pickle or fake-quantized checkpoint."""
    output_gain = _output_gain_float32(output_gain)
    if not isinstance(model, SpectralTCN):
        raise ValueError("EDNSI8 export requires a supported SpectralTCN graph")
    if model.config.feature_layout != "erb_complex":
        raise ValueError("Integer export currently supports only the erb_complex feature layout")
    if model.config.activation_mode != "signed":
        raise ValueError("EDNSI8-v2 requires signed hardtanh residual states")
    if not isinstance(model.input_proj, QuantConv1d):
        raise ValueError("Call configure_qat and train/evaluate its quantized graph before export")
    modules = [model.input_proj]
    for block in model.blocks:
        modules.extend([block.depthwise, block.pointwise])
    modules.append(model.head)
    if any(not isinstance(layer, QuantConv1d) or not layer.enabled for layer in modules):
        raise ValueError("Every neural convolution must have active quantization")
    layers = [_quantized_layer(module) for module in modules]
    input_dim, width, output_dim = layers[0].input_channels, layers[0].output_channels, layers[-1].output_channels
    blocks = len(model.blocks)
    if any(layer.input_channels != width or layer.output_channels != width for layer in layers[1:-1]):
        raise ValueError("Residual blocks must keep a constant hidden width")
    input_exp, hidden_exp, output_exp = layers[0].input_exponent, layers[0].output_exponent, layers[-1].output_exponent
    boundaries = [(model.input_quant, input_exp), (model.head_quant, output_exp)]
    boundaries.extend((block.residual_quant, hidden_exp) for block in model.blocks)
    if any(not isinstance(layer, QuantActivation) or not layer.enabled or int(layer.exponent) != exponent
           for layer, exponent in boundaries):
        raise ValueError("Quantization boundaries must use the deployed activation grids")
    if not np.isfinite(model.config.mask_scale) or not 0 < model.config.mask_scale <= 8:
        raise ValueError("mask_scale must be finite and in (0, 8]")
    if not (-16 <= input_exp <= 8 and -12 <= hidden_exp <= 0 and output_exp == -7):
        raise ValueError("Unsupported activation exponents; output must represent hardtanh at 2^-7")
    for layer in layers[1:]:
        if layer.input_exponent != hidden_exp or (layer is not layers[-1] and layer.output_exponent != hidden_exp):
            raise ValueError("All hidden tensors and residual branches must share one scale")
    state_bytes = 4 * blocks + sum(2 * layer.dilation * width for layer in layers if layer.kind == 1) + 3 * width
    payload = bytearray(HEADER.size + LAYER.size * len(layers))
    for index, layer in enumerate(layers):
        weights_offset = len(payload)
        payload.extend(layer.weights.tobytes())
        payload.extend(bytes((-len(payload)) % 4))
        bias_offset = len(payload)
        payload.extend(layer.bias.astype("<i4").tobytes())
        exponents_offset = len(payload)
        payload.extend(layer.weight_exponents.tobytes())
        LAYER.pack_into(payload, HEADER.size + index * LAYER.size, layer.kind,
                        layer.input_exponent, layer.output_exponent, layer.kernel,
                        layer.input_channels, layer.output_channels, layer.dilation, 0,
                        weights_offset, bias_offset, exponents_offset)
    # Include every model-specific DSP constant in the reported deployed size.
    # Float32 FFT/window/filterbank arithmetic belongs to the external DSP path;
    # the neural runtime never uses floating-point arithmetic.
    payload.extend(bytes((-len(payload)) % 4))
    dsp_offset = len(payload)
    dsp_version = 1 if output_gain == 1.0 else 2
    payload.extend(DSP_HEADER.pack(b"DSP1" if dsp_version == 1 else b"DSP2", model.config.sample_rate, model.config.n_fft,
                                  model.config.hop_length, model.low_bins, model.high_bands,
                                  model.config.mask_scale))
    for name, dtype in (("window", "<f4"), ("erb_lower", "u1"), ("erb_upper", "u1"),
                        ("erb_lower_weight", "<f4"), ("erb_upper_weight", "<f4")):
        values = getattr(model, name).detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite DSP constants: {name}")
        payload.extend(values.astype(dtype).tobytes())
    if dsp_version == 2:
        payload.extend(struct.pack("<f", output_gain))
    HEADER.pack_into(payload, 0, MAGIC, FORMAT_VERSION, input_dim, width, output_dim, blocks,
                     input_exp, hidden_exp, output_exp, dsp_version, len(payload), state_bytes)
    if len(payload) > max_bytes:
        raise ValueError(f"Integer model is {len(payload):,} bytes, exceeding {max_bytes:,}-byte limit")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    report = {"format": f"EDNSI8-v{FORMAT_VERSION}", "model_bytes": len(payload), "state_bytes": state_bytes,
              "weight_bytes": sum(layer.weights.size for layer in layers),
              "bias_bytes": sum(layer.bias.nbytes for layer in layers),
              "input_channels": input_dim, "hidden_channels": width, "output_channels": output_dim,
              "blocks": blocks, "input_exponent": input_exp, "hidden_exponent": hidden_exp,
              "activation_mode": "signed",
              "output_exponent": output_exp,
              "dsp_offset": dsp_offset, "dsp_bytes": len(payload) - dsp_offset,
              "mask_scale": model.config.mask_scale,
              "dsp_version": dsp_version, "output_gain": output_gain,
              "macs_per_frame": sum(layer.weights.size for layer in layers),
              "teacher_parameter_bytes": 11_854_856,
              "parameter_payload_reduction": 11_854_856 / len(payload),
              "precision": "INT8 weights, activations and state; INT32 bias and accumulation",
              "audio_quality": "unmeasured", "esp32_runtime": "unmeasured"}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _output_gain_float32(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("DSP output gain must be a finite positive scalar at most 8")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("DSP output gain is outside float32 range") from error
    if not np.isfinite(value) or not 0 < value <= 8:
        raise ValueError("DSP output gain must be finite and in (0,8]")
    result = struct.unpack("<f", struct.pack("<f", value))[0]
    if result <= 0:
        raise ValueError("DSP output gain must remain positive in float32")
    return result


def with_output_gain(source: str | Path | bytes, destination: str | Path, gain: float,
                     *, max_bytes: int = MAX_MODEL_BYTES) -> dict:
    """Replace one external-DSP gain, preserving every learned integer byte.

    This low-level serializer does not fit a gain or verify its training source.
    Preserve the calibration report separately and bind it to the source hash.
    A gain of one restores the byte-identical legacy DSP1 representation.
    """
    gain = _output_gain_float32(gain)
    model = IntegerDenoiser(source)
    if (model.sample_rate, model.n_fft, model.hop_length, model.low_bins, model.high_bands,
        model.input_channels, model.output_channels) != (16000, 512, 256, 65, 64, 387, 514):
        raise ValueError("Unsupported DSP contract for output-gain serialization")
    payload = bytearray(model.data[:model.dsp_offset + 3988])
    version = 1 if gain == 1.0 else 2
    payload[model.dsp_offset:model.dsp_offset + 4] = b"DSP1" if version == 1 else b"DSP2"
    if version == 2:
        payload.extend(struct.pack("<f", gain))
    values = list(HEADER.unpack_from(payload))
    values[9], values[10] = version, len(payload)
    HEADER.pack_into(payload, 0, *values)
    IntegerDenoiser(bytes(payload))
    if len(payload) > max_bytes:
        raise ValueError(f"Integer model is {len(payload):,} bytes, exceeding {max_bytes:,}-byte limit")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    report = {"format": f"EDNSI8-v{FORMAT_VERSION}", "dsp_version": version,
              "source_sha256": hashlib.sha256(model.data).hexdigest(),
              "model_sha256": hashlib.sha256(payload).hexdigest(), "model_bytes": len(payload),
              "previous_output_gain": model.output_gain, "output_gain": gain,
              "neural_weights_and_state_unchanged": True,
              "scope": "float32 external DSP multiply after overlap-add, before PCM16 output conversion"}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def requantize(accumulator: np.ndarray, shift: np.ndarray | int) -> np.ndarray:
    """Integer-only nearest/away-from-zero power-of-two rescaling."""
    acc = np.asarray(accumulator, dtype=np.int64)
    shifts = np.broadcast_to(shift, acc.shape).astype(np.int64)
    result = np.empty_like(acc)
    for power in np.unique(shifts):
        mask = shifts == power
        values = acc[mask]
        if power >= 7:
            result[mask] = np.sign(values) * 128
        elif power >= 0:
            # Once outside the target range, left shifting further changes nothing.
            result[mask] = np.clip(values, -128, 127) * (1 << int(power))
        elif power < -32:
            result[mask] = 0
        else:
            divisor = 1 << int(-power)
            result[mask] = np.sign(values) * ((np.abs(values) + divisor // 2) // divisor)
    return result.clip(-128, 127).astype(np.int8)


class IntegerDenoiser:
    """Numpy INT8/INT32 streaming oracle for the exact deployment format."""

    def __init__(self, source: str | Path | bytes):
        self.data = bytes(source) if isinstance(source, bytes) else Path(source).read_bytes()
        values = HEADER.unpack_from(self.data)
        magic, version, self.input_channels, self.width, self.output_channels, self.blocks, self.input_exponent, self.hidden_exponent, self.output_exponent, dsp_version, total, self.state_bytes = values
        if magic != MAGIC or version != FORMAT_VERSION or total != len(self.data) or dsp_version not in (1, 2):
            raise ValueError("Invalid integer denoiser model header")
        self.dsp_version = dsp_version
        self.layers = []
        last_offset = 0
        for index in range(2 + 2 * self.blocks):
            kind, ie, oe, kernel, inputs, outputs, dilation, _, wo, bo, eo = LAYER.unpack_from(self.data, HEADER.size + index * LAYER.size)
            count = outputs * kernel * (1 if kind else inputs)
            weights = np.frombuffer(self.data, dtype=np.int8, count=count, offset=wo).copy()
            weights = weights.reshape(outputs, 1 if kind else inputs, kernel)
            bias = np.frombuffer(self.data, dtype="<i4", count=outputs, offset=bo).copy()
            exponents = np.frombuffer(self.data, dtype=np.int8, count=outputs, offset=eo).copy()
            last_offset = max(last_offset, wo + count, bo + outputs * 4, eo + outputs)
            self.layers.append(IntegerLayer(kind, ie, oe, kernel, inputs, outputs, dilation, weights, bias, exponents))
        self.dsp_offset = (last_offset + 3) & ~3
        dsp_magic, self.sample_rate, self.n_fft, self.hop_length, self.low_bins, self.high_bands, self.mask_scale = DSP_HEADER.unpack_from(self.data, self.dsp_offset)
        if dsp_magic != (b"DSP1" if dsp_version == 1 else b"DSP2"):
            raise ValueError("Missing DSP metadata")
        cursor = self.dsp_offset + DSP_HEADER.size
        high_bins = self.n_fft // 2 + 1 - self.low_bins
        for name, dtype, count in (("window", "<f4", self.n_fft), ("erb_lower", "u1", high_bins),
                                   ("erb_upper", "u1", high_bins), ("erb_lower_weight", "<f4", high_bins),
                                   ("erb_upper_weight", "<f4", high_bins)):
            values = np.frombuffer(self.data, dtype=dtype, count=count, offset=cursor).copy()
            setattr(self, name, values)
            cursor += values.nbytes
        self.output_gain = 1.0
        if dsp_version == 2:
            if cursor + 4 > len(self.data):
                raise ValueError("Missing DSP output gain")
            self.output_gain = _output_gain_float32(struct.unpack_from("<f", self.data, cursor)[0])
            cursor += 4
        if cursor != len(self.data):
            raise ValueError("Unexpected integer-model trailing data")
        self.reset()

    def reset(self) -> None:
        self.history = [np.zeros((2 * self.layers[1 + 2 * i].dilation, self.width), dtype=np.int8) for i in range(self.blocks)]
        self.positions = [0] * self.blocks

    @staticmethod
    def _dense(layer: IntegerLayer, x: np.ndarray) -> np.ndarray:
        acc = layer.weights[:, :, 0].astype(np.int32) @ x.astype(np.int32) + layer.bias
        return requantize(acc, layer.input_exponent + layer.weight_exponents.astype(np.int32) - layer.output_exponent)

    def step(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features)
        if x.dtype != np.int8 or x.shape != (self.input_channels,):
            raise ValueError(f"Expected INT8 features with shape ({self.input_channels},)")
        hidden_max = min(127, 6 * 2 ** -self.hidden_exponent)
        hidden_min = max(-128, -6 * 2 ** -self.hidden_exponent)
        x = np.clip(self._dense(self.layers[0], x), hidden_min, hidden_max).astype(np.int8)
        for i in range(self.blocks):
            depthwise, pointwise = self.layers[1 + 2*i:3 + 2*i]
            history, position = self.history[i], self.positions[i]
            taps = np.stack((history[position], history[(position + depthwise.dilation) % len(history)], x), axis=1).astype(np.int32)
            acc = (taps * depthwise.weights[:, 0, :].astype(np.int32)).sum(axis=1, dtype=np.int32) + depthwise.bias
            history[position] = x
            self.positions[i] = (position + 1) % len(history)
            y = requantize(acc, depthwise.input_exponent + depthwise.weight_exponents.astype(np.int32) - depthwise.output_exponent)
            y = np.clip(y, 0, hidden_max).astype(np.int8)
            residual = self._dense(pointwise, y)
            x = np.clip(x.astype(np.int32) + residual.astype(np.int32), hidden_min, hidden_max).astype(np.int8)
        return self._dense(self.layers[-1], x)

    def process(self, features: np.ndarray) -> np.ndarray:
        """Process (frames, input_channels) while retaining history across calls."""
        x = np.asarray(features)
        if x.ndim != 2 or x.shape[1] != self.input_channels or x.dtype != np.int8:
            raise ValueError("Expected a (frames, input_channels) INT8 matrix")
        return np.stack([self.step(frame) for frame in x]) if len(x) else np.empty((0, self.output_channels), dtype=np.int8)
