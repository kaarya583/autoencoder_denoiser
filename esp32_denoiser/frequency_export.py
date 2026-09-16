"""Versioned INT8 frequency-U-Net format and a streaming integer reference.

Header (64 bytes): magic[8], version:u32, c1/c2/c3/global:u16,
local/global/layer counts:u8, reserved:u8, input/hidden/output exponents:i8,
reserved:u8, total/dsp_offset/history_bytes/reserved:u32, reserved[20].
Layer (32 bytes): depthwise/kt/kf/stride_f:u8, inputs/outputs/dilation_t/pad_f:u16,
input/output exponents:i8, reserved:u16, weight/bias/exponent offsets:u32,
reserved[4]. All multi-byte values are little endian.

Weights are [out,in/group,time,frequency]. Arrays start on a 16-byte boundary
and have at least 16 readable guard bytes after the weights. This padding is
included in model size. The graph order is fixed and validated on import.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct

import numpy as np
import torch
from torch import nn

from .export import requantize
from .frequency_model import FrequencyUNet, FrequencyUNetConfig
from .quantization import QuantActivation, round_away

MAGIC = b"EDNFQ8\0\0"
VERSION = 1
HEADER = struct.Struct("<8sI4H4B3bx4I20x")
LAYER = struct.Struct("<4B4H2bH3I4x")
DSP_HEADER = struct.Struct("<4s3H2xf")
MAX_HISTORY_BYTES = 65_536


def expected_layer_shapes(config):
    """Topology validation without allocating a floating-point neural network."""
    c1, c2, c3 = config.encoder_channels
    width = config.global_width
    shapes = []

    def dense(inputs, outputs, kf=1, stride=1, padding=0):
        shapes.append((int(inputs == outputs == 1), 1, kf, stride, inputs, outputs, 1, padding))

    def depth(channels, kt=1, kf=5, stride=1, dilation=1, padding=2):
        shapes.append((1, kt, kf, stride, channels, channels, dilation, padding))

    dense(3,c1,5,2,2)
    for inputs, outputs in ((c1,c2),(c2,c3)):
        depth(inputs,stride=2)
        dense(inputs,outputs)
    for dilation in config.local_dilations:
        depth(c3,kt=3,kf=3,dilation=dilation,padding=1)
        dense(c3,c3)
    dense(c3*33,width)
    for dilation in config.global_dilations:
        depth(width,kt=3,kf=1,dilation=dilation,padding=0)
        dense(width,width)
    dense(width,c3*33)
    for inputs, outputs in ((c3,c2),(c2,c1)):
        dense(inputs,outputs)
        depth(outputs)
    depth(c1)
    dense(c1,2)
    return shapes


def graph_modules(model):
    layers = [("stem", model.stem)]
    for name in ("down1", "down2"):
        block = getattr(model, name)
        layers.extend(((name + ".depthwise", block.depthwise), (name + ".pointwise", block.pointwise)))
    for i, block in enumerate(model.local_blocks):
        layers.extend(((f"local_blocks.{i}.depthwise", block.depthwise),
                       (f"local_blocks.{i}.pointwise", block.pointwise)))
    layers.append(("global_in", model.global_in))
    for i, block in enumerate(model.global_blocks):
        layers.extend(((f"global_blocks.{i}.depthwise", block.depthwise),
                       (f"global_blocks.{i}.pointwise", block.pointwise)))
    layers.append(("global_out", model.global_out))
    for name in ("up1", "up2"):
        block = getattr(model, name)
        layers.extend(((name + ".pointwise", block.pointwise), (name + ".depthwise", block.depthwise)))
    layers.extend((("head_dw", model.head_dw), ("head", model.head)))
    return layers


@dataclass
class FrequencyLayer:
    depthwise: int
    kt: int
    kf: int
    stride: int
    inputs: int
    outputs: int
    dilation: int
    padding: int
    input_exponent: int
    output_exponent: int
    weights: np.ndarray
    bias: np.ndarray
    exponents: np.ndarray


def _shape(module):
    if isinstance(module, nn.Conv1d):
        kt, kf, stride, dilation, padding = module.kernel_size[0], 1, 1, module.dilation[0], 0
        if module.stride != (1,) or module.padding != (0,):
            raise ValueError("Global Conv1d requires stride one and explicit causal padding")
    else:
        kt, kf = module.kernel_size
        stride, dilation, padding = module.stride[1], module.dilation[0], module.padding[1]
        if module.stride[0] != 1 or module.dilation[1] != 1 or module.padding[0] != 0:
            raise ValueError("Unsupported temporal/frequency convolution layout")
        if kt == 3:
            padding = 1  # Local blocks explicitly pad frequency as well as time.
    depthwise = int(module.groups == module.in_channels and module.out_channels == module.in_channels)
    if module.groups not in (1, module.in_channels) or (module.groups != 1 and not depthwise):
        raise ValueError("Only dense and multiplier-one depthwise convolutions are supported")
    return depthwise, kt, kf, stride, module.in_channels, module.out_channels, dilation, padding


def _encode_layer(module):
    if not hasattr(module, "weight_exponents") or not getattr(module, "enabled", False):
        raise ValueError("Every convolution must have enabled quantization before export")
    shape = _shape(module)
    weight = module.weight.detach().cpu()
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Nonfinite model weights")
    if weight.ndim == 3:
        weight = weight.unsqueeze(-1)
    exponents = module.weight_exponents().cpu()
    if bool(((exponents < -24) | (exponents > 16)).any()):
        raise ValueError("Unsupported weight exponents")
    ie, oe = int(module.input_exponent), int(module.output_exponent)
    qweight = round_away(weight / (2.0 ** exponents[:, None, None, None])).clamp(-127, 127).to(torch.int8)
    bias = module.bias.detach().cpu() if module.bias is not None else torch.zeros(module.out_channels)
    qbias = round_away(bias.double() / (2.0 ** (exponents.double() + ie)))
    bound = weight[0].numel() * 128 * 127
    if not bool(torch.isfinite(qbias).all()) or bool((qbias.abs() > 2**31 - 1 - bound).any()):
        raise ValueError("Potential INT32 accumulator overflow")
    return FrequencyLayer(*shape, ie, oe, qweight.numpy(), qbias.to(torch.int32).numpy(),
                          exponents.to(torch.int8).numpy())


def export_frequency_model(model, destination, *, max_bytes=99_000):
    if isinstance(model, FrequencyUNet) and not isinstance(model.config, FrequencyUNetConfig):
        raise ValueError(f"{type(model).__name__} INT8 export is not implemented")
    if not isinstance(model, FrequencyUNet):
        raise ValueError("Expected FrequencyUNet")
    layers = [_encode_layer(module) for _, module in graph_modules(model)]
    ie, he, oe = layers[0].input_exponent, layers[0].output_exponent, layers[-1].output_exponent
    if not (-16 <= ie <= 8 and -12 <= he <= 0 and oe == -7):
        raise ValueError("Unsupported activation scales")
    if any(layer.input_exponent != he or layer.output_exponent != (oe if i == len(layers)-1 else he)
           for i, layer in enumerate(layers) if i):
        raise ValueError("All hidden tensors must share one activation grid")
    boundaries = [(model.input_quant, ie), (model.head_quant, oe), (model.global_residual_quant, he),
                  (model.up1.skip_quant, he), (model.up2.skip_quant, he)]
    boundaries += [(block.residual_quant, he) for block in (*model.local_blocks, *model.global_blocks)]
    if any(not isinstance(node, QuantActivation) or not node.enabled or int(node.exponent) != exponent
           for node, exponent in boundaries):
        raise ValueError("Missing or inconsistent quantized addition/input/output boundary")
    if not np.isfinite(model.config.mask_scale) or not 0 < model.config.mask_scale <= 8:
        raise ValueError("Invalid mask scale")
    if len(layers) > 255 or any(v > 65535 for v in (*model.config.encoder_channels, model.config.global_width)):
        raise ValueError("Architecture exceeds format dimensions")
    payload = bytearray(HEADER.size + LAYER.size * len(layers))
    for index, layer in enumerate(layers):
        payload.extend(bytes((-len(payload)) % 16))
        wo = len(payload)
        payload.extend(layer.weights.tobytes())
        payload.extend(bytes(16))
        payload.extend(bytes((-len(payload)) % 4))
        bo = len(payload)
        payload.extend(layer.bias.astype("<i4").tobytes())
        eo = len(payload)
        payload.extend(layer.exponents.tobytes())
        LAYER.pack_into(payload, HEADER.size + index * LAYER.size,
                        layer.depthwise, layer.kt, layer.kf, layer.stride, layer.inputs,
                        layer.outputs, layer.dilation, layer.padding, layer.input_exponent,
                        layer.output_exponent, 0, wo, bo, eo)
    payload.extend(bytes((-len(payload)) % 4))
    dsp_offset = len(payload)
    window = model.window.detach().cpu().numpy().astype("<f4")
    if not np.isfinite(window).all():
        raise ValueError("Nonfinite window")
    payload.extend(DSP_HEADER.pack(b"FDS1", 16000, 512, 256, model.config.mask_scale))
    payload.extend(window.tobytes())
    history_bytes = int(model.model_stats()["neural_state_bytes_int8"])
    if history_bytes > MAX_HISTORY_BYTES:
        raise ValueError("Neural histories exceed the 64 KiB export screening limit")
    HEADER.pack_into(payload, 0, MAGIC, VERSION, *model.config.encoder_channels, model.config.global_width,
                     len(model.local_blocks), len(model.global_blocks), len(layers), 0,
                     ie, he, oe, len(payload), dsp_offset, history_bytes, 0)
    if len(payload) > max_bytes:
        raise ValueError(f"Integer model is {len(payload):,} bytes, exceeding {max_bytes:,}-byte limit")
    # Parse the actual payload before writing it, including topology and guards.
    IntegerFrequencyDenoiser(bytes(payload))
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    report = {"format": "EDNFQ8-v1", "model_bytes": len(payload), "neural_history_bytes": history_bytes,
              "neural_workspace_bytes": "not yet measured in C", "dsp_offset": dsp_offset,
              "dsp_bytes": len(payload)-dsp_offset, "input_exponent": ie, "hidden_exponent": he,
              "output_exponent": oe, "macs_per_second": model.model_stats()["macs_per_second"],
              "weight_bytes": sum(layer.weights.size for layer in layers),
              "bias_bytes": sum(layer.bias.nbytes for layer in layers),
              "parameter_payload_reduction": 11_854_856 / len(payload),
              "precision": "INT8 neural weights/activations/state; INT32 biases/accumulation; external float32 DSP",
              "audio_quality": "unmeasured", "esp32_runtime": "unmeasured"}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(report, indent=2)+"\n")
    return report


class IntegerFrequencyDenoiser:
    """Slow NumPy reference, with exact integer frame state and rescaling."""
    def __init__(self, source):
        self.data = source if isinstance(source, bytes) else Path(source).read_bytes()
        if len(self.data) < HEADER.size:
            raise ValueError("Truncated frequency model")
        values = HEADER.unpack_from(self.data)
        magic, version, c1, c2, c3, width, nl, ng, count, reserved, ie, he, oe, total, dsp, history, reserved2 = values
        if magic != MAGIC or version != VERSION or total != len(self.data) or reserved or reserved2:
            raise ValueError("Invalid frequency model header")
        if min(c1, c2, c3, width, nl, ng) < 1 or count != 13 + 2 * (nl + ng):
            raise ValueError("Invalid graph dimensions")
        if history > MAX_HISTORY_BYTES:
            raise ValueError("Neural histories exceed the supported limit")
        if not (-16 <= ie <= 8 and -12 <= he <= 0 and oe == -7):
            raise ValueError("Unsupported activation scales")
        table_end = HEADER.size + count * LAYER.size
        if not table_end <= dsp or dsp + DSP_HEADER.size + 2048 != len(self.data):
            raise ValueError("Invalid frequency model data range")
        self.input_exponent, self.hidden_exponent, self.output_exponent = ie, he, oe
        self.layers = []
        cursor = table_end
        for i in range(count):
            fields = LAYER.unpack_from(self.data, HEADER.size + i * LAYER.size)
            dw, kt, kf, stride, inputs, outputs, dilation, padding, lie, loe, pad, wo, bo, eo = fields
            if dw not in (0, 1) or min(kt, kf, stride, inputs, outputs, dilation) < 1 or pad:
                raise ValueError("Invalid layer fields")
            elements = outputs * (1 if dw else inputs) * kt * kf
            if dw and inputs != outputs:
                raise ValueError("Depthwise multiplier must be one")
            if wo % 16 or bo % 4 or not (cursor <= wo and wo+elements+16 <= bo and bo+outputs*4 == eo and eo+outputs <= dsp):
                raise ValueError("Invalid layer data or missing SIMD read guards")
            if lie != (ie if i == 0 else he) or loe != (oe if i == count-1 else he):
                raise ValueError("Inconsistent activation scales")
            weights = np.frombuffer(self.data, np.int8, elements, wo).reshape(outputs, 1 if dw else inputs, kt, kf).copy()
            bias = np.frombuffer(self.data, "<i4", outputs, bo).copy()
            exponents = np.frombuffer(self.data, np.int8, outputs, eo).copy()
            bound = np.abs(weights.astype(np.int64)).reshape(outputs,-1).sum(1) * 128
            if np.any((exponents < -24) | (exponents > 16)) or np.any(np.abs(bias.astype(np.int64)) > 2**31-1-bound):
                raise ValueError("Invalid exponents or accumulator range")
            self.layers.append(FrequencyLayer(dw, kt, kf, stride, inputs, outputs, dilation, padding,
                                              lie, loe, weights, bias, exponents))
            cursor = eo + outputs
        local_dilations = tuple(self.layers[5 + 2*i].dilation for i in range(nl))
        global_start = 6 + 2*nl
        global_dilations = tuple(self.layers[global_start + 2*i].dilation for i in range(ng))
        dm, sr, fft, hop, scale = DSP_HEADER.unpack_from(self.data, dsp)
        if dm != b"FDS1" or (sr, fft, hop) != (16000, 512, 256) or not np.isfinite(scale) or not 0 < scale <= 8:
            raise ValueError("Invalid DSP contract")
        self.config = FrequencyUNetConfig((c1,c2,c3), width, local_dilations, global_dilations, mask_scale=scale)
        expected_history = 2 * sum(local_dilations)*c3*33 + 2*sum(global_dilations)*width
        if history != expected_history:
            raise ValueError("Invalid persistent history size")
        # Validate every operator against the fixed graph; do not accept an
        # arbitrary blob's dimensions as an instruction to reshape allocations.
        for layer, expected in zip(self.layers, expected_layer_shapes(self.config), strict=True):
            actual = (layer.depthwise,layer.kt,layer.kf,layer.stride,layer.inputs,layer.outputs,layer.dilation,layer.padding)
            if actual != expected:
                raise ValueError("Layer topology differs from the fixed frequency graph")
        self.window = np.frombuffer(self.data, "<f4", 512, dsp+DSP_HEADER.size).copy()
        if not np.isfinite(self.window).all():
            raise ValueError("Invalid window")
        self.reset()

    def reset(self):
        self.histories = {}
        self.positions = {}

    def _conv(self, index, x):
        layer = self.layers[index]
        if x.dtype != np.int8 or x.ndim != 2 or x.shape[0] != layer.inputs:
            raise ValueError("Invalid integer activation shape")
        if layer.kt == 1:
            taps = [x]
        else:
            length = 2 * layer.dilation
            history = self.histories.get(index)
            if history is None:
                history = np.zeros((length,*x.shape), np.int8)
                self.histories[index] = history
            pos = self.positions.get(index, 0)
            taps = [history[pos].copy(), history[(pos+layer.dilation)%length].copy(), x]
            history[pos] = x
            self.positions[index] = (pos+1)%length
        size = (x.shape[1] + 2*layer.padding-layer.kf)//layer.stride + 1
        acc = np.broadcast_to(layer.bias[:,None], (layer.outputs,size)).copy()
        positions = np.arange(size)*layer.stride-layer.padding
        for t, values in enumerate(taps):
            for f in range(layer.kf):
                source = positions + f
                valid = (source >= 0) & (source < x.shape[1])
                chunk = values[:, source[valid]].astype(np.int32)
                weight = layer.weights[:,:,t,f].astype(np.int32)
                acc[:,valid] += weight[:,0,None] * chunk if layer.depthwise else weight @ chunk
        return requantize(acc, (layer.input_exponent+layer.exponents.astype(np.int32)-layer.output_exponent)[:,None])

    def _bounded(self, x, relu=False):
        maximum = min(127, int(6 * 2.0**-self.hidden_exponent))
        minimum = 0 if relu else max(-128, -int(6 * 2.0**-self.hidden_exponent))
        return np.clip(x, minimum, maximum).astype(np.int8)

    def _add(self, x, y):
        return self._bounded(x.astype(np.int32) + y.astype(np.int32))

    def step(self, features):
        if features.dtype != np.int8 or features.shape != (3,257):
            raise ValueError("Expected one INT8 feature frame [3,257]")
        conv = self._conv
        skip1 = self._bounded(conv(0, features))
        skip2 = self._bounded(conv(2, self._bounded(conv(1, skip1), True)))
        x = self._bounded(conv(4, self._bounded(conv(3, skip2), True)))
        cursor = 5
        for _ in self.config.local_dilations:
            x = self._add(x, conv(cursor+1, self._bounded(conv(cursor,x), True)))
            cursor += 2
        hidden = self._bounded(conv(cursor, x.reshape(-1,1)))
        cursor += 1
        for _ in self.config.global_dilations:
            hidden = self._add(hidden, conv(cursor+1, self._bounded(conv(cursor,hidden),True)))
            cursor += 2
        x = self._add(x, conv(cursor,hidden).reshape(x.shape))
        cursor += 1
        for skip in (skip2,skip1):
            x = np.repeat(x,2,axis=-1)[:,:skip.shape[-1]]
            x = self._add(conv(cursor+1,self._bounded(conv(cursor,x),True)),skip)
            cursor += 2
        x = np.repeat(x,2,axis=-1)[:,:257]
        result = conv(cursor+1,self._bounded(conv(cursor,x),True))
        return result.reshape(514)

    def process(self, features):
        if features.ndim != 3 or features.shape[1:] != (3,257) or features.dtype != np.int8:
            raise ValueError("Expected INT8 [frames,3,257] features")
        return np.stack([self.step(frame) for frame in features]) if len(features) else np.empty((0,514),np.int8)
