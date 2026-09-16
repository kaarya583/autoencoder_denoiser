"""Isolated calibrated INT8 operators for a future GTCRN streaming graph.

NumPy execution uses INT8 neural tensors, INT32 dots/biases and checked INT64
requantization intermediates. Float work occurs only in parameter preparation
and activation calibration. This is not a graph exporter, full-model QAT,
native implementation or evidence of trained integer speech quality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re

import numpy as np
import torch
from torch import nn

from .experimental_gru import _round_numpy, _shift_numpy
from .vendor.gtcrn.convolution import StreamConv2d, StreamConvTranspose2d


@dataclass(frozen=True)
class ActivationGrid:
    exponent: int

    def __post_init__(self):
        if isinstance(self.exponent, bool) or not isinstance(self.exponent, int) or not -16 <= self.exponent <= 8:
            raise ValueError("Activation exponent must be an integer from -16 to 8")

    @property
    def scale(self):
        return 2.0**self.exponent

    def quantize(self, value):
        value = _float_array(value)
        return _round_numpy(value / self.scale).clip(-128, 127).astype(np.int8)


def _float_array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.floating) or not value.size or not np.isfinite(value).all():
        raise ValueError("Require nonempty finite floating values")
    return value.astype(np.float64)


def _codes(value):
    value = np.asarray(value)
    if value.dtype != np.int8 or not value.size:
        raise ValueError("Neural tensor must be nonempty INT8")
    return value


def _grids(*values):
    if any(not isinstance(value, ActivationGrid) for value in values):
        raise TypeError("Explicit ActivationGrid values are required")


def _readonly(value, dtype):
    value = np.array(value, dtype=dtype, copy=True, order="C")
    value.flags.writeable = False
    return value


class ActivationObserver:
    """Streaming min/max calibration; no waveform retention or model hooks.

    The caller supplies actual training activations and their manifest hash.
    This records the declared lineage; it does not audit dataset membership.
    """

    def __init__(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("An activation boundary name is required")
        self.name, self.count = name, 0
        self.minimum, self.maximum = math.inf, -math.inf
        self.shapes = set()

    def observe(self, value):
        value = _float_array(value)
        self.minimum = min(self.minimum, float(value.min()))
        self.maximum = max(self.maximum, float(value.max()))
        self.count += value.size
        self.shapes.add(tuple(value.shape))

    def finish(self, source_manifest_sha256, *, checkpoint_sha256, split="train"):
        hashes = (source_manifest_sha256, checkpoint_sha256)
        if split != "train" or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
            raise ValueError("Calibration requires a declared train split and manifest/checkpoint SHA256 hashes")
        if not self.count:
            raise ValueError("No activations were observed")
        required = max(max(0., self.maximum) / 127, -min(0., self.minimum) / 128)
        exponent = -16 if required <= 2.0**-16 else math.ceil(math.log2(required))
        grid = ActivationGrid(exponent)  # Fail rather than silently clip a range above the contract.
        return dict(name=self.name, grid=asdict(grid), observed_values=self.count,
                    observed_minimum=self.minimum, observed_maximum=self.maximum,
                    observed_shapes=[list(shape) for shape in sorted(self.shapes)],
                    method="min/max covering all observed values; no percentile clipping",
                    split=split, source_manifest_sha256=source_manifest_sha256,
                    checkpoint_sha256=checkpoint_sha256,
                    lineage_scope="Caller-declared training activations; membership is not audited by this observer")


def folded_parameters(layer, batch_norm=None):
    """Canonical output-major float64 weights/bias with optional eval BN fold.

    Grouped ConvTranspose weights are explicitly changed from
    [input,output/group,kT,kF] to [output,input/group,kT,kF], without flipping
    kernels. Converted StreamConvTranspose2d already contains an ordinary
    Conv2d with reversed kernels; pass that contained layer, not its wrapper.
    """
    if not isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        raise TypeError("Expected Linear, Conv2d or ConvTranspose2d")
    weight = _float_array(layer.weight)
    if isinstance(layer, nn.ConvTranspose2d):
        groups = layer.groups
        weight = weight.reshape(groups, layer.in_channels // groups, layer.out_channels // groups,
                                *layer.kernel_size).transpose(0, 2, 1, 3, 4).reshape(
                                    layer.out_channels, layer.in_channels // groups, *layer.kernel_size)
    outputs = weight.shape[0]
    bias = np.zeros(outputs) if layer.bias is None else _float_array(layer.bias)
    if batch_norm is not None:
        expected = nn.BatchNorm1d if isinstance(layer, nn.Linear) else nn.BatchNorm2d
        if (not isinstance(batch_norm, expected) or layer.training or batch_norm.training
                or not batch_norm.track_running_stats or batch_norm.num_features != outputs):
            raise ValueError("BN folding requires matching eval modules and running statistics")
        mean, variance = _float_array(batch_norm.running_mean), _float_array(batch_norm.running_var)
        if not math.isfinite(batch_norm.eps) or batch_norm.eps <= 0 or np.any(variance < 0):
            raise ValueError("Invalid BatchNorm variance/epsilon")
        gamma = _float_array(batch_norm.weight) if batch_norm.affine else np.ones(outputs)
        beta = _float_array(batch_norm.bias) if batch_norm.affine else np.zeros(outputs)
        scale = gamma / np.sqrt(variance + batch_norm.eps)
        weight = weight * scale.reshape((-1,) + (1,) * (weight.ndim - 1))
        bias = (bias - mean) * scale + beta
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Folded parameters must be finite")
    return weight.copy(), bias.copy()


def _weight_grid(weight):
    peak = np.abs(weight).reshape(len(weight), -1).max(axis=1)
    exponent = np.ceil(np.log2(np.maximum(peak, 127 * 2.0**-20) / 127)).astype(np.int64)
    if np.any(exponent > 4):
        raise ValueError("Weight magnitude exceeds the supported per-output grids")
    return exponent


class IntegerAffine:
    """INT8 Conv/ConvTranspose/Linear with an explicit INT8 output boundary."""

    @classmethod
    def from_torch(cls, layer, input_grid, output_grid, *, batch_norm=None):
        _grids(input_grid, output_grid)
        weight, bias = folded_parameters(layer, batch_norm)
        self = cls()
        self.input_grid, self.output_grid = input_grid, output_grid
        self.kind = "linear" if isinstance(layer, nn.Linear) else (
            "conv_transpose2d" if isinstance(layer, nn.ConvTranspose2d) else "conv2d")
        self.input_channels = layer.in_features if self.kind == "linear" else layer.in_channels
        self.output_channels = weight.shape[0]
        self.groups = 1 if self.kind == "linear" else layer.groups
        if self.kind != "linear":
            if layer.padding_mode != "zeros" or not isinstance(layer.padding, tuple):
                raise ValueError("Only explicit symmetric zero padding is supported")
            self.kernel_size, self.stride, self.padding, self.dilation = (
                tuple(getattr(layer, name)) for name in ("kernel_size", "stride", "padding", "dilation"))
            self.output_padding = tuple(layer.output_padding) if self.kind == "conv_transpose2d" else (0, 0)
        self.exponents = _readonly(_weight_grid(weight), np.int8)
        scales = np.exp2(self.exponents.astype(np.float64))
        packed = _round_numpy(weight / scales.reshape((-1,) + (1,) * (weight.ndim - 1))).clip(-128, 127)
        bias_codes = _round_numpy(bias / (scales * input_grid.scale))
        if np.any(np.abs(bias_codes) > np.iinfo(np.int32).max):
            raise OverflowError("Affine bias exceeds INT32")
        self.weights, self.bias = _readonly(packed, np.int8), _readonly(bias_codes, np.int32)
        bound = np.abs(self.bias.astype(np.int64)) + 128 * np.abs(self.weights.astype(np.int64)).reshape(len(weight), -1).sum(1)
        if np.any(bound > np.iinfo(np.int32).max):
            raise OverflowError("Affine partial dot-plus-bias bound exceeds INT32")
        self.accumulator_bounds = _readonly(bound, np.int64)
        # Allowed exponents imply shifts [-44,28]. INT32 * 2**28 and every
        # rounded right shift are strictly bounded by signed INT64.
        self.shifts = _readonly(input_grid.exponent + self.exponents.astype(np.int64) - output_grid.exponent, np.int64)
        self.batch_norm_folded = batch_norm is not None
        return self

    def __call__(self, codes, *, return_accumulator=False):
        x = _codes(codes)
        if self.kind == "linear":
            if x.ndim < 1 or x.shape[-1] != self.input_channels:
                raise ValueError("Linear input has the wrong trailing feature count")
            accumulator = x.astype(np.int32) @ self.weights.astype(np.int32).T + self.bias
            shifts = self.shifts
        else:
            if x.ndim != 4 or x.shape[1] != self.input_channels:
                raise ValueError("Convolution input must be INT8 [batch,channel,time,frequency]")
            accumulator = self._convolution(x)
            shifts = self.shifts[None, :, None, None]
        output = _shift_numpy(accumulator, shifts).clip(-128, 127).astype(np.int8)
        return (output, accumulator) if return_accumulator else output

    def _convolution(self, x):
        batch, _, time, frequency = x.shape
        kt, kf = self.kernel_size
        st, sf = self.stride
        pt, pf = self.padding
        dt, df = self.dilation
        if self.kind == "conv2d":
            ot, of = (time + 2*pt - dt*(kt-1) - 1)//st + 1, (frequency + 2*pf - df*(kf-1) - 1)//sf + 1
        else:
            ot = (time-1)*st - 2*pt + dt*(kt-1) + self.output_padding[0] + 1
            of = (frequency-1)*sf - 2*pf + df*(kf-1) + self.output_padding[1] + 1
        if min(ot, of) < 1:
            raise ValueError("Convolution has an empty output")
        accumulator = np.broadcast_to(self.bias[None, :, None, None], (batch, self.output_channels, ot, of)).copy()
        inputs, outputs = self.input_channels // self.groups, self.output_channels // self.groups
        wi, xi = self.weights.astype(np.int32), x.astype(np.int32)
        for group in range(self.groups):
            ins, outs = slice(group*inputs, (group+1)*inputs), slice(group*outputs, (group+1)*outputs)
            for t in range(ot if self.kind == "conv2d" else time):
                for f in range(of if self.kind == "conv2d" else frequency):
                    for i in range(kt):
                        for j in range(kf):
                            if self.kind == "conv2d":
                                it, jf = t*st - pt + i*dt, f*sf - pf + j*df
                                if 0 <= it < time and 0 <= jf < frequency:
                                    accumulator[:, outs, t, f] += xi[:, ins, it, jf] @ wi[outs, :, i, j].T
                            else:
                                it, jf = t*st - pt + i*dt, f*sf - pf + j*df
                                if 0 <= it < ot and 0 <= jf < of:
                                    accumulator[:, outs, it, jf] += xi[:, ins, t, f] @ wi[outs, :, i, j].T
        return accumulator

    def metadata(self):
        result = dict(kind=self.kind, input_grid=asdict(self.input_grid), output_grid=asdict(self.output_grid),
                      input_channels=self.input_channels, output_channels=self.output_channels, groups=self.groups,
                      weight_shape=list(self.weights.shape), weight_layout="output,input/group,kT,kF" if self.kind != "linear" else "output,input",
                      weight_exponents=self.exponents.tolist(), batch_norm_folded=self.batch_norm_folded,
                      maximum_abs_accumulator_bound=int(self.accumulator_bounds.max()),
                      parameter_array_bytes=self.weights.nbytes+self.bias.nbytes+self.exponents.nbytes,
                      scope="NumPy operator snapshot; no serialized graph metadata or native workspace measurement")
        if self.kind != "linear":
            result.update({name: list(getattr(self, name)) for name in ("kernel_size", "stride", "padding", "dilation", "output_padding")})
        return result


def _frequency_pad(x, left, right):
    start, end = max(0, -left), x.shape[-1] - max(0, -right)
    if end <= start:
        raise ValueError("Frequency cropping removed every input")
    return np.pad(x[..., start:end], ((0,0),(0,0),(0,0),(max(0,left),max(0,right))))


class IntegerStreamConv:
    """The vendored causal wrapper around an already-converted Conv2d.

    Temporal and frequency axes are already reversed by the upstream converter.
    This adapter only reproduces its frequency upsampling/padding and cache.
    """

    @classmethod
    def from_torch(cls, layer, input_grid, output_grid, *, batch_norm=None):
        if not isinstance(layer, (StreamConv2d, StreamConvTranspose2d)):
            raise TypeError("Expected a vendored streaming convolution wrapper")
        self = cls()
        self.transpose = isinstance(layer, StreamConvTranspose2d)
        conv = layer.ConvTranspose2d if self.transpose else layer.Conv2d
        if conv.stride[0] != 1 or conv.padding[0] != 0:
            raise ValueError("Streaming convolution requires time stride 1 and no time padding")
        self.affine = IntegerAffine.from_torch(conv, input_grid, output_grid, batch_norm=batch_norm)
        self.history_frames = (conv.kernel_size[0]-1)*conv.dilation[0]
        if self.transpose:
            self.frequency_stride = layer.F_stride
            self.frequency_padding = (layer.F_size-1)*layer.F_dilation-layer.F_pad
            if layer.F_stride > 1 and (layer.F_size <= 1 or layer.F_stride > layer.F_size):
                raise ValueError("Unsupported frequency upsampling in the vendored wrapper")
        return self

    def initial_state(self, batch, frequency):
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (batch, frequency)):
            raise ValueError("Positive batch/frequency dimensions are required")
        return np.zeros((batch, self.affine.input_channels, self.history_frames, frequency), dtype=np.int8)

    def step(self, frame, state=None):
        frame = _codes(frame)
        if frame.ndim != 4 or frame.shape[1:3] != (self.affine.input_channels, 1):
            raise ValueError("Streaming frame must be INT8 [batch,channel,1,frequency]")
        expected = (frame.shape[0], frame.shape[1], self.history_frames, frame.shape[3])
        state = self.initial_state(frame.shape[0], frame.shape[3]) if state is None else np.asarray(state)
        if state.dtype != np.int8 or state.shape != expected:
            raise ValueError("Streaming history must have the exact INT8 cache shape")
        joined = np.concatenate([state, frame], axis=2)
        next_state = joined[:, :, 1:].copy()
        if self.transpose:
            stride = self.frequency_stride
            if stride > 1:
                expanded = np.zeros((*joined.shape[:-1], joined.shape[-1]*stride), dtype=np.int8)
                expanded[..., ::stride] = joined
                joined = expanded
            joined = _frequency_pad(joined, self.frequency_padding, self.frequency_padding-(stride-1))
        return self.affine(joined), next_state

    def metadata(self):
        result = self.affine.metadata()
        result.update(streaming=True, history_frames=self.history_frames,
                      history_dtype="INT8 on input_grid; [batch,input_channels,history_frames,frequency]",
                      converted_transpose_wrapper=self.transpose)
        if self.transpose:
            result.update(frequency_upsampling=self.frequency_stride,
                          frequency_padding=[self.frequency_padding, self.frequency_padding-(self.frequency_stride-1)],
                          kernel_conversion="Use the already-converted Conv2d weights; no further flips")
        return result


class IntegerPReLU:
    def __init__(self, layer, input_grid, output_grid, *, channel_axis=1):
        if not isinstance(layer, nn.PReLU):
            raise TypeError("Expected PReLU")
        _grids(input_grid, output_grid)
        slopes = _float_array(layer.weight)
        self.exponent = int(_weight_grid(slopes.reshape(1, -1))[0])
        self.slopes = _readonly(_round_numpy(slopes / 2.0**self.exponent).clip(-128, 127), np.int8)
        self.input_grid, self.output_grid, self.channel_axis = input_grid, output_grid, channel_axis

    def __call__(self, codes):
        x = _codes(codes)
        shape = [1] * x.ndim
        if self.slopes.size != 1:
            if not -x.ndim <= self.channel_axis < x.ndim:
                raise ValueError("PReLU channel axis is outside the input dimensions")
            axis = self.channel_axis % x.ndim
            if x.shape[axis] != self.slopes.size:
                raise ValueError("PReLU channel count does not match its learned slopes")
            shape[axis] = self.slopes.size
        slope = self.slopes.reshape(shape).astype(np.int64)
        positive = _shift_numpy(x, self.input_grid.exponent-self.output_grid.exponent)
        negative = _shift_numpy(x.astype(np.int64)*slope, self.input_grid.exponent+self.exponent-self.output_grid.exponent)
        return np.where(x >= 0, positive, negative).clip(-128, 127).astype(np.int8)

    def metadata(self):
        return dict(kind="prelu", input_grid=asdict(self.input_grid), output_grid=asdict(self.output_grid),
                    slope_exponent=self.exponent, slope_codes=self.slopes.tolist(),
                    slope_array_bytes=self.slopes.nbytes, channel_axis=self.channel_axis,
                    scope="INT8 learned slopes; binary descriptors and native workspace excluded")


def requantize_activation(codes, input_grid, output_grid):
    _grids(input_grid, output_grid)
    return _shift_numpy(_codes(codes), input_grid.exponent-output_grid.exponent).clip(-128, 127).astype(np.int8)


def residual_add(left, right, left_grid, right_grid, output_grid):
    _grids(left_grid, right_grid, output_grid)
    left, right = _codes(left), _codes(right)
    if left.shape != right.shape:
        raise ValueError("Residual inputs must have equal shapes")
    finest = min(left_grid.exponent, right_grid.exponent, output_grid.exponent)
    accumulator = _shift_numpy(left, left_grid.exponent-finest) + _shift_numpy(right, right_grid.exponent-finest)
    return _shift_numpy(accumulator, finest-output_grid.exponent).clip(-128, 127).astype(np.int8)


def _rational_grid(value, shift, denominator):
    value = np.asarray(value, dtype=np.int64)
    if shift >= 0:
        value = value * (np.int64(1) << shift)
    else:
        denominator *= 1 << -shift
    result = (np.abs(value) + denominator//2) // denominator
    return np.where(value < 0, -result, result).clip(-128, 127).astype(np.int8)


def attention_energy(codes, input_grid, output_grid, *, frequency_axis=-1):
    _grids(input_grid, output_grid)
    x = _codes(codes)
    frequency = x.shape[frequency_axis]
    if not 1 <= frequency <= 1024:
        raise ValueError("Attention energy supports 1..1024 frequency bins")
    # Sum <=2**24. The permitted grid shift is [-40,32], so both the scaled
    # numerator and denominator, including rounding, fit signed INT64.
    squares = np.square(x.astype(np.int32)).sum(axis=frequency_axis, dtype=np.int32)
    return _rational_grid(squares, 2*input_grid.exponent-output_grid.exponent, frequency)


def attention_product(codes, probability_codes, input_grid, output_grid):
    _grids(input_grid, output_grid)
    x, gate = np.broadcast_arrays(_codes(codes), _codes(probability_codes))
    # Signed probability code -128..127 maps to 0..255, including both endpoints.
    product = x.astype(np.int64) * (gate.astype(np.int64)+128)
    return _rational_grid(product, input_grid.exponent-output_grid.exponent, 255)


def subband_features(codes, kernel_size=3):
    x = _codes(codes)
    if x.ndim != 4 or kernel_size != 3:
        raise ValueError("The GTCRN SFE contract is [B,C,T,F] with a 3-bin neighborhood")
    padded = np.pad(x, ((0,0),(0,0),(0,0),(1,1)))
    return np.stack([padded[..., j:j+x.shape[-1]] for j in range(3)], axis=2).reshape(
        x.shape[0], x.shape[1]*3, x.shape[2], x.shape[3])


def shuffle_pair(left, right):
    left, right = _codes(left), _codes(right)
    if left.ndim != 4 or left.shape != right.shape:
        raise ValueError("Shuffle inputs must have equal [B,C,T,F] shapes on the same grid")
    return np.stack([left, right], axis=2).reshape(left.shape[0], 2*left.shape[1], *left.shape[2:])
