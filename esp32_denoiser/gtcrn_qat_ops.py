"""Device-resident trainable operators for the fixed GTCRN integer contract.

Float masterweights and gradient surrogates are training machinery. Forward
matmul operands are exact integer codes with verified FP32 partial-sum bounds;
requantization and LayerNorm use explicit INT64 arithmetic. No CPU snapshots
are taken in forward. TF32/AMP must be disabled by the calling numerical model.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .experimental_gtcrn_ops import IntegerStreamConv
from .experimental_layer_norm import IntegerLayerNormParameters


class ExactForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, surrogate, exact):
        return exact

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def round_away(value):
    # Avoid abs(x)+.5 rounding up a representable value immediately below a
    # half-integer in FP32. The fractional comparison has no such cancellation.
    magnitude = value.abs()
    whole = torch.floor(magnitude)
    return value.sign() * (whole + (magnitude-whole >= .5).to(value.dtype))


def fake_codes(value, exponent, lower=-128, upper=127):
    scaled = value / (2.0**exponent if isinstance(exponent, int) else torch.exp2(exponent))
    exact = round_away(scaled).clamp(lower, upper)
    return ExactForward.apply(scaled, exact)


def quantize(value, exponent):
    return fake_codes(value, exponent) * 2.0**exponent


def shift_integer(value, shift):
    value = value.to(torch.int64)
    shift = torch.as_tensor(shift, device=value.device, dtype=torch.int64)
    left = torch.bitwise_left_shift(value, shift.clamp_min(0))
    right = (-shift).clamp_min(1)
    magnitude = torch.bitwise_right_shift(value.abs() + torch.bitwise_left_shift(torch.ones_like(shift), right-1), right)
    rounded = torch.where(value < 0, -magnitude, magnitude)
    return torch.where(shift >= 0, left, rounded)


def divide_integer(value, denominator):
    rounded = torch.div(value.abs()+denominator//2, denominator, rounding_mode="floor")
    return torch.where(value < 0, -rounded, rounded)


def rational_integer(value, shift, denominator):
    if shift >= 0:
        value = value * 2**shift
    else:
        denominator *= 2**-shift
    return divide_integer(value, denominator).clamp(-128, 127)


def _readonly(value, dtype):
    array = value.detach().cpu().numpy().astype(dtype, copy=True)
    array.flags.writeable = False
    return array


def _folded(source, bn, converted_transpose=False):
    weight = source.weight.double()
    if isinstance(source, nn.ConvTranspose2d):
        groups = source.groups
        weight = weight.reshape(groups, source.in_channels//groups, source.out_channels//groups,
                                *source.kernel_size).permute(0, 2, 1, 3, 4).reshape(
                                    source.out_channels, source.in_channels//groups, *source.kernel_size)
    if converted_transpose:
        weight = weight.flip((-2, -1))
    bias = weight.new_zeros(weight.shape[0]) if source.bias is None else source.bias.double()
    if bn is not None:
        factor = bn.weight.double() / torch.sqrt(bn.running_var.double()+bn.eps)
        weight = weight * factor.reshape((-1,)+(1,)*(weight.ndim-1))
        bias = (bias-bn.running_mean.double())*factor+bn.bias.double()
    return weight, bias


class QATAffine(nn.Module):
    """Fixed-grid quantized affine; source modules supply trainable masters."""
    def __init__(self, snapshot):
        super().__init__()
        self.template = deepcopy(snapshot)
        self.streaming = isinstance(snapshot, IntegerStreamConv)
        self.affine = snapshot.affine if self.streaming else snapshot
        self.register_buffer("weight_exponents", torch.from_numpy(self.affine.exponents.copy()))
        self.input_exponent = self.affine.input_grid.exponent
        self.output_exponent = self.affine.output_grid.exponent

    def parameters_for_forward(self, source, bn=None):
        converted = self.streaming and self.template.transpose
        weight, bias = _folded(source, bn, converted)
        exponents = self.weight_exponents.double()
        scales = exponents.reshape((-1,)+(1,)*(weight.ndim-1))
        w = fake_codes(weight, scales)
        b = fake_codes(bias, exponents+self.input_exponent, -(2**31), 2**31-1)
        bound = b.detach().abs()+128*w.detach().abs().reshape(len(b), -1).sum(1)
        # This includes every signed partial sum, not only the final dot.
        valid = torch.isfinite(weight).all() & torch.isfinite(bias).all() & (bound < 2**24).all()
        if not bool(valid):
            raise ValueError("QAT affine requires finite masters and exact FP32 dot bounds below2^24")
        return w.float(), b.float(), bound

    def forward(self, value, source, bn=None):
        value = quantize(value, self.input_exponent)
        x = value / 2.0**self.input_exponent
        weights, bias, _ = self.parameters_for_forward(source, bn)
        a = self.affine
        if a.kind == "linear":
            accumulator = F.linear(x, weights, bias)
            exponents = self.weight_exponents.to(torch.int64)
        else:
            if self.streaming:
                x = F.pad(x, (0, 0, self.template.history_frames, 0))
                if self.template.transpose:
                    x = F.pad(x, (self.template.frequency_padding, self.template.frequency_padding))
            kernel = weights
            padding = a.padding
            if a.kind == "conv_transpose2d":
                if a.kernel_size[0] != 1 or a.stride[0] != 1:
                    raise ValueError("Pinned nonstreaming transpose affines have temporal kernel/stride1")
                stride = a.stride[1]
                if stride > 1:
                    x = F.pad(x[..., None], (0, stride-1)).reshape(*x.shape[:-1], x.shape[-1]*stride)[..., :-(stride-1)]
                padding = (0, a.kernel_size[1]-1-a.padding[1])
                kernel = weights.flip(-1)
            stride = (1, 1) if a.kind == "conv_transpose2d" else a.stride
            patches = F.unfold(x, a.kernel_size, dilation=a.dilation, padding=padding, stride=stride)
            batch, _, columns = patches.shape
            groups = a.groups
            patches = patches.reshape(batch, groups, -1, columns)
            kernel = kernel.reshape(groups, a.output_channels//groups, -1)
            accumulator = torch.matmul(kernel[None], patches).reshape(batch, a.output_channels, columns) + bias[None, :, None]
            time = (x.shape[-2]+2*padding[0]-a.dilation[0]*(a.kernel_size[0]-1)-1)//stride[0]+1
            frequency = (x.shape[-1]+2*padding[1]-a.dilation[1]*(a.kernel_size[1]-1)-1)//stride[1]+1
            accumulator = accumulator.reshape(batch, a.output_channels, time, frequency)
            exponents = self.weight_exponents.to(torch.int64)[None, :, None, None]
        exact = shift_integer(accumulator.detach().to(torch.int64), self.input_exponent+exponents-self.output_exponent).clamp(-128, 127)
        surrogate = accumulator * torch.exp2((self.input_exponent+exponents).to(torch.float32))
        return ExactForward.apply(surrogate, exact.float()*2.0**self.output_exponent)

    def snapshot(self, source, bn=None):
        result = deepcopy(self.template)
        affine = result.affine if self.streaming else result
        w, b, bound = self.parameters_for_forward(source, bn)
        affine.weights, affine.bias = _readonly(w, np.int8), _readonly(b, np.int32)
        affine.exponents = _readonly(self.weight_exponents, np.int8)
        affine.accumulator_bounds = _readonly(bound, np.int64)
        affine.shifts = self.input_exponent + affine.exponents.astype(np.int64)-self.output_exponent
        affine.shifts.flags.writeable = False
        return result


class QATPReLU(nn.Module):
    def __init__(self, snapshot):
        super().__init__()
        self.template = deepcopy(snapshot)
        self.input_exponent, self.output_exponent, self.exponent = snapshot.input_grid.exponent, snapshot.output_grid.exponent, snapshot.exponent

    def forward(self, value, source):
        if not bool(torch.isfinite(source.weight).all()):
            raise ValueError("QAT PReLU requires finite masterweights")
        value = quantize(value, self.input_exponent)
        slope = fake_codes(source.weight.double(), self.exponent).float()
        q = (value.detach()/2.0**self.input_exponent).to(torch.int64)
        negative = shift_integer(q*slope.detach().to(torch.int64), self.input_exponent+self.exponent-self.output_exponent)
        positive = shift_integer(q, self.input_exponent-self.output_exponent)
        exact = torch.where(q < 0, negative, positive).clamp(-128, 127)
        surrogate = torch.where(value < 0, value*slope*2.0**self.exponent, value)
        return ExactForward.apply(surrogate, exact.float()*2.0**self.output_exponent)

    def snapshot(self, source):
        if not bool(torch.isfinite(source.weight).all()):
            raise ValueError("QAT PReLU requires finite masterweights")
        result = deepcopy(self.template)
        result.slopes = _readonly(fake_codes(source.weight.double(), self.exponent), np.int8)
        return result


class QATLayerNorm(nn.Module):
    """Batched exact INT64 forward with the float LayerNorm derivative.

    A float64 square-root estimate is corrected using integer comparisons.
    Under the primitive's<2^60 squared-denominator bound, it is within one
    integer; correction recovers the exact floor root on CPU/CUDA.
    """
    def __init__(self, snapshot):
        super().__init__()
        self.template = snapshot
        self.input_exponent, self.output_exponent = snapshot.input_exponent, snapshot.output_exponent
        self.gamma_exponent, self.epsilon_code = snapshot.gamma_exponent, snapshot.epsilon_code
        self.shift = snapshot.affine_shift

    def parameters_for_forward(self, source):
        gamma = fake_codes(source.weight.double(), self.gamma_exponent).float()
        beta = fake_codes(source.bias.double(), self.output_exponent, -(2**31), 2**31-1)
        bound = self.template.bounds()
        numerator = (255*527*gamma.detach().abs().max().double()*2**max(0, self.shift)
                     +beta.detach().abs().max()*bound["max_affine_denominator"]+bound["max_affine_denominator"]//2)
        if not bool(torch.isfinite(source.weight).all() & torch.isfinite(source.bias).all() & (numerator < 2**63-1024)):
            raise ValueError("QAT LayerNorm parameters exceed the safe INT64 forward contract")
        return gamma, beta

    def forward(self, value, source):
        value = quantize(value, self.input_exponent)
        gamma, beta = self.parameters_for_forward(source)
        q = (value.detach()/2.0**self.input_exponent).to(torch.int64)
        total = q.sum((-2, -1), keepdim=True)
        variance = 528*q.square().sum((-2, -1), keepdim=True)-total.square()
        squared = (variance << 24)+self.epsilon_code
        root = torch.sqrt(squared.double()).to(torch.int64)
        root = root-(root.square()>squared).to(torch.int64)
        root = root+((root+1).square()<=squared).to(torch.int64)
        denominator = root * 2**max(0, -self.shift)
        numerator = (528*q-total)*gamma.detach().to(torch.int64)*2**max(0, self.shift)+beta.detach().to(torch.int64)*denominator
        exact = divide_integer(numerator, denominator).clamp(-128, 127)
        surrogate = F.layer_norm(value, (33, 16), gamma*2.0**self.gamma_exponent,
                                 beta.float()*2.0**self.output_exponent, eps=self.template.effective_epsilon)
        return ExactForward.apply(surrogate, exact.float()*2.0**self.output_exponent)

    def snapshot(self, source):
        gamma, beta = self.parameters_for_forward(source)
        return IntegerLayerNormParameters(_readonly(gamma, np.int8), _readonly(beta, np.int32),
                                           self.input_exponent, self.gamma_exponent, self.output_exponent,
                                           self.template.epsilon, 24)
