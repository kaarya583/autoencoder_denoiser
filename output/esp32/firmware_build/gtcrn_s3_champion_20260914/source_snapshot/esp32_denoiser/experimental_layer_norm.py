"""Numerical feasibility of GTCRN's framewise INT8 LayerNorm.

This is an isolated Python integer reference and CPU fake-quantization
prototype, not a GTCRN exporter, C implementation, or GPU training kernel.
Input/output and gamma are INT8; beta is INT32 on the output grid. There is
no persistent neural state. Exact reduction, square-root and affine arithmetic
requires bounded INT64 intermediates; Python reference integers are explicitly
checked against those bounds when parameters are constructed.

For N input codes q with scale s, let A=sum(q), B=sum(q*q), D=N*B-A*A.
The normalized value is (N*q-A)/sqrt(D + epsilon*N*N/(s*s)). We encode
the denominator's squared value on a Q24 grid and take its integer square
root. There is no prematurely rounded mean and no separate normalized INT8
activation. The affine transform rounds/saturates only at the output boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


INT64_MAX = 2**63 - 1
# Keeps every square used by the independent integer square-root search safe.
MAX_SQUARED_DENOMINATOR = (2**31 - 1)**2


def _exponent(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or not -16 <= value <= 8:
        raise ValueError(f"{name} must be an integer from -16 to 8")


def _round_divide(numerator: int, denominator: int) -> int:
    """Nearest integer, with exact ties away from zero; denominator > 0."""
    magnitude = (abs(numerator) + denominator // 2) // denominator
    return -magnitude if numerator < 0 else magnitude


@dataclass(frozen=True)
class IntegerLayerNormParameters:
    """Immutable quantized affine arrays and a checked arithmetic contract.

``gamma`` has normalized_shape and dtype int8. ``beta`` has the same shape
and dtype int32, representing beta * 2**output_exponent. Gamma uses one shared
power-of-two exponent per normalization layer. Metadata is not serialized.
"""

    gamma: np.ndarray
    beta: np.ndarray
    input_exponent: int = -4
    gamma_exponent: int = -6
    output_exponent: int = -4
    epsilon: float = 1e-8
    variance_fractional_bits: int = 24
    _epsilon_code: int = field(init=False, repr=False)

    def __post_init__(self):
        gamma, beta = np.asarray(self.gamma), np.asarray(self.beta)
        if (gamma.dtype != np.int8 or beta.dtype != np.int32 or gamma.shape != beta.shape
                or gamma.ndim < 1 or not gamma.size or gamma.size > 1024):
            raise ValueError("Require matching nonempty INT8 gamma / INT32 beta shapes with at most 1024 values")
        for name in ("input_exponent", "gamma_exponent", "output_exponent"):
            _exponent(getattr(self, name), name)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        bits = self.variance_fractional_bits
        if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(0, 25, 2):
            raise ValueError("variance_fractional_bits must be an even integer from 0 to 24")
        for name, array in (("gamma", gamma), ("beta", beta)):
            array = array.copy()
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        value = (Fraction(float(self.epsilon)) * self.size**2 * 2**bits
                 / Fraction(2)**(2 * self.input_exponent))
        object.__setattr__(self, "_epsilon_code", max(1, _round_divide(value.numerator, value.denominator)))
        if self.bounds()["max_squared_denominator"] > MAX_SQUARED_DENOMINATOR:
            raise ValueError("LayerNorm variance/epsilon exceeds the checked INT64 square-root range")
        if self.bounds()["max_rounded_numerator"] > INT64_MAX:
            raise ValueError("LayerNorm affine requantization can overflow signed INT64")

    @property
    def normalized_shape(self) -> tuple[int, ...]:
        return self.gamma.shape

    @property
    def size(self) -> int:
        return self.gamma.size

    @property
    def epsilon_code(self) -> int:
        # Encoded during preparation; actual integer inference never reads the
        # floating metadata or performs epsilon conversion.
        return self._epsilon_code

    @property
    def effective_epsilon(self) -> float:
        return self.epsilon_code * 2.0**(2 * self.input_exponent - self.variance_fractional_bits) / self.size**2

    @property
    def affine_shift(self) -> int:
        return self.variance_fractional_bits // 2 + self.gamma_exponent - self.output_exponent

    def bounds(self) -> dict:
        """Worst-case input-code bounds, including affine rounding addition."""
        variance = (self.size**2 // 4) * 255**2
        squared = variance * 2**self.variance_fractional_bits + self.epsilon_code
        root = math.isqrt(squared)
        shift = self.affine_shift
        denominator = root * 2**max(0, -shift)
        center = 255 * (self.size - 1)
        gamma_peak = int(np.abs(self.gamma.astype(np.int64)).max())
        beta_peak = int(np.abs(self.beta.astype(np.int64)).max())
        numerator = center * gamma_peak * 2**max(0, shift) + beta_peak * denominator
        return {"max_abs_sum": self.size * 128,
                "max_sum_of_squares": self.size * 128**2,
                "max_variance_numerator": variance,
                "max_squared_denominator": squared,
                "max_abs_centered_numerator": center,
                "max_affine_denominator": denominator,
                "max_abs_affine_numerator": numerator,
                "max_rounded_numerator": numerator + denominator // 2}

    def memory_accounting(self) -> dict:
        return {"normalized_shape": list(self.normalized_shape),
                "gamma_bytes_int8": self.gamma.nbytes, "beta_bytes_int32": self.beta.nbytes,
                "parameter_array_bytes": self.gamma.nbytes + self.beta.nbytes,
                "input_output_bytes_per_frame": 2 * self.size,
                "persistent_neural_state_bytes": 0,
                "arithmetic": "INT32 suffices for sums/sumsquares; INT64 variance, square-root search and affine requantization",
                "workspace": "Two passes and bounded scalar arithmetic are possible; Python/Torch object allocation is not a native RAM measurement",
                "metadata": "Scales, epsilon code, dimensions and descriptors require additional bytes; no serialized format exists",
                "four_gtcrn_affine_array_bytes": 4 * (self.gamma.nbytes + self.beta.nbytes)
                if self.normalized_shape == (33, 16) else None}


def quantize_layer_norm(gamma, beta, *, input_exponent: int = -4,
                       output_exponent: int = -4, gamma_exponent: int | None = None,
                       epsilon: float = 1e-8, variance_fractional_bits: int = 24) -> IntegerLayerNormParameters:
    """Prepare a snapshot from finite float affine values; no audio calibration.

An omitted gamma exponent covers its maximum absolute value. An explicit
gamma grid clips weights to INT8; beta overflow fails instead of wrapping.
Input/output scales must ultimately be calibrated on training activations.
"""
    def array(value):
        if isinstance(value, Tensor):
            # NumPy cannot consume BF16 tensors directly. Convert before the
            # handoff, preserving every represented affine value exactly.
            value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
        return np.asarray(value, dtype=np.float64)

    gamma, beta = array(gamma), array(beta)
    if gamma.shape != beta.shape or gamma.ndim < 1 or not gamma.size:
        raise ValueError("gamma and beta must have the same nonempty shape")
    if not np.isfinite(gamma).all() or not np.isfinite(beta).all():
        raise ValueError("LayerNorm affine values must be finite")
    _exponent(input_exponent, "input_exponent")
    _exponent(output_exponent, "output_exponent")
    if gamma_exponent is None:
        peak = float(np.abs(gamma).max())
        # Select the minimum grid before division so subnormal finite weights
        # cannot underflow peak / 127 to zero and enter log2(0).
        gamma_exponent = -16 if peak <= 127 * 2.0**-16 else math.ceil(math.log2(peak / 127))
    _exponent(gamma_exponent, "gamma_exponent")

    def rounded(value):
        return np.sign(value) * np.floor(np.abs(value) + .5)

    gamma_codes = np.clip(rounded(gamma / 2.0**gamma_exponent), -128, 127).astype(np.int8)
    beta_codes = rounded(beta / 2.0**output_exponent)
    if np.any(beta_codes < -(2**31)) or np.any(beta_codes > 2**31 - 1):
        raise ValueError("LayerNorm beta exceeds INT32")
    return IntegerLayerNormParameters(gamma_codes, beta_codes.astype(np.int32), input_exponent,
                                      gamma_exponent, output_exponent, epsilon, variance_fractional_bits)


def _shape(codes, parameters):
    shape = parameters.normalized_shape
    if codes.ndim < len(shape) or tuple(codes.shape[-len(shape):]) != shape or math.prod(codes.shape) == 0:
        raise ValueError(f"Input must have nonempty trailing normalized shape {shape}")


def integer_layer_norm(codes: np.ndarray, parameters: IntegerLayerNormParameters,
                       *, return_diagnostics: bool = False):
    """Actual integer reference; every frame is independent and has INT8 I/O."""
    codes = np.asarray(codes)
    if codes.dtype != np.int8:
        raise ValueError("Integer LayerNorm input must have dtype INT8")
    _shape(codes, parameters)
    frames = codes.reshape(-1, parameters.size)
    output = np.empty_like(frames)
    gamma, beta = parameters.gamma.reshape(-1), parameters.beta.reshape(-1)
    epsilon = parameters.epsilon_code
    shift = parameters.affine_shift
    diagnostics = {"frames": len(frames), "zero_variance_frames": 0,
                   "saturated_outputs": 0, "max_squared_denominator": 0,
                   "max_abs_affine_numerator": 0}
    for index, frame in enumerate(frames):
        total = sum(int(value) for value in frame)
        squares = sum(int(value)**2 for value in frame)
        variance = parameters.size * squares - total**2
        squared = (variance << parameters.variance_fractional_bits) + epsilon
        root = math.isqrt(squared)
        denominator = root << max(0, -shift)
        diagnostics["zero_variance_frames"] += int(variance == 0)
        diagnostics["max_squared_denominator"] = max(diagnostics["max_squared_denominator"], squared)
        for position, value in enumerate(frame):
            center = parameters.size * int(value) - total
            numerator = (center * int(gamma[position]) << max(0, shift)) + int(beta[position]) * denominator
            result = _round_divide(numerator, denominator)
            diagnostics["saturated_outputs"] += int(result < -128 or result > 127)
            diagnostics["max_abs_affine_numerator"] = max(diagnostics["max_abs_affine_numerator"], abs(numerator))
            output[index, position] = min(127, max(-128, result))
    output = output.reshape(codes.shape)
    return (output, diagnostics) if return_diagnostics else output


def _torch_isqrt(values: Tensor) -> Tensor:
    """Exact bounded integer binary search, independent of math.isqrt."""
    low, high = torch.zeros_like(values), torch.full_like(values, 2**31 - 1)
    for _ in range(31):
        middle = (low + high + 1) // 2
        fits = middle * middle <= values
        low = torch.where(fits, middle, low)
        high = torch.where(fits, high, middle - 1)
    return low


def torch_integer_layer_norm(codes: Tensor, parameters: IntegerLayerNormParameters) -> Tensor:
    """All-integer Torch forward with an independently implemented square root.

This CPU numerical helper uses INT64 temporary tensors, not an optimized
activation workspace. Those temporaries are arithmetic intermediates, and
there is no persistent state or floating-point fallback in the integer path.
FakeQuantLayerNorm calls this forward directly; comparing those two alone is
not an independent normalization-correctness check.
"""
    if codes.dtype != torch.int8 or codes.device.type != "cpu":
        raise ValueError("The LayerNorm feasibility helper requires CPU INT8 codes")
    _shape(codes, parameters)
    frame = codes.reshape(-1, parameters.size).to(torch.int64)
    total = frame.sum(-1, keepdim=True)
    variance = parameters.size * (frame * frame).sum(-1, keepdim=True) - total * total
    root = _torch_isqrt((variance << parameters.variance_fractional_bits) + parameters.epsilon_code)
    shift = parameters.affine_shift
    denominator = root << max(0, -shift)
    gamma = torch.tensor(parameters.gamma.reshape(-1).astype(np.int64))
    beta = torch.tensor(parameters.beta.reshape(-1).astype(np.int64))
    numerator = ((parameters.size * frame - total) * gamma << max(0, shift)) + beta * denominator
    result = torch.sign(numerator) * ((numerator.abs() + denominator // 2) // denominator)
    return result.clamp(-128, 127).to(torch.int8).reshape(codes.shape)


class _ExactForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, surrogate, exact):
        return exact

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def _fake_codes(value: Tensor, exponent: int, lower: int, upper: int) -> Tensor:
    clipped = (value.to(torch.float64) / 2.0**exponent).clamp(lower, upper)
    rounded = clipped.sign() * torch.floor(clipped.abs() + .5)
    return clipped + (rounded - clipped).detach()


class FakeQuantLayerNorm(nn.Module):
    """CPU feasibility module: exact integer forward, floating STE gradients.

Master affine parameters are floats as in ordinary QAT. Forward materializes
the same INT8 codes as the independent reference. Backward uses LayerNorm
through quantized inputs/affines and the encoded epsilon, treating output
rounding/saturation as straight-through. This is not full-model GPU QAT.
"""

    def __init__(self, normalized_shape=(33, 16), *, input_exponent=-4,
                 output_exponent=-4, gamma_exponent=-6, epsilon=1e-8,
                 variance_fractional_bits=24):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(tuple(normalized_shape)))
        self.bias = nn.Parameter(torch.zeros(tuple(normalized_shape)))
        self.options = dict(input_exponent=input_exponent, output_exponent=output_exponent,
                            gamma_exponent=gamma_exponent, epsilon=epsilon,
                            variance_fractional_bits=variance_fractional_bits)
        self.snapshot()

    @classmethod
    def from_float(cls, layer: nn.LayerNorm, *, input_exponent=-4, output_exponent=-4,
                   gamma_exponent=None):
        if not layer.elementwise_affine or layer.bias is None:
            raise ValueError("The GTCRN prototype requires LayerNorm gamma and beta")
        parameters = quantize_layer_norm(layer.weight, layer.bias, input_exponent=input_exponent,
                                         output_exponent=output_exponent, gamma_exponent=gamma_exponent,
                                         epsilon=layer.eps)
        result = cls(layer.normalized_shape, input_exponent=input_exponent,
                     output_exponent=output_exponent, gamma_exponent=parameters.gamma_exponent,
                     epsilon=layer.eps).to(dtype=layer.weight.dtype)
        with torch.no_grad():
            result.weight.copy_(layer.weight.detach().cpu())
            result.bias.copy_(layer.bias.detach().cpu())
        return result

    def snapshot(self) -> IntegerLayerNormParameters:
        return quantize_layer_norm(self.weight, self.bias, **self.options)

    def forward(self, value: Tensor) -> Tensor:
        if value.device.type != "cpu" or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("FakeQuantLayerNorm is a CPU prototype requiring finite floating input")
        parameters = self.snapshot()
        _shape(value, parameters)
        codes = _fake_codes(value, parameters.input_exponent, -128, 127)
        gamma = _fake_codes(self.weight, parameters.gamma_exponent, -128, 127) * 2.0**parameters.gamma_exponent
        beta = _fake_codes(self.bias, parameters.output_exponent, -(2**31), 2**31 - 1) * 2.0**parameters.output_exponent
        surrogate = F.layer_norm(codes * 2.0**parameters.input_exponent,
                                 parameters.normalized_shape, gamma, beta,
                                 eps=parameters.effective_epsilon)
        exact = torch_integer_layer_norm(codes.detach().to(torch.int8), parameters).to(torch.float64)
        exact = exact * 2.0**parameters.output_exponent
        return _ExactForward.apply(surrogate, exact).to(value.dtype)


def layer_norm_error_report(codes: np.ndarray, parameters: IntegerLayerNormParameters) -> dict:
    """Per-frame errors against float LayerNorm on the same quantized inputs/affines.

This isolates denominator approximation, encoded epsilon and output rounding
from input/affine quantization. It is not a trained-model audio-quality result.
"""
    actual, diagnostics = integer_layer_norm(codes, parameters, return_diagnostics=True)
    x = codes.reshape(-1, parameters.size).astype(np.float64) * 2.0**parameters.input_exponent
    center = x - x.mean(-1, keepdims=True)
    normalized = center / np.sqrt(np.mean(center**2, axis=-1, keepdims=True) + parameters.epsilon)
    expected = (normalized * parameters.gamma.reshape(-1) * 2.0**parameters.gamma_exponent
                + parameters.beta.reshape(-1) * 2.0**parameters.output_exponent)
    decoded = actual.reshape(x.shape).astype(np.float64) * 2.0**parameters.output_exponent
    error = decoded - expected
    encoded_float = expected / 2.0**parameters.output_exponent
    rounded_float = np.clip(np.sign(encoded_float) * np.floor(np.abs(encoded_float) + .5), -128, 127).astype(np.int8)
    return {"normalized_shape": list(parameters.normalized_shape),
            "frames": len(x), "reference": "float64 LayerNorm; identical quantized inputs and affine parameters",
            "mean_absolute_error": float(np.mean(np.abs(error))),
            "maximum_absolute_error": float(np.max(np.abs(error))),
            "per_frame_mean_absolute_error": np.mean(np.abs(error), axis=-1).tolist(),
            "per_frame_maximum_absolute_error": np.max(np.abs(error), axis=-1).tolist(),
            "rounded_float_code_disagreements": int(np.count_nonzero(actual.reshape(x.shape) != rounded_float)),
            "requested_epsilon": parameters.epsilon, "effective_epsilon": parameters.effective_epsilon,
            "integer_diagnostics": diagnostics, "bounds": parameters.bounds(),
            "memory": parameters.memory_accounting()}
