"""Experimental eight-bit GRU primitive, not a GTCRN exporter/runtime.

The neural tensor contract is INT8 input, matrix weights, sigmoid outputs,
tanh outputs and hidden state, with INT32 biases and dot accumulators.
Sigmoid probabilities use all 256 codes: p=(signed_code+128)/255, so both
endpoints are representable. Candidates/state use signed Q0.7. Affine paths
are aligned as checked INT32 accumulator intermediates before gate products;
INT64 is used for safe shifting and multiplication, never persistent state.

FakeQuantGRUCell has exactly the same forward rounding and table lookup as
IntegerGRUCell. It uses straight-through rounding and analytic sigmoid/tanh
surrogates for gradients. This is a local QAT building block, not evidence of
audio quality, complete-model INT8 support, C parity or MCU performance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GRUQuantizationConfig:
    input_exponent: int = -5
    state_exponent: int = -7
    logit_exponent: int = -4
    accumulator_exponent: int = -12

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) for v in asdict(self).values()):
            raise ValueError("GRU exponents must be integers")
        if not -12 <= self.input_exponent <= 0 or not -6 <= self.logit_exponent <= -2:
            raise ValueError("Unsupported experimental input/logit grid")
        if self.state_exponent != -7 or self.accumulator_exponent != -12:
            raise ValueError("This prototype requires Q0.7 state and Q12 accumulators")


def _round_numpy(value):
    return np.copysign(np.floor(np.abs(value) + 0.5), value)


def _shift_numpy(value, shift):
    """Round signed power-of-two division away from zero at a tie."""
    value, shift = np.broadcast_arrays(np.asarray(value, np.int64), np.asarray(shift, np.int64))
    result = np.empty(value.shape, np.int64)
    left = shift >= 0
    result[left] = value[left] << shift[left]
    right = -shift[~left]
    magnitude = (np.abs(value[~left]) + (np.int64(1) << (right - 1))) >> right
    result[~left] = np.where(value[~left] < 0, -magnitude, magnitude)
    return result


def _divide255_numpy(value):
    value = np.asarray(value, np.int64)
    magnitude = (np.abs(value) + 127) // 255
    return np.where(value < 0, -magnitude, magnitude)


def _tables(config):
    x = np.arange(-128, 128, dtype=np.float64) * 2.0 ** config.logit_exponent
    sigmoid = (_round_numpy(255 / (1 + np.exp(-x))).clip(0, 255) - 128).astype(np.int8)
    tanh = _round_numpy(np.tanh(x) / 2.0 ** config.state_exponent).clip(-128, 127).astype(np.int8)
    return sigmoid, tanh


def _source_parameters(source, direction="forward"):
    if isinstance(source, nn.GRU):
        if source.num_layers != 1 or direction not in {"forward", "reverse"}:
            raise ValueError("Extract exactly one GRU layer/direction")
        if direction == "reverse" and not source.bidirectional:
            raise ValueError("The source GRU has no reverse direction")
        suffix = "_l0" + ("_reverse" if direction == "reverse" else "")
    elif isinstance(source, (nn.GRUCell, FakeQuantGRUCell)):
        if direction != "forward":
            raise ValueError("A GRU cell has no reverse direction")
        suffix = ""
    else:
        raise TypeError("Expected torch GRU/GRUCell or FakeQuantGRUCell")
    if not source.bias:
        raise ValueError("This prototype requires explicit input/recurrent biases")
    return [getattr(source, name + suffix) for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]


def _weight_exponents(weights):
    maximum = weights.detach().double().abs().amax(dim=1).clamp_min(2.0 ** -20)
    return torch.ceil(torch.log2(maximum / 127)).clamp(-20, 4).to(torch.int8)


def _validate_shapes(parameters):
    wi, wh, bi, bh = parameters
    hidden = wh.shape[1] if wh.ndim == 2 else 0
    if (not 1 <= hidden <= 128 or wi.ndim != 2 or not 1 <= wi.shape[1] <= 512 or
            wi.shape[0] != 3 * hidden or wh.shape != (3 * hidden, hidden) or
            bi.shape != (3 * hidden,) or bh.shape != (3 * hidden,)):
        raise ValueError("Invalid or unsupported GRU matrix dimensions")
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("GRU parameters must be finite")


def _i32_checked(value):
    value = np.asarray(value, np.int64)
    if np.any(value < np.iinfo(np.int32).min) or np.any(value > np.iinfo(np.int32).max):
        raise OverflowError("GRU accumulator exceeds INT32")
    return value.astype(np.int32)


class IntegerGRUCell:
    """Immutable parameter snapshot with caller-owned INT8 state [batch,H].

    Use from_torch() to quantize a single direction. step() accepts only INT8
    arrays; quantize_input() defines the external float-to-integer boundary.
    Statistics are diagnostic counters, not recurrent model state.
    """

    @classmethod
    def from_torch(cls, source, config=None, *, direction="forward"):
        if config is None and isinstance(source, FakeQuantGRUCell):
            config = source.config
        self = cls()
        self.config = config or GRUQuantizationConfig()
        parameters = _source_parameters(source, direction)
        _validate_shapes(parameters)
        self.input_size, self.hidden_size = parameters[0].shape[1], parameters[1].shape[1]
        self.weight_ih, self.bias_ih, self.exponent_ih = self._pack(parameters[0], parameters[2], self.config.input_exponent)
        self.weight_hh, self.bias_hh, self.exponent_hh = self._pack(parameters[1], parameters[3], self.config.state_exponent)
        bounds = []
        for weight, bias, exponent, input_exponent in (
                (self.weight_ih, self.bias_ih, self.exponent_ih, self.config.input_exponent),
                (self.weight_hh, self.bias_hh, self.exponent_hh, self.config.state_exponent)):
            raw_bound = np.abs(bias.astype(np.int64)) + 128 * np.abs(weight.astype(np.int64)).sum(axis=1)
            bounds.append(_shift_numpy(raw_bound, input_exponent + exponent.astype(np.int64) - self.config.accumulator_exponent))
        # A gate never amplifies its candidate branch. This bound therefore
        # covers reset/update sums and the gated candidate sum for all inputs.
        _i32_checked(bounds[0] + bounds[1])
        self.sigmoid_lut, self.tanh_lut = _tables(self.config)
        if isinstance(source, FakeQuantGRUCell) and self.config == source.config:
            for name in ("sigmoid_lut", "tanh_lut"):
                if not np.array_equal(getattr(self, name), getattr(source, name).detach().cpu().numpy()):
                    raise ValueError("QAT lookup tables differ from the configured integer contract")
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh", "exponent_ih", "exponent_hh",
                     "sigmoid_lut", "tanh_lut"):
            getattr(self, name).flags.writeable = False
        self.reset_statistics()
        return self

    @staticmethod
    def _pack(weight, bias, input_exponent):
        exponents = _weight_exponents(weight).cpu().numpy()
        scales = np.exp2(exponents.astype(np.float64))
        packed = _round_numpy(weight.detach().double().cpu().numpy() / scales[:, None]).clip(-128, 127).astype(np.int8)
        bias_codes = _round_numpy(bias.detach().double().cpu().numpy() / (scales * 2.0 ** input_exponent))
        if np.any(np.abs(bias_codes) > np.iinfo(np.int32).max):
            raise OverflowError("GRU bias does not fit INT32")
        biases = bias_codes.astype(np.int32)
        bounds = np.abs(biases.astype(np.int64)) + 128 * np.abs(packed.astype(np.int64)).sum(axis=1)
        if np.any(bounds > np.iinfo(np.int32).max):
            raise OverflowError("GRU dot-plus-bias bound exceeds INT32")
        return packed, biases, exponents

    def reset_statistics(self):
        self.statistics = dict(frames=0, input_values=0, input_clipped=0, state_values=0,
                               state_saturated=0, state_zero=0, state_unchanged=0,
                               logit_values=0, logit_clipped=0)

    def quantize_input(self, values):
        values = np.asarray(values)
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError("Input quantization requires finite floating values")
        codes = _round_numpy(values.astype(np.float64) / 2.0 ** self.config.input_exponent)
        self.statistics["input_values"] += codes.size
        self.statistics["input_clipped"] += int(((codes < -128) | (codes > 127)).sum())
        return codes.clip(-128, 127).astype(np.int8)

    def initial_state(self, batch_size=1):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return np.zeros((batch_size, self.hidden_size), dtype=np.int8)

    def _affine(self, inputs, weight, bias, exponent, input_exponent):
        # Export checked each row's worst-case bound, including bias.
        accumulator = inputs.astype(np.int32) @ weight.astype(np.int32).T + bias
        aligned = _shift_numpy(accumulator, input_exponent + exponent.astype(np.int64) - self.config.accumulator_exponent)
        return _i32_checked(aligned)

    def _logit(self, accumulator):
        codes = _shift_numpy(accumulator, self.config.accumulator_exponent - self.config.logit_exponent)
        self.statistics["logit_values"] += codes.size
        self.statistics["logit_clipped"] += int(((codes < -128) | (codes > 127)).sum())
        return codes.clip(-128, 127).astype(np.int8)

    def step(self, inputs, state=None, *, return_trace=False):
        inputs = np.asarray(inputs)
        if inputs.dtype != np.int8 or inputs.ndim != 2 or inputs.shape[1] != self.input_size or not len(inputs):
            raise ValueError("GRU input must be INT8 [positive batch,input_size]")
        state = self.initial_state(len(inputs)) if state is None else np.asarray(state)
        if state.dtype != np.int8 or state.shape != (len(inputs), self.hidden_size):
            raise ValueError("GRU state must be INT8 [batch,hidden_size]")
        a = self._affine(inputs, self.weight_ih, self.bias_ih, self.exponent_ih, self.config.input_exponent)
        b = self._affine(state, self.weight_hh, self.bias_hh, self.exponent_hh, self.config.state_exponent)
        ar, az, an = np.split(a, 3, axis=-1)
        br, bz, bn = np.split(b, 3, axis=-1)
        r_logit = self._logit(_i32_checked(ar.astype(np.int64) + br))
        z_logit = self._logit(_i32_checked(az.astype(np.int64) + bz))
        r = self.sigmoid_lut[r_logit.astype(np.int16) + 128]
        z = self.sigmoid_lut[z_logit.astype(np.int16) + 128]
        candidate_accumulator = _i32_checked(an.astype(np.int64) + _divide255_numpy((r.astype(np.int64) + 128) * bn))
        n_logit = self._logit(candidate_accumulator)
        n = self.tanh_lut[n_logit.astype(np.int16) + 128]
        probability = z.astype(np.int64) + 128
        updated = _divide255_numpy((255 - probability) * n.astype(np.int64) + probability * state.astype(np.int64))
        updated = updated.clip(-128, 127).astype(np.int8)
        self.statistics["frames"] += 1
        self.statistics["state_values"] += updated.size
        self.statistics["state_saturated"] += int(((updated == -128) | (updated == 127)).sum())
        self.statistics["state_zero"] += int((updated == 0).sum())
        self.statistics["state_unchanged"] += int((updated == state).sum())
        if return_trace:
            return updated, {"reset_logit": r_logit, "update_logit": z_logit, "candidate_logit": n_logit,
                             "reset": r, "update": z, "candidate": n, "hidden": updated}
        return updated

    def process(self, inputs, state=None):
        inputs = np.asarray(inputs)
        if inputs.dtype != np.int8 or inputs.ndim != 3 or min(inputs.shape) < 1 or inputs.shape[-1] != self.input_size:
            raise ValueError("GRU sequence must be INT8 [batch,positive time,input_size]")
        outputs = []
        for frame in range(inputs.shape[1]):
            state = self.step(inputs[:, frame], state)
            outputs.append(state)
        return np.stack(outputs, axis=1), state.copy()

    def storage_stats(self):
        arrays = (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh,
                  self.exponent_ih, self.exponent_hh, self.sigmoid_lut, self.tanh_lut)
        probability = self.sigmoid_lut.astype(np.int16) + 128
        return {"parameter_and_table_bytes": sum(x.nbytes for x in arrays),
                "state_bytes_per_stream": self.hidden_size,
                "matrix_macs_per_step": 3 * self.hidden_size * (self.input_size + self.hidden_size),
                "sigmoid_probability_code_range": [int(probability.min()), int(probability.max())],
                "sigmoid_reaches_both_endpoints": bool(probability.min() == 0 and probability.max() == 255),
                "scope": "single GRU direction; Python objects, scratch, binary metadata and full-model operators excluded"}


def _round_torch(value):
    rounded = torch.sign(value) * torch.floor(value.abs() + 0.5)
    return value + (rounded - value).detach()


def _codes_torch(value, exponent):
    return _round_torch(value / 2.0 ** exponent).clamp(-128, 127)


class FakeQuantGRUCell(nn.Module):
    """Differentiable cell whose quantized forward matches IntegerGRUCell.

    Export a frozen snapshot with IntegerGRUCell.from_torch(cell). Scales are
    explicit; per-row weight exponents are recomputed from detached weights.
    Wider float64 tensors here model integer accumulators exactly, not a
    deployment fallback. The eventual C implementation must match the integer
    reference. Forward_float() is the unquantized PyTorch-equivalent control.
    """

    bias = True

    def __init__(self, source, config=None, *, direction="forward"):
        super().__init__()
        if config is None and isinstance(source, FakeQuantGRUCell):
            config = source.config
        self.config = config or GRUQuantizationConfig()
        parameters = _source_parameters(source, direction)
        _validate_shapes(parameters)
        self.input_size, self.hidden_size = parameters[0].shape[1], parameters[1].shape[1]
        for name, parameter in zip(("weight_ih", "weight_hh", "bias_ih", "bias_hh"), parameters):
            setattr(self, name, nn.Parameter(parameter.detach().clone()))
        sigmoid, tanh = _tables(self.config)
        self.register_buffer("sigmoid_lut", torch.from_numpy(sigmoid).to(parameters[0].device))
        self.register_buffer("tanh_lut", torch.from_numpy(tanh).to(parameters[0].device))

    def _check(self, inputs, state):
        if (inputs.ndim != 2 or inputs.shape[-1] != self.input_size or not len(inputs) or
                not inputs.is_floating_point() or not torch.isfinite(inputs).all()):
            raise ValueError("GRU input must be finite floating [batch,input_size]")
        state = inputs.new_zeros((len(inputs), self.hidden_size)) if state is None else state
        if state.shape != (len(inputs), self.hidden_size) or not state.is_floating_point() or not torch.isfinite(state).all():
            raise ValueError("GRU state must be finite floating [batch,hidden_size]")
        return state

    def forward_float(self, inputs, state=None):
        state = self._check(inputs, state)
        a = F.linear(inputs, self.weight_ih, self.bias_ih)
        b = F.linear(state, self.weight_hh, self.bias_hh)
        ar, az, an = a.chunk(3, -1)
        br, bz, bn = b.chunk(3, -1)
        r, z = torch.sigmoid(ar + br), torch.sigmoid(az + bz)
        n = torch.tanh(an + r * bn)
        return (1 - z) * n + z * state

    def _affine(self, inputs, weight, bias, input_exponent):
        exponents = _weight_exponents(weight).to(weight.device).double()
        scale = torch.pow(2.0, exponents)
        weight_codes = _round_torch(weight.double() / scale[:, None]).clamp(-128, 127)
        bias_codes = _round_torch(bias.double() / (scale * 2.0 ** input_exponent))
        accumulator = F.linear(inputs, weight_codes, bias_codes)
        return _round_torch(accumulator * torch.pow(2.0, input_exponent + exponents - self.config.accumulator_exponent))

    def _nonlinear(self, accumulator, kind):
        logits = _round_torch(accumulator * 2.0 ** (self.config.accumulator_exponent - self.config.logit_exponent)).clamp(-128, 127)
        real = logits * 2.0 ** self.config.logit_exponent
        if kind == "sigmoid":
            approximate = torch.sigmoid(real) * 255
            codes = self.sigmoid_lut[logits.long() + 128].double() + 128
        else:
            approximate = torch.tanh(real) / 2.0 ** self.config.state_exponent
            codes = self.tanh_lut[logits.long() + 128].double()
        return approximate + (codes - approximate).detach()

    def forward(self, inputs, state=None):
        state = self._check(inputs, state)
        # Disable caller AMP: float64 represents integer accumulator arithmetic.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            x = _codes_torch(inputs.double(), self.config.input_exponent)
            h = _codes_torch(state.double(), self.config.state_exponent)
            a = self._affine(x, self.weight_ih, self.bias_ih, self.config.input_exponent)
            b = self._affine(h, self.weight_hh, self.bias_hh, self.config.state_exponent)
            ar, az, an = a.chunk(3, -1)
            br, bz, bn = b.chunk(3, -1)
            r = self._nonlinear(ar + br, "sigmoid")
            z = self._nonlinear(az + bz, "sigmoid")
            n = self._nonlinear(an + _round_torch(r * bn / 255), "tanh")
            updated = _round_torch(((255 - z) * n + z * h) / 255).clamp(-128, 127)
            return (updated * 2.0 ** self.config.state_exponent).to(inputs.dtype)

    def process(self, inputs, state=None):
        if inputs.ndim != 3 or min(inputs.shape) < 1 or inputs.shape[-1] != self.input_size:
            raise ValueError("GRU sequence must be floating [batch,positive time,input_size]")
        outputs = []
        for frame in range(inputs.shape[1]):
            state = self(inputs[:, frame], state)
            outputs.append(state)
        return torch.stack(outputs, dim=1), state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=816)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    reports = []
    for hidden in (4, 8, 16):
        source = nn.GRU(8, hidden, batch_first=True).eval()
        integer = IntegerGRUCell.from_torch(source)
        inputs = generator.normal(0, .3, (1, args.frames, 8)).astype(np.float32)
        inputs[:, args.frames // 3:2 * args.frames // 3] = 0
        codes = integer.quantize_input(inputs)
        output, state = integer.process(codes)
        with torch.no_grad():
            control = source(torch.from_numpy(inputs))[0].numpy()
        error = output.astype(np.float32) / 128 - control
        reports.append({"input_size": 8, "hidden_size": hidden, "frames": args.frames,
                        "config": asdict(integer.config), "storage": integer.storage_stats(),
                        "statistics": integer.statistics,
                        "mean_abs_state_error_vs_float": float(np.abs(error).mean()),
                        "max_abs_state_error_vs_float": float(np.abs(error).max()),
                        "final_state_dtype": str(state.dtype)})
    print(json.dumps({"scope": "random untrained GRU primitive; no audio-quality or MCU timing claim",
                      "seed": args.seed, "results": reports}, indent=2))


if __name__ == "__main__":
    main()
