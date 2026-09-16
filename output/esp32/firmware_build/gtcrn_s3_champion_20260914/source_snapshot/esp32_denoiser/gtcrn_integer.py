"""Isolated streaming GTCRN integer graph and its independently checked shadow.

Learned neural tensors, weights and persistent histories are INT8. Affine
biases/dots are INT32; GRU, normalization, rescaling and energy use disclosed
wider integer intermediates. FFT, ERB, complex masks and overlap-add are
external float32 DSP. This is a numerical reference, not a model serializer,
full C implementation, QAT trainer or ESP32 performance result.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import io
from itertools import islice
import json
import math
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn

from .experimental_gru import GRUQuantizationConfig, IntegerGRUCell, _round_numpy
from .experimental_gtcrn_ops import (
    ActivationGrid, ActivationObserver, IntegerAffine, IntegerPReLU, IntegerStreamConv,
    attention_energy, attention_product, requantize_activation, residual_add, shuffle_pair, subband_features,
)
from .experimental_layer_norm import integer_layer_norm, quantize_layer_norm
from .gtcrn_erb import SparseGTCRNERB
from .gtcrn_model import GTCRNConfig, GTCRNDenoiser
from .vendor.gtcrn.convolution import StreamConv2d, StreamConvTranspose2d


@dataclass
class _Edge:
    data: object
    name: str


def _view(edge, data):
    return _Edge(data, edge.name)


class _Graph:
    """One-frame control flow shared by the shadow and integer backends.

    The shadow is compared to the untouched converted upstream network;
    integer operator semantics have independent primitive tests.
    """
    def __init__(self, backend):
        self.backend = backend

    def grouped_gru(self, name, x, state=None):
        backend = self.backend
        half = x.data.shape[-1] // 2
        y, histories = [], []
        for index in range(2):
            part = _view(x, x.data[..., index*half:(index+1)*half])
            history = None if state is None else state[..., index*(state.shape[-1]//2):(index+1)*(state.shape[-1]//2)]
            result, history = backend.gru(f"{name}.rnn{index+1}", part, history)
            y.append(result)
            histories.append(history)
        return backend.concat(name + ".output", y, -1), backend.cat(histories, -1)

    def conv_block(self, name, x):
        y, _ = self.backend.affine(name + ".conv", x, name + ".bn")
        return self.backend.activation(name + ".act", y)

    def gt_block(self, name, x, state):
        b = self.backend
        left, right = _view(x, x.data[:, :8]), _view(x, x.data[:, 8:])
        left = _view(left, b.sfe(left.data))
        left, _ = b.affine(name + ".point_conv1", left, name + ".point_bn1")
        left = b.activation(name + ".point_act", left)
        left, state[name + ".depth_conv"] = b.affine(
            name + ".depth_conv", left, name + ".depth_bn", state.get(name + ".depth_conv"))
        left = b.activation(name + ".depth_act", left)
        left, _ = b.affine(name + ".point_conv2", left, name + ".point_bn2")
        energy = b.energy(name + ".tra.energy", left)
        energy = _view(energy, energy.data.swapaxes(1, 2))
        attention, state[name + ".tra.att_gru"] = b.gru(
            name + ".tra.att_gru", energy, state.get(name + ".tra.att_gru"))
        attention, _ = b.affine(name + ".tra.att_fc", attention)
        attention = _view(attention, attention.data.swapaxes(1, 2))
        gate = b.activation(name + ".tra.att_act", attention)
        gate = _view(gate, gate.data[..., None])
        left = b.product(name + ".tra.product", left, gate)
        return b.shuffle(name + ".output", left, right)

    def dual_path(self, name, x, state):
        b = self.backend
        original = _view(x, x.data.swapaxes(1, 3))  # B,F,T,C -> B,T,F,C below.
        original = _view(original, original.data.swapaxes(1, 2))
        frames = original.data.shape[1]
        intra = _view(original, original.data.reshape(frames, 33, 16))
        intra, _ = self.grouped_gru(name + ".intra_rnn", intra)
        intra, _ = b.affine(name + ".intra_fc", intra)
        intra = _view(intra, intra.data.reshape(1, frames, 33, 16))
        intra = b.layer_norm(name + ".intra_ln", intra)
        intra = b.add(name + ".intra_add", original, intra)
        inter = _view(intra, intra.data.swapaxes(1, 2).reshape(33, frames, 16))
        inter, state[name + ".inter_rnn"] = self.grouped_gru(
            name + ".inter_rnn", inter, state.get(name + ".inter_rnn"))
        inter, _ = b.affine(name + ".inter_fc", inter)
        inter = _view(inter, inter.data.reshape(1, 33, frames, 16).swapaxes(1, 2))
        inter = b.layer_norm(name + ".inter_ln", inter)
        output = b.add(name + ".inter_add", intra, inter)
        return _view(output, output.data.swapaxes(1, 2).swapaxes(1, 3))

    def frame(self, features, state=None):
        state = {} if state is None else dict(state)
        b = self.backend
        x = b.input("erb.output", features)
        x = _view(x, b.sfe(x.data))
        skips = []
        for index in range(5):
            name = f"encoder.en_convs.{index}"
            x = self.conv_block(name, x) if index < 2 else self.gt_block(name, x, state)
            skips.append(x)
        for name in ("dpgrnn1", "dpgrnn2"):
            x = self.dual_path(name, x, state)
        for index in range(5):
            name = f"decoder.de_convs.{index}"
            x = b.add(name + ".skip_add", x, skips[4-index])
            x = self.gt_block(name, x, state) if index < 3 else self.conv_block(name, x)
        return x, state


class _FloatBackend:
    def __init__(self, network, observe=False):
        self.modules = dict(network.named_modules())
        self.observers, self.recipes = {}, {}
        self.observe = observe

    def record(self, name, value):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite float-shadow edge: {name}")
        if self.observe:
            self.observers.setdefault(name, ActivationObserver(name)).observe(value)
        return _Edge(value, name)

    def input(self, name, value):
        return self.record(name, value)

    @staticmethod
    def cat(values, axis):
        return torch.cat(values, dim=axis)

    @staticmethod
    def sfe(value):
        return nn.functional.unfold(value, (1, 3), padding=(0, 1)).reshape(1, value.shape[1]*3, value.shape[2], value.shape[3])

    def affine(self, name, x, bn_name=None, state=None):
        self.record(name + ".input", x.data)
        layer = self.modules[name]
        if isinstance(layer, (StreamConv2d, StreamConvTranspose2d)):
            conv = layer.Conv2d if isinstance(layer, StreamConv2d) else layer.ConvTranspose2d
            if state is None:
                state = x.data.new_zeros(1, conv.in_channels, (conv.kernel_size[0]-1)*conv.dilation[0], x.data.shape[-1])
            value, state = layer(x.data, state)
            history = (conv.kernel_size[0]-1)*conv.dilation[0]
            state = state[:, :, -history:].clone() if history else state[:, :, :0].clone()
            kind = "stream"
        else:
            value, kind = layer(x.data), "affine"
        if bn_name:
            value = self.modules[bn_name](value)
        self.recipes[name] = {"kind": kind, "batch_norm": bn_name}
        return self.record(name + ".output", value), state

    def activation(self, name, x):
        self.record(name + ".input", x.data)
        layer = self.modules[name]
        kind = {nn.PReLU: "prelu", nn.Tanh: "tanh", nn.Sigmoid: "sigmoid"}.get(type(layer))
        if kind is None:
            raise ValueError(f"Unsupported GTCRN activation: {name}")
        self.recipes[name] = {"kind": kind}
        return self.record(name + ".output", layer(x.data))

    def gru(self, name, x, state):
        self.record(name + ".input", x.data)
        value, state = self.modules[name](x.data, state)
        self.recipes[name] = {"kind": "gru"}
        return self.record(name + ".output", value), state

    def layer_norm(self, name, x):
        self.record(name + ".input", x.data)
        self.recipes[name] = {"kind": "layer_norm"}
        return self.record(name + ".output", self.modules[name](x.data))

    def energy(self, name, x):
        self.record(name + ".input", x.data)
        self.recipes[name] = {"kind": "energy"}
        return self.record(name + ".output", x.data.square().mean(-1))

    def product(self, name, x, gate):
        self.record(name + ".input", x.data)
        self.recipes[name] = {"kind": "product"}
        return self.record(name + ".output", x.data * gate.data)

    def add(self, name, x, y):
        return self.record(name, x.data + y.data)

    def concat(self, name, values, axis):
        return self.record(name, self.cat([value.data for value in values], axis))

    def shuffle(self, name, left, right):
        value = torch.stack([left.data, right.data], dim=2).reshape(1, 16, left.data.shape[2], left.data.shape[-1])
        return self.record(name, value)


class GTCRNFloatShadow:
    """Frozen independently wired frame graph checked against upstream."""
    def __init__(self, source, *, observe=False):
        _validate_source(source)
        self.streamer = source.make_streaming()
        self.network = self.streamer.network
        self.backend = _FloatBackend(self.network, observe=observe)
        self.graph = _Graph(self.backend)

    @torch.inference_mode()
    def frame(self, spectrum, state=None):
        if spectrum.shape != (1, 257, 1, 2) or spectrum.dtype != torch.float32:
            raise ValueError("Float shadow requires spectrum[1,257,1,2] float32")
        return self.sequence(spectrum, state)

    @torch.inference_mode()
    def sequence(self, spectrum, state=None):
        if spectrum.ndim != 4 or spectrum.shape[:2] != (1, 257) or spectrum.shape[2] < 1 or spectrum.shape[3] != 2 or spectrum.dtype != torch.float32 or spectrum.device.type != "cpu":
            raise ValueError("Float shadow requires spectrum[1,257,T,2] float32")
        real, imag = spectrum[..., 0].swapaxes(1, 2), spectrum[..., 1].swapaxes(1, 2)
        features = torch.stack([(real.square() + imag.square() + 1e-12).sqrt(), real, imag], dim=1)
        features = self.network.erb.bm(features)
        mask, state = self.graph.frame(features, state)
        mask = self.network.erb.bs(mask.data)
        output = torch.stack([real*mask[:, 0]-imag*mask[:, 1], imag*mask[:, 0]+real*mask[:, 1]], -1).swapaxes(1, 2)
        return output, state

    @torch.inference_mode()
    def verify(self, spectra):
        state, original = {}, self.streamer.init_stream_state()
        maximum, state_maximum = 0.0, 0.0
        count = 0
        for spectrum in spectra:
            actual, state = self.frame(spectrum, state)
            expected, original.convolution, original.attention, original.recurrent = self.network(
                spectrum, original.convolution, original.attention, original.recurrent)
            error = float((actual - expected).abs().max())
            maximum = max(maximum, error)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            # Compare all six convolution/attention histories and both
            # temporal dual-path histories, not just their output spectrum.
            for side, prefix, indices, intervals in (
                    (0, "encoder.en_convs", (2, 3, 4), ((0, 2), (2, 6), (6, 16))),
                    (1, "decoder.de_convs", (0, 1, 2), ((6, 16), (2, 6), (0, 2)))):
                for slot, (index, (start, stop)) in enumerate(zip(indices, intervals)):
                    for name, reference in (
                            (f"{prefix}.{index}.depth_conv", original.convolution[side, :, :, start:stop]),
                            (f"{prefix}.{index}.tra.att_gru", original.attention[side, slot])):
                        state_maximum = max(state_maximum, float((state[name] - reference).abs().max()))
                        torch.testing.assert_close(state[name], reference, rtol=2e-5, atol=2e-6)
            for index in range(2):
                history = state[f"dpgrnn{index+1}.inter_rnn"]
                reference = original.recurrent[index]
                state_maximum = max(state_maximum, float((history-reference).abs().max()))
                torch.testing.assert_close(history, reference, rtol=2e-5, atol=2e-6)
            count += 1
        if not count:
            raise ValueError("Float-shadow verification needs at least one frame")
        return {"frames": count, "max_abs_spectral_error": maximum,
                "max_abs_history_error": state_maximum, "histories_checked_per_frame": 14,
                "reference": "untouched pinned converted StreamGTCRN on identical spectra and reset states"}


def _network_frames(waveform, window, config):
    waveform = np.asarray(waveform, np.float32)
    padded = np.pad(waveform, (256, (-len(waveform)) % 256 + 256))
    for offset in range(0, len(padded)-511, 256):
        frame = padded[offset:offset+512]
        spectrum = np.fft.rfft(frame * window).astype(np.complex64)
        scale = np.float32(1)
        if config.normalize_input:
            scale = np.float32(512) * np.sqrt(np.maximum(np.mean(frame*frame, dtype=np.float32), np.float32(config.rms_floor**2)))
        yield frame, spectrum, (spectrum / scale).astype(np.complex64), scale


def _validate_source(model):
    if not isinstance(model, GTCRNDenoiser) or model.training:
        raise ValueError("A CPU float32 eval GTCRNDenoiser is required")
    if any(value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float32)
           for value in model.state_dict().values()):
        raise ValueError("A CPU float32 eval GTCRNDenoiser is required")


def _source_fingerprint(model):
    _validate_source(model)
    digest = hashlib.sha256(json.dumps(asdict(model.config), sort_keys=True).encode())
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _implementation_hashes():
    root = Path(__file__).parent
    names = ("gtcrn_integer.py", "gtcrn_erb.py", "experimental_gtcrn_ops.py", "experimental_gru.py",
             "experimental_layer_norm.py", "gtcrn_model.py", "vendor/gtcrn/network.py",
             "vendor/gtcrn/streaming.py", "vendor/gtcrn/convolution.py", "vendor/gtcrn/convert.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


@torch.inference_mode()
def calibrate_gtcrn_integer(model, batches, *, max_batches=32, base_config=None):
    """Numerical calibration callback; caller owns training-membership audit.

    The real-data factory below reconstructs audited checkpoint training
    representatives instead of trusting a caller-declared split.
    """
    if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1:
        raise ValueError("max_batches must be positive")
    base_config = base_config or GRUQuantizationConfig()
    implementation = _implementation_hashes()
    source_fingerprint = _source_fingerprint(model)
    shadow = GTCRNFloatShadow(model, observe=True)
    window = model.window.detach().numpy().copy()
    probe = []
    counts = {"batches": 0, "utterances": 0, "frames": 0}
    for batch in islice(batches, max_batches):
        if not isinstance(batch, torch.Tensor) or batch.device.type != "cpu" or batch.ndim != 2 or min(batch.shape) < 1 or not batch.is_floating_point() or not bool(torch.isfinite(batch).all()):
            raise ValueError("Calibration batches must be finite CPU waveforms[B,N]")
        for waveform in batch.detach().float().numpy():
            values = []
            for _, _, spectrum, _ in _network_frames(waveform, window, model.config):
                value = torch.view_as_real(torch.from_numpy(spectrum))[None, :, None]
                if len(probe) < 16:
                    probe.append(value.clone())
                values.append(value)
                counts["frames"] += 1
            # Only the float observer is vectorized in time. The deployed
            # numerical graph below still processes one causal frame.
            shadow.sequence(torch.cat(values, dim=2))
            counts["utterances"] += 1
        counts["batches"] += 1
    if not counts["batches"]:
        raise ValueError("At least one training representative is required")
    grids, observations = {}, {}
    for name, observer in shadow.backend.observers.items():
        required = max(max(0., observer.maximum)/127, -min(0., observer.minimum)/128)
        exponent = -16 if required <= 2.0**-16 else math.ceil(math.log2(required))
        grids[name] = ActivationGrid(exponent).exponent
        observations[name] = {"observed_values": observer.count, "observed_minimum": observer.minimum,
                              "observed_maximum": observer.maximum}
    probability_encodings = {}
    for name, recipe in shadow.backend.recipes.items():
        if recipe["kind"] == "gru":
            if grids[name + ".input"] > 0:
                raise ValueError(f"Calibrated GRU edge exceeds supported input grid: {name}; exponent={grids[name+'.input']}")
            grids[name + ".input"] = max(-12, grids[name + ".input"])
        if recipe["kind"] in {"gru", "tanh"}:
            grids[name + ".output"] = -7
        if recipe["kind"] == "sigmoid":
            # A probability code is not a symmetric power-of-two activation.
            # Keep its observed float range, but never label it with a false
            # ordinary grid that a future serializer could accidentally use.
            grids.pop(name + ".output")
            probability_encodings[name + ".output"] = {"kind": "signed_probability", "offset": 128, "denominator": 255}
    # Every grouped GRU concatenates hidden values on the same Q7 grid.
    for name in grids:
        if name.endswith(("intra_rnn.output", "inter_rnn.output")):
            grids[name] = -7
    verify_shadow = GTCRNFloatShadow(model)
    gate = verify_shadow.verify(probe)
    if _source_fingerprint(model) != source_fingerprint:
        raise ValueError("Source model changed during integer calibration")
    if _implementation_hashes() != implementation:
        raise ValueError("Integer graph implementation changed during calibration")
    return grids, {**counts, "method": "float-shadow min/max on caller-supplied training representatives",
                   "scope": "Numerical interface only; training lineage is bound by the checkpoint-derived wrapper",
                   "grids": grids, "observations": observations, "recipes": shadow.backend.recipes,
                   "probability_encodings": probability_encodings,
                   "gru_config": asdict(base_config), "float_shadow": gate,
                   "implementation_sha256": implementation,
                   "numerical_versions": {"numpy": np.__version__, "torch": torch.__version__},
                   "source_state_sha256": source_fingerprint,
                   "fixed_nonlinear_outputs": {"gru_and_tanh": "signed Q7", "sigmoid": "signed code(q+128)/255"}}


class _IntegerBackend:
    def __init__(self, network, calibration):
        self.grids = {name: ActivationGrid(value) for name, value in calibration["grids"].items()}
        self.recipes = calibration["recipes"]
        self.probability_encodings = calibration["probability_encodings"]
        modules = dict(network.named_modules())
        self.ops, self.tables, self.edges = {}, {}, {}
        self.base_gru = GRUQuantizationConfig(**calibration["gru_config"])
        for name, recipe in self.recipes.items():
            kind = recipe["kind"]
            if kind in {"affine", "stream"}:
                cls = IntegerStreamConv if kind == "stream" else IntegerAffine
                self.ops[name] = cls.from_torch(modules[name], self.grid(name + ".input"), self.grid(name + ".output"),
                                                batch_norm=modules.get(recipe.get("batch_norm")))
            elif kind == "prelu":
                self.ops[name] = IntegerPReLU(modules[name], self.grid(name + ".input"), self.grid(name + ".output"))
            elif kind in {"tanh", "sigmoid"}:
                if kind == "tanh" and self.grid(name + ".output").exponent != -7:
                    raise ValueError("Tanh outputs require the fixed Q7 grid")
                if kind == "sigmoid" and self.probability_encodings.get(name + ".output") != {
                        "kind": "signed_probability", "offset": 128, "denominator": 255}:
                    raise ValueError("Sigmoid outputs require explicit endpoint-inclusive probability encoding")
                x = np.arange(-128, 128, dtype=np.float64) * self.grid(name + ".input").scale
                if kind == "tanh":
                    values = _round_numpy(np.tanh(x) * 128).clip(-128, 127)
                else:
                    values = _round_numpy(255 / (1 + np.exp(-np.clip(x, -700, 700)))).clip(0, 255) - 128
                self.tables[name] = values.astype(np.int8)
                self.tables[name].flags.writeable = False
            elif kind == "gru":
                if self.grid(name + ".output").exponent != -7:
                    raise ValueError("GRU outputs require the fixed Q7 grid")
                config = replace(self.base_gru, input_exponent=self.grid(name + ".input").exponent)
                directions = ("forward", "reverse") if modules[name].bidirectional else ("forward",)
                self.ops[name] = tuple(IntegerGRUCell.from_torch(modules[name], config, direction=direction) for direction in directions)
            elif kind == "layer_norm":
                layer = modules[name]
                self.ops[name] = quantize_layer_norm(layer.weight, layer.bias,
                                                     input_exponent=self.grid(name + ".input").exponent,
                                                     output_exponent=self.grid(name + ".output").exponent,
                                                     epsilon=layer.eps)
            elif kind not in {"energy", "product"}:
                raise ValueError(f"Unsupported calibrated recipe: {name}: {kind}")

    def grid(self, name):
        if name in self.probability_encodings:
            raise ValueError(f"Probability edge has no symmetric activation grid: {name}")
        if name not in self.grids:
            raise ValueError(f"Missing calibrated activation grid: {name}")
        return self.grids[name]

    def record(self, name, value):
        if not isinstance(value, np.ndarray) or value.dtype != np.int8 or not value.size:
            raise ValueError(f"Learned neural edge is not nonempty INT8: {name}")
        stats = self.edges.setdefault(name, {"values": 0, "zeros": 0, "rail_values": 0})
        stats["values"] += value.size
        stats["zeros"] += int(np.count_nonzero(value == 0))
        stats["rail_values"] += int(np.count_nonzero((value == -128) | (value == 127)))
        return _Edge(value, name)

    def input(self, name, value):
        return self.record(name, self.grid(name).quantize(value))

    def regrid(self, edge, name):
        result = requantize_activation(edge.data, self.grid(edge.name), self.grid(name))
        return self.record(name, result)

    @staticmethod
    def cat(values, axis):
        return np.concatenate(values, axis=axis)

    sfe = staticmethod(subband_features)

    def affine(self, name, x, bn_name=None, state=None):
        x = self.regrid(x, name + ".input")
        operation = self.ops[name]
        if isinstance(operation, IntegerStreamConv):
            value, state = operation.step(x.data, state)
        else:
            value = operation(x.data)
        return self.record(name + ".output", value), state

    def activation(self, name, x):
        x = self.regrid(x, name + ".input")
        value = self.ops[name](x.data) if name in self.ops else self.tables[name][x.data.astype(np.int16) + 128]
        return self.record(name + ".output", value)

    def gru(self, name, x, state):
        x = self.regrid(x, name + ".input")
        cells = self.ops[name]
        expected = (len(cells), x.data.shape[0], cells[0].hidden_size)
        if state is None:
            state = np.zeros(expected, np.int8)
        if not isinstance(state, np.ndarray) or state.dtype != np.int8 or state.shape != expected:
            raise ValueError(f"Invalid INT8 recurrent history: {name}")
        outputs, states = [], []
        for index, cell in enumerate(cells):
            sequence = x.data if index == 0 else x.data[:, ::-1]
            value, last = cell.process(sequence, state[index])
            outputs.append(value if index == 0 else value[:, ::-1])
            states.append(last)
        return self.record(name + ".output", self.cat(outputs, -1)), np.stack(states, axis=0)

    def layer_norm(self, name, x):
        x = self.regrid(x, name + ".input")
        return self.record(name + ".output", integer_layer_norm(x.data, self.ops[name]))

    def energy(self, name, x):
        x = self.regrid(x, name + ".input")
        return self.record(name + ".output", attention_energy(x.data, self.grid(x.name), self.grid(name + ".output")))

    def product(self, name, x, gate):
        x = self.regrid(x, name + ".input")
        if self.recipes[gate.name.removesuffix(".output")]["kind"] != "sigmoid":
            raise ValueError("Attention multiplication requires endpoint-inclusive probability codes")
        value = attention_product(x.data, gate.data, self.grid(x.name), self.grid(name + ".output"))
        return self.record(name + ".output", value)

    def add(self, name, x, y):
        return self.record(name, residual_add(x.data, y.data, self.grid(x.name), self.grid(y.name), self.grid(name)))

    def concat(self, name, values, axis):
        codes = [requantize_activation(value.data, self.grid(value.name), self.grid(name)) for value in values]
        return self.record(name, self.cat(codes, axis))

    def shuffle(self, name, left, right):
        left = requantize_activation(left.data, self.grid(left.name), self.grid(name))
        right = requantize_activation(right.data, self.grid(right.name), self.grid(name))
        return self.record(name, shuffle_pair(left, right))

    def parameter_accounting(self):
        arrays = []
        for operation in self.ops.values():
            if isinstance(operation, IntegerStreamConv):
                operation = operation.affine
            if isinstance(operation, IntegerAffine):
                arrays.extend((operation.weights, operation.bias, operation.exponents))
            elif isinstance(operation, IntegerPReLU):
                arrays.append(operation.slopes)
            elif isinstance(operation, tuple):
                for cell in operation:
                    arrays.extend((cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh,
                                   cell.exponent_ih, cell.exponent_hh, cell.sigmoid_lut, cell.tanh_lut))
            else:
                arrays.extend((operation.gamma, operation.beta))
        arrays.extend(self.tables.values())
        return {"parameter_and_lut_array_bytes": sum(value.nbytes for value in arrays),
                "int8_array_bytes": sum(value.nbytes for value in arrays if value.dtype == np.int8),
                "int32_bias_array_bytes": sum(value.nbytes for value in arrays if value.dtype == np.int32),
                "scope": "Includes per-cell duplicated LUT arrays currently owned by this Python reference; excludes graph/grid descriptors, Python objects and temporaries"}


@dataclass
class GTCRNIntegerState:
    neural: dict
    analysis: np.ndarray
    synthesis: np.ndarray
    synthesis_weight: np.ndarray


class GTCRNIntegerDenoiser:
    """Frozen NumPy neural graph; no source float network is retained.

    Constructor is a lower-level numerical interface. Use
    ``from_checkpoint_training`` for audited real-data calibration.
    """
    def __init__(self, source, calibration):
        calibration = deepcopy(calibration)
        if not calibration.get("float_shadow", {}).get("frames"):
            raise ValueError("A verified float-shadow calibration is required")
        if calibration.get("source_state_sha256") != _source_fingerprint(source):
            raise ValueError("Calibration source state/config differs from the frozen source")
        if calibration.get("implementation_sha256") != _implementation_hashes():
            raise ValueError("Calibrated integer implementation differs from the current source")
        stream = source.make_streaming()
        self.config = source.config
        self.window = source.window.detach().cpu().numpy().copy()
        self.window.flags.writeable = False
        self.erb = SparseGTCRNERB.from_torch(stream.network.erb)
        self.backend = _IntegerBackend(stream.network, calibration)
        self.graph = _Graph(self.backend)
        self.calibration = calibration
        self.source_sha256 = calibration.get("source_checkpoint_sha256")
        self.state_shapes = {}
        for name, operation in self.backend.ops.items():
            if isinstance(operation, IntegerStreamConv):
                self.state_shapes[name] = (1, operation.affine.input_channels, operation.history_frames, 33)
            elif name.endswith(".tra.att_gru"):
                self.state_shapes[name] = (1, 1, operation[0].hidden_size)
        for name in ("dpgrnn1.inter_rnn", "dpgrnn2.inter_rnn"):
            self.state_shapes[name] = (1, 33, 16)

    def initial_state(self):
        return GTCRNIntegerState({}, np.zeros(256, np.float32), np.zeros(256, np.float32), np.zeros(256, np.float32))

    def spectrum_frame(self, spectrum, neural_state=None):
        spectrum = np.asarray(spectrum)
        if spectrum.dtype != np.complex64 or spectrum.shape != (257,) or not np.isfinite(spectrum).all():
            raise ValueError("Integer graph expects one finite complex64 spectrum with 257 bins")
        if neural_state is not None:
            if not isinstance(neural_state, dict) or (neural_state and neural_state.keys() != self.state_shapes.keys()):
                raise ValueError("Neural state must be empty or contain the complete named histories")
            for name, value in neural_state.items():
                if not isinstance(value, np.ndarray) or value.dtype != np.int8 or value.shape != self.state_shapes[name]:
                    raise ValueError(f"Invalid INT8 neural history: {name}")
        real, imag = spectrum.real, spectrum.imag
        features = np.stack([np.sqrt(real*real + imag*imag + np.float32(1e-12)), real, imag])
        compressed = self.erb.forward(features)[None, :, None]
        mask, state = self.graph.frame(compressed, neural_state)
        mask = mask.data[0, :, 0].astype(np.float32) * np.float32(self.backend.grid(mask.name).scale)
        expanded = self.erb.inverse(mask)
        output = np.empty(257, np.complex64)
        output.real = real*expanded[0] - imag*expanded[1]
        output.imag = imag*expanded[0] + real*expanded[1]
        return output, state

    def stream_step(self, chunk, state=None):
        chunk = np.asarray(chunk)
        if chunk.dtype != np.float32 or chunk.shape != (256,) or not np.isfinite(chunk).all():
            raise ValueError("Streaming audio requires 256 finite float32 samples")
        state = self.initial_state() if state is None else state
        if not isinstance(state, GTCRNIntegerState):
            raise ValueError("Invalid integer denoiser state")
        for value in (state.analysis, state.synthesis, state.synthesis_weight):
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (256,) or not np.isfinite(value).all():
                raise ValueError("Invalid float32 DSP history")
        frame = np.concatenate([state.analysis, chunk])
        spectrum = np.fft.rfft(frame*self.window).astype(np.complex64)
        scale = np.float32(1)
        if self.config.normalize_input:
            scale = np.float32(512) * np.sqrt(np.maximum(np.mean(frame*frame, dtype=np.float32), np.float32(self.config.rms_floor**2)))
        enhanced, neural = self.spectrum_frame((spectrum/scale).astype(np.complex64), state.neural)
        synthesis = np.fft.irfft(enhanced*scale, n=512).astype(np.float32) * self.window
        weight = self.window*self.window
        output = (state.synthesis + synthesis[:256]) / np.maximum(state.synthesis_weight + weight[:256], np.float32(1e-8))
        next_state = GTCRNIntegerState(neural, chunk.copy(), synthesis[256:].copy(), weight[256:].copy())
        if not np.isfinite(output).all():
            raise ValueError("Integer waveform DSP overflowed")
        return output.astype(np.float32), next_state

    def __call__(self, noisy):
        is_torch = isinstance(noisy, torch.Tensor)
        if is_torch:
            if noisy.device.type != "cpu" or not noisy.is_floating_point():
                raise ValueError("Integer waveform evaluation requires CPU floating audio")
            value = noisy.detach().float().numpy()
        else:
            value = np.asarray(noisy)
        if value.dtype != np.float32 or value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
            raise ValueError("Expected finite float32 waveforms[B,N]")
        samples, results = value.shape[-1], []
        for row in value:
            padded = np.pad(row, (0, (-samples) % 256 + 256))
            state, chunks = self.initial_state(), []
            for chunk in padded.reshape(-1, 256):
                output, state = self.stream_step(chunk, state)
                chunks.append(output)
            results.append(np.concatenate(chunks)[256:256+samples])
        result = np.stack(results)
        return torch.from_numpy(result) if is_torch else result

    forward = __call__

    def statistics(self):
        recurrent = {name: [dict(cell.statistics) for cell in cells] for name, cells in self.backend.ops.items() if isinstance(cells, tuple)}
        return {"edges": self.backend.edges, "recurrent": recurrent,
                "interpretation": "Edge rail contact is not necessarily clipping; recurrent logit clipping is counted before saturation"}

    def model_stats(self):
        return {"precision": "INT8 learned weights/activations/histories; INT32 biases/dots, bounded wider integer arithmetic; float32 external DSP",
                **self.backend.parameter_accounting(), "erb_payload_bytes": len(self.erb.data),
                "window_bytes": self.window.nbytes, "neural_state_bytes_int8": sum(math.prod(shape) for shape in self.state_shapes.values()),
                "float32_dsp_history_bytes": 3*256*4,
                "numpy_version": np.__version__,
                "dsp_precision": "float32 arrays; internal FFT precision follows the installed NumPy backend",
                "deployment_status": "NumPy reference only; complete serialization, C graph, QAT and MCU timing remain unimplemented"}


def from_checkpoint_training(checkpoint_path, evaluation_manifest, *, crops=32, seed=483, base_config=None):
    """Load one frozen float artifact; audit/replay its actual training recipe."""
    from .gtcrn_recurrent_probe import calibrate_checkpoint_training
    checkpoint_path = Path(checkpoint_path).resolve()
    contents = checkpoint_path.read_bytes()
    saved = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=False)
    if saved.get("model_kind") != "gtcrn" or saved.get("phase", "float") != "float":
        raise ValueError("Integer GTCRN preparation requires a frozen float GTCRN checkpoint")
    with torch.random.fork_rng(devices=[]):
        source = GTCRNDenoiser(GTCRNConfig.from_checkpoint(saved["model_config"]))
    source.load_state_dict(saved["model"], strict=True)
    source.eval().requires_grad_(False)
    _, calibration = calibrate_checkpoint_training(source, saved, evaluation_manifest, crops=crops,
                                                   seed=seed, base_config=base_config, calibrator=calibrate_gtcrn_integer)
    if checkpoint_path.read_bytes() != contents:
        raise ValueError("Frozen GTCRN checkpoint changed during preparation")
    calibration["source_checkpoint_sha256"] = hashlib.sha256(contents).hexdigest()
    calibration["source_checkpoint"] = str(checkpoint_path)
    result = GTCRNIntegerDenoiser(source, calibration)
    return result, source, calibration


def evaluate_gtcrn_integer(integer, source, manifest, *, max_utterances=4, seed=482, perceptual=False):
    """Score a common seeded development cohort without changing metrics.

    The same seed/cohort rule is used by the GRU and LayerNorm sensitivity
    probes. Float waveform input/output has no independent gain correction
    or PCM16 saturation; the evaluation reports preservation as well as SI-SDR.
    """
    from .evaluate import evaluate_manifest
    from .comparison import compare_evaluations
    from .gtcrn_recurrent_probe import _development_records
    if isinstance(max_utterances, bool) or not isinstance(max_utterances, int) or max_utterances < 1:
        raise ValueError("max_utterances must be a positive integer")
    manifest = Path(manifest)
    contents = manifest.read_bytes()
    records = _development_records(manifest)
    if manifest.read_bytes() != contents:
        raise ValueError("Development manifest changed while loading the cohort")
    indices = sorted(random.Random(seed).sample(range(len(records)), min(max_utterances, len(records))))
    cohort = [records[index] for index in indices]
    with tempfile.TemporaryDirectory(prefix="gtcrn-integer-cohort-") as directory:
        path = Path(directory) / "cohort.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cohort))
        control = evaluate_manifest(source, path, perceptual=perceptual)
        actual = evaluate_manifest(integer, path, perceptual=perceptual)
    if manifest.read_bytes() != contents:
        raise ValueError("Development manifest changed during evaluation")
    return {"scope": "Frozen whole-graph INT8 NumPy sensitivity, float32 DSP; no QAT recovery, official test or MCU claim",
            "manifest_sha256": hashlib.sha256(contents).hexdigest(),
            "cohort": {"seed": seed, "indices": indices, "ids": [row["id"] for row in cohort],
                       "available_utterances": len(records), "selected_utterances": len(cohort)},
            "comparison": compare_evaluations(actual, control), "float": control, "integer": actual}


def main():
    from .evaluate import _json_finite
    from .gtcrn_recurrent_probe import _development_records
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-crops", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=483)
    parser.add_argument("--max-utterances", type=int, default=4)
    parser.add_argument("--seed", type=int, default=482)
    parser.add_argument("--perceptual", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    _development_records(args.manifest)  # Explicitly reject final-test evaluation.
    torch.set_num_threads(args.threads)
    integer, source, calibration = from_checkpoint_training(args.checkpoint, args.manifest,
                                                           crops=args.calibration_crops, seed=args.calibration_seed)
    result = evaluate_gtcrn_integer(integer, source, args.manifest, max_utterances=args.max_utterances,
                                    seed=args.seed, perceptual=args.perceptual)
    result.update(calibration=calibration, model_stats=integer.model_stats(), statistics=integer.statistics())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_finite(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "comparison": result["comparison"]}))


if __name__ == "__main__":
    main()
