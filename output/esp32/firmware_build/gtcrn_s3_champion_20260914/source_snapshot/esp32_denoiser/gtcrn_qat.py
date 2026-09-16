"""Isolated batched QAT graph with fixed GTCRN integer deployment grids.

Original float masterweights are trainable; BatchNorm running statistics and
all activation/weight grids remain frozen. Full-sequence affines are batched.
Every recurrent step uses the actual quantized hidden-state contract. Exact
integer forward values use analytic floating gradient surrogates, not a claim
that the deployment graph has floating learned operators. GPU throughput and
quality recovery must be measured separately.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .experimental_gtcrn_ops import IntegerAffine, IntegerPReLU, IntegerStreamConv
from .gtcrn_integer import GTCRNIntegerDenoiser, _Graph, _Edge, _view, _source_fingerprint
from .gtcrn_integer_export import load_gtcrn_integer, pack_gtcrn_integer
from .gtcrn_model import GTCRNConfig, GTCRNDenoiser
from .gtcrn_qat_gru import BatchedQATGRU
from .gtcrn_qat_ops import (ExactForward, QATAffine, QATPReLU, QATLayerNorm, quantize,
                           rational_integer, shift_integer, round_away)


OPTIONS = {"activation_grids": "frozen packed snapshot", "weight_grids": "frozen packed per-output powers of two",
           "batch_norm": "trainable affine, frozen running statistics, differentiable eval fold",
           "dot_forward": "FP32 integer codes with strict partial-sum bound below2^24; TF32 and AMP disabled",
           "wider_forward": "INT64 requantization/GRU gates/LayerNorm with analytic floating backward surrogates"}


def _code_hashes():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ("gtcrn_qat.py", "gtcrn_qat_ops.py", "gtcrn_qat_gru.py")}


class _BatchedGraph(_Graph):
    def dual_path(self, name, x, state):
        b = self.backend
        original = _view(x, x.data.permute(0, 2, 3, 1))
        batch, frames, frequency, channels = original.data.shape
        intra = _view(original, original.data.reshape(batch*frames, frequency, channels))
        intra, _ = self.grouped_gru(name+".intra_rnn", intra)
        intra, _ = b.affine(name+".intra_fc", intra)
        intra = _view(intra, intra.data.reshape(batch, frames, frequency, channels))
        intra = b.layer_norm(name+".intra_ln", intra)
        intra = b.add(name+".intra_add", original, intra)
        inter = _view(intra, intra.data.permute(0, 2, 1, 3).reshape(batch*frequency, frames, channels))
        inter, state[name+".inter_rnn"] = self.grouped_gru(name+".inter_rnn", inter, state.get(name+".inter_rnn"))
        inter, _ = b.affine(name+".inter_fc", inter)
        inter = _view(inter, inter.data.reshape(batch, frequency, frames, channels).permute(0, 2, 1, 3))
        inter = b.layer_norm(name+".inter_ln", inter)
        result = b.add(name+".inter_add", intra, inter)
        return _view(result, result.data.permute(0, 3, 1, 2))


class _Backend:
    def __init__(self, model):
        self.model = model

    def exponent(self, name):
        return self.model.grids[name]

    def input(self, name, value):
        return _Edge(quantize(value, self.exponent(name)), name)

    def regrid(self, edge, name):
        return self.input(name, edge.data)

    @staticmethod
    def cat(values, axis):
        return torch.cat(values, dim=axis)

    @staticmethod
    def sfe(value):
        return F.unfold(value, (1, 3), padding=(0, 1)).reshape(value.shape[0], value.shape[1]*3, value.shape[2], value.shape[3])

    def affine(self, name, x, bn_name=None, state=None):
        x = self.regrid(x, name+".input")
        operation = self.model.operation(name)
        source = self.model.master.core.get_submodule(name)
        bn = self.model.master.core.get_submodule(bn_name) if bn_name else None
        value = operation(x.data, source, bn)
        history = None
        if operation.streaming:
            if state is not None:
                raise ValueError("Batched QAT starts each training clip from reset convolution histories")
            h = operation.template.history_frames
            history = F.pad(x.data, (0, 0, h, 0))[:, :, -h:]
        return _Edge(value, name+".output"), history

    def activation(self, name, x):
        x = self.regrid(x, name+".input")
        kind = self.model.recipes[name]["kind"]
        if kind == "prelu":
            value = self.model.operation(name)(x.data, self.model.master.core.get_submodule(name))
        else:
            code = (x.data.detach()/2.0**self.exponent(x.name)).to(torch.int64)
            table = self.model.lookup(name)
            selected = table[code+128].float()
            if kind == "sigmoid":
                exact, surrogate = (selected+128)/255, torch.sigmoid(x.data)
            else:
                exact, surrogate = selected/128, torch.tanh(x.data)
            value = ExactForward.apply(surrogate, exact)
        return _Edge(value, name+".output")

    def gru(self, name, x, state):
        x = self.regrid(x, name+".input")
        value, state = self.model.operation(name)(x.data, state)
        return _Edge(value, name+".output"), state

    def layer_norm(self, name, x):
        x = self.regrid(x, name+".input")
        return _Edge(self.model.operation(name)(x.data, self.model.master.core.get_submodule(name)), name+".output")

    def energy(self, name, x):
        x = self.regrid(x, name+".input")
        input_exp, output_exp = self.exponent(x.name), self.exponent(name+".output")
        q = (x.data.detach()/2.0**input_exp).to(torch.int64)
        exact = rational_integer(q.square().sum(-1), 2*input_exp-output_exp, q.shape[-1]).float()*2.0**output_exp
        return _Edge(ExactForward.apply(x.data.square().mean(-1), exact), name+".output")

    def product(self, name, x, gate):
        x = self.regrid(x, name+".input")
        input_exp, output_exp = self.exponent(x.name), self.exponent(name+".output")
        q = (x.data.detach()/2.0**input_exp).to(torch.int64)
        probability = round_away(gate.data.detach()*255).to(torch.int64)
        exact = rational_integer(q*probability, input_exp-output_exp, 255).float()*2.0**output_exp
        return _Edge(ExactForward.apply(x.data*gate.data, exact), name+".output")

    def add(self, name, x, y):
        # Exact integer single-rounding sum on the finest participating grid.
        ex, ey, out = self.exponent(x.name), self.exponent(y.name), self.exponent(name)
        finest = min(ex, ey, out)
        left = (x.data.detach()/2.0**ex).to(torch.int64)
        right = (y.data.detach()/2.0**ey).to(torch.int64)
        summed = shift_integer(left, ex-finest)+shift_integer(right, ey-finest)
        exact = shift_integer(summed, finest-out).clamp(-128, 127).float()*2.0**out
        return _Edge(ExactForward.apply(x.data+y.data, exact), name)

    def concat(self, name, values, axis):
        return _Edge(torch.cat([quantize(value.data, self.exponent(name)) for value in values], dim=axis), name)

    def shuffle(self, name, left, right):
        a, b = (quantize(value.data, self.exponent(name)) for value in (left, right))
        return _Edge(torch.stack((a, b), dim=2).reshape(a.shape[0], 16, a.shape[2], a.shape[3]), name)


class GTCRNQAT(nn.Module):
    """Trainable fixed-grid numerical model; use the class factories.

    All clips start from reset histories. Returned state values are floating
    tensors lying exactly on the deployed INT8 grids, preserving autograd.
    The GRU implementation quantizes its actual recurrence at every step.
    """
    model_kind = "gtcrn_qat"

    @classmethod
    def from_float(cls, source, packed_model, *, calibration=None):
        snapshot = load_gtcrn_integer(packed_model, calibration=calibration)
        if not isinstance(source, GTCRNDenoiser) or source.config != snapshot.config or _source_fingerprint(source) != snapshot.packed_metadata["source_state_sha256"]:
            raise ValueError("QAT source model differs from the calibrated packed parent")
        result = cls()
        result._setup(source, snapshot, calibration)
        return result

    def _setup(self, source, snapshot, calibration):
        self.master = deepcopy(source).float()
        for name, parameter in self.master.named_parameters():
            parameter.requires_grad_(not name.startswith("core.erb."))
        self.config = self.master.config
        self.initial_packed_model = snapshot.packed_data
        self.initial_calibration = deepcopy(calibration)
        self.parent_metadata = dict(snapshot.packed_metadata)
        self.qat_options = dict(OPTIONS)
        self.grids = {name: grid.exponent for name, grid in snapshot.backend.grids.items()}
        self.recipes = deepcopy(snapshot.backend.recipes)
        self.base_gru = snapshot.backend.base_gru
        self.names = {name: str(index) for index, name in enumerate(snapshot.backend.ops)}
        self.operations = nn.ModuleDict()
        for name, operation in snapshot.backend.ops.items():
            if isinstance(operation, (IntegerAffine, IntegerStreamConv)):
                module = QATAffine(operation)
            elif isinstance(operation, IntegerPReLU):
                module = QATPReLU(operation)
            elif isinstance(operation, tuple):
                module = BatchedQATGRU.from_float(self.master.core.get_submodule(name), config=operation[0].config, snapshots=operation)
            else:
                module = QATLayerNorm(operation)
            self.operations[self.names[name]] = module
        self.table_names = {name: "lookup_"+str(index) for index, name in enumerate(snapshot.backend.tables)}
        for name, table in snapshot.backend.tables.items():
            self.register_buffer(self.table_names[name], torch.from_numpy(table.copy()))
        self.register_buffer("erb_matrix", torch.from_numpy(snapshot.erb.dense_matrix()))
        self.backend, self.graph = _Backend(self), None
        self.graph = _BatchedGraph(self.backend)
        self.train(source.training)

    def operation(self, name):
        return self.operations[self.names[name]]

    def lookup(self, name):
        return getattr(self, self.table_names[name])

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "master"):
            for layer in self.master.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        return self

    def _precision_guard(self, value):
        if value.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("Exact QAT acceptance requires torch.backends.cuda.matmul.allow_tf32=False")
        if any(parameter.dtype != torch.float32 for parameter in self.master.parameters()):
            raise ValueError("QAT masterweights must stay FP32; AMP is disabled inside the numerical graph")

    def forward_features(self, features, *, return_state=False):
        if features.ndim != 4 or features.shape[1] != 3 or features.shape[-1] != 129 or min(features.shape) < 1 or not features.is_floating_point() or not bool(torch.isfinite(features).all()):
            raise ValueError("QAT features must have shape[B,3,T,129]")
        self._precision_guard(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            mask, state = self.graph.frame(features.float())
        return (mask.data, state) if return_state else mask.data

    def forward(self, waveform):
        if waveform.ndim != 2 or min(waveform.shape) < 1 or not waveform.is_floating_point() or not bool(torch.isfinite(waveform).all()):
            raise ValueError("QAT requires finite waveforms[B,N]")
        self._precision_guard(waveform)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            samples = waveform.shape[-1]
            padded = F.pad(waveform.float(), (256, (-samples)%256+256))
            frames = padded.unfold(-1, 512, 256)
            spectrum = torch.fft.rfft(frames*self.master.window, n=512)
            network_spectrum = spectrum
            if self.config.normalize_input:
                rms = frames.square().mean(-1, keepdim=True).clamp_min(self.config.rms_floor**2).sqrt()
                network_spectrum = spectrum/(512*rms)
            real, imag = network_spectrum.real, network_spectrum.imag
            features = torch.stack(((real.square()+imag.square()+1e-12).sqrt(), real, imag), dim=1)
            features = torch.cat((features[..., :65], F.linear(features[..., 65:], self.erb_matrix)), dim=-1)
            mask = self.forward_features(features)
            mask = torch.cat((mask[..., :65], F.linear(mask[..., 65:], self.erb_matrix.T)), dim=-1)
            enhanced = torch.complex(spectrum.real*mask[:, 0]-spectrum.imag*mask[:, 1], spectrum.imag*mask[:, 0]+spectrum.real*mask[:, 1])
            time = torch.fft.irfft(enhanced, n=512)*self.master.window
            arguments = dict(output_size=(1, padded.shape[-1]), kernel_size=(1, 512), stride=(1, 256))
            output = F.fold(time.transpose(1, 2), **arguments).reshape(waveform.shape[0], -1)
            weights = F.fold(self.master.window.square()[None, :, None].expand(1, 512, frames.shape[1]), **arguments).reshape(1, -1)
            return (output/weights.clamp_min(1e-8))[:, 256:256+samples]

    def float_master(self):
        return deepcopy(self.master).cpu().float().eval()

    def _validate_recipe(self):
        parent = load_gtcrn_integer(self.initial_packed_model, calibration=self.initial_calibration)
        if self.grids != {name: grid.exponent for name, grid in parent.backend.grids.items()} or self.qat_options != OPTIONS:
            raise ValueError("QAT immutable activation/precision recipe changed")
        for name, operation in parent.backend.ops.items():
            current = self.operation(name)
            if isinstance(operation, (IntegerAffine, IntegerStreamConv)):
                expected = operation.affine.exponents if isinstance(operation, IntegerStreamConv) else operation.exponents
                if not np.array_equal(current.weight_exponents.detach().cpu().numpy(), expected):
                    raise ValueError("QAT frozen affine weight grids changed")
            elif isinstance(operation, tuple):
                actual = current.integer_snapshots()
                for left, right in zip(actual, operation):
                    if left.config != right.config or any(not np.array_equal(getattr(left, attr), getattr(right, attr)) for attr in ("exponent_ih", "exponent_hh", "sigmoid_lut", "tanh_lut")):
                        raise ValueError("QAT frozen recurrent recipe changed")
        if any(not np.array_equal(self.lookup(name).detach().cpu().numpy(), table) for name, table in parent.backend.tables.items()):
            raise ValueError("QAT frozen lookup table changed")
        if not np.array_equal(self.erb_matrix.detach().cpu().numpy(), parent.erb.dense_matrix()):
            raise ValueError("QAT fixed ERB transform changed")
        return parent

    def integer_snapshot(self, *, checkpoint_sha256=None, training_metadata=None):
        """Export current codes on the original fixed grids; never recalibrate."""
        parent = self._validate_recipe()
        result = GTCRNIntegerDenoiser.__new__(GTCRNIntegerDenoiser)
        result.__dict__.update({key: value for key, value in parent.__dict__.items() if not key.startswith("packed_")})
        for name, operation in parent.backend.ops.items():
            module, source = self.operation(name), self.master.core.get_submodule(name)
            if isinstance(operation, (IntegerAffine, IntegerStreamConv)):
                bn_name = self.recipes[name].get("batch_norm")
                bn = self.master.core.get_submodule(bn_name) if bn_name else None
                snapshot = module.snapshot(source, bn)
            elif isinstance(operation, tuple):
                snapshot = module.integer_snapshots()
            else:
                snapshot = module.snapshot(source)
            result.backend.ops[name] = snapshot
        result.backend.edges = {}
        result.graph = _Graph(result.backend)
        master_hash = _source_fingerprint(self.float_master())
        result.source_sha256 = checkpoint_sha256
        result.calibration = {"schema": "gtcrn_qat_snapshot_v1", "source_state_sha256": master_hash,
                              "source_checkpoint_sha256": checkpoint_sha256, "implementation_sha256": _code_hashes(),
                              "initial_packed_sha256": self.parent_metadata["packed_sha256"],
                              "initial_calibration_sha256": self.parent_metadata["calibration_sha256"],
                              "initial_calibration": deepcopy(self.initial_calibration),
                              "grids": dict(self.grids), "recipes": deepcopy(self.recipes),
                              "gru_config": asdict(self.base_gru), "probability_encodings": deepcopy(result.backend.probability_encodings),
                              "qat_options": dict(self.qat_options), "training_metadata": deepcopy(training_metadata),
                              "scope": "Updated masterweights on immutable parent grids; initial training calibration remains nested unchanged; no recalibration"}
        pack_gtcrn_integer(result)  # Check complete snapshot/serialization bounds.
        return result

    def checkpoint_payload(self):
        self._validate_recipe()
        return {"model_kind": self.model_kind, "qat_schema": 1, "model_config": asdict(self.config),
                "qat_options": dict(self.qat_options), "initial_packed_model": self.initial_packed_model,
                "initial_calibration": deepcopy(self.initial_calibration), "current_master_sha256": _source_fingerprint(self.float_master()),
                "model": {name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
                          for name, value in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, payload):
        if payload.get("model_kind") != "gtcrn_qat" or payload.get("qat_schema") != 1 or payload.get("qat_options") != OPTIONS:
            raise ValueError("Unknown GTCRN QAT checkpoint schema/recipe")
        parent = load_gtcrn_integer(payload["initial_packed_model"], calibration=payload.get("initial_calibration"))
        config = GTCRNConfig.from_checkpoint(payload["model_config"])
        if config != parent.config:
            raise ValueError("QAT checkpoint configuration differs from packed parent")
        with torch.random.fork_rng(devices=[]):
            master = GTCRNDenoiser(config).eval()
        master.load_state_dict({name.removeprefix("master."): value for name, value in payload["model"].items() if name.startswith("master.")}, strict=True)
        if _source_fingerprint(master) != payload["current_master_sha256"]:
            raise ValueError("QAT checkpoint master-state fingerprint mismatch")
        result = cls()
        result._setup(master, parent, payload.get("initial_calibration"))
        result.load_state_dict(payload["model"], strict=True)
        result._validate_recipe()
        if _source_fingerprint(result.float_master()) != payload["current_master_sha256"]:
            raise ValueError("QAT parameter aliases changed the saved master-state fingerprint")
        return result
