"""Bounded packed GTCRN snapshots, independent of float checkpoints at load time.

The versioned binary contains every neural array, nonlinear table, activation
grid and external DSP constant used by the NumPy reference. Fixed topology
descriptors avoid executable/object serialization. INT8 data and INT32 biases
are little-endian and 16-byte aligned; duplicate immutable arrays share bytes.
Calibration provenance is a separately hash-bound audit, never an inference
dependency. Loading validates numerical safety, not the truth of that audit.
This is a packed NumPy model, not a complete C runtime or an MCU timing result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from .experimental_gru import GRUQuantizationConfig, IntegerGRUCell, _shift_numpy, _tables
from .experimental_gtcrn_ops import ActivationGrid, IntegerAffine, IntegerPReLU, IntegerStreamConv
from .experimental_layer_norm import IntegerLayerNormParameters
from .gtcrn_erb import SparseGTCRNERB
from .gtcrn_integer import GTCRNIntegerDenoiser, _Graph, _IntegerBackend
from .gtcrn_model import GTCRNConfig


MAGIC = b"GTI8PK01"
VERSION, TOPOLOGY_ID, MAX_MODEL_BYTES = 1, 1, 99_000
# prefix44, integrity32, checkpoint32, source-state32, calibration32, zero20.
HEADER = struct.Struct("<8sHHIIIIHHd4b32s32s32s32s20s")
RECORD = struct.Struct("<HBBbbBb8H10I16s")
DIGEST_OFFSET = 44
AFFINE, STREAM, PRELU, GRU, LAYER_NORM, TANH, SIGMOID, ERB, WINDOW = range(1, 10)
KINDS = {"linear": 1, "conv2d": 2, "conv_transpose2d": 3}
PROBABILITY = {"kind": "signed_probability", "offset": 128, "denominator": 255}


@dataclass(frozen=True)
class _Spec:
    name: str
    kind: int
    flags: int = 0
    subtype: int = 0
    dimensions: tuple = (0,) * 8


def _layout():
    """Pinned operator shapes and edge order, independently stated as data.

    Affine dimensions: I,O,groups,kT,kF,strideF,padF,dilationT. All temporal
    strides are1, temporal padding0, frequency dilation1, output padding0.
    Converted transpose stream kernels are already reversed and stay Conv2d.
    """
    records, grids, recipes, probabilities = [], ["erb.output"], {}, {}

    def edges(name, kind, bn=None):
        recipes[name] = {"kind": kind}
        if kind in {"affine", "stream"}:
            recipes[name]["batch_norm"] = bn
        grids.append(name + ".input")
        if kind == "sigmoid":
            probabilities[name + ".output"] = PROBABILITY.copy()
        else:
            grids.append(name + ".output")

    def affine(name, i, o, *, kind="conv2d", groups=1, kt=1, kf=1, stride=1, pad=0, dilation=1, bn=None, stream=False, transpose=False):
        edges(name, "stream" if stream else "affine", bn)
        dims = (i, o, groups, kt, kf, stride, pad, dilation)
        if kind == "linear":
            dims = (i, o, 1, 0, 0, 0, 0, 0)
        records.append(_Spec(name, STREAM if stream else AFFINE, int(bn is not None) | (2 if transpose else 0), KINDS[kind], dims))

    def activation(name, kind="prelu"):
        edges(name, kind)
        records.append(_Spec(name, {"prelu": PRELU, "tanh": TANH, "sigmoid": SIGMOID}[kind]))

    def gru(name, hidden, directions=1):
        edges(name, "gru")
        for direction in range(directions):
            records.append(_Spec(name, GRU, subtype=direction, dimensions=(8, hidden, 0, 0, 0, 0, 0, 0)))

    def conv_block(name, i, o, groups=1, transpose=False, final=False):
        affine(name + ".conv", i, o, kind="conv_transpose2d" if transpose else "conv2d", groups=groups,
               kf=5, stride=2, pad=2, bn=name + ".bn")
        activation(name + ".act", "tanh" if final else "prelu")

    def gt_block(name, dilation, transpose=False):
        kind = "conv_transpose2d" if transpose else "conv2d"
        affine(name + ".point_conv1", 24, 16, kind=kind, bn=name + ".point_bn1")
        activation(name + ".point_act")
        affine(name + ".depth_conv", 16, 16, groups=16, kt=3, kf=3, pad=0 if transpose else 1,
               dilation=dilation, bn=name + ".depth_bn", stream=True, transpose=transpose)
        activation(name + ".depth_act")
        affine(name + ".point_conv2", 16, 8, kind=kind, bn=name + ".point_bn2")
        edges(name + ".tra.energy", "energy")
        gru(name + ".tra.att_gru", 16)
        affine(name + ".tra.att_fc", 16, 8, kind="linear")
        activation(name + ".tra.att_act", "sigmoid")
        edges(name + ".tra.product", "product")
        grids.append(name + ".output")

    conv_block("encoder.en_convs.0", 9, 16)
    conv_block("encoder.en_convs.1", 16, 16, groups=2)
    for index, dilation in enumerate((1, 2, 5), 2):
        gt_block(f"encoder.en_convs.{index}", dilation)
    for block in ("dpgrnn1", "dpgrnn2"):
        for path in ("intra", "inter"):
            for group in (1, 2):
                gru(f"{block}.{path}_rnn.rnn{group}", 4 if path == "intra" else 8, 2 if path == "intra" else 1)
            grids.append(f"{block}.{path}_rnn.output")
            affine(f"{block}.{path}_fc", 16, 16, kind="linear")
            name = f"{block}.{path}_ln"
            edges(name, "layer_norm")
            records.append(_Spec(name, LAYER_NORM, dimensions=(33, 16, 24, 0, 0, 0, 0, 0)))
            grids.append(f"{block}.{path}_add")
    for index in range(5):
        name = f"decoder.de_convs.{index}"
        grids.append(name + ".skip_add")
        if index < 3:
            gt_block(name, (5, 2, 1)[index], transpose=True)
        else:
            conv_block(name, 16, 16 if index == 3 else 2, groups=2 if index == 3 else 1, transpose=True, final=index == 4)
    records += [_Spec("erb", ERB), _Spec("window", WINDOW)]
    return tuple(records), tuple(grids), recipes, probabilities


SPECS, GRID_NAMES, RECIPES, PROBABILITIES = _layout()
AFFINE_NAMES = tuple(spec.name for spec in SPECS if spec.kind in (AFFINE, STREAM))
PRELU_NAMES = tuple(spec.name for spec in SPECS if spec.kind == PRELU)
GRU_DIRECTIONS = tuple((spec.name, spec.subtype) for spec in SPECS if spec.kind == GRU)
LN_NAMES = tuple(spec.name for spec in SPECS if spec.kind == LAYER_NORM)
LUT_NAMES = tuple(spec.name for spec in SPECS if spec.kind in (TANH, SIGMOID))
GRID_OFFSET = HEADER.size
RECORD_OFFSET = (GRID_OFFSET + len(GRID_NAMES) + 15) & ~15
ARRAY_OFFSET = (RECORD_OFFSET + len(SPECS) * RECORD.size + 15) & ~15


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value, *, optional=False):
    if optional and value is None:
        return bytes(32)
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a lowercase SHA256 digest")
    return bytes.fromhex(value)


def _digest(data):
    return hashlib.sha256(data[:DIGEST_OFFSET] + bytes(32) + data[DIGEST_OFFSET+32:]).digest()


def render_packed_tables_header():
    """Reproducible C parser constants; no float nonlinear math at C load time."""
    from fractions import Fraction
    from .experimental_gru import _round_numpy
    lines = ["/* Generated by gtcrn_integer_export.render_packed_tables_header(). */",
             "#ifndef EDNG_PACKED_TABLES_H", "#define EDNG_PACKED_TABLES_H", "#include <stdint.h>"]
    for kind in ("tanh", "sigmoid"):
        lines.append(f"static const uint8_t edng_{kind}_hashes[25][32] = {{")
        for exponent in range(-16, 9):
            x = np.arange(-128, 128, dtype=np.float64) * 2.0**exponent
            values = (_round_numpy(np.tanh(x)*128).clip(-128, 127) if kind == "tanh" else
                      _round_numpy(255/(1+np.exp(-np.clip(x, -700, 700)))).clip(0, 255)-128)
            digest = hashlib.sha256(values.astype(np.int8).tobytes()).digest()
            lines.append("    {" + ",".join(f"0x{byte:02x}" for byte in digest) + "},")
        lines.append("};")
    codes = []
    for exponent in range(-16, 9):
        value = Fraction(1e-8) * 528**2 * 2**24 / Fraction(2)**(2*exponent)
        codes.append(max(1, (value.numerator + value.denominator//2)//value.denominator))
    lines += ["static const uint64_t edng_epsilon_codes[25] = {",
              "    " + ",".join(f"UINT64_C({code})" for code in codes), "};", "#endif", ""]
    return "\n".join(lines)


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _affine_shape(spec):
    i, o, groups, kt, kf, *_ = spec.dimensions
    return (o, i) if spec.subtype == KINDS["linear"] else (o, i // groups, kt, kf)


def _validate_affine(operation, spec):
    i, o, groups, kt, kf, stride, pad, dilation = spec.dimensions
    if (operation.kind != next(kind for kind, code in KINDS.items() if code == spec.subtype)
            or (operation.input_channels, operation.output_channels, operation.groups) != (i, o, groups)
            or operation.batch_norm_folded != bool(spec.flags & 1)):
        raise ValueError(f"Affine topology differs: {spec.name}")
    if operation.kind != "linear" and any(getattr(operation, key) != expected for key, expected in (
            ("kernel_size", (kt, kf)), ("stride", (1, stride)), ("padding", (0, pad)),
            ("dilation", (dilation, 1)), ("output_padding", (0, 0)))):
        raise ValueError(f"Affine kernel/stride differs: {spec.name}")


def _grid_pair(backend, spec):
    if spec.kind in (ERB, WINDOW):
        return 0, 0
    # A sigmoid output is a probability encoding, never an exponent.
    return backend.grid(spec.name + ".input").exponent, 0 if spec.kind == SIGMOID else backend.grid(spec.name + ".output").exponent


def pack_gtcrn_integer(model: GTCRNIntegerDenoiser) -> bytes:
    """Serialize a prepared numerical snapshot; audit lineage stays separate.

    Use from_checkpoint_training before this function for real-data exports.
    The final parser also checks every exported accumulator/normalization bound.
    """
    if not isinstance(model, GTCRNIntegerDenoiser):
        raise TypeError("Expected a prepared GTCRNIntegerDenoiser")
    backend = model.backend
    if set(backend.grids) != set(GRID_NAMES) or backend.recipes != RECIPES or backend.probability_encodings != PROBABILITIES:
        raise ValueError("Prepared graph differs from the pinned packed topology")
    if set(backend.ops) != {spec.name for spec in SPECS if spec.kind in (AFFINE, STREAM, PRELU, GRU, LAYER_NORM)}:
        raise ValueError("Unexpected or missing neural operators")
    if set(backend.tables) != {spec.name for spec in SPECS if spec.kind in (TANH, SIGMOID)}:
        raise ValueError("Unexpected or missing nonlinear tables")
    data, blocks = bytearray(ARRAY_OFFSET), {}

    def array(value, dtype, shape):
        value = np.asarray(value)
        if value.dtype != np.dtype(dtype) or value.shape != shape:
            raise ValueError("Snapshot array dtype/shape differs from pinned topology")
        raw = value.astype(np.dtype(dtype).newbyteorder("<"), copy=False).tobytes(order="C")
        key = (np.dtype(dtype).str, raw)
        if key not in blocks:
            data.extend(bytes((-len(data)) % 16))
            blocks[key] = len(data)
            data.extend(raw)
        return blocks[key]

    for ordinal, spec in enumerate(SPECS):
        refs, auxiliary, tail = [], 0, bytes(16)
        input_exp, output_exp = _grid_pair(backend, spec)
        operation = backend.ops.get(spec.name)
        if spec.kind in (AFFINE, STREAM):
            if spec.kind == STREAM:
                if (not isinstance(operation, IntegerStreamConv) or operation.transpose != bool(spec.flags & 2)
                        or operation.history_frames != 2 * spec.dimensions[7]
                        or operation.transpose and (operation.frequency_stride, operation.frequency_padding) != (1, 1)):
                    raise ValueError("Streaming convolution cache/topology differs")
                operation = operation.affine
            if not isinstance(operation, IntegerAffine):
                raise ValueError("Unexpected affine operator")
            _validate_affine(operation, spec)
            refs = [array(operation.weights, "i1", _affine_shape(spec)),
                    array(operation.bias, "i4", (spec.dimensions[1],)),
                    array(operation.exponents, "i1", (spec.dimensions[1],))]
        elif spec.kind == PRELU:
            if not isinstance(operation, IntegerPReLU) or operation.channel_axis != 1:
                raise ValueError("Unsupported PReLU channel axis")
            auxiliary = operation.exponent
            refs = [array(operation.slopes, "i1", (1,))]
        elif spec.kind == GRU:
            if len(operation) != (2 if ".intra_rnn." in spec.name else 1):
                raise ValueError("Unexpected GRU direction count")
            cell = operation[spec.subtype]
            i, h = spec.dimensions[:2]
            if (cell.input_size, cell.hidden_size) != (i, h) or cell.config != replace(backend.base_gru, input_exponent=input_exp):
                raise ValueError("GRU dimensions/config differ")
            refs = [array(getattr(cell, name), dtype, shape) for name, dtype, shape in (
                ("weight_ih", "i1", (3*h, i)), ("weight_hh", "i1", (3*h, h)),
                ("bias_ih", "i4", (3*h,)), ("bias_hh", "i4", (3*h,)),
                ("exponent_ih", "i1", (3*h,)), ("exponent_hh", "i1", (3*h,)),
                ("sigmoid_lut", "i1", (256,)), ("tanh_lut", "i1", (256,)))]
        elif spec.kind == LAYER_NORM:
            if operation.normalized_shape != (33, 16) or operation.variance_fractional_bits != 24 or operation.epsilon != 1e-8:
                raise ValueError("LayerNorm shape/epsilon contract differs")
            auxiliary, tail = operation.gamma_exponent, struct.pack("<Qd", operation.epsilon_code, operation.epsilon)
            refs = [array(operation.gamma, "i1", (33, 16)), array(operation.beta, "i4", (33, 16))]
        elif spec.kind in (TANH, SIGMOID):
            refs = [array(backend.tables[spec.name], "i1", (256,))]
        elif spec.kind == ERB:
            if model.erb.nonzero_count != 382:
                raise ValueError("Packed pinned ERB requires all382 exact nonzeros")
            refs = [array(np.frombuffer(model.erb.data, dtype=np.uint8), "u1", (2040,))]
        else:
            refs = [array(model.window, "f4", (512,))]
        if spec.kind in (AFFINE, STREAM, PRELU):
            if (operation.input_grid.exponent, operation.output_grid.exponent) != (input_exp, output_exp):
                raise ValueError("Operator grids disagree with graph edges")
        if spec.kind == LAYER_NORM and (operation.input_exponent, operation.output_exponent) != (input_exp, output_exp):
            raise ValueError("LayerNorm grids disagree with graph edges")
        RECORD.pack_into(data, RECORD_OFFSET + ordinal * RECORD.size, ordinal, spec.kind, spec.flags,
                         input_exp, output_exp, spec.subtype, auxiliary, *spec.dimensions, *refs, *([0]*(10-len(refs))), tail)
    data.extend(bytes((-len(data)) % 16))
    if len(data) > MAX_MODEL_BYTES:
        raise ValueError(f"Packed model exceeds {MAX_MODEL_BYTES} bytes")
    data[GRID_OFFSET:GRID_OFFSET+len(GRID_NAMES)] = np.asarray([backend.grid(name).exponent for name in GRID_NAMES], np.int8).tobytes()
    # Preserve the original audit hash when reserializing a model loaded without
    # its optional audit file. Do not manufacture a new training-provenance claim.
    metadata = getattr(model, "packed_metadata", {})
    calibration_sha = metadata.get("calibration_sha256") or hashlib.sha256(_json_bytes(model.calibration)).hexdigest()
    state_sha = metadata.get("source_state_sha256") or model.calibration.get("source_state_sha256")
    HEADER.pack_into(data, 0, MAGIC, VERSION, HEADER.size, len(data), TOPOLOGY_ID, int(model.config.normalize_input), RECORD.size,
                     len(GRID_NAMES), len(SPECS), model.config.rms_floor, *asdict(backend.base_gru).values(),
                     bytes(32), _sha(model.source_sha256, optional=True), _sha(state_sha), _sha(calibration_sha), bytes(20))
    data[DIGEST_OFFSET:DIGEST_OFFSET+32] = _digest(bytes(data))
    result = bytes(data)
    load_gtcrn_integer(result)  # Export fails if any arithmetic or table guard fails.
    return result


class PackedGTCRNIntegerDenoiser(GTCRNIntegerDenoiser):
    """A validated packed model with no retained floating learned network."""

    def model_stats(self):
        return {**super().model_stats(), **self.packed_metadata,
                "deployment_status": "Packed loadable NumPy reference; complete C graph, QAT and MCU timing remain unverified"}


def load_gtcrn_integer(source: bytes | str | Path, *, calibration=None) -> PackedGTCRNIntegerDenoiser:
    """Parse at most99,000 bytes, check every extent and numerical bound.

    Optional calibration must match its SHA256. That checks file association,
    not source membership or authenticity. Nothing is unpickled or evaluated.
    """
    if isinstance(source, bytes):
        data = source
    else:
        with Path(source).open("rb") as handle:
            data = handle.read(MAX_MODEL_BYTES + 1)
    if not HEADER.size <= len(data) <= MAX_MODEL_BYTES:
        raise ValueError("Packed GTCRN size is outside the bounded format")
    (magic, version, header_size, size, topology, flags, record_size, grids, records, rms_floor,
     input_exp, state_exp, logit_exp, accumulator_exp, digest, checkpoint_sha, state_sha, calibration_sha, reserved) = HEADER.unpack_from(data)
    if (magic != MAGIC or version != VERSION or header_size != HEADER.size or size != len(data) or topology != TOPOLOGY_ID
            or flags not in (0, 1) or record_size != RECORD.size or grids != len(GRID_NAMES) or records != len(SPECS)
            or reserved != bytes(20) or len(data) < ARRAY_OFFSET or len(data) % 16):
        raise ValueError("Invalid packed GTCRN header/topology")
    if _digest(data) != digest:
        raise ValueError("Packed GTCRN integrity hash mismatch")
    if state_sha == bytes(32) or calibration_sha == bytes(32):
        raise ValueError("Packed GTCRN requires state and calibration hashes")
    if calibration is not None and hashlib.sha256(_json_bytes(calibration)).digest() != calibration_sha:
        raise ValueError("Calibration audit hash differs from packed model")
    config = GTCRNConfig(normalize_input=bool(flags), rms_floor=rms_floor)
    base_gru = GRUQuantizationConfig(input_exp, state_exp, logit_exp, accumulator_exp)
    backend = _IntegerBackend.__new__(_IntegerBackend)
    backend.grids = {name: ActivationGrid(int(value)) for name, value in zip(GRID_NAMES, np.frombuffer(data, np.int8, len(GRID_NAMES), GRID_OFFSET))}
    backend.recipes = {name: dict(recipe) for name, recipe in RECIPES.items()}
    backend.probability_encodings = {name: dict(value) for name, value in PROBABILITIES.items()}
    backend.ops, backend.tables, backend.edges, backend.base_gru = {}, {}, {}, base_gru
    spans, descriptors, read_cursor = {}, [], ARRAY_OFFSET

    def array(offset, dtype, shape):
        nonlocal read_cursor
        dtype = np.dtype(dtype).newbyteorder("<")
        length = math.prod(shape) * dtype.itemsize
        if offset < ARRAY_OFFSET or offset % 16 or offset + length > len(data):
            raise ValueError("Packed array has an invalid alignment/extent")
        identity = (length, dtype.str)
        if offset in spans and spans[offset] != identity:
            raise ValueError("Packed arrays alias incompatible types/lengths")
        if offset not in spans:
            if offset != ((read_cursor+15) & ~15) or any(data[read_cursor:offset]):
                raise ValueError("Packed arrays violate canonical first-reference order/padding")
            read_cursor = offset+length
        spans[offset] = identity
        return np.frombuffer(data, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape)

    def exponent(value, lower=-20, upper=4):
        if np.any(value < lower) or np.any(value > upper):
            raise ValueError("Packed weight/affine exponent outside contract")

    def dot_bound(weights, bias):
        bound = np.abs(bias.astype(np.int64)) + 128 * np.abs(weights.astype(np.int64)).reshape(len(bias), -1).sum(1)
        if np.any(bound > np.iinfo(np.int32).max):
            raise ValueError("Packed dot-plus-bias can overflow INT32")
        return bound

    for ordinal, spec in enumerate(SPECS):
        values = RECORD.unpack_from(data, RECORD_OFFSET + ordinal * RECORD.size)
        number, kind, record_flags, in_exp, out_exp, subtype, auxiliary = values[:7]
        dimensions, refs, tail = values[7:15], values[15:25], values[25]
        if (number, kind, record_flags, subtype, dimensions) != (ordinal, spec.kind, spec.flags, spec.subtype, spec.dimensions):
            raise ValueError(f"Packed descriptor differs from pinned topology: {spec.name}")
        if (in_exp, out_exp) != _grid_pair(backend, spec):
            raise ValueError("Packed operator grid differs from graph grid")
        if kind != LAYER_NORM and tail != bytes(16) or kind not in (PRELU, LAYER_NORM) and auxiliary != 0:
            raise ValueError("Packed descriptor reserved fields must be zero")
        use = {AFFINE: 3, STREAM: 3, PRELU: 1, GRU: 8, LAYER_NORM: 2, TANH: 1, SIGMOID: 1, ERB: 1, WINDOW: 1}[kind]
        if any(refs[use:]):
            raise ValueError("Packed descriptor has unexpected array references")
        descriptors.append({"index": ordinal, "name": spec.name, "kind": kind, "direction": subtype if kind == GRU else None,
                            "record_offset": RECORD_OFFSET + ordinal * RECORD.size, "array_offsets": list(refs[:use])})
        if kind in (AFFINE, STREAM):
            op = IntegerAffine()
            i, o, groups, kt, kf, stride, pad, dilation = dimensions
            op.kind = next(key for key, value in KINDS.items() if value == subtype)
            op.input_channels, op.output_channels, op.groups = i, o, groups
            op.input_grid, op.output_grid = ActivationGrid(in_exp), ActivationGrid(out_exp)
            op.batch_norm_folded = bool(record_flags & 1)
            if op.kind != "linear":
                op.kernel_size, op.stride, op.padding, op.dilation, op.output_padding = (kt, kf), (1, stride), (0, pad), (dilation, 1), (0, 0)
            op.weights, op.bias, op.exponents = array(refs[0], "i1", _affine_shape(spec)), array(refs[1], "i4", (o,)), array(refs[2], "i1", (o,))
            exponent(op.exponents)
            op.accumulator_bounds = _readonly(dot_bound(op.weights, op.bias))
            op.shifts = _readonly(in_exp + op.exponents.astype(np.int64) - out_exp)
            if kind == STREAM:
                wrapper = IntegerStreamConv()
                wrapper.affine, wrapper.transpose, wrapper.history_frames = op, bool(record_flags & 2), 2*dilation
                if wrapper.transpose:
                    wrapper.frequency_stride, wrapper.frequency_padding = 1, 1
                op = wrapper
            backend.ops[spec.name] = op
        elif kind == PRELU:
            exponent(auxiliary)
            op = IntegerPReLU.__new__(IntegerPReLU)
            op.input_grid, op.output_grid = ActivationGrid(in_exp), ActivationGrid(out_exp)
            op.exponent, op.channel_axis, op.slopes = auxiliary, 1, array(refs[0], "i1", (1,))
            backend.ops[spec.name] = op
        elif kind == GRU:
            if out_exp != -7:
                raise ValueError("Packed GRU outputs require signed Q7")
            cell = IntegerGRUCell()
            cell.input_size, cell.hidden_size = i, h = dimensions[:2]
            cell.config = replace(base_gru, input_exponent=in_exp)
            for offset, (name, dtype, shape) in zip(refs, (
                    ("weight_ih", "i1", (3*h, i)), ("weight_hh", "i1", (3*h, h)),
                    ("bias_ih", "i4", (3*h,)), ("bias_hh", "i4", (3*h,)),
                    ("exponent_ih", "i1", (3*h,)), ("exponent_hh", "i1", (3*h,)),
                    ("sigmoid_lut", "i1", (256,)), ("tanh_lut", "i1", (256,)))):
                setattr(cell, name, array(offset, dtype, shape))
            bounds = []
            for suffix, grid in (("ih", in_exp), ("hh", -7)):
                exponents = getattr(cell, "exponent_" + suffix)
                exponent(exponents)
                bound = dot_bound(getattr(cell, "weight_" + suffix), getattr(cell, "bias_" + suffix))
                bounds.append(_shift_numpy(bound, grid + exponents.astype(np.int64) - base_gru.accumulator_exponent))
            if np.any(bounds[0] + bounds[1] > np.iinfo(np.int32).max):
                raise ValueError("Packed GRU aligned gate sum can overflow INT32")
            if any(not np.array_equal(actual, expected) for actual, expected in zip((cell.sigmoid_lut, cell.tanh_lut), _tables(cell.config))):
                raise ValueError("Packed GRU table differs from integer contract")
            cell.reset_statistics()
            backend.ops[spec.name] = backend.ops.get(spec.name, ()) + (cell,)
        elif kind == LAYER_NORM:
            epsilon_code, epsilon = struct.unpack("<Qd", tail)
            if epsilon != 1e-8:
                raise ValueError("Packed LayerNorm epsilon differs from pinned network")
            op = IntegerLayerNormParameters(array(refs[0], "i1", (33, 16)), array(refs[1], "i4", (33, 16)),
                                            in_exp, auxiliary, out_exp, epsilon, 24)
            if epsilon_code != op.epsilon_code:
                raise ValueError("Packed LayerNorm epsilon code mismatch")
            backend.ops[spec.name] = op
        elif kind in (TANH, SIGMOID):
            table = array(refs[0], "i1", (256,))
            from .experimental_gru import _round_numpy
            x = np.arange(-128, 128, dtype=np.float64) * 2.0**in_exp
            if kind == TANH:
                if out_exp != -7:
                    raise ValueError("Packed tanh outputs require signed Q7")
                expected = _round_numpy(np.tanh(x)*128).clip(-128, 127).astype(np.int8)
            else:
                expected = (_round_numpy(255/(1+np.exp(-np.clip(x, -700, 700)))).clip(0, 255)-128).astype(np.int8)
            if not np.array_equal(table, expected):
                raise ValueError("Packed nonlinear table differs from integer contract")
            backend.tables[spec.name] = table
        elif kind == ERB:
            erb = SparseGTCRNERB(array(refs[0], "u1", (2040,)).tobytes())
            if erb.nonzero_count != 382:
                raise ValueError("Packed ERB nonzero count differs")
        else:
            window = array(refs[0], "f4", (512,))
            if not np.isfinite(window).all() or np.any(window < 0) or np.any(window > 1) or window[0] != 0 or window[256] != 1:
                raise ValueError("Packed window is not a finite bounded sqrt-Hann window")
    # Account for every byte: aliases must have identical spans/types, distinct
    # arrays cannot overlap, and all alignment/trailing padding is zero.
    previous = ARRAY_OFFSET
    for offset, (length, _) in sorted(spans.items()):
        if offset < previous or offset != ((previous + 15) & ~15) or any(data[previous:offset]):
            raise ValueError("Packed arrays overlap or contain noncanonical gaps")
        previous = offset + length
    if len(data) != ((previous + 15) & ~15) or any(data[previous:]):
        raise ValueError("Packed payload has unreferenced trailing bytes")
    if any(data[GRID_OFFSET+len(GRID_NAMES):RECORD_OFFSET]) or any(data[RECORD_OFFSET+len(SPECS)*RECORD.size:ARRAY_OFFSET]):
        raise ValueError("Packed metadata padding must be zero")
    for name in GRID_NAMES:
        if name.endswith(("intra_rnn.output", "inter_rnn.output")) and backend.grid(name).exponent != -7:
            raise ValueError("Packed grouped GRU output requires signed Q7")
    result = PackedGTCRNIntegerDenoiser.__new__(PackedGTCRNIntegerDenoiser)
    result.backend, result.graph, result.config, result.erb, result.window = backend, _Graph(backend), config, erb, window
    result.source_sha256 = None if checkpoint_sha == bytes(32) else checkpoint_sha.hex()
    result.calibration = calibration if calibration is not None else {"source_state_sha256": state_sha.hex(), "audit_loaded": False}
    result.state_shapes = {}
    for name, operation in backend.ops.items():
        if isinstance(operation, IntegerStreamConv):
            result.state_shapes[name] = (1, 16, operation.history_frames, 33)
        elif name.endswith(".tra.att_gru"):
            result.state_shapes[name] = (1, 1, 16)
    for name in ("dpgrnn1.inter_rnn", "dpgrnn2.inter_rnn"):
        result.state_shapes[name] = (1, 33, 16)
    result.packed_data = data
    result.packed_descriptors = descriptors
    result.packed_metadata = dict(format="GTI8PK01", format_version=VERSION, packed_bytes=len(data),
                                  packed_sha256=hashlib.sha256(data).hexdigest(), source_state_sha256=state_sha.hex(),
                                  source_checkpoint_sha256=result.source_sha256,
                                  calibration_sha256=calibration_sha.hex(), calibration_audit_loaded=calibration is not None,
                                  implementation_sha256=None if calibration is None else calibration.get("implementation_sha256"),
                                  metadata_bytes=ARRAY_OFFSET, unique_array_bytes=sum(size for size, _ in spans.values()),
                                  padding_bytes=len(data)-ARRAY_OFFSET-sum(size for size, _ in spans.values()),
                                  parameter_format="INT8 neural arrays, INT32 biases, exact stored LUTs; float32 external DSP constants",
                                  parser_scope="Checks bounded topology, extents, integrity and arithmetic; hashes do not authenticate training lineage")
    return result


def main():
    """Calibrate only from the checkpoint's audited training recipe, then pack."""
    import torch
    from .gtcrn_integer import from_checkpoint_training
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Development manifest used for disjointness auditing, never calibration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-crops", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=483)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.output.suffix != ".bin":
        parser.error("output must end in .bin to keep binary, audit and summary paths distinct")
    torch.set_num_threads(args.threads)
    integer, _, audit = from_checkpoint_training(args.checkpoint, args.manifest, crops=args.calibration_crops, seed=args.calibration_seed)
    data = pack_gtcrn_integer(integer)
    loaded = load_gtcrn_integer(data, calibration=audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    args.output.with_suffix(".calibration.json").write_bytes(_json_bytes(audit) + b"\n")
    args.output.with_suffix(".json").write_text(json.dumps(loaded.model_stats(), indent=2) + "\n")
    print(json.dumps(loaded.packed_metadata))


if __name__ == "__main__":
    main()
