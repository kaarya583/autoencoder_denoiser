"""Host bindings for isolated portable INT8 GRU and LayerNorm primitives.

This is an exact native arithmetic check, not a GTCRN model runtime, an ESP32
firmware build, or a hardware performance result. Inputs, outputs and GRU
state are actual INT8. The C code contains no floating neural arithmetic.
Parameter preparation still belongs to the audited Python prototypes.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .experimental_gru import IntegerGRUCell
from .experimental_layer_norm import IntegerLayerNormParameters
from .runtime_cache import runtime_fingerprint


class _Buffer(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("bytes", ctypes.c_size_t)]


class _GRUSpec(ctypes.Structure):
    _fields_ = [("input_size", ctypes.c_uint32), ("hidden_size", ctypes.c_uint32),
                ("input_exponent", ctypes.c_int), ("state_exponent", ctypes.c_int),
                ("logit_exponent", ctypes.c_int), ("accumulator_exponent", ctypes.c_int),
                *[(name, _Buffer) for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh",
                                             "exponent_ih", "exponent_hh", "sigmoid_lut", "tanh_lut")]]


class _GRUTrace(ctypes.Structure):
    _fields_ = [("codes", ctypes.c_void_p), ("codes_bytes", ctypes.c_size_t),
                ("candidate_accumulators", ctypes.c_void_p), ("candidate_count", ctypes.c_size_t)]


class _LayerNormSpec(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("input_exponent", ctypes.c_int),
                ("gamma_exponent", ctypes.c_int), ("output_exponent", ctypes.c_int),
                ("variance_fractional_bits", ctypes.c_uint32), ("epsilon_code", ctypes.c_uint64),
                ("gamma", _Buffer), ("beta", _Buffer)]


def _native_runtime():
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("The experimental native primitives require a C99 compiler")
    source = Path(__file__).resolve().parents[1] / "firmware/experimental_int8"
    fingerprint = runtime_fingerprint(source, ("primitives.c", "primitives.h"))
    return _runtime_for_source(source, compiler, fingerprint)


@lru_cache(maxsize=1)
def _runtime_for_source(source, compiler, fingerprint):
    directory = tempfile.TemporaryDirectory(prefix="experimental-int8-")
    library_path = Path(directory.name) / "primitives.so"
    try:
        subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-DEDNX_TESTING", "-shared", "-fPIC", str(source / "primitives.c"),
                        "-o", str(library_path)], check=True, capture_output=True, text=True)
        library = ctypes.CDLL(str(library_path))
    except Exception:
        directory.cleanup()
        raise
    for name in ("ednx_gru_handle_bytes", "ednx_layer_norm_handle_bytes"):
        getattr(library, name).argtypes = []
        getattr(library, name).restype = ctypes.c_size_t
    library.ednx_gru_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GRUSpec)]
    library.ednx_layer_norm_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(_LayerNormSpec)]
    library.ednx_gru_scratch_bytes.argtypes = [ctypes.c_void_p]
    library.ednx_gru_scratch_bytes.restype = ctypes.c_size_t
    library.ednx_gru_step.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_size_t, ctypes.POINTER(_GRUTrace)]
    library.ednx_layer_norm_frame.argtypes = [ctypes.c_void_p] * 3
    library.ednx_test_round_divide.argtypes = [ctypes.c_int64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int64)]
    library.ednx_test_isqrt.argtypes = [ctypes.c_uint64]
    library.ednx_test_isqrt.restype = ctypes.c_uint64
    for name in ("ednx_gru_init", "ednx_layer_norm_init", "ednx_gru_step",
                 "ednx_layer_norm_frame", "ednx_test_round_divide"):
        getattr(library, name).restype = ctypes.c_int
    return library, directory


def _handle(size):
    # The opaque C structs need at most uint64_t's natural alignment.
    return (ctypes.c_uint64 * ((size + 7) // 8))()


def _copy_array(value, dtype):
    value = np.asarray(value)
    if value.dtype != np.dtype(dtype):
        raise ValueError(f"Native parameter arrays must have dtype {np.dtype(dtype)}")
    result = np.array(value, copy=True, order="C")
    result.flags.writeable = False
    return result


def _buffer(value):
    return _Buffer(value.ctypes.data, value.nbytes)


class CIntegerGRUCell:
    """A frozen native cell with caller-owned INT8 state; one stream per C call.

    The Python convenience methods loop over independent batch items. One cell
    instance reuses its scratch buffer and must not be called concurrently.
    """

    def __init__(self, snapshot: IntegerGRUCell):
        if not isinstance(snapshot, IntegerGRUCell):
            raise TypeError("Prepare an audited IntegerGRUCell snapshot first")
        self.config = snapshot.config
        self.input_size, self.hidden_size = snapshot.input_size, snapshot.hidden_size
        names = ("weight_ih", "weight_hh", "bias_ih", "bias_hh", "exponent_ih",
                 "exponent_hh", "sigmoid_lut", "tanh_lut")
        self.arrays = {name: _copy_array(getattr(snapshot, name), np.int32 if name.startswith("bias_") else np.int8)
                       for name in names}
        self.spec = _GRUSpec(self.input_size, self.hidden_size,
                             self.config.input_exponent, self.config.state_exponent,
                             self.config.logit_exponent, self.config.accumulator_exponent,
                             *[_buffer(self.arrays[name]) for name in names])
        self.library, self._directory = _native_runtime()
        self.handle_bytes = self.library.ednx_gru_handle_bytes()
        self.handle = _handle(self.handle_bytes)
        if self.library.ednx_gru_init(self.handle, ctypes.byref(self.spec)):
            raise ValueError("The C GRU rejected the parameter dimensions, grids or integer bounds")
        self.scratch_bytes = self.library.ednx_gru_scratch_bytes(self.handle)
        self.scratch = np.empty(self.scratch_bytes, dtype=np.int8)

    def initial_state(self, batch_size=1):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return np.zeros((batch_size, self.hidden_size), dtype=np.int8)

    def step(self, inputs, state=None, *, return_trace=False):
        inputs = np.asarray(inputs)
        if inputs.dtype != np.int8 or inputs.ndim != 2 or inputs.shape[1] != self.input_size or not len(inputs):
            raise ValueError("C GRU input must be INT8 [positive batch,input_size]")
        state = self.initial_state(len(inputs)) if state is None else np.asarray(state)
        if state.dtype != np.int8 or state.shape != (len(inputs), self.hidden_size):
            raise ValueError("C GRU state must be INT8 [batch,hidden_size]")
        inputs, state = np.ascontiguousarray(inputs), np.ascontiguousarray(state)
        output = np.empty_like(state)
        if return_trace:
            traces = np.empty((len(inputs), 6, self.hidden_size), dtype=np.int8)
            candidates = np.empty_like(state, dtype=np.int32)
        for index in range(len(inputs)):
            trace = (_GRUTrace(traces[index].ctypes.data, traces[index].nbytes,
                                candidates[index].ctypes.data, self.hidden_size) if return_trace else None)
            status = self.library.ednx_gru_step(
                self.handle, inputs[index].ctypes.data, state[index].ctypes.data,
                output[index].ctypes.data, self.scratch.ctypes.data, self.scratch_bytes,
                ctypes.byref(trace) if trace is not None else None)
            if status:
                raise RuntimeError("Native GRU step rejected its buffers")
        if return_trace:
            names = ("reset_logit", "update_logit", "candidate_logit", "reset", "update", "candidate")
            trace = {name: traces[:, index].copy() for index, name in enumerate(names)}
            trace.update(hidden=output, candidate_accumulator=candidates)
            return output, trace
        return output

    def process(self, inputs, state=None):
        inputs = np.asarray(inputs)
        if inputs.dtype != np.int8 or inputs.ndim != 3 or min(inputs.shape) < 1 or inputs.shape[-1] != self.input_size:
            raise ValueError("C GRU sequence must be INT8 [batch,positive time,input_size]")
        output = np.empty((*inputs.shape[:2], self.hidden_size), dtype=np.int8)
        for frame in range(inputs.shape[1]):
            state = self.step(inputs[:, frame], state)
            output[:, frame] = state
        return output, state.copy()

    def memory_accounting(self):
        return dict(parameter_and_table_bytes=sum(value.nbytes for value in self.arrays.values()),
                    persistent_state_bytes_per_stream=self.hidden_size,
                    explicit_scratch_bytes=self.scratch_bytes, host_handle_bytes=self.handle_bytes,
                    scope="Arrays and host sizeof(handle); input/output, compiler stack, Python objects, serialized metadata and S3 layout excluded")


class CIntegerLayerNorm:
    """Two-pass native LayerNorm, with no persistent state or allocated C heap."""

    def __init__(self, parameters: IntegerLayerNormParameters):
        if not isinstance(parameters, IntegerLayerNormParameters):
            raise TypeError("Prepare audited IntegerLayerNormParameters first")
        self.parameters = parameters
        self.gamma = _copy_array(parameters.gamma, np.int8)
        self.beta = _copy_array(parameters.beta, np.int32)
        self.spec = _LayerNormSpec(parameters.size, parameters.input_exponent,
                                   parameters.gamma_exponent, parameters.output_exponent,
                                   parameters.variance_fractional_bits, parameters.epsilon_code,
                                   _buffer(self.gamma), _buffer(self.beta))
        self.library, self._directory = _native_runtime()
        self.handle_bytes = self.library.ednx_layer_norm_handle_bytes()
        self.handle = _handle(self.handle_bytes)
        if self.library.ednx_layer_norm_init(self.handle, ctypes.byref(self.spec)):
            raise ValueError("The C LayerNorm rejected its dimensions, grids or integer bounds")

    def __call__(self, codes):
        codes = np.asarray(codes)
        shape = self.parameters.normalized_shape
        if codes.dtype != np.int8 or codes.ndim < len(shape) or codes.shape[-len(shape):] != shape or not codes.size:
            raise ValueError(f"C LayerNorm requires nonempty INT8 input with trailing shape {shape}")
        frames = np.ascontiguousarray(codes).reshape(-1, self.parameters.size)
        output = np.empty_like(frames)
        for index, frame in enumerate(frames):
            if self.library.ednx_layer_norm_frame(self.handle, frame.ctypes.data, output[index].ctypes.data):
                raise RuntimeError("Native LayerNorm rejected its frame")
        return output.reshape(codes.shape)

    def memory_accounting(self):
        return dict(parameter_array_bytes=self.gamma.nbytes + self.beta.nbytes,
                    persistent_neural_state_bytes=0, explicit_scratch_bytes=0,
                    host_handle_bytes=self.handle_bytes, input_output_bytes_per_frame=2 * self.parameters.size,
                    scope="Native arrays/host handle; scalar compiler stack, Python objects, serialized metadata and S3 layout excluded")
