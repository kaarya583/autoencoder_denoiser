"""Host parity bindings for isolated portable INT8 GTCRN operators.

This module prepares frozen C operators from the audited NumPy snapshots.
It does not serialize or dispatch a full GTCRN graph. All inference arithmetic
in operators.c is integer, and all native buffers remain caller-owned.
"""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .experimental_gtcrn_ops import ActivationGrid, IntegerAffine, IntegerPReLU, IntegerStreamConv
from .experimental_native import _Buffer, _buffer, _copy_array, _handle
from .runtime_cache import runtime_fingerprint


class _AffineSpec(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "kind", "input_channels", "output_channels", "groups", "kernel_time", "kernel_frequency",
        "stride_time", "stride_frequency", "padding_time", "padding_frequency", "dilation_time", "dilation_frequency",
        "output_padding_time", "output_padding_frequency")] + [
        ("input_exponent", ct.c_int), ("output_exponent", ct.c_int),
        ("weights", _Buffer), ("bias", _Buffer), ("exponents", _Buffer)]


def _runtime():
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("A C99 compiler is required for the isolated integer operator checks")
    source = Path(__file__).resolve().parents[1] / "firmware/experimental_int8"
    fingerprint = runtime_fingerprint(source, ("operators.c", "operators.h", "primitives.h"))
    return _compiled(source, compiler, fingerprint)


@lru_cache(maxsize=1)
def _compiled(source, compiler, fingerprint):
    directory = tempfile.TemporaryDirectory(prefix="experimental-int8-ops-")
    path = Path(directory.name) / "operators.so"
    try:
        subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-shared", "-fPIC",
                        str(source / "operators.c"), "-o", str(path)], check=True, capture_output=True, text=True)
        lib = ct.CDLL(str(path))
    except Exception:
        directory.cleanup()
        raise
    p, n, u, i = ct.c_void_p, ct.c_size_t, ct.c_uint32, ct.c_int
    signatures = {
        "ednx_affine_handle_bytes": ([], n),
        "ednx_affine_init": ([p, ct.POINTER(_AffineSpec)], i),
        "ednx_affine_output_shape": ([p,u,u,ct.POINTER(u),ct.POINTER(u)], i),
        "ednx_affine_run": ([p,p,n,u,u,p,n,p,n], i),
        "ednx_stream_history_bytes": ([p,u],n),
        "ednx_stream_workspace_bytes": ([p,u,u,i,i],n),
        "ednx_stream_conv": ([p,p,n,p,n,u,u,i,i,p,n,p,n,p,n],i),
        "ednx_regrid": ([p,p,n,i,i],i),
        "ednx_residual": ([p,p,p,n,i,i,i],i),
        "ednx_prelu": ([p,p,u,n,p,n,i,i,i],i),
        "ednx_lut": ([p,p,n,p],i),
        "ednx_attention_energy": ([p,p,n,u,i,i],i),
        "ednx_attention_product": ([p,p,p,n,u,i,i],i),
        "ednx_subband": ([p,p,u,u,u],i),
        "ednx_shuffle": ([p,p,p,u,n],i),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    return lib, directory


def _codes(value):
    value = np.asarray(value)
    if value.dtype != np.int8 or not value.size:
        raise ValueError("Native neural tensors must be nonempty INT8")
    return np.ascontiguousarray(value)


def _exponent(grid):
    if not isinstance(grid, ActivationGrid):
        raise TypeError("An explicit ActivationGrid is required")
    return grid.exponent


def _check(status):
    if status:
        raise ValueError("Native integer operator rejected its dimensions, grids, bounds or buffers")


class CIntegerAffine:
    def __init__(self, snapshot):
        if not isinstance(snapshot, IntegerAffine):
            raise TypeError("Prepare an IntegerAffine snapshot first")
        self.kind = snapshot.kind
        self.input_channels, self.output_channels = snapshot.input_channels, snapshot.output_channels
        self.arrays = {name: _copy_array(getattr(snapshot, name), np.int32 if name == "bias" else np.int8)
                       for name in ("weights", "bias", "exponents")}
        kernel, stride, padding, dilation, output_padding = ((1,1),(1,1),(0,0),(1,1),(0,0)) if self.kind == "linear" else (
            snapshot.kernel_size, snapshot.stride, snapshot.padding, snapshot.dilation, snapshot.output_padding)
        self.spec = _AffineSpec({"linear":0,"conv2d":1,"conv_transpose2d":2}[self.kind],
                                self.input_channels, self.output_channels, snapshot.groups,
                                *kernel, *stride, *padding, *dilation, *output_padding,
                                snapshot.input_grid.exponent, snapshot.output_grid.exponent,
                                *[_buffer(self.arrays[name]) for name in ("weights", "bias", "exponents")])
        self.library, self._directory = _runtime()
        self.handle_bytes = self.library.ednx_affine_handle_bytes()
        self.handle = _handle(self.handle_bytes)
        _check(self.library.ednx_affine_init(self.handle, ct.byref(self.spec)))

    def output_shape(self, time, frequency):
        ot, of = ct.c_uint32(), ct.c_uint32()
        _check(self.library.ednx_affine_output_shape(self.handle, time, frequency, ct.byref(ot), ct.byref(of)))
        return ot.value, of.value

    def __call__(self, value, *, return_accumulator=False):
        value = _codes(value)
        if self.kind == "linear":
            if value.shape[-1] != self.input_channels:
                raise ValueError("Linear input has the wrong trailing feature count")
            inputs = value.reshape(1,-1,self.input_channels)
            t, f = inputs.shape[1], 1
            self.output_shape(t, f)
            shape = (*value.shape[:-1], self.output_channels)
            output = np.empty((1,t,self.output_channels), np.int8)
        else:
            if value.ndim != 4 or value.shape[1] != self.input_channels:
                raise ValueError("Convolution expects [B,C,T,F]")
            inputs, (t,f) = value, value.shape[-2:]
            ot, of = self.output_shape(t, f)
            shape = (len(value),self.output_channels,ot,of)
            output = np.empty(shape,np.int8)
        accumulators = np.empty_like(output,dtype=np.int32) if return_accumulator else None
        for row in range(len(inputs)):
            _check(self.library.ednx_affine_run(self.handle, inputs[row].ctypes.data, inputs[row].nbytes, t, f,
                   output[row].ctypes.data, output[row].nbytes,
                   accumulators[row].ctypes.data if return_accumulator else None,
                   accumulators[row].size if return_accumulator else 0))
        result = output.reshape(shape)
        return (result,accumulators.reshape(shape)) if return_accumulator else result


class CIntegerStreamConv:
    def __init__(self, snapshot):
        if not isinstance(snapshot, IntegerStreamConv):
            raise TypeError("Prepare an IntegerStreamConv snapshot first")
        self.affine = CIntegerAffine(snapshot.affine)
        self.history_frames = snapshot.history_frames
        self.stride = snapshot.frequency_stride if snapshot.transpose else 1
        self.left = snapshot.frequency_padding if snapshot.transpose else 0
        self.right = self.left-(self.stride-1)

    def initial_state(self, batch, frequency):
        if any(isinstance(n,bool) or not isinstance(n,int) or n < 1 for n in (batch,frequency)):
            raise ValueError("Positive batch and frequency are required")
        return np.zeros((batch,self.affine.input_channels,self.history_frames,frequency),np.int8)

    def workspace_bytes(self, frequency):
        return self.affine.library.ednx_stream_workspace_bytes(self.affine.handle, frequency,self.stride,self.left,self.right)

    def step(self, frame, state=None):
        frame = _codes(frame)
        if frame.ndim != 4 or frame.shape[1:3] != (self.affine.input_channels,1):
            raise ValueError("Streaming input must be [B,C,1,F]")
        batch, frequency = len(frame), frame.shape[-1]
        state = self.initial_state(batch,frequency) if state is None else np.asarray(state)
        expected = (batch,self.affine.input_channels,self.history_frames,frequency)
        if state.dtype != np.int8 or state.shape != expected:
            raise ValueError("Streaming history must have the exact INT8 shape")
        state = np.ascontiguousarray(state)
        workbytes = self.workspace_bytes(frequency)
        if not workbytes:
            raise ValueError("Invalid native streaming workspace dimensions")
        work = np.empty(workbytes,np.int8)
        ot,of = self.affine.output_shape(self.history_frames+1,frequency*self.stride+self.left+self.right)
        output = np.empty((batch,self.affine.output_channels,ot,of),np.int8)
        next_state = np.empty_like(state)
        for row in range(batch):
            _check(self.affine.library.ednx_stream_conv(self.affine.handle,
                   frame[row].ctypes.data,frame[row].nbytes,state[row].ctypes.data,state[row].nbytes,
                   frequency,self.stride,self.left,self.right,output[row].ctypes.data,output[row].nbytes,
                   next_state[row].ctypes.data,next_state[row].nbytes,work.ctypes.data,work.nbytes))
        return output,next_state


class CIntegerOps:
    """Independent C elementwise/reshape helpers, with no learned graph."""
    def __init__(self):
        self.library, self._directory = _runtime()

    def regrid(self, value, input_grid, output_grid):
        x=_codes(value); y=np.empty_like(x)
        _check(self.library.ednx_regrid(x.ctypes.data,y.ctypes.data,x.size,_exponent(input_grid),_exponent(output_grid)))
        return y

    def residual(self, left, right, left_grid, right_grid, output_grid):
        a,b=_codes(left),_codes(right)
        if a.shape != b.shape: raise ValueError("Residual shapes must match")
        y=np.empty_like(a)
        _check(self.library.ednx_residual(a.ctypes.data,b.ctypes.data,y.ctypes.data,a.size,
               _exponent(left_grid),_exponent(right_grid),_exponent(output_grid)))
        return y

    def prelu(self, value, snapshot):
        if not isinstance(snapshot,IntegerPReLU): raise TypeError("Prepare an IntegerPReLU snapshot first")
        x=_codes(value)
        if x.ndim < 1 or (snapshot.slopes.size != 1 and not -x.ndim <= snapshot.channel_axis < x.ndim):
            raise ValueError("PReLU requires a valid channel axis on a non-scalar tensor")
        axis=snapshot.channel_axis%x.ndim
        if snapshot.slopes.size != 1 and x.shape[axis] != snapshot.slopes.size:
            raise ValueError("PReLU channel count does not match slopes")
        moved=np.ascontiguousarray(np.moveaxis(x,axis,0)); y=np.empty_like(moved)
        _check(self.library.ednx_prelu(moved.ctypes.data,y.ctypes.data,len(moved),moved.size//len(moved),
               snapshot.slopes.ctypes.data,snapshot.slopes.size,snapshot.exponent,
               snapshot.input_grid.exponent,snapshot.output_grid.exponent))
        return np.ascontiguousarray(np.moveaxis(y,0,axis))

    def lut(self,value,table):
        x,table=_codes(value),_codes(table)
        if table.shape != (256,): raise ValueError("A 256-code LUT is required")
        y=np.empty_like(x)
        _check(self.library.ednx_lut(x.ctypes.data,y.ctypes.data,x.size,table.ctypes.data))
        return y

    def energy(self,value,input_grid,output_grid):
        x=_codes(value); y=np.empty(x.shape[:-1],np.int8)
        _check(self.library.ednx_attention_energy(x.ctypes.data,y.ctypes.data,y.size,x.shape[-1],
               _exponent(input_grid),_exponent(output_grid)))
        return y

    def product(self,value,probabilities,input_grid,output_grid):
        x,p=_codes(value),_codes(probabilities)
        x,p=np.broadcast_arrays(x,p)
        x,p=np.ascontiguousarray(x),np.ascontiguousarray(p)
        y=np.empty_like(x)
        # Arbitrary broadcast shapes reduce to one probability per scalar.
        _check(self.library.ednx_attention_product(x.ctypes.data,p.ctypes.data,y.ctypes.data,x.size,1,
               _exponent(input_grid),_exponent(output_grid)))
        return y

    def subband(self,value):
        x=_codes(value)
        if x.ndim != 4: raise ValueError("Subband input must be [B,C,T,F]")
        b,c,t,f=x.shape; y=np.empty((b,c*3,t,f),np.int8)
        for row in range(b):
            _check(self.library.ednx_subband(x[row].ctypes.data,y[row].ctypes.data,c,t,f))
        return y

    def shuffle(self,left,right):
        a,b=_codes(left),_codes(right)
        if a.ndim != 4 or a.shape != b.shape: raise ValueError("Shuffle expects matching [B,C,T,F]")
        batch,c,t,f=a.shape; y=np.empty((batch,c*2,t,f),np.int8)
        for row in range(batch):
            _check(self.library.ednx_shuffle(a[row].ctypes.data,b[row].ctypes.data,y[row].ctypes.data,c,t*f))
        return y
