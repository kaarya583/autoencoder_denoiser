"""Batched, device-resident GRU QAT with the existing exact INT8 forward.

This is an isolated recurrent primitive, not a complete GTCRN trainer. Input
projections are batched across all sequence positions; hidden recurrence is
batched across streams and quantized at every step. No CPU array snapshot
occurs in forward. One combined scalar bound check synchronizes each call.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .experimental_gru import GRUQuantizationConfig,IntegerGRUCell,_tables


class _ExactForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,surrogate,exact):return exact
    @staticmethod
    def backward(ctx,gradient):return gradient,None


def _round(value):return value.sign()*torch.floor(value.abs()+.5)


def _shift(value,shift):
    """Checked INT64 signed nearest shift, without negative shift operands."""
    shift=torch.as_tensor(shift,device=value.device,dtype=torch.int64)
    left=shift.clamp_min(0);right=(-shift).clamp_min(0)
    half=torch.where(right>0,torch.bitwise_left_shift(torch.ones_like(right),(right-1).clamp_min(0)),0)
    magnitude=torch.bitwise_right_shift(value.abs()+half,right)
    rounded=torch.where(value<0,-magnitude,magnitude)
    return torch.where(shift>=0,value*torch.bitwise_left_shift(torch.ones_like(left),left),rounded)


def _divide255(value):
    magnitude=torch.div(value.abs()+127,255,rounding_mode="floor")
    return torch.where(value<0,-magnitude,magnitude)


def _code_ste(value,scale):
    continuous=(value.float()/scale).clamp(-128,127)
    codes=_round(continuous).to(torch.int8)
    return _ExactForward.apply(continuous,codes.float()),codes


@contextmanager
def _ieee_matmul(device):
    if device.type != "cuda":
        yield
        return
    # F.linear uses matmul, not cuDNN convolution/RNN. Restore caller policy
    # even on error. This process-global setting requires externally serialized
    # precision-policy changes; the full trainer should set IEEE once globally.
    previous=torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision="ieee"
    try:yield
    finally:torch.backends.cuda.matmul.fp32_precision=previous


class BatchedQATGRU(nn.Module):
    """One-layer batch-first GRU, with shared trainable float32 masterweights.

    ``forward(x[B,T,I], state[D,B,H]|None)`` returns float32 values exactly on
    the Q7 output/state grid, like nn.GRU's shapes. The exact recurrence uses
    INT8 hidden codes; wider tensors are temporary training arithmetic.
    ``integer_snapshots`` is the explicit offline export/CPU-transfer boundary.
    Per-output weight exponents are frozen buffers, not recomputed after an
    optimizer update. The C serializer must use those exported arrays.
    """
    def __init__(self,source,config=None,*,snapshots=None):
        super().__init__()
        if (not isinstance(source,nn.GRU) or source.num_layers != 1 or not source.batch_first
                or not source.bias or source.dropout != 0):
            raise ValueError("Require a one-layer batch-first GRU with explicit biases and zero dropout")
        self.input_size,self.hidden_size=source.input_size,source.hidden_size
        if not 1 <= self.input_size <= 512 or not 1 <= self.hidden_size <= 128:
            raise ValueError("Unsupported integer GRU matrix dimensions")
        self.bidirectional=source.bidirectional
        self.num_directions=2 if source.bidirectional else 1
        if snapshots is not None and (not isinstance(snapshots,(tuple,list)) or len(snapshots) != self.num_directions):
            raise ValueError("Snapshot direction count differs")
        self.config=config or (snapshots[0].config if snapshots is not None else GRUQuantizationConfig())
        if not isinstance(self.config,GRUQuantizationConfig):raise TypeError("Explicit GRUQuantizationConfig required")
        if snapshots is None:
            snapshots=tuple(IntegerGRUCell.from_torch(source,self.config,direction=direction)
                            for direction in (("forward","reverse") if self.bidirectional else ("forward",)))
        if len(snapshots) != self.num_directions:raise ValueError("Snapshot direction count differs")
        for index,snapshot in enumerate(snapshots):
            if (not isinstance(snapshot,IntegerGRUCell) or snapshot.config != self.config
                    or snapshot.input_size != self.input_size or snapshot.hidden_size != self.hidden_size):
                raise ValueError("Frozen snapshot dimensions/grids differ from source")
            suffix="_l0"+("_reverse" if index else "")
            for name in ("weight_ih","weight_hh","bias_ih","bias_hh"):
                parameter=getattr(source,name+suffix)
                if parameter.dtype != torch.float32:raise ValueError("QAT masterweights must be float32")
                setattr(self,name+suffix,parameter)
            for name in ("exponent_ih","exponent_hh"):
                self.register_buffer(name+suffix,torch.from_numpy(getattr(snapshot,name).copy()).to(source.weight_ih_l0.device))
        sigmoid,tanh=_tables(self.config)
        self.register_buffer("sigmoid_lut",torch.from_numpy(sigmoid).to(source.weight_ih_l0.device))
        self.register_buffer("tanh_lut",torch.from_numpy(tanh).to(source.weight_ih_l0.device))
        self._prepare()  # Reject initial unsupported raw/aligned precision bounds.

    @classmethod
    def from_float(cls,source,config=None,*,snapshots=None):return cls(source,config,snapshots=snapshots)

    def get_extra_state(self):
        return dict(version=1,config=asdict(self.config),input_size=self.input_size,
                    hidden_size=self.hidden_size,bidirectional=self.bidirectional)

    def set_extra_state(self,state):
        if (not isinstance(state,dict) or state.get("version") != 1 or state.get("input_size") != self.input_size
                or state.get("hidden_size") != self.hidden_size or state.get("bidirectional") != self.bidirectional):
            raise ValueError("Saved GRU QAT metadata differs from module topology")
        self.config=GRUQuantizationConfig(**state["config"])

    def _prepare(self,extra_checks=()):
        """Quantize once per sequence, staying on-device; validate one scalar."""
        prepared=[];checks=list(extra_checks)
        for index in range(self.num_directions):
            suffix="_l0"+("_reverse" if index else "")
            paths=[]
            for side,input_exp in (("ih",self.config.input_exponent),("hh",self.config.state_exponent)):
                weight,bias=getattr(self,"weight_"+side+suffix),getattr(self,"bias_"+side+suffix)
                exponent=getattr(self,"exponent_"+side+suffix)
                if weight.dtype != torch.float32 or bias.dtype != torch.float32:
                    raise ValueError("QAT masterweights must remain float32")
                if exponent.dtype != torch.int8 or exponent.shape != (3*self.hidden_size,):
                    raise ValueError("Frozen GRU weight exponents must be INT8 per output row")
                checks.extend((torch.isfinite(weight).all(),torch.isfinite(bias).all(),((exponent>=-20)&(exponent<=4)).all()))
                # Corrupt checkpoints must fail without first executing an
                # oversized shift or overflowing preparation intermediates.
                safe_exponent=exponent.clamp(-20,4)
                scale=torch.pow(2.0,safe_exponent.float())
                # The small bias vector uses double only during encoding:
                # adding .5 to a large float32 bias code can round incorrectly.
                bias_raw=_round(bias.double()/(scale.double()*(2.0**input_exp)))
                checks.append((torch.isfinite(bias_raw)&(bias_raw.abs()<=2**31-1)).all())
                safe_bias=torch.nan_to_num(bias_raw.detach(),nan=0,posinf=2**31-1,neginf=-(2**31-1)).clamp(-(2**31-1),2**31-1)
                bias_integer=safe_bias.to(torch.int64)
                weight_continuous=(weight/scale[:,None]).clamp(-128,127)
                weight_integer=_round(torch.nan_to_num(weight_continuous.detach(),nan=0)).to(torch.int8)
                bound=bias_integer.abs()+128*weight_integer.to(torch.int64).abs().sum(1)
                checks.append((bound<=2**24-1).all())
                shift=input_exp+safe_exponent.to(torch.int64)-self.config.accumulator_exponent
                aligned_bound=_shift(bound,shift)
                w=_ExactForward.apply(weight_continuous,weight_integer.float())
                b=_ExactForward.apply((bias/(scale*(2.0**input_exp))),bias_integer.float())
                paths.append(dict(weight=w,bias=b,weight_integer=weight_integer,bias_integer=bias_integer,
                                  exponent=exponent,shift=shift,aligned_bound=aligned_bound,
                                  real_scale=scale*(2.0**input_exp),raw_bound=bound))
            checks.append((paths[0]["aligned_bound"]+paths[1]["aligned_bound"]<=2**31-1).all())
            prepared.append(paths)
        if not bool(torch.stack(checks).all()):
            raise ValueError("GRU QAT requires finite tensors, frozen legal grids, raw FP32-exact bounds below2^24 and aligned INT32 bounds")
        return prepared

    def _logit(self,surrogate,exact_accumulator):
        codes=_shift(exact_accumulator,self.config.accumulator_exponent-self.config.logit_exponent).clamp(-128,127).to(torch.int8)
        scale=2.0**self.config.logit_exponent
        continuous=surrogate.clamp(-128*scale,127*scale)
        return _ExactForward.apply(continuous,codes.float()*scale),codes

    def _gate(self,surrogate,exact_accumulator):
        real,logit=self._logit(surrogate,exact_accumulator)
        stored=self.sigmoid_lut[logit.long()+128]
        probability=stored.to(torch.int64)+128
        return _ExactForward.apply(torch.sigmoid(real),probability.float()/255),probability

    def forward(self,inputs,state=None):
        if (not isinstance(inputs,torch.Tensor) or inputs.ndim != 3 or min(inputs.shape)<1
                or inputs.shape[-1] != self.input_size
                or inputs.dtype not in (torch.float16,torch.bfloat16,torch.float32)):
            raise ValueError("GRU input must be FP32/FP16/BF16 [positive batch,time,input_size]")
        if inputs.device != self.weight_ih_l0.device:raise ValueError("GRU input and masterweights must share a device")
        expected=(self.num_directions,inputs.shape[0],self.hidden_size)
        state=inputs.new_zeros(expected,dtype=torch.float32) if state is None else state
        if (not isinstance(state,torch.Tensor) or state.shape != expected
                or state.dtype not in (torch.float16,torch.bfloat16,torch.float32)
                or state.device != inputs.device):raise ValueError("GRU state must be floating [directions,batch,hidden_size] on the input device")
        with torch.autocast(device_type=inputs.device.type,enabled=False),_ieee_matmul(inputs.device):
            prepared=self._prepare((torch.isfinite(inputs).all(),torch.isfinite(state).all()))
            x,_=_code_ste(inputs,2.0**self.config.input_exponent)
            sequences,states=[],[]
            for direction,(input_path,hidden_path) in enumerate(prepared):
                # One GEMM for every input position, rather than one per step.
                input_raw=F.linear(x,input_path["weight"],input_path["bias"])
                exact_input=_shift(input_raw.detach().to(torch.int64),input_path["shift"])
                real_input=input_raw*input_path["real_scale"]
                h_codes,h_integer=_code_ste(state[direction],2.0**self.config.state_exponent)
                h=h_codes*(2.0**self.config.state_exponent)
                outputs=[]
                positions=range(inputs.shape[1]-1,-1,-1) if direction else range(inputs.shape[1])
                for position in positions:
                    hidden_raw=F.linear(h_codes,hidden_path["weight"],hidden_path["bias"])
                    exact_hidden=_shift(hidden_raw.detach().to(torch.int64),hidden_path["shift"])
                    real_hidden=hidden_raw*hidden_path["real_scale"]
                    ar,az,an=exact_input[:,position].chunk(3,-1);br,bz,bn=exact_hidden.chunk(3,-1)
                    sar,saz,san=real_input[:,position].chunk(3,-1);sbr,sbz,sbn=real_hidden.chunk(3,-1)
                    reset,reset_code=self._gate(sar+sbr,ar+br)
                    update,update_code=self._gate(saz+sbz,az+bz)
                    candidate_accumulator=an+_divide255(reset_code*bn)
                    candidate_real,candidate_logit=self._logit(san+reset*sbn,candidate_accumulator)
                    candidate_integer=self.tanh_lut[candidate_logit.long()+128]
                    candidate=_ExactForward.apply(torch.tanh(candidate_real),candidate_integer.float()/128)
                    next_integer=_divide255((255-update_code)*candidate_integer.to(torch.int64)+update_code*h_integer.to(torch.int64)).clamp(-128,127).to(torch.int8)
                    h=_ExactForward.apply((1-update)*candidate+update*h,next_integer.float()/128)
                    h_integer=next_integer;h_codes=h*128
                    outputs.append(h)
                sequence=torch.stack(outputs,dim=1)
                sequences.append(sequence.flip(1) if direction else sequence);states.append(h)
            return torch.cat(sequences,dim=-1),torch.stack(states,dim=0)

    @torch.no_grad()
    def integer_snapshots(self):
        """Explicit offline preparation; preserves the frozen weight grids."""
        prepared=self._prepare()
        sigmoid,tanh=_tables(self.config)
        for name,expected in (("sigmoid_lut",sigmoid),("tanh_lut",tanh)):
            if not np.array_equal(getattr(self,name).cpu().numpy(),expected):
                raise ValueError("QAT lookup table differs from saved integer contract")
        snapshots=[]
        for paths in prepared:
            cell=IntegerGRUCell();cell.config=self.config
            cell.input_size,cell.hidden_size=self.input_size,self.hidden_size
            for side,path in zip(("ih","hh"),paths):
                for name,key,dtype in (("weight_","weight_integer",np.int8),("bias_","bias_integer",np.int32),("exponent_","exponent",np.int8)):
                    array=path[key].detach().cpu().numpy().astype(dtype,copy=True);array.flags.writeable=False
                    setattr(cell,name+side,array)
            cell.sigmoid_lut=sigmoid.copy();cell.tanh_lut=tanh.copy()
            cell.sigmoid_lut.flags.writeable=cell.tanh_lut.flags.writeable=False
            cell.reset_statistics();snapshots.append(cell)
        return tuple(snapshots)


__all__=["BatchedQATGRU"]
