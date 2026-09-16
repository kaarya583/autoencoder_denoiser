"""Audited training-only power-of-two MSE calibration for the fixed INT8 graph.

The min/max control remains available. This alternative scores every float
training activation, without a histogram/sample approximation. It changes
only fixed deployment grid metadata, preserving the current C operators.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
from itertools import islice, product
import json
from pathlib import Path

import numpy as np
import torch

from .experimental_gru import IntegerGRUCell, _round_numpy
from .experimental_gtcrn_ops import ActivationGrid, IntegerAffine, IntegerStreamConv
from .experimental_layer_norm import quantize_layer_norm
from .gtcrn_integer import GTCRNFloatShadow, GTCRNIntegerDenoiser, _network_frames, _source_fingerprint, calibrate_gtcrn_integer
from .gtcrn_model import GTCRNConfig, GTCRNDenoiser


DEFAULT_OFFSETS=(-3,-2,-1,0,1)


class _CandidateErrors:
    """Exact streaming float64 SSE and counts; stores no activation samples."""
    def __init__(self,exponents):
        self.values=0
        self.reference_squared=0.0
        self.minimum=self.maximum=None
        self.digest=hashlib.sha256()
        self.candidates={int(e):dict(squared_error=0.0,clipped_values=0,zero_codes=0,maximum_absolute_error=0.0) for e in exponents}

    def observe(self,value):
        if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
        value=np.asarray(value)
        if not np.issubdtype(value.dtype,np.floating) or not value.size or not np.isfinite(value).all():
            raise ValueError("MSE calibration requires nonempty finite float activations")
        # Shape framing prevents concatenation ambiguity; bytes are always LE
        # float32, the pinned shadow's actual precision.
        array=np.ascontiguousarray(value,dtype="<f4")
        shape=json.dumps(list(array.shape),separators=(",",":")).encode()
        self.digest.update(len(shape).to_bytes(4,"little"));self.digest.update(shape);self.digest.update(array.tobytes())
        x=value.astype(np.float64,copy=False)
        self.values+=x.size;self.reference_squared+=float(np.square(x).sum())
        self.minimum=float(x.min()) if self.minimum is None else min(self.minimum,float(x.min()))
        self.maximum=float(x.max()) if self.maximum is None else max(self.maximum,float(x.max()))
        for exponent,row in self.candidates.items():
            raw=_round_numpy(x/(2.0**exponent));codes=np.clip(raw,-128,127)
            error=codes*(2.0**exponent)-x
            row["squared_error"]+=float(np.square(error).sum())
            row["clipped_values"]+=int(np.count_nonzero((raw < -128)|(raw > 127)))
            row["zero_codes"]+=int(np.count_nonzero(codes == 0))
            row["maximum_absolute_error"]=max(row["maximum_absolute_error"],float(np.abs(error).max()))

    def choose(self,original):
        if not self.values or original not in self.candidates:raise ValueError("Need observed activations and the min/max control candidate")
        return min(self.candidates,key=lambda e:(self.candidates[e]["squared_error"],e != original,abs(e-original),e))

    def report(self):
        if not self.values:raise ValueError("No MSE activations observed")
        return dict(values=self.values,observed_minimum=self.minimum,observed_maximum=self.maximum,
            reference_mean_square=self.reference_squared/self.values,activation_stream_sha256=self.digest.hexdigest(),
            candidates={str(e):dict(**row,mse=row["squared_error"]/self.values,
                clipping_fraction=row["clipped_values"]/self.values,zero_code_fraction=row["zero_codes"]/self.values,
                represented_range=[-128*(2.0**e),127*(2.0**e)]) for e,row in self.candidates.items()})


def _fixed_names(calibration):
    fixed={name+".output" for name,recipe in calibration["recipes"].items() if recipe["kind"] in {"gru","tanh"}}
    fixed.update(name for name in calibration["grids"] if name.endswith(("intra_rnn.output","inter_rnn.output")))
    return fixed


def _validate_candidate(name,exponent,integer,modules):
    """Check input-dependent bias/GRU bounds and single-edge LN bounds."""
    base,suffix=name.rsplit(".",1)
    recipe=integer.backend.recipes.get(base)
    if recipe is None or suffix not in {"input","output"}:return
    kind=recipe["kind"]
    if suffix == "input" and kind in {"affine","stream"}:
        cls=IntegerStreamConv if kind == "stream" else IntegerAffine
        cls.from_torch(modules[base],ActivationGrid(exponent),integer.backend.grid(base+".output"),
                       batch_norm=modules.get(recipe.get("batch_norm")))
    elif suffix == "input" and kind == "gru":
        config=replace(integer.backend.base_gru,input_exponent=exponent)
        for direction in (("forward","reverse") if modules[base].bidirectional else ("forward",)):
            IntegerGRUCell.from_torch(modules[base],config,direction=direction)
    elif kind == "layer_norm":
        layer=modules[base]
        quantize_layer_norm(layer.weight,layer.bias,
            input_exponent=exponent if suffix == "input" else integer.backend.grid(base+".input").exponent,
            output_exponent=exponent if suffix == "output" else integer.backend.grid(base+".output").exponent,epsilon=layer.eps)


@torch.inference_mode()
def calibrate_gtcrn_mse(model,batches,*,max_batches=32,base_config=None,offsets=DEFAULT_OFFSETS):
    """Numerical callback; audited checkpoint wrapper establishes membership.

    There is one dataset selection iteration and two float neural passes on
    the retained crops: unchanged min/max calibration, then exact SSE scoring.
    Every grid stays fixed for a complete deployed stream. Independent local
    MSE minimization is not a guarantee of lower whole-model acoustic loss.
    """
    if isinstance(max_batches,bool) or not isinstance(max_batches,int) or max_batches < 1:
        raise ValueError("max_batches must be positive")
    if (not isinstance(offsets,(tuple,list)) or not offsets
            or any(isinstance(e,bool) or not isinstance(e,int) or not -8 <= e <= 8 for e in offsets)
            or 0 not in offsets or len(set(offsets)) != len(offsets)):
        raise ValueError("Candidate offsets must be distinct integers in [-8,8], including zero")
    selected=list(islice(batches,max_batches))
    source_hash=_source_fingerprint(model)
    control_grids,control=calibrate_gtcrn_integer(model,selected,max_batches=max_batches,base_config=base_config)
    integer=GTCRNIntegerDenoiser(model,control)
    shadow=GTCRNFloatShadow(model)
    fixed=_fixed_names(control);observers={};rejections={}
    for name,original in control_grids.items():
        candidates=[original] if name in fixed else sorted({original+offset for offset in offsets if -16 <= original+offset <= 8})
        accepted=[]
        for exponent in candidates:
            try:_validate_candidate(name,exponent,integer,shadow.backend.modules)
            except (ValueError,OverflowError) as error:
                rejections.setdefault(name,{})[str(exponent)]=str(error)
            else:accepted.append(exponent)
        if original not in accepted:raise ValueError(f"The original min/max grid failed its contract: {name}")
        observers[name]=_CandidateErrors(accepted)
    original_record=shadow.backend.record
    def record(name,value):
        if name in observers:observers[name].observe(value)
        return original_record(name,value)
    shadow.backend.record=record
    frames=0
    try:
        for batch in selected:
            for waveform in batch.detach().float().numpy():
                spectra=[]
                for _,_,spectrum,_ in _network_frames(waveform,integer.window,model.config):
                    spectra.append(torch.view_as_real(torch.from_numpy(spectrum))[None,:,None]);frames+=1
                shadow.sequence(torch.cat(spectra,dim=2))
    finally:shadow.backend.record=original_record
    chosen={name:observer.choose(control_grids[name]) for name,observer in observers.items()}
    joint=[]
    # Input epsilon and output beta scales interact. If individually admitted
    # minima form an invalid pair, choose the best jointly valid pair using
    # the sum of each boundary's error divided by its observed signal energy.
    for name,recipe in control["recipes"].items():
        if recipe["kind"] != "layer_norm":continue
        a,b=name+".input",name+".output";layer=shadow.backend.modules[name]
        def valid_pair(ie,oe):
            try:quantize_layer_norm(layer.weight,layer.bias,input_exponent=ie,output_exponent=oe,epsilon=layer.eps)
            except (ValueError,OverflowError):return False
            return True
        if valid_pair(chosen[a],chosen[b]):continue
        scores=[]
        for ie,oe in product(observers[a].candidates,observers[b].candidates):
            if valid_pair(ie,oe):
                score=sum(observers[n].candidates[e]["squared_error"]/max(observers[n].reference_squared,1e-30)
                          for n,e in ((a,ie),(b,oe)))
                scores.append((score,(ie != control_grids[a])+(oe != control_grids[b]),ie,oe))
        if not scores:raise ValueError(f"No jointly valid LayerNorm grids: {name}")
        _,_,ie,oe=min(scores)
        joint.append(dict(name=name,independent_choice=[chosen[a],chosen[b]],valid_choice=[ie,oe],
                          reason="joint variance/beta arithmetic bound"))
        chosen[a],chosen[b]=ie,oe
    report=deepcopy(control)
    report["method"]="training-only power-of-two activation MSE, every observed float value; original min/max candidate retained"
    report["grids"]=chosen
    report["mse_calibration"]=dict(candidate_offsets=list(offsets),selection_passes=1,float_neural_passes=2,
        frames_per_pass=frames,retained_training_waveform_bytes=sum(batch.numel()*batch.element_size() for batch in selected),
        retained_activation_sample_bytes=0,minmax_grids=control_grids,
        fixed_q7_edges=sorted(fixed),fixed_probability_encodings=deepcopy(control["probability_encodings"]),
        rejected_candidates=rejections,joint_constraint_adjustments=joint,
        edges={name:dict(**observer.report(),minmax_exponent=control_grids[name],selected_exponent=chosen[name],
                         fixed_contract=name in fixed) for name,observer in observers.items()},
        changed_edges={name:dict(minmax=control_grids[name],mse=chosen[name]) for name in chosen if chosen[name] != control_grids[name]},
        scoring="Sum of squared reconstruction error over every float training activation at each edge, accumulated in float64",
        ties="Prefer original min/max exponent, then nearest exponent, then finer exponent",
        limitations="Local MSE does not model upstream quantization, edge coupling or waveform quality; no development audio is used",
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    GTCRNIntegerDenoiser(model,report)  # Validate all selected graph contracts together.
    if source_hash != _source_fingerprint(model):raise ValueError("Source changed during MSE calibration")
    return chosen,report


def from_checkpoint_training_mse(checkpoint_path,evaluation_manifest,*,crops=32,seed=483,offsets=DEFAULT_OFFSETS):
    from .gtcrn_recurrent_probe import calibrate_checkpoint_training
    path=Path(checkpoint_path).resolve();contents=path.read_bytes()
    saved=torch.load(io.BytesIO(contents),map_location="cpu",weights_only=False)
    if saved.get("model_kind") != "gtcrn" or saved.get("phase","float") != "float":
        raise ValueError("MSE preparation requires a frozen float GTCRN checkpoint")
    with torch.random.fork_rng(devices=[]):source=GTCRNDenoiser(GTCRNConfig.from_checkpoint(saved["model_config"]))
    source.load_state_dict(saved["model"],strict=True);source.eval().requires_grad_(False)
    def calibrator(model,batches,*,max_batches,base_config=None):
        return calibrate_gtcrn_mse(model,batches,max_batches=max_batches,base_config=base_config,offsets=offsets)
    _,audit=calibrate_checkpoint_training(source,saved,evaluation_manifest,crops=crops,seed=seed,calibrator=calibrator)
    if path.read_bytes() != contents:raise ValueError("Frozen checkpoint changed during calibration")
    audit.update(source_checkpoint=str(path),source_checkpoint_sha256=hashlib.sha256(contents).hexdigest(),
        development_manifest_usage="Split/source-disjointness audit only; no development waveform loading or fitting")
    return GTCRNIntegerDenoiser(source,audit),source,audit


def main():
    from .gtcrn_integer_export import _json_bytes,pack_gtcrn_integer,load_gtcrn_integer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--calibration-crops",type=int,default=32)
    parser.add_argument("--calibration-seed",type=int,default=483)
    parser.add_argument("--candidate-offsets",type=int,nargs="+",default=list(DEFAULT_OFFSETS))
    parser.add_argument("--threads",type=int,default=1)
    args=parser.parse_args()
    if args.output.suffix != ".bin":parser.error("Output must end in .bin")
    if args.threads < 1:parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    integer,_,audit=from_checkpoint_training_mse(args.checkpoint,args.manifest,crops=args.calibration_crops,
        seed=args.calibration_seed,offsets=args.candidate_offsets)
    data=pack_gtcrn_integer(integer);loaded=load_gtcrn_integer(data,calibration=audit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(data)
    args.output.with_suffix(".calibration.json").write_bytes(_json_bytes(audit)+b"\n")
    args.output.with_suffix(".json").write_text(json.dumps(loaded.model_stats(),indent=2)+"\n")
    print(json.dumps(dict(**loaded.packed_metadata,changed_grids=len(audit["mse_calibration"]["changed_edges"]))))


if __name__ == "__main__":main()
