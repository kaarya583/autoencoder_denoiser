"""Training-only localization of whole-graph GTCRN PTQ error.

One audited dataset iteration selects the calibration crops. Those frozen
tensors supply calibration and a bounded numerical diagnostic pass. No
development waveform is read, no grid/weight is fitted to development data,
and no checkpoint or graph parameter is modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
from itertools import islice
import json
import math
from pathlib import Path

import numpy as np
import torch

from .experimental_gru import _round_numpy
from .experimental_gtcrn_ops import IntegerAffine, IntegerPReLU, IntegerStreamConv, folded_parameters
from .gtcrn_integer import (
    GTCRNFloatShadow, GTCRNIntegerDenoiser, _network_frames, _source_fingerprint, calibrate_gtcrn_integer,
)
from .gtcrn_model import GTCRNConfig, GTCRNDenoiser


class _Error:
    def __init__(self):
        self.count=0
        self.squared_error=self.squared_reference=self.absolute_error=self.maximum_error=0.0

    def add(self, actual, reference):
        actual,reference=np.asarray(actual,np.float64),np.asarray(reference,np.float64)
        if actual.shape != reference.shape or not actual.size or not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ValueError("Diagnostic tensors must have matching nonempty finite shapes")
        error=actual-reference
        self.count+=actual.size
        self.squared_error+=float(np.square(error).sum())
        self.squared_reference+=float(np.square(reference).sum())
        self.absolute_error+=float(np.abs(error).sum())
        self.maximum_error=max(self.maximum_error,float(np.abs(error).max()))

    def report(self):
        return dict(values=self.count,rmse=math.sqrt(self.squared_error/self.count),
                    reference_rms=math.sqrt(self.squared_reference/self.count),
                    mean_absolute_error=self.absolute_error/self.count,max_absolute_error=self.maximum_error,
                    relative_l2=math.sqrt(self.squared_error/self.squared_reference) if self.squared_reference else None,
                    sqnr_db=10*math.log10(self.squared_reference/self.squared_error)
                    if self.squared_error and self.squared_reference else None,
                    exact=self.squared_error == 0)


def _decode(name,codes,backend):
    codes=np.asarray(codes,np.float64)
    if name in backend.probability_encodings:
        return (codes+128)/255
    return codes*backend.grid(name).scale


def _encode(name,value,backend):
    if name in backend.probability_encodings:
        raw=_round_numpy(np.asarray(value,np.float64)*255)-128
    else:
        raw=_round_numpy(np.asarray(value,np.float64)/backend.grid(name).scale)
    return raw.clip(-128,127).astype(np.int8),int(np.count_nonzero((raw < -128)|(raw > 127)))


def _parameter_error(actual,reference):
    summary=_Error();summary.add(actual,reference)
    actual,reference=np.asarray(actual),np.asarray(reference)
    result=summary.report()
    result.update(quantized_zero_fraction=float(np.mean(actual == 0)),
                  original_zero_fraction=float(np.mean(reference == 0)),
                  newly_zero_fraction=float(np.mean((actual == 0)&(reference != 0))))
    return result


def _parameter_reports(integer,modules):
    reports={}
    for name,operation in integer.backend.ops.items():
        if isinstance(operation,IntegerStreamConv):operation=operation.affine
        layer=modules[name]
        if isinstance(operation,IntegerAffine):
            if hasattr(layer,"Conv2d"):layer=layer.Conv2d
            elif hasattr(layer,"ConvTranspose2d"):layer=layer.ConvTranspose2d
            bn=modules.get(integer.backend.recipes[name].get("batch_norm"))
            weight,bias=folded_parameters(layer,bn)
            scales=np.exp2(operation.exponents.astype(np.float64))
            decoded=operation.weights.astype(np.float64)*scales.reshape((-1,)+(1,)*(weight.ndim-1))
            rows=[_parameter_error(decoded[i],weight[i]) for i in range(len(weight))]
            reports[name]=dict(kind="folded_affine",weights=_parameter_error(decoded,weight),
                               bias=_parameter_error(operation.bias.astype(np.float64)*scales*operation.input_grid.scale,bias),
                               worst_output_rows=sorted((dict(row=i,**row) for i,row in enumerate(rows)),
                                    key=lambda r:r["relative_l2"] if r["relative_l2"] is not None else -1,reverse=True)[:3])
        elif isinstance(operation,IntegerPReLU):
            reports[name]=dict(kind="prelu",slopes=_parameter_error(operation.slopes.astype(np.float64)*2.0**operation.exponent,
                                                layer.weight.detach().numpy()))
        elif isinstance(operation,tuple):
            directions={}
            for index,cell in enumerate(operation):
                suffix="_reverse" if index else ""
                arrays={}
                for side in ("ih","hh"):
                    scales=np.exp2(getattr(cell,"exponent_"+side).astype(np.float64))
                    weights=getattr(cell,"weight_"+side).astype(np.float64)*scales[:,None]
                    input_scale=2.0**(cell.config.input_exponent if side == "ih" else cell.config.state_exponent)
                    bias=getattr(cell,"bias_"+side).astype(np.float64)*scales*input_scale
                    arrays["weight_"+side]=_parameter_error(weights,getattr(layer,"weight_"+side+"_l0"+suffix).detach().numpy())
                    arrays["bias_"+side]=_parameter_error(bias,getattr(layer,"bias_"+side+"_l0"+suffix).detach().numpy())
                directions["reverse" if index else "forward"]=arrays
            reports[name]=dict(kind="gru",directions=directions)
        else:
            reports[name]=dict(kind="layer_norm",
                gamma=_parameter_error(operation.gamma.astype(np.float64)*2.0**operation.gamma_exponent,layer.weight.detach().numpy()),
                beta=_parameter_error(operation.beta.astype(np.float64)*2.0**operation.output_exponent,layer.bias.detach().numpy()))
    return reports


def _prior_gru_state(name,previous):
    if name in previous:return previous[name]
    if ".inter_rnn.rnn" in name:
        prefix,part=name.rsplit(".rnn",1)
        if prefix in previous:
            state=previous[prefix];half=state.shape[-1]//2;index=int(part)-1
            return state[...,index*half:(index+1)*half]
    return None  # Bidirectional frequency GRUs reset for every causal frame.


@torch.inference_mode()
def _local_reference(name,recipe,trace,previous,backend,modules):
    value=_decode(name+".input",trace[name+".input"],backend)
    x=torch.from_numpy(value.astype(np.float32))
    kind=recipe["kind"]
    if kind == "stream":
        operation=backend.ops[name]
        history=previous.get(name)
        if history is None:history=operation.initial_state(1,x.shape[-1])
        history=torch.from_numpy(history.astype(np.float32)*np.float32(operation.affine.input_grid.scale))
        y,_=modules[name](x,history)
    elif kind == "gru":
        history=_prior_gru_state(name,previous)
        history=None if history is None else torch.from_numpy(history.astype(np.float32)/128)
        y,_=modules[name](x,history)
    elif kind == "energy":y=x.square().mean(-1)
    elif kind == "product":
        gate_name=name.removesuffix(".product")+".att_act.output"
        gate=torch.from_numpy(_decode(gate_name,trace[gate_name],backend).astype(np.float32))
        y=x*gate[...,None]
    else:y=modules[name](x)
    if recipe.get("batch_norm"):y=modules[recipe["batch_norm"]](y)
    return y.detach().numpy()


@torch.inference_mode()
def diagnose_integer_training(source,integer,batches,*,probe_crops=4,max_frames_per_crop=64):
    """Numerical helper; production callers must use the audited factory.

    All operator-local float references receive the *actual decoded integer*
    input and old integer state. Their discrepancy therefore excludes error
    already introduced upstream. GRU local error includes recurrence within
    that one operator call (33 frequency steps for intra-frame GRUs).
    """
    for name,value in (("probe_crops",probe_crops),("max_frames_per_crop",max_frames_per_crop)):
        if isinstance(value,bool) or not isinstance(value,int) or value < 1:raise ValueError(f"{name} must be positive")
    if integer.calibration["source_state_sha256"] != _source_fingerprint(source):
        raise ValueError("Diagnostic source differs from the calibrated source")
    shadow=GTCRNFloatShadow(source)
    modules=shadow.backend.modules
    float_trace,int_trace={},{}
    original_float,original_integer=shadow.backend.record,integer.backend.record
    def capture_float(name,value):
        float_trace[name]=value.detach().numpy().copy()
        return original_float(name,value)
    def capture_integer(name,value):
        int_trace[name]=value.copy()
        return original_integer(name,value)
    shadow.backend.record=capture_float
    integer.backend.record=capture_integer
    edges,local={},{}
    crop_reports=[]
    try:
        for crop_index,batch in enumerate(islice(batches,probe_crops)):
            if not isinstance(batch,torch.Tensor) or batch.device.type != "cpu" or batch.ndim != 2 or batch.shape[0] != 1:
                raise ValueError("Diagnostics require individual CPU training crops [1,N]")
            waveform=batch.detach().float().numpy()[0]
            float_state,int_state={},{}
            count=0
            for _,_,spectrum,_ in islice(_network_frames(waveform,integer.window,source.config),max_frames_per_crop):
                float_trace.clear();int_trace.clear()
                tensor=torch.view_as_real(torch.from_numpy(spectrum))[None,:,None]
                _,float_state=shadow.frame(tensor,float_state)
                previous=int_state
                _,int_state=integer.spectrum_frame(spectrum,int_state)
                if float_trace.keys() != int_trace.keys():raise ValueError("Float/integer trace boundaries differ")
                for name,reference in float_trace.items():
                    codes=int_trace[name];decoded=_decode(name,codes,integer.backend)
                    q_reference,outside=_encode(name,reference,integer.backend)
                    entry=edges.setdefault(name,dict(propagated=_Error(),float_edge_grid_only=_Error(),
                        float_edge_values_outside_grid=0,integer_rail_contacts=0,integer_zero_codes=0))
                    entry["propagated"].add(decoded,reference)
                    entry["float_edge_grid_only"].add(_decode(name,q_reference,integer.backend),reference)
                    entry["float_edge_values_outside_grid"]+=outside
                    entry["integer_rail_contacts"]+=int(np.count_nonzero((codes == -128)|(codes == 127)))
                    entry["integer_zero_codes"]+=int(np.count_nonzero(codes == 0))
                for name,recipe in integer.backend.recipes.items():
                    reference=_local_reference(name,recipe,int_trace,previous,integer.backend,modules)
                    out=name+".output";decoded=_decode(out,int_trace[out],integer.backend)
                    q_reference,outside=_encode(out,reference,integer.backend)
                    rounded=_decode(out,q_reference,integer.backend)
                    entry=local.setdefault(name,dict(kind=recipe["kind"],total_local=_Error(),
                        after_reference_output_rounding=_Error(),reference_output_grid_only=_Error(),
                        local_reference_values_outside_grid=0))
                    entry["total_local"].add(decoded,reference)
                    entry["after_reference_output_rounding"].add(decoded,rounded)
                    entry["reference_output_grid_only"].add(rounded,reference)
                    entry["local_reference_values_outside_grid"]+=outside
                count+=1
            crop_reports.append(dict(index=crop_index,frames=count,
                noisy_crop_sha256=hashlib.sha256(waveform.astype("<f4").tobytes()).hexdigest()))
    finally:
        shadow.backend.record=original_float
        integer.backend.record=original_integer
    if not crop_reports:raise ValueError("At least one diagnostic training crop is required")
    for rows in (edges,local):
        for entry in rows.values():
            for key,value in list(entry.items()):
                if isinstance(value,_Error):entry[key]=value.report()
    top=lambda rows,key:[dict(name=name,**entry) for name,entry in sorted(rows.items(),
        key=lambda pair:pair[1][key]["relative_l2"] if pair[1][key]["relative_l2"] is not None else -1,reverse=True)[:20]]
    return dict(scope="Training-only numerical localization; no sensitivity recovery, changed grids, development calibration or quality claim",
        frames=sum(row["frames"] for row in crop_reports),probe_crops=crop_reports,
        max_frames_per_crop=max_frames_per_crop,edges=edges,operators=local,
        parameters=_parameter_reports(integer,modules),
        largest_relative_propagated_edges=top(edges,"propagated"),
        largest_relative_local_operators=top(local,"total_local"),
        interpretation={"propagated":"Integer decoded edge versus original float graph on identical spectra; includes upstream error",
            "float_edge_grid_only":"Original float edge quantized once on its assigned encoding; isolates boundary resolution/range",
            "total_local":"Actual integer output versus original float operator on actual decoded integer inputs and prior state",
            "after_reference_output_rounding":"Same local reference rounded to actual output encoding; exposes weight/internal arithmetic effects",
            "integer_rail_contacts":"Contacts include valid endpoints and do not establish clipping",
            "integer_zero_codes":"A signed probability code of zero means128/255, not a zero probability",
            "ranking":"Relative L2 may be unstable on near-zero references; inspect reference RMS and absolute error alongside it",
            "limits":"Does not independently toggle classes to float or attribute waveform SI-SDR causally; residual/reshape errors appear in edge statistics"})


def diagnose_checkpoint_training(checkpoint_path,evaluation_manifest,*,crops=32,seed=483,probe_crops=4,max_frames_per_crop=64):
    from .gtcrn_recurrent_probe import calibrate_checkpoint_training
    for name,value in (("crops",crops),("probe_crops",probe_crops),("max_frames_per_crop",max_frames_per_crop)):
        if isinstance(value,bool) or not isinstance(value,int) or value < 1:raise ValueError(f"{name} must be positive")
    if probe_crops > crops:raise ValueError("Diagnostic crops must be a subset of calibration crops")
    path=Path(checkpoint_path).resolve();contents=path.read_bytes()
    saved=torch.load(io.BytesIO(contents),map_location="cpu",weights_only=False)
    if saved.get("model_kind") != "gtcrn" or saved.get("phase","float") != "float":
        raise ValueError("Diagnostics require a frozen float GTCRN checkpoint")
    with torch.random.fork_rng(devices=[]):source=GTCRNDenoiser(GTCRNConfig.from_checkpoint(saved["model_config"]))
    source.load_state_dict(saved["model"],strict=True);source.eval().requires_grad_(False)
    def calibrator(model,batches,*,max_batches,base_config=None):
        # Exactly one dataset iteration; reuse these bounded tensors instead
        # of selecting another set or reading any development waveforms.
        selected=list(islice(batches,max_batches))
        grids,report=calibrate_gtcrn_integer(model,selected,max_batches=max_batches,base_config=base_config)
        integer=GTCRNIntegerDenoiser(model,report)
        report["localization"]=diagnose_integer_training(model,integer,selected,probe_crops=probe_crops,
                                                         max_frames_per_crop=max_frames_per_crop)
        return grids,report
    _,report=calibrate_checkpoint_training(source,saved,evaluation_manifest,crops=crops,seed=seed,calibrator=calibrator)
    if path.read_bytes() != contents:raise ValueError("Frozen checkpoint changed during diagnosis")
    report.update(source_checkpoint=str(path),source_checkpoint_sha256=hashlib.sha256(contents).hexdigest(),
                  model_config=asdict(source.config),selection_passes=1,
                  development_manifest_usage="Audit of split/IDs/source overlap only; no development audio read or parameters fitted")
    return report


def main():
    from .evaluate import _json_finite
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True,help="Development manifest used only to audit training disjointness")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--calibration-crops",type=int,default=32)
    parser.add_argument("--seed",type=int,default=483)
    parser.add_argument("--probe-crops",type=int,default=4)
    parser.add_argument("--max-frames-per-crop",type=int,default=64)
    parser.add_argument("--threads",type=int,default=1)
    args=parser.parse_args()
    if args.threads < 1:parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    report=diagnose_checkpoint_training(args.checkpoint,args.manifest,crops=args.calibration_crops,seed=args.seed,
                                        probe_crops=args.probe_crops,max_frames_per_crop=args.max_frames_per_crop)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(_json_finite(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output),frames=report["localization"]["frames"],selection_passes=1)))


if __name__ == "__main__":main()
