"""Training-only range selection, exact SSE, fixed contracts and packed C parity."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from esp32_denoiser.gtcrn_integer import GTCRNIntegerDenoiser,calibrate_gtcrn_integer,_source_fingerprint
from esp32_denoiser.gtcrn_integer_export import pack_gtcrn_integer,load_gtcrn_integer
from esp32_denoiser.gtcrn_mse_calibration import _CandidateErrors,calibrate_gtcrn_mse,from_checkpoint_training_mse,main
from esp32_denoiser.gtcrn_native import CIntegerGTCRN
from test_gtcrn_integer import _source,_audio
from test_gtcrn_recurrent_probe import broad_checkpoint


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


def test_exact_streamed_sse_chooses_a_finer_clipping_grid_without_sampling():
    values=np.r_[np.full(100000,.02,np.float32),np.float32(7.9)]
    observer=_CandidateErrors([-7,-6,-5,-4,-3])
    observer.observe(values[:54321]);observer.observe(values[54321:])
    report=observer.report()
    assert report["values"] == len(values) and observer.choose(-4) == -5
    for exponent,row in report["candidates"].items():
        scale=2.0**int(exponent);x=values.astype(np.float64)
        raw=np.copysign(np.floor(np.abs(x/scale)+.5),x)
        error=np.clip(raw,-128,127)*scale-x
        assert row["squared_error"] == pytest.approx(float(np.dot(error,error)),rel=1e-13)
        assert row["clipped_values"] == int(np.count_nonzero((raw < -128)|(raw > 127)))
    assert report["candidates"]["-5"]["clipped_values"] == 1
    assert report["candidates"]["-4"]["clipped_values"] == 0
    zero=_CandidateErrors([-7,-6,-5]);zero.observe(np.zeros(25,np.float32))
    assert zero.choose(-6) == -6


def test_mse_preserves_control_and_fixed_encodings_and_is_deterministic():
    source=_source();batch=[_audio(769)]
    _,control=calibrate_gtcrn_integer(source,batch,max_batches=1)
    before=deepcopy(control);source_sha=_source_fingerprint(source);rng=torch.random.get_rng_state().clone()
    grids,audit=calibrate_gtcrn_mse(source,batch,max_batches=1)
    repeated,repeated_audit=calibrate_gtcrn_mse(source,batch,max_batches=1)
    assert grids == repeated and audit == repeated_audit
    assert control == before and source_sha == _source_fingerprint(source)
    assert torch.equal(rng,torch.random.get_rng_state())
    detail=audit["mse_calibration"]
    assert detail["minmax_grids"] == control["grids"]
    assert len(detail["edges"]) == 182 and len(detail["fixed_probability_encodings"]) == 6
    assert audit["probability_encodings"] == control["probability_encodings"]
    assert detail["changed_edges"]
    for name,entry in detail["edges"].items():
        assert str(entry["minmax_exponent"]) in entry["candidates"]
        if not detail["joint_constraint_adjustments"]:
            assert entry["candidates"][str(entry["selected_exponent"])]["mse"] <= entry["candidates"][str(entry["minmax_exponent"])]["mse"]
        if entry["fixed_contract"]:assert entry["selected_exponent"] == entry["minmax_exponent"] == -7
        assert len(entry["activation_stream_sha256"]) == 64
    for name,recipe in audit["recipes"].items():
        if recipe["kind"] == "gru":assert -12 <= grids[name+".input"] <= 0
    assert detail["selection_passes"] == 1 and detail["float_neural_passes"] == 2
    assert detail["retained_activation_sample_bytes"] == 0
    assert detail["retained_training_waveform_bytes"] == batch[0].numel()*4


def test_zero_offset_reproduces_minmax_model_and_candidate_stream_is_bounded():
    source=_source();batch=_audio(257)
    calls=[]
    def selected():
        calls.append(1);yield batch
        raise AssertionError("Overconsumed calibration stream")
    grids,audit=calibrate_gtcrn_mse(source,selected(),max_batches=1,offsets=(0,))
    _,control=calibrate_gtcrn_integer(source,[batch],max_batches=1)
    assert calls == [1] and grids == control["grids"]
    assert not audit["mse_calibration"]["changed_edges"]
    expected=GTCRNIntegerDenoiser(source,control)(batch)
    torch.testing.assert_close(GTCRNIntegerDenoiser(source,audit)(batch),expected,rtol=0,atol=0)


def test_selected_mse_grids_pack_and_run_exactly_in_complete_c_graph():
    source=_source();_,audit=calibrate_gtcrn_mse(source,[_audio(769)],max_batches=1)
    integer=GTCRNIntegerDenoiser(source,audit);data=pack_gtcrn_integer(integer)
    reference=load_gtcrn_integer(data,calibration=audit);native=CIntegerGTCRN(data,calibration=audit)
    assert len(data) <= 99000 and reference.packed_metadata["calibration_audit_loaded"]
    waveform=_audio(513)
    torch.testing.assert_close(reference(waveform),integer(waveform),rtol=0,atol=0)
    torch.testing.assert_close(native(waveform),integer(waveform),rtol=0,atol=0)
    tampered=deepcopy(audit);tampered["mse_calibration"]["candidate_offsets"]=[0]
    with pytest.raises(ValueError,match="audit hash"):
        load_gtcrn_integer(data,calibration=tampered)


def test_audited_cli_selects_training_once_and_exports_without_development_audio(broad_checkpoint,tmp_path,monkeypatch):
    from esp32_denoiser.data import PairedAudioDataset
    from esp32_denoiser.extra_data import DynamicMixtureDataset
    _,saved,validation=broad_checkpoint;saved["model_config"]["normalize_input"]=True
    checkpoint=tmp_path/"frozen.pt";torch.save(saved,checkpoint)
    contents=checkpoint.read_bytes();calls=[]
    for cls,label in ((PairedAudioDataset,"paired"),(DynamicMixtureDataset,"synthetic")):
        original=cls.__getitem__
        def wrapped(self,index,_original=original,_label=label):
            calls.append(_label);return _original(self,index)
        monkeypatch.setattr(cls,"__getitem__",wrapped)
    for row in map(json.loads,validation.read_text().splitlines()):
        for role in ("clean","noisy"):Path(row[role]).write_bytes(b"not decodable audio")
    output=tmp_path/"mse.bin"
    monkeypatch.setattr("sys.argv",["mse","--checkpoint",str(checkpoint),"--manifest",str(validation),
        "--output",str(output),"--calibration-crops","4","--calibration-seed","483"])
    main()
    audit=json.loads(output.with_suffix(".calibration.json").read_text())
    assert len(calls) == 4 and set(calls) == {"paired","synthetic"}
    assert audit["source_checkpoint_sha256"] == hashlib.sha256(contents).hexdigest()
    assert checkpoint.read_bytes() == contents
    assert len(audit["crops"]) == 4 and audit["mse_calibration"]["selection_passes"] == 1
    assert "no development waveform" in audit["development_manifest_usage"]
    assert CIntegerGTCRN(output,calibration=audit).model_stats()["packed_bytes"] <= 99000
    rows=[dict(json.loads(line),source_split="test") for line in validation.read_text().splitlines()]
    validation.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError,match="test remains sealed"):
        from_checkpoint_training_mse(checkpoint,validation,crops=1)


@pytest.mark.parametrize("offsets",[(),(-1,1),(0,0),(0,True),(0,9)])
def test_invalid_candidates_fail_without_consuming_training(offsets):
    class Unreadable:
        def __iter__(self):raise AssertionError("Consumed invalid request")
    with pytest.raises(ValueError,match="Candidate offsets"):
        calibrate_gtcrn_mse(_source(),Unreadable(),offsets=offsets)
