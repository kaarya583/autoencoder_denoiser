"""Training-only localization, honest decomposition, and immutable source."""
import hashlib
import json

import numpy as np
import pytest
import torch

from esp32_denoiser.gtcrn_integer import GTCRNIntegerDenoiser, _source_fingerprint, calibrate_gtcrn_integer
from esp32_denoiser.gtcrn_integer_diagnostics import (
    _decode, _encode, _prior_gru_state, diagnose_checkpoint_training, diagnose_integer_training,
)
from esp32_denoiser.gtcrn_model import GTCRNConfig, GTCRNDenoiser
from test_gtcrn_recurrent_probe import broad_checkpoint


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


@pytest.fixture(scope="module")
def prepared():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(413)
        source=GTCRNDenoiser(GTCRNConfig(normalize_input=True)).eval()
        batches=[torch.randn(1,769)*.04]
    _,calibration=calibrate_gtcrn_integer(source,batches,max_batches=1)
    return source,calibration,batches


def test_localization_tracks_every_boundary_and_preserves_source(prepared):
    source,calibration,batches=prepared
    before=_source_fingerprint(source)
    integer=GTCRNIntegerDenoiser(source,calibration)
    original=integer.backend.record
    result=diagnose_integer_training(source,integer,batches,probe_crops=1,max_frames_per_crop=2)
    assert integer.backend.record == original
    assert _source_fingerprint(source) == before
    assert result["frames"] == 2 and len(result["edges"]) == 188 and len(result["operators"]) == 84
    assert len(result["parameters"]) == 65
    for entry in result["operators"].values():
        assert entry["total_local"]["values"] == entry["after_reference_output_rounding"]["values"]
        assert entry["total_local"]["values"] == entry["reference_output_grid_only"]["values"]
    assert result["probe_crops"][0]["noisy_crop_sha256"] == hashlib.sha256(batches[0].numpy().astype("<f4").tobytes()).hexdigest()
    assert "not establish clipping" in result["interpretation"]["integer_rail_contacts"]
    assert "33 frequency steps" in diagnose_integer_training.__doc__
    json.dumps(result,allow_nan=False)


def test_direct_weight_error_is_separated_from_upstream_and_grid_error(prepared):
    source,calibration,batches=prepared
    baseline=GTCRNIntegerDenoiser(source,calibration)
    changed=GTCRNIntegerDenoiser(source,calibration)
    name="encoder.en_convs.0.conv"
    operation=changed.backend.ops[name]
    operation.weights=np.zeros_like(operation.weights)
    operation.bias=np.zeros_like(operation.bias)
    before=diagnose_integer_training(source,baseline,batches,probe_crops=1,max_frames_per_crop=1)
    after=diagnose_integer_training(source,changed,batches,probe_crops=1,max_frames_per_crop=1)
    # Identical input-grid loss; local original-weight reference is the same.
    assert before["edges"][name+".input"] == after["edges"][name+".input"]
    assert before["operators"][name]["reference_output_grid_only"] == after["operators"][name]["reference_output_grid_only"]
    assert after["operators"][name]["total_local"]["relative_l2"] == pytest.approx(1)
    assert after["operators"][name]["after_reference_output_rounding"]["rmse"] > before["operators"][name]["after_reference_output_rounding"]["rmse"]
    assert after["parameters"][name]["weights"]["quantized_zero_fraction"] == 1


def test_probability_encoding_and_temporal_state_slices_are_explicit(prepared):
    integer=GTCRNIntegerDenoiser(prepared[0],prepared[1])
    name=next(iter(integer.backend.probability_encodings))
    codes=np.arange(-128,128,dtype=np.int16).astype(np.int8)
    decoded=_decode(name,codes,integer.backend)
    np.testing.assert_array_equal(decoded,(codes.astype(np.float64)+128)/255)
    assert decoded[0] == 0 and decoded[-1] == 1 and decoded[128] == 128/255
    actual,outside=_encode(name,decoded,integer.backend)
    np.testing.assert_array_equal(actual,codes);assert outside == 0
    state=np.arange(33*16,dtype=np.int16).astype(np.int8).reshape(1,33,16)
    previous={"dpgrnn1.inter_rnn":state}
    np.testing.assert_array_equal(_prior_gru_state("dpgrnn1.inter_rnn.rnn1",previous),state[...,:8])
    np.testing.assert_array_equal(_prior_gru_state("dpgrnn1.inter_rnn.rnn2",previous),state[...,8:])
    assert _prior_gru_state("dpgrnn1.intra_rnn.rnn1",previous) is None


def test_checkpoint_diagnostics_select_once_and_never_read_development_audio(broad_checkpoint,tmp_path,monkeypatch):
    from esp32_denoiser.data import PairedAudioDataset
    from esp32_denoiser.extra_data import DynamicMixtureDataset
    source,saved,validation=broad_checkpoint
    saved["model_config"]["normalize_input"]=True
    checkpoint=tmp_path/"frozen.pt";torch.save(saved,checkpoint)
    original=checkpoint.read_bytes()
    calls=[]
    for cls,label in ((PairedAudioDataset,"paired"),(DynamicMixtureDataset,"synthetic")):
        method=cls.__getitem__
        def capture(self,index,_method=method,_label=label):
            calls.append(_label)
            return _method(self,index)
        monkeypatch.setattr(cls,"__getitem__",capture)
    from pathlib import Path
    for row in map(json.loads,validation.read_text().splitlines()):
        for role in ("noisy","clean"):Path(row[role]).write_bytes(b"deliberately not audio")
    report=diagnose_checkpoint_training(checkpoint,validation,crops=4,probe_crops=4,max_frames_per_crop=1)
    assert len(calls) == 4 and set(calls) == {"paired","synthetic"}
    assert report["selection_passes"] == 1 and report["localization"]["frames"] == 4
    assert report["source_checkpoint_sha256"] == hashlib.sha256(original).hexdigest()
    assert checkpoint.read_bytes() == original
    assert [r["noisy_crop_sha256"] for r in report["crops"]] == [r["noisy_crop_sha256"] for r in report["localization"]["probe_crops"]]
    assert "no development audio" in report["development_manifest_usage"]
    rows=[dict(json.loads(line),source_split="test") for line in validation.read_text().splitlines()]
    validation.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError,match="test remains sealed"):
        diagnose_checkpoint_training(checkpoint,validation,crops=1,probe_crops=1,max_frames_per_crop=1)


def test_invalid_controls_fail_before_checkpoint_io(tmp_path):
    for kwargs in (dict(crops=0),dict(probe_crops=True),dict(max_frames_per_crop=-1),dict(crops=1,probe_crops=2)):
        with pytest.raises(ValueError):diagnose_checkpoint_training(tmp_path/"missing.pt",tmp_path/"missing.jsonl",**kwargs)


def test_record_hook_restored_when_local_diagnostic_fails(prepared,monkeypatch):
    import esp32_denoiser.gtcrn_integer_diagnostics as diagnostics
    source,calibration,batches=prepared
    integer=GTCRNIntegerDenoiser(source,calibration);original=integer.backend.record
    def fail(*args,**kwargs):raise RuntimeError("local diagnostic failed")
    monkeypatch.setattr(diagnostics,"_local_reference",fail)
    with pytest.raises(RuntimeError,match="local diagnostic failed"):
        diagnose_integer_training(source,integer,batches,probe_crops=1,max_frames_per_crop=1)
    assert integer.backend.record == original
