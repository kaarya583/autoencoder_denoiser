"""Packed-model replay and deliberately rehashed malformed payloads."""
from copy import deepcopy
import hashlib
import json
import struct

import numpy as np
import pytest
import torch
from torch import nn

from esp32_denoiser.gtcrn_integer import GTCRNIntegerDenoiser, calibrate_gtcrn_integer
from esp32_denoiser.gtcrn_integer_export import (
    AFFINE, ARRAY_OFFSET, DIGEST_OFFSET, GRU, GRID_NAMES, GRID_OFFSET, HEADER,
    LAYER_NORM, MAX_MODEL_BYTES, RECORD, RECORD_OFFSET, SPECS, WINDOW,
    load_gtcrn_integer, pack_gtcrn_integer, _digest,
)
from test_gtcrn_integer import _audio, _source
from test_gtcrn_recurrent_probe import broad_checkpoint


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module", params=[False, True], ids=["raw", "frame_rms"])
def packed(request):
    torch.set_num_threads(1)
    source = _source(request.param)
    # Avoid the artificially high deduplication of untrained identical LN
    # parameters; signed gamma and nonzero beta also exercise wider bounds.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(849)
        for layer in source.modules():
            if isinstance(layer, nn.LayerNorm):
                layer.weight.uniform_(-1.2, 1.2)
                layer.bias.uniform_(-.1, .1)
    _, calibration = calibrate_gtcrn_integer(source, [_audio(513)], max_batches=1)
    integer = GTCRNIntegerDenoiser(source, calibration)
    blob = pack_gtcrn_integer(integer)
    return integer, blob


def _resign(data):
    data = bytearray(data)
    data[DIGEST_OFFSET:DIGEST_OFFSET+32] = _digest(bytes(data))
    return bytes(data)


def _record(blob, kind):
    index = next(index for index, spec in enumerate(SPECS) if spec.kind == kind)
    return RECORD_OFFSET + index * RECORD.size


def _ref(blob, kind, slot=0):
    return struct.unpack_from("<I", blob, _record(blob, kind)+24+slot*4)[0]


def test_complete_packed_size_audit_binding_and_identical_reserialization(packed, tmp_path):
    integer, blob = packed
    assert HEADER.size == 192 and RECORD.size == 80
    assert (len(GRID_NAMES), len(SPECS), ARRAY_OFFSET) == (182, 78, 6624)
    assert len(blob) < 60_000 < MAX_MODEL_BYTES
    path = tmp_path / "model.bin"
    path.write_bytes(blob)
    loaded = load_gtcrn_integer(path, calibration=integer.calibration)
    assert pack_gtcrn_integer(loaded) == blob
    assert pack_gtcrn_integer(load_gtcrn_integer(blob)) == blob
    assert loaded.config == integer.config
    assert loaded.backend.probability_encodings == integer.backend.probability_encodings
    assert loaded.packed_metadata["packed_sha256"] == hashlib.sha256(blob).hexdigest()
    stats = loaded.model_stats()
    assert stats["packed_bytes"] == stats["metadata_bytes"] + stats["unique_array_bytes"] + stats["padding_bytes"]
    assert stats["neural_state_bytes_int8"] == 18_048
    assert stats["calibration_audit_loaded"]
    assert "complete C graph" in stats["deployment_status"]
    altered = deepcopy(integer.calibration)
    altered["frames"] += 1
    with pytest.raises(ValueError, match="audit hash"):
        load_gtcrn_integer(blob, calibration=altered)
    directions = [row for row in loaded.packed_descriptors if row["kind"] == GRU]
    # All18 GRU directions share one exact table pair at the common logit grid.
    assert len(directions) == 18
    assert len({tuple(row["array_offsets"][6:]) for row in directions}) == 1


def test_loaded_graph_exact_streaming_histories_and_waveform_without_float_model(packed, monkeypatch):
    integer, blob = packed
    def forbidden(*args, **kwargs):
        raise AssertionError("Packed loading/inference attempted a learned float operation")
    from esp32_denoiser.gtcrn_model import GTCRNDenoiser
    monkeypatch.setattr(GTCRNDenoiser, "__init__", forbidden)
    for kind in (nn.Conv2d, nn.ConvTranspose2d, nn.GRU, nn.Linear, nn.LayerNorm, nn.PReLU):
        monkeypatch.setattr(kind, "forward", forbidden)
    loaded = load_gtcrn_integer(blob)
    spectra = np.fft.rfft(_audio(512, batch=12).numpy(), axis=-1).astype(np.complex64) * np.float32(.1)
    original_state, loaded_state = {}, {}
    for spectrum in spectra:
        expected, original_state = integer.spectrum_frame(spectrum, original_state)
        actual, loaded_state = loaded.spectrum_frame(spectrum, loaded_state)
        np.testing.assert_array_equal(actual, expected)
        for name in original_state:
            np.testing.assert_array_equal(loaded_state[name], original_state[name])
            assert loaded_state[name].dtype == np.int8
    assert sum(value.nbytes for value in loaded_state.values()) == 18_048
    audio = _audio(1031, batch=2)
    torch.testing.assert_close(loaded(audio), integer(audio), rtol=0, atol=0)
    torch.testing.assert_close(loaded(audio), loaded(audio), rtol=0, atol=0)
    assert not loaded.window.flags.writeable
    first = loaded.backend.ops[SPECS[0].name]
    assert not first.weights.flags.writeable and not first.bias.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first.weights.flat[0] = 1


@pytest.mark.parametrize("mutation,match", [
    ("truncated", "header|size"), ("oversized", "size"), ("corrupt", "integrity"),
    ("version", "header"), ("reserved", "header"), ("shape", "topology"),
    ("grid", "Activation exponent"), ("record_grid", "operator grid"),
    ("unaligned", "alignment"), ("outside", "extent"),
    ("unused_ref", "unexpected array"), ("dot_overflow", "overflow INT32"),
    ("weight_exponent", "exponent"), ("gru_table", "GRU table"),
    ("ln_epsilon", "epsilon code"), ("ln_overflow", "overflow signed INT64"),
    ("window", "window"), ("padding", "padding"),
])
def test_parser_rejects_corruption_and_rehashed_unsafe_payloads(packed, mutation, match):
    _, original = packed
    with pytest.raises(ValueError, match=match):
        load_gtcrn_integer(_mutated_blob(original, mutation))


def _mutated_blob(original, mutation):
    blob = bytearray(original)
    if mutation == "truncated":
        blob = blob[:-1]
    elif mutation == "oversized":
        blob = bytearray(MAX_MODEL_BYTES+1)
    elif mutation == "corrupt":
        blob[-17] ^= 1
    elif mutation == "version":
        struct.pack_into("<H", blob, 8, 99)
    elif mutation == "reserved":
        blob[HEADER.size-1] = 1
    elif mutation == "shape":
        struct.pack_into("<H", blob, _record(blob, AFFINE)+8, 65535)
    elif mutation == "grid":
        blob[GRID_OFFSET] = 127
    elif mutation == "record_grid":
        blob[_record(blob, AFFINE)+4] = 0
    elif mutation in ("unaligned", "outside"):
        struct.pack_into("<I", blob, _record(blob, AFFINE)+24, ARRAY_OFFSET+1 if mutation == "unaligned" else len(blob)+16)
    elif mutation == "unused_ref":
        struct.pack_into("<I", blob, _record(blob, AFFINE)+24+9*4, ARRAY_OFFSET)
    elif mutation == "dot_overflow":
        struct.pack_into("<i", blob, _ref(blob, AFFINE, 1), 2**31-1)
    elif mutation == "weight_exponent":
        blob[_ref(blob, AFFINE, 2)] = 127
    elif mutation == "gru_table":
        blob[_ref(blob, GRU, 6)] ^= 1
    elif mutation == "ln_epsilon":
        blob[_record(blob, LAYER_NORM)+64] ^= 1
    elif mutation == "ln_overflow":
        struct.pack_into("<i", blob, _ref(blob, LAYER_NORM, 1), 2**31-1)
        first_ln = next(spec for spec in SPECS if spec.kind == LAYER_NORM)
        blob[GRID_OFFSET + GRID_NAMES.index(first_ln.name + ".output")] = 8
        blob[_record(blob, LAYER_NORM)+5] = 8
        struct.pack_into("<b", blob, _record(blob, LAYER_NORM)+7, -16)
    elif mutation == "window":
        struct.pack_into("<f", blob, _ref(blob, WINDOW)+4, float("nan"))
    elif mutation == "padding":
        blob[GRID_OFFSET+len(GRID_NAMES)] = 1
    if mutation not in ("truncated", "oversized", "corrupt"):
        blob = _resign(blob)
    return bytes(blob)


def test_export_rejects_mutated_runtime_topology_or_grids(packed):
    integer, _ = packed
    changed = deepcopy(integer)
    operation = changed.backend.ops[SPECS[0].name]
    operation.groups = 9
    with pytest.raises(ValueError, match="topology"):
        pack_gtcrn_integer(changed)
    changed = deepcopy(integer)
    changed.backend.grids.pop("erb.output")
    with pytest.raises(ValueError, match="topology"):
        pack_gtcrn_integer(changed)


def test_real_data_cli_preserves_audited_recipe_and_writes_bound_artifacts(broad_checkpoint, tmp_path, monkeypatch):
    from esp32_denoiser.gtcrn_integer_export import main
    _, checkpoint, validation = broad_checkpoint
    path = tmp_path / "teacher.pt"
    torch.save(checkpoint, path)
    output = tmp_path / "export" / "model.bin"
    monkeypatch.setattr("sys.argv", ["export", "--checkpoint", str(path), "--manifest", str(validation),
                                    "--output", str(output), "--calibration-crops", "2", "--threads", "1"])
    main()
    audit = json.loads(output.with_suffix(".calibration.json").read_text())
    summary = json.loads(output.with_suffix(".json").read_text())
    loaded = load_gtcrn_integer(output, calibration=audit)
    assert audit["recipe"] == "checkpoint training policy"
    assert audit["seed"] == 483 and audit["paired_crops"] + audit["synthetic_crops"] == 2
    assert summary["source_checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert summary["implementation_sha256"] == audit["implementation_sha256"]
    assert summary["packed_bytes"] == len(output.read_bytes()) < MAX_MODEL_BYTES
    assert loaded.packed_metadata["calibration_audit_loaded"]
