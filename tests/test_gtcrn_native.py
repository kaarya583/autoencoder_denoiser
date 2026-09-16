"""Complete C graph parity, persistent-state layout and buffer bounds."""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from esp32_denoiser.experimental_gru import IntegerGRUCell
from esp32_denoiser.experimental_gtcrn_ops import IntegerAffine, IntegerPReLU, IntegerStreamConv
from esp32_denoiser.gtcrn_integer import calibrate_gtcrn_integer, GTCRNIntegerDenoiser
from esp32_denoiser.gtcrn_integer_export import load_gtcrn_integer, pack_gtcrn_integer
from esp32_denoiser.gtcrn_native import CIntegerGTCRN, STATE_LAYOUT, _layout_header, main

from test_gtcrn_integer import _audio, _source
from test_gtcrn_recurrent_probe import _manifest


@pytest.fixture(scope="module")
def blob():
    torch.set_num_threads(1)
    source = _source()
    _, calibration = calibrate_gtcrn_integer(source, [_audio()], max_batches=1)
    return pack_gtcrn_integer(GTCRNIntegerDenoiser(source, calibration))


def test_generated_native_topology_matches_python_packed_schema():
    path = Path(__file__).parents[1] / "firmware/experimental_gtcrn/layout.h"
    assert path.read_text() == _layout_header()
    assert STATE_LAYOUT[-1][-1] == 18_048
    assert len(STATE_LAYOUT) == 14


def test_entire_native_graph_matches_numpy_mask_and_every_history(blob):
    reference, native = load_gtcrn_integer(blob), CIntegerGTCRN(blob)
    random = np.random.default_rng(7001)
    frames = [random.integers(-128, 128, (3, 129), dtype=np.int8), np.zeros((3, 129), np.int8),
              random.integers(-20, 21, (3, 129), dtype=np.int8),
              np.tile(np.array([-128, 127, 0], np.int8), 129).reshape(3, 129)]
    state, native_state = {}, None
    scale = reference.backend.grid("erb.output").scale
    for frame in frames:
        expected, state = reference.graph.frame(frame.astype(np.float32)[None, :, None]*scale, state)
        actual, native_state = native.neural_step(frame, native_state)
        np.testing.assert_array_equal(actual, expected.data.reshape(2, 129))
        for name, shape, start, stop in STATE_LAYOUT:
            np.testing.assert_array_equal(native_state[start:stop].reshape(shape), state[name], err_msg=name)
    assert native.statistics()["native_frames"] == len(frames)
    assert native.native_workspace_bytes == 17_536
    assert native.native_state_bytes == 18_048
    assert native.model_stats()["native_neural_buffer_subtotal"] < 200*1024
    assert "excludes compiler stack" in native.model_stats()["native_memory_scope"]


def test_native_waveform_exact_parity_and_reset_with_nontrivial_masks(blob):
    reference, native = load_gtcrn_integer(blob), CIntegerGTCRN(blob)
    audio = _audio(1031, 2)
    audio[:, 0] += .2
    audio[:, -1] -= .1
    expected = reference(audio)
    actual = native(audio)
    assert not torch.equal(expected, audio)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(native(audio), expected, rtol=0, atol=0)
    state, chunks = native.initial_state(), []
    for chunk in np.pad(audio[0].numpy(), (0, (-1031)%256+256)).reshape(-1, 256):
        output, state = native.stream_step(chunk, state)
        chunks.append(output)
    np.testing.assert_array_equal(np.concatenate(chunks)[256:1287], expected[0].numpy())
    assert all(value.dtype == np.int8 for value in state.neural.values())


def test_native_neural_call_does_not_use_numpy_learned_kernels_and_preserves_states(blob, monkeypatch):
    native = CIntegerGTCRN(blob)
    frame = np.arange(387, dtype=np.int16).astype(np.int8).reshape(3, 129)
    first, state = native.neural_step(frame)
    original = state.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy learned kernel executed instead of the full C graph")
    monkeypatch.setattr(IntegerAffine, "__call__", forbidden)
    monkeypatch.setattr(IntegerPReLU, "__call__", forbidden)
    monkeypatch.setattr(IntegerStreamConv, "step", forbidden)
    monkeypatch.setattr(IntegerGRUCell, "process", forbidden)
    second, continued = native.neural_step(frame, state)
    repeated, repeated_state = native.neural_step(frame, state)
    np.testing.assert_array_equal(state, original)
    np.testing.assert_array_equal(second, repeated)
    np.testing.assert_array_equal(continued, repeated_state)
    np.testing.assert_array_equal(native.neural_step(frame)[0], first)
    assert bool(torch.isfinite(native(_audio(257))).all())


def test_native_c_buffer_guards_alias_rejection_and_long_stream(blob):
    native = CIntegerGTCRN(blob)
    state = np.full(native.native_state_bytes+32, 85, np.int8)
    workspace = np.full(native.native_workspace_bytes+32, 85, np.int8)
    output = np.full(258+32, 85, np.int8)
    state_ptr, workspace_ptr, output_ptr = state.ctypes.data+16, workspace.ctypes.data+16, output.ctypes.data+16
    assert native.library.edng_reset(native._handle, state_ptr, native.native_state_bytes) == 0
    rng = np.random.default_rng(778)
    for index in range(48):
        frame = rng.integers(-128, 128, 387, dtype=np.int8) if index%3 else np.zeros(387, np.int8)
        status = native.library.edng_process_frame(native._handle, state_ptr, native.native_state_bytes,
                                                    frame.ctypes.data, frame.nbytes, output_ptr, 258,
                                                    workspace_ptr, native.native_workspace_bytes)
        assert status == 0
    for values in (state, workspace, output):
        np.testing.assert_array_equal(values[:16], np.full(16, 85, np.int8))
        np.testing.assert_array_equal(values[-16:], np.full(16, 85, np.int8))
    old = state.copy()
    assert native.library.edng_process_frame(native._handle, state_ptr, native.native_state_bytes,
                                              frame.ctypes.data, 386, output_ptr, 258,
                                              workspace_ptr, native.native_workspace_bytes) != 0
    assert native.library.edng_process_frame(native._handle, state_ptr, native.native_state_bytes,
                                              frame.ctypes.data, 387, output_ptr, 258,
                                              state_ptr, native.native_workspace_bytes) != 0
    np.testing.assert_array_equal(state, old)
    for invalid in (np.zeros(387, np.float32), np.zeros(386, np.int8)):
        with pytest.raises(ValueError, match="Native features"):
            native.neural_step(invalid)
    with pytest.raises(ValueError, match="flat INT8"):
        native.neural_step(np.zeros(387, np.int8), np.zeros(1, np.int8))


def test_c_loader_rejects_truncation_and_integrity_change_without_python_parser(blob):
    native = CIntegerGTCRN(blob)
    handle = (ctypes.c_uint64 * ((native.native_handle_bytes+7)//8))()
    assert native.library.edng_init(handle, native.native_handle_bytes, native._blob, len(blob)-1) != 0
    altered = bytearray(blob)
    altered[-16] ^= 1
    storage = (ctypes.c_uint64 * ((len(blob)+7)//8))()
    ctypes.memmove(storage, bytes(altered), len(blob))
    assert native.library.edng_init(handle, native.native_handle_bytes, storage, len(blob)) != 0


def test_native_evaluation_cli_records_actual_audio_and_rejects_test(blob, tmp_path, monkeypatch):
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(blob)
    manifest, _ = _manifest(tmp_path, "development", "p226")
    output = tmp_path / "native.json"
    monkeypatch.setattr("sys.argv", ["gtcrn-native", "--integer-model", str(model_path),
                                    "--manifest", str(manifest), "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert len(report["utterances"]) == 2
    assert len(report["utterances"][0]["audio_sha256"]["clean"]) == 64
    assert report["model_stats"]["packed_bytes"] == len(blob)
    assert report["artifact_verification"]["packed_file_sha256"] == hashlib.sha256(blob).hexdigest()
    assert report["artifact_verification"]["checkpoint_file_verified"] is False
    assert "no PCM16" in report["io_contract"]
    assert "not ESP32 performance" in report["timing"]["measurement"]
    rows = [dict(json.loads(line), source_split="test") for line in manifest.read_text().splitlines()]
    manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
    output.unlink()
    with pytest.raises(ValueError, match="test remains sealed"):
        main()
    assert not output.exists()


def test_native_cli_checks_checkpoint_and_calibration_association(blob, tmp_path, monkeypatch):
    # This is an explicitly synthetic file-association fixture, not an audit
    # claiming that these arbitrary bytes were a real training checkpoint.
    checkpoint = tmp_path / "fixture-checkpoint.bin"
    checkpoint.write_bytes(b"numerical fixture checkpoint identity")
    snapshot = load_gtcrn_integer(blob)
    snapshot.source_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audit = {"scope": "synthetic numerical test only", "source_state_sha256": snapshot.packed_metadata["source_state_sha256"]}
    snapshot.calibration = audit
    snapshot.packed_metadata.pop("calibration_sha256")
    model = tmp_path / "packed.bin"
    model.write_bytes(pack_gtcrn_integer(snapshot))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(audit))
    manifest, _ = _manifest(tmp_path, "development", "p226")
    output = tmp_path / "verified.json"
    monkeypatch.setattr("sys.argv", ["native", "--integer-model", str(model), "--calibration", str(calibration),
                                    "--checkpoint", str(checkpoint), "--manifest", str(manifest), "--output", str(output),
                                    "--max-utterances", "1"])
    main()
    report = json.loads(output.read_text())["artifact_verification"]
    assert report["checkpoint_file_verified"] and report["calibration_audit_file_verified"]
    assert report["source_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output.unlink()
    checkpoint.write_bytes(b"changed identity")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        main()
    assert not output.exists()
    checkpoint.write_bytes(b"numerical fixture checkpoint identity")
    calibration.write_text("{}")
    with pytest.raises(ValueError, match="Calibration audit hash"):
        main()
    assert not output.exists()
