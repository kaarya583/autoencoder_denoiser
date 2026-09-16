"""Known-answer full C DSP, PCM16 rounding and causal state behavior."""
import ctypes
import json

import numpy as np
import pytest
import torch

from esp32_denoiser.gtcrn_embedded import EmbeddedGTCRN, main
from esp32_denoiser.gtcrn_integer import GTCRNIntegerDenoiser, calibrate_gtcrn_integer
from esp32_denoiser.gtcrn_integer_export import pack_gtcrn_integer
from esp32_denoiser.gtcrn_native import CIntegerGTCRN
from test_gtcrn_integer import _source, _audio
from test_gtcrn_integer_export import packed, threads
from test_gtcrn_recurrent_probe import _manifest


@pytest.fixture(scope="module", params=[False, True], ids=["raw", "frame_rms"])
def identity_blob(request):
    torch.set_num_threads(1)
    model = _source(request.param, constant_mask=True)
    _, calibration = calibrate_gtcrn_integer(model, [_audio(513)], max_batches=1)
    return pack_gtcrn_integer(GTCRNIntegerDenoiser(model, calibration))


@pytest.mark.parametrize("samples", [1, 255, 256, 257, 1031])
def test_constant_mask_known_answer_including_flush_dc_nyquist(identity_blob, samples):
    audio = _audio(samples, batch=2)
    audio += torch.arange(samples).remainder(2)*.02+.03
    audio[:, 0] += .2
    audio[:, -1] -= .1
    model = EmbeddedGTCRN(identity_blob, io_format="float32")
    expected = audio*(127/128)
    actual = model(audio)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-7)
    torch.testing.assert_close(model(audio), actual, rtol=0, atol=0)
    torch.testing.assert_close(model(torch.zeros_like(audio)), torch.zeros_like(audio), rtol=0, atol=0)
    assert model.model_stats()["native_buffer_bytes"]["audio"] % 16 == 0
    assert model.model_stats()["native_buffer_bytes"]["neural"] == 18_048


def test_pcm16_matches_float_on_the_identical_rounded_input_and_counts_clipping(identity_blob):
    waveform = _audio(1537)*20
    waveform[:, :8] = torch.tensor([-1.4, 1.4, -1., 1., .5/32768, -.5/32768, 1.5/32768, -1.5/32768])
    encoded = np.rint(waveform.numpy().astype(np.float64)*32768)
    rounded = torch.from_numpy(encoded.clip(-32768, 32767).astype(np.float32)/32768)
    floating = EmbeddedGTCRN(identity_blob, io_format="float32")(rounded)
    expected = np.rint(floating.numpy().astype(np.float64)*32768).clip(-32768, 32767).astype(np.float32)/32768
    model = EmbeddedGTCRN(identity_blob)
    actual = model(waveform)
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert model.io_statistics["samples"] == waveform.numel()
    assert model.io_statistics["input_clipped_samples"] == int(np.count_nonzero((encoded < -32768)|(encoded > 32767)))
    assert actual.min() >= -1 and actual.max() <= 32767/32768


def test_complete_c_matches_reference_dsp_and_future_changes_preserve_prefix(packed):
    _, blob = packed
    waveform = _audio(1025)
    model = EmbeddedGTCRN(blob, io_format="float32")
    actual, expected = model(waveform), CIntegerGTCRN(blob)(waveform)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    changed = torch.cat((waveform[:, :768], _audio(519)*-3), dim=-1)
    torch.testing.assert_close(model(changed)[:, :512], actual[:, :512], rtol=0, atol=0)
    assert model.model_stats()["native_ram_subtotal_bytes"] == sum(value.nbytes for name, value in model.buffers.items() if name != "blob")


def test_native_audio_rejects_overlap_nonfinite_and_closed_instances(identity_blob):
    model = EmbeddedGTCRN(identity_blob, io_format="float32")
    source, output = np.zeros(256, np.float32), np.zeros(256, np.float32)
    source[0] = np.nan
    assert model.library.edng_audio_process(model.handle, source.ctypes.data, output.ctypes.data) != 0
    assert model.library.edng_audio_reset(model.handle) == 0
    source.fill(0)
    assert model.library.edng_audio_process(model.handle, source.ctypes.data, model.buffers["neural"].ctypes.data) != 0
    assert model.library.edng_audio_process(model.handle, source.ctypes.data+1, output.ctypes.data) != 0
    # Exact input/output alias is allowed after complete frame analysis.
    assert model.library.edng_audio_process(model.handle, source.ctypes.data, source.ctypes.data) == 0
    for bad in (torch.zeros(0, 256), torch.zeros(1, 0), torch.full((1, 256), float("inf"))):
        with pytest.raises(ValueError, match="finite nonempty"):
            model(bad)
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model(torch.zeros(1, 256))


def test_audio_failed_reinitialization_invalidates_old_state(identity_blob):
    model = EmbeddedGTCRN(identity_blob)
    values = model.buffers
    status = model.library.edng_audio_init(model.handle, values["audio"].nbytes, None,
                                          values["neural"].ctypes.data, values["neural"].nbytes,
                                          values["workspace"].ctypes.data, values["workspace"].nbytes)
    assert status != 0
    assert model.library.edng_audio_reset(model.handle) != 0


def test_complete_audio_cli_uses_full_c_pcm16_and_keeps_test_sealed(identity_blob, tmp_path, monkeypatch):
    blob = tmp_path / "model.bin"
    blob.write_bytes(identity_blob)
    manifest, _ = _manifest(tmp_path, "development", "heldout", split="development")
    output = tmp_path / "evaluation.json"
    arguments = ["embedded", "--integer-model", str(blob), "--manifest", str(manifest), "--output", str(output),
                 "--threads", "1", "--max-utterances", "1", "--compare-reference"]
    monkeypatch.setattr("sys.argv", arguments)
    main()
    report = json.loads(output.read_text())
    assert report["model_stats"]["io_format"] == "pcm16"
    assert report["io_statistics"]["samples"] > 0
    assert "numpy_dsp_reference" in report and "dsp_comparison" in report
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row["source_split"] = "test"
    manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="test remains sealed"):
        main()
