"""Training-only output calibration uses actual C inference and strict holdouts."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf
import torch

from esp32_denoiser import output_gain
from esp32_denoiser.development import prepare_development
from esp32_denoiser.export import export_model, with_output_gain
from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig
from esp32_denoiser.quantization import configure_qat
from test_esp32_extra_data import prepared


def _rewrite(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@pytest.fixture
def files(prepared):
    root, extra = prepared
    development = prepare_development(root, num_mixtures=5, clean_examples=1, crop_seconds=.04)
    rng = np.random.default_rng(674)
    paired = []
    for index, speaker in enumerate(("p225", "p225", "p227", "p226")):
        audio = rng.normal(0, .12, 1001 + index * 7).astype(np.float32)
        path = root / f"{speaker}_{index:03d}.wav"
        sf.write(path, audio, 16000, subtype="FLOAT")
        paired.append({"id": path.stem, "speaker": speaker, "sample_rate": 16000,
                       "samples": len(audio), "source_split": "train", "clean": str(path), "noisy": str(path)})
    train, primary = root / "paired_train.jsonl", root / "primary_dev.jsonl"
    _rewrite(train, paired[:-1])
    _rewrite(primary, paired[-1:])
    model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2)))
    with torch.no_grad():
        model.head.bias[:257].fill_(.25)  # Gain1.5, with unity serialized DSP output gain.
    binary = root / "source.bin"
    export_model(configure_qat(model), binary)
    arguments = dict(paired_train_manifest=train, speech_train_manifest=extra["speech_train"],
                     primary_development_manifest=primary, speech_development_manifest=extra["speech_val"],
                     external_mixtures_manifest=development["mixtures"], external_clean_manifest=development["clean"],
                     examples_per_source=2, crop_seconds=.04, seed=41)
    return binary, arguments


@pytest.mark.skipif(shutil.which("cc") is None, reason="C99 compiler required")
def test_actual_c_balanced_fit_is_deterministic_level_aware_and_records_sources(files):
    binary, arguments = files
    original = binary.read_bytes()
    torch_rng = torch.get_rng_state().clone()
    result = output_gain.calibrate_integer_output_gain(binary, **arguments)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert result == output_gain.calibrate_integer_output_gain(binary, **arguments)
    assert binary.read_bytes() == original
    assert result["source_model"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert result["source_model"]["output_gain"] == 1
    assert result["fit"]["valid_examples"] == 4
    assert [row["source"] for row in result["selected_examples"]].count("paired_train") == 2
    assert [row["source"] for row in result["selected_examples"]].count("speech_train") == 2
    assert result["output_gain"] == pytest.approx(2/3, abs=1e-5)
    assert result["diagnostics"]["normalized_mse_before"] > .24
    assert result["diagnostics"]["normalized_mse_after"] < 1e-7
    assert result["exclusions"]["heldout_audio_files_hashed"] > 3
    cross, energy = 0, 0
    for record in result["selected_examples"]:
        path = Path(record["path"])
        assert record["audio_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        target, _ = sf.read(path, start=record["offset"], frames=record["samples"], dtype="float32")
        predicted = np.rint(target * 32768).clip(-32768, 32767).astype(np.float64) / 32768 * 1.5
        target = target.astype(np.float64)
        cross += float(np.mean(predicted * target) / np.mean(target**2))
        energy += float(np.mean(predicted**2) / np.mean(target**2))
    assert result["output_gain"] == pytest.approx(cross / energy, abs=1e-6)
    assert result["fit"]["cross_sum"] == pytest.approx(cross, abs=1e-6)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("overlap", ("id", "speaker", "group", "path", "content"))
def test_development_overlap_cannot_reach_a_gain_fit(files, monkeypatch, overlap):
    binary, arguments = files
    train = _rows(arguments["paired_train_manifest"])[0]
    heldout = _rows(arguments["primary_development_manifest"])[0]
    if overlap in {"id", "speaker"}:
        train[overlap] = heldout[overlap]
    elif overlap == "group":
        train["group"] = heldout["speaker"]
    elif overlap == "path":
        train["clean"] = heldout["clean"]
    else:
        Path(train["clean"]).write_bytes(Path(heldout["clean"]).read_bytes())
        train["samples"] = heldout["samples"]
    _rewrite(arguments["paired_train_manifest"], [train])
    arguments["examples_per_source"] = 1
    called = False

    class NoInference:
        def __init__(self, *args):
            pass
        def __call__(self, audio):
            nonlocal called
            called = True
            raise AssertionError("Held-out audio must not enter the fit")
        def close(self):
            pass

    monkeypatch.setattr(output_gain, "EmbeddedWaveformEnhancer", NoInference)
    with pytest.raises(ValueError, match="overlaps development"):
        output_gain.calibrate_integer_output_gain(binary, **arguments)
    assert not called


@pytest.mark.parametrize("defect", ("train_partition", "extra_partition", "external_test", "hash", "length", "insufficient"))
def test_gain_calibration_rejects_unapproved_or_stale_sources(files, defect):
    binary, arguments = files
    if defect == "insufficient":
        arguments["examples_per_source"] = 64
    else:
        key = {"train_partition": "paired_train_manifest", "extra_partition": "speech_train_manifest",
               "external_test": "external_clean_manifest", "hash": "speech_development_manifest",
               "length": "primary_development_manifest"}[defect]
        records = _rows(arguments[key])
        if defect == "train_partition":
            records[0]["source_split"] = "test"
        elif defect == "extra_partition":
            for row in records:
                row["split"] = "val"
        elif defect == "external_test":
            records[0]["source_split"] = "test"
        elif defect == "hash":
            records[0]["sha256"] = "0" * 64
        else:
            records[0]["samples"] -= 1
        _rewrite(arguments[key], records)
    with pytest.raises(ValueError):
        output_gain.calibrate_integer_output_gain(binary, **arguments)


def test_corrected_source_is_rejected_before_manifest_io(files):
    binary, arguments = files
    corrected = binary.with_name("already_corrected.bin")
    with_output_gain(binary, corrected, .8)
    arguments["paired_train_manifest"] = "missing"
    with pytest.raises(ValueError, match="uncorrected"):
        output_gain.calibrate_integer_output_gain(corrected, **arguments)


@pytest.mark.skipif(shutil.which("cc") is None, reason="C99 compiler required")
@pytest.mark.parametrize("case", ("silent_training", "zero_response"))
def test_fit_requires_complete_nonsilent_cohort_and_positive_response(files, case):
    binary, arguments = files
    if case == "silent_training":
        for row in _rows(arguments["paired_train_manifest"]):
            sf.write(row["clean"], np.zeros(row["samples"], np.float32), 16000, subtype="FLOAT")
        reason = "Insufficient nonsilent"
    else:
        model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2)))
        with torch.no_grad():
            model.head.bias[:257].fill_(-.5)  # Exact zero complex gain.
        export_model(configure_qat(model), binary)
        reason = "positive least-squares"
    with pytest.raises(ValueError, match=reason):
        output_gain.calibrate_integer_output_gain(binary, **arguments)


@pytest.mark.skipif(shutil.which("cc") is None, reason="C99 compiler required")
@pytest.mark.parametrize("mutation", ("binary", "manifest", "audio"))
def test_frozen_calibration_inputs_cannot_change_during_inference(files, monkeypatch, mutation):
    binary, arguments = files
    original_class = output_gain.EmbeddedWaveformEnhancer
    called = False

    class MutatingEnhancer(original_class):
        def forward(self, audio):
            nonlocal called
            result = super().forward(audio)
            if not called:
                called = True
                if mutation == "binary":
                    binary.write_bytes(binary.read_bytes() + b"changed")
                elif mutation == "manifest":
                    path = arguments["paired_train_manifest"]
                    path.write_text(path.read_text() + "\n")
                else:
                    # A development file is never inferred, but changing it
                    # after the exclusion audit must still invalidate the fit.
                    path = Path(_rows(arguments["primary_development_manifest"])[0]["clean"])
                    path.write_bytes(path.read_bytes() + b"changed")
            return result

    monkeypatch.setattr(output_gain, "EmbeddedWaveformEnhancer", MutatingEnhancer)
    with pytest.raises(ValueError, match="changed during calibration"):
        output_gain.calibrate_integer_output_gain(binary, **arguments)
