"""Fixed external development audio, group isolation, and cohort accounting."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from esp32_denoiser.data import PairedAudioDataset, read_manifest
from esp32_denoiser.development import common_valid_ids, prepare_development, preservation_metrics, summarize_development
from esp32_denoiser.extra_data import SOURCES, _active_mask, _asset_identity


@pytest.fixture
def sources(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for kind in ("speech", "noise"):
        rows = {"train": [], "val": []}
        for index in range(1, 4):
            member = (f"LibriSpeech/train-clean-100/{index}/10/{index}-10-0001.flac" if kind == "speech"
                      else f"musan/noise/free-sound/noise-{index:04d}.wav")
            path = tmp_path / "sources" / kind / member
            path.parent.mkdir(parents=True, exist_ok=True)
            audio = (.3 * np.sin(np.arange(2400) * (.07 * index)) if kind == "speech" else
                     np.random.default_rng(index).uniform(-.2, .2, 400))
            sf.write(path, audio, 16000, subtype="PCM_16")
            identifier, group, speaker = _asset_identity(kind, member)
            split = "train" if index == 1 else "val"
            rows[split].append({"id": identifier, "group": group, "speaker": speaker,
                                "member": member, "path": str(path), "kind": kind, "split": split,
                                "source": SOURCES[kind].source, "license": SOURCES[kind].license,
                                "samples": len(audio), "sample_rate": 16000,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for split, records in rows.items():
            (manifests / f"{kind}_{split}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    return tmp_path


def test_development_fixed_matched_snrs_rng_and_float_audio(sources):
    torch.manual_seed(426)
    before = torch.random.get_rng_state().clone()
    paths = prepare_development(sources, num_mixtures=10, clean_examples=3, crop_seconds=.05)
    assert torch.equal(before, torch.random.get_rng_state())
    records = read_manifest(paths["mixtures"])
    assert len(records) == 10
    for base in range(2):
        group = [row for row in records if row["base_crop"] == base]
        for key in ("speech_id", "speech_offset", "noise_id", "noise_offset", "item_seed"):
            assert len({row[key] for row in group}) == 1
        assert {row["snr_db"] for row in group} == {-5, 0, 5, 10, 20}
        for row in group:
            clean, rate = sf.read(row["clean"], dtype="float32")
            noisy, _ = sf.read(row["noisy"], dtype="float32")
            active = _active_mask(clean)
            noise = noisy - clean
            actual = 10 * np.log10(np.mean(clean[active] ** 2) / np.mean(noise[active] ** 2))
            assert actual == pytest.approx(row["snr_db"], abs=1e-5)
            assert rate == 16000 and len(clean) == row["samples"] == 800
            assert sf.info(row["clean"]).subtype == "FLOAT"
            assert hashlib.sha256(Path(row["noisy"]).read_bytes()).hexdigest() == row["noisy_sha256"]
            assert row["source_split"] == "development" and row["split"] == "val"
    clean_set = PairedAudioDataset(paths["clean"], crop_seconds=None, random_crop=False)
    for item in clean_set:
        assert torch.equal(item["clean"], item["noisy"])
    specification = json.loads(paths["provenance"].read_text())["specification"]
    assert not specification["official_test_accessed"]
    assert not (sources / "manifests/test.jsonl").exists()
    # A second output location produces the same WAV, JSONL, and provenance bytes.
    repeated = prepare_development(sources, output_dir=sources / "another", num_mixtures=10,
                                   clean_examples=3, crop_seconds=.05)
    for key in paths:
        assert paths[key].read_bytes() == repeated[key].read_bytes()


def test_development_freezes_specification_and_requires_heldout_assets(sources):
    prepare_development(sources, num_mixtures=5, clean_examples=1, crop_seconds=.05)
    with pytest.raises(ValueError, match="specification changed"):
        prepare_development(sources, num_mixtures=10, clean_examples=1, crop_seconds=.05)
    manifest = sources / "manifests/speech_val.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["split"] = "train"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="partition"):
        prepare_development(sources, output_dir=sources / "invalid", num_mixtures=5, clean_examples=1)


def test_development_detects_overlap_and_changed_source_waveforms(sources):
    manifest = sources / "manifests/noise_val.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    original = manifest.read_text()
    train = json.loads((sources / "manifests/noise_train.jsonl").read_text())
    rows[0]["path"], rows[0]["sha256"] = train["path"], train["sha256"]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="overlaps training"):
        prepare_development(sources, num_mixtures=5, clean_examples=1)
    manifest.write_text(original)
    # All held-out noise is altered so any selected mixture must detect it.
    for row in [json.loads(line) for line in original.splitlines()]:
        sf.write(row["path"], np.random.default_rng(77).uniform(-.3, .3, 400), 16000)
    with pytest.raises(ValueError, match="waveform changed"):
        prepare_development(sources, num_mixtures=5, clean_examples=1, crop_seconds=.05)


def test_stratum_summary_uses_equal_utterances_and_explicit_common_cohort():
    metadata = [{"id": str(index), "source_split": "development", "snr_db": level, "speaker": "2"}
                for index, level in enumerate((-5, -5, 20, 20))]
    first = [{"id": "0", "si_sdr_noisy": -5, "si_sdr_enhanced": 5},
             {"id": "1", "si_sdr_noisy": -5, "si_sdr_enhanced": 15},
             {"id": "2", "si_sdr_noisy": 20, "si_sdr_enhanced": None}]
    second = [{"id": "0", "si_sdr_noisy": -5, "si_sdr_enhanced": 3},
              {"id": "1", "si_sdr_noisy": -5, "si_sdr_enhanced": None},
              {"id": "2", "si_sdr_noisy": 20, "si_sdr_enhanced": 25}]
    summary = summarize_development(first, metadata)
    assert summary["summary"]["si_sdri"] == 15
    assert summary["summary"]["invalid_utterances"] == 1
    assert summary["missing_utterances"] == 1
    assert summary["by_snr_db"]["20"]["si_sdri"] is None
    common = common_valid_ids({"utterances": first}, second)
    assert common == {"0"}
    shared = summarize_development(first, metadata, common_ids=common)
    assert shared["summary"]["si_sdri"] == 10
    assert shared["excluded_by_common_cohort"] == 2
    assert shared["summary"]["valid_utterances"] == 1
    assert summarize_development(first, metadata, common_ids=set())["summary"]["si_sdri"] is None
    with pytest.raises(ValueError, match="missing or invalid"):
        summarize_development(first, metadata, common_ids={"2"})
    with pytest.raises(ValueError, match="absent"):
        summarize_development(first + [{"id": "unknown", "si_sdr_noisy": 0, "si_sdr_enhanced": 1}], metadata)
    with pytest.raises(ValueError, match="Duplicate"):
        common_valid_ids(first + first)


def test_perceptual_common_cohort_is_metric_specific():
    records = [{"id": "0", "si_sdr_noisy": 0, "si_sdr_enhanced": 5,
                "pesq_wb_noisy": 1.2, "pesq_wb_enhanced": 2.1},
               {"id": "1", "si_sdr_noisy": 0, "si_sdr_enhanced": 5,
                "pesq_wb_noisy": 1.2, "pesq_wb_enhanced": None}]
    assert common_valid_ids(records) == {"0", "1"}
    assert common_valid_ids(records, metric_keys=("pesq_wb_noisy", "pesq_wb_enhanced")) == {"0"}


def test_preservation_detects_attenuation_and_polarity_inversion():
    clean = np.array([-.4, -.2, .2, .4], dtype=np.float32)
    result = preservation_metrics(clean, clean, torch.from_numpy(clean) * .5)
    assert result["noisy"]["projection_gain"] == pytest.approx(1)
    assert result["noisy"]["waveform_l1"] == 0
    assert result["enhanced"]["projection_gain"] == pytest.approx(.5)
    assert result["enhanced"]["rms_ratio"] == pytest.approx(.5)
    assert result["enhanced"]["normalized_waveform_l1"] == pytest.approx(.15 / np.sqrt(.1))
    inverted = preservation_metrics(clean, clean, -clean)["enhanced"]
    assert inverted["projection_gain"] == pytest.approx(-1)
    assert inverted["rms_ratio"] == pytest.approx(1)
    assert inverted["normalized_waveform_l1"] > 1


def test_preservation_rails_padding_and_silent_denominators():
    clean = np.zeros(5)
    values = np.array([-1.1, -1, 32767 / 32768, 1, float("nan")])
    result = preservation_metrics(np.zeros(5), clean, values, length=4)
    assert not result["valid_reference"] and result["samples"] == 4
    enhanced = result["enhanced"]
    assert enhanced["projection_gain"] is enhanced["rms_ratio"] is enhanced["normalized_waveform_l1"] is None
    assert enhanced["waveform_l1"] > 0
    assert enhanced["pcm16_out_of_range_samples"] == 2
    assert enhanced["pcm16_rail_or_exceeds_samples"] == 4
    with pytest.raises(ValueError, match="Nonfinite"):
        preservation_metrics(np.zeros(5), clean, values)
