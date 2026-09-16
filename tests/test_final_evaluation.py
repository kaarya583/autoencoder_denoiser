"""Synthetic fixtures only; never downloads or opens the real final corpus."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import soundfile as sf
import torch

from esp32_denoiser import final_evaluation as final
from esp32_denoiser.data import PairedAudioDataset, read_manifest
from esp32_denoiser.development import prepare_development
from esp32_denoiser.development_checkpoints import _validated_dataset
from esp32_denoiser.extra_data import SOURCES, _active_mask, _asset_identity
from esp32_denoiser.train import check_training_split
from test_esp32_development import sources
from test_frequency_factory import _pilot_data


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


@pytest.fixture
def sealed(sources, monkeypatch):
    # Extend the small development fixture to30 held-out noise recordings,
    # render one five-SNR base crop, and reserve the29 it did not use.
    root = sources
    noise_path = root / "manifests/noise_val.jsonl"
    noise = [json.loads(line) for line in noise_path.read_text().splitlines()]
    for index in range(4, 32):
        member = f"musan/noise/free-sound/noise-{index:04d}.wav"
        path = root / "sources/noise" / member
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.random.default_rng(index).uniform(-.2, .2, 400), 16000, subtype="PCM_16")
        identifier, group, speaker = _asset_identity("noise", member)
        noise.append({"id": identifier, "group": group, "speaker": speaker, "member": member,
                      "path": str(path), "kind": "noise", "split": "val", "sample_rate": 16000,
                      "samples": 400, "sha256": _hash(path), "source": SOURCES["noise"].source,
                      "license": SOURCES["noise"].license})
    _jsonl(noise_path, noise)
    development = prepare_development(root, num_mixtures=5, clean_examples=1, crop_seconds=.05)
    used = {row["noise_id"] for row in read_manifest(development["mixtures"])}
    reserved = [{key: row[key] for key in ("id", "group", "member", "sha256", "samples", "sample_rate", "source", "license")}
                for row in noise if row["id"] not in used]
    assert len(reserved) == 29
    paired_root = root / "paired"
    paired_root.mkdir()
    paired = _pilot_data(paired_root)
    speech = root / "fixture_final.flac"
    samples = .18 * np.sin(np.arange(60000) * .061)
    sf.write(speech, samples, 16000, subtype="PCM_16")
    archive = root / "fixture_test_clean.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(speech, arcname="LibriSpeech/test-clean/999/10/999-10-0001.flac")
    # The public code has no checksum-bypass argument. Only this synthetic
    # fixture patches the publisher constant; no real final audio is used.
    fixture_md5 = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
    monkeypatch.setattr(final, "TEST_CLEAN_MD5", fixture_md5)
    plan_path = root / "sealed_plan.json"
    plan = {"speech_source": {"dataset": "LibriSpeech test-clean", "official_md5": fixture_md5},
            "reserved_noise_count": 29, "reserved_noise_records": reserved,
            "manifest_sha256": {"manifests/noise_val.jsonl": _hash(noise_path),
                                "development/mixtures.jsonl": _hash(development["mixtures"]),
                                "development/clean.jsonl": _hash(development["clean"])},
            "mixture_plan": {"base_crops": 200, "snr_db": [-5, 0, 5, 10, 20],
                             "crop_seconds": 3, "clean_examples": 100, "seed": 20260913}}
    plan_path.write_text(json.dumps(plan))
    manifests = {role: root / "manifests" / f"{role}.jsonl" for role in ("speech_train", "speech_val", "noise_train", "noise_val")}
    manifests.update(paired_train=Path(paired["train"]), paired_development=Path(paired["val"]),
                     development_mixtures=development["mixtures"], development_clean=development["clean"])
    artifacts = []
    for role in ("model", "baseline"):
        path = root / f"{role}.bin"
        path.write_bytes(f"frozen fixture {role}".encode())
        artifacts.append({"role": role, "path": str(path), "sha256": _hash(path)})
    freeze_path = root / "freeze.json"
    freeze = {"version": 1, "selection_frozen": True, "test_used_for_selection": False,
              "plan_sha256": _hash(plan_path), "artifacts": artifacts,
              "inference_specification": {"architecture": "fixture", "grids": "fixture", "gain": 1, "DSP": "fixture"},
              "source_manifests": {role: {"path": str(path), "sha256": _hash(path)} for role, path in manifests.items()}}
    freeze_path.write_text(json.dumps(freeze))
    return {"root": root, "plan": plan_path, "freeze": freeze_path, "archive": archive,
            "plan_data": plan, "freeze_data": freeze, "noise": noise, "used_noise": used, "speech": speech}


def _audit(sealed):
    return final.audit_final_inputs(plan_path=sealed["plan"], freeze_path=sealed["freeze"], speech_archive=sealed["archive"])


def _refresh_plan(sealed):
    sealed["plan"].write_text(json.dumps(sealed["plan_data"]))
    sealed["freeze_data"]["plan_sha256"] = _hash(sealed["plan"])
    sealed["freeze"].write_text(json.dumps(sealed["freeze_data"]))


def test_model_freeze_is_required_before_any_final_archive_access(sealed, monkeypatch):
    sealed["freeze_data"]["artifacts"] = []
    sealed["freeze"].write_text(json.dumps(sealed["freeze_data"]))
    original_hash = final._hash_file

    def guarded(path):
        assert Path(path) != sealed["archive"], "Final archive was accessed without a valid freeze"
        return original_hash(path)

    monkeypatch.setattr(final, "_hash_file", guarded)
    with pytest.raises(ValueError, match="nonempty"):
        _audit(sealed)
    assert not (sealed["root"] / "final").exists()


@pytest.mark.parametrize("change,reason", [("artifact", "Frozen file hash"), ("plan", "plan hash"),
                                           ("archive", "archive MD5"), ("noise", "noise waveform hash")])
def test_changed_frozen_inputs_are_rejected_without_rendering(sealed, change, reason):
    if change == "artifact":
        Path(sealed["freeze_data"]["artifacts"][0]["path"]).write_bytes(b"changed model")
    elif change == "plan":
        sealed["plan"].write_text(sealed["plan"].read_text() + " ")
    elif change == "archive":
        with sealed["archive"].open("ab") as stream:
            stream.write(b"changed")
    else:
        row = next(row for row in sealed["noise"] if row["id"] not in sealed["used_noise"])
        Path(row["path"]).write_bytes(b"changed noise")
    with pytest.raises(ValueError, match=reason):
        final.prepare_final_evaluation(plan_path=sealed["plan"], freeze_path=sealed["freeze"],
                                       speech_archive=sealed["archive"], output_dir=sealed["root"] / "final")
    assert not (sealed["root"] / "final").exists()


def test_reservation_rejects_previously_used_development_noise(sealed):
    row = next(row for row in sealed["noise"] if row["id"] in sealed["used_noise"])
    sealed["plan_data"]["reserved_noise_records"][0] = {key: row[key] for key in sealed["plan_data"]["reserved_noise_records"][0]}
    _refresh_plan(sealed)
    with pytest.raises(ValueError, match="overlaps training/development id"):
        _audit(sealed)


@pytest.mark.parametrize("member,kind", [("../escape.flac", "file"), ("/absolute.flac", "file"),
                                        ("LibriSpeech/test-clean/link", "symlink"),
                                        ("LibriSpeech/test-clean/link", "hardlink")])
def test_archive_path_and_link_guards(tmp_path, member, kind):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        item = tarfile.TarInfo(member)
        if kind == "file":
            item.size = 1
            bundle.addfile(item, io.BytesIO(b"x"))
        else:
            item.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            item.linkname = "../../escape"
            bundle.addfile(item)
    with pytest.raises(ValueError, match="Unsafe|Links"):
        final._extract_speech(archive, tmp_path / "extracted", final._empty_exclusions())
    assert not (tmp_path.parent / "escape.flac").exists()


@pytest.mark.parametrize("field,value", [("speaker", "999"), ("group", "librispeech:speaker:999"),
                                        ("id", "librispeech:999-10-0001"), ("sha256", None)])
def test_final_speech_disjointness_checks_all_original_source_identifiers(sealed, field, value):
    audited = _audit(sealed)
    audited["excluded"][field].add(_hash(sealed["speech"]) if value is None else value)
    with pytest.raises(ValueError, match=f"overlaps training/development {field}"):
        final._extract_speech(audited["archive"], sealed["root"] / "rejected_sources", audited["excluded"])


def test_fixture_renderer_is_deterministic_matched_snr_and_excluded_from_training(sealed):
    audited = _audit(sealed)
    speech = final._extract_speech(audited["archive"], sealed["root"] / "extracted", audited["excluded"])
    small = replace(audited["recipe"], base_crops=2, clean_examples=2, crop_seconds=.05)
    rng_before = torch.random.get_rng_state().clone()
    outputs = []
    for name in ("first", "second"):
        output = sealed["root"] / name
        rows = final._render_suites(speech, audited["noise"], output, small)
        for suite, records in rows.items():
            _jsonl(output / f"{suite}.jsonl", records)
        outputs.append((output, rows))
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert outputs[0][1] == outputs[1][1]
    directory, rows = outputs[0]
    assert len(rows["mixtures"]) == 10 and len(rows["clean"]) == 2
    for base in range(2):
        group = [row for row in rows["mixtures"] if row["base_crop"] == base]
        for key in ("speech_id", "speech_offset", "noise_id", "noise_offset", "item_seed"):
            assert len({row[key] for row in group}) == 1
        for row in group:
            clean, rate = sf.read(directory / row["clean"], dtype="float32")
            noisy, _ = sf.read(directory / row["noisy"], dtype="float32")
            active = _active_mask(clean)
            measured = 10 * np.log10(np.mean(clean[active].astype(np.float64) ** 2) / np.mean((noisy - clean)[active].astype(np.float64) ** 2))
            assert measured == pytest.approx(row["snr_db"], abs=2e-5)
            assert rate == 16000 and len(clean) == 800 and row["source_split"] == "test"
            assert sf.info(directory / row["clean"]).subtype == "FLOAT"
            assert _hash(directory / row["noisy"]) == row["noisy_sha256"]
    mixed_crops = {(r["speech_id"], r["speech_offset"]) for r in rows["mixtures"]}
    clean_crops = {(r["speech_id"], r["speech_offset"]) for r in rows["clean"]}
    assert not mixed_crops & clean_crops
    for row in rows["clean"]:
        assert (directory / row["clean"]).read_bytes() == (directory / row["noisy"]).read_bytes()
    dataset = PairedAudioDataset(directory / "mixtures.jsonl", random_crop=False)
    validation = PairedAudioDataset(audited["freeze"]["source_paths"]["paired_development"], random_crop=False)
    with pytest.raises(ValueError, match="training-origin"):
        check_training_split(dataset, validation)
    with pytest.raises(ValueError, match="development-only"):
        _validated_dataset(directory / "mixtures.jsonl")


def test_public_recipe_publishes_exact_1000_plus100_only_after_reverification(sealed, monkeypatch):
    calls = []

    def light_fixture_writer(path, audio):
        assert len(audio) == 48000 and np.isfinite(audio).all() and np.abs(audio).max() <= .990001
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-only WAV substitute")
        calls.append(path)
        return _hash(path)

    # Full public orchestration/recipe is tested without writing422 MB of WAVs.
    # The preceding test exercises the real deterministic float WAV writer.
    monkeypatch.setattr(final, "_write_audio", light_fixture_writer)
    output = sealed["root"] / "final"
    paths = final.prepare_final_evaluation(plan_path=sealed["plan"], freeze_path=sealed["freeze"],
                                           speech_archive=sealed["archive"], output_dir=output)
    mixtures, clean = read_manifest(paths["mixtures"]), read_manifest(paths["clean"])
    assert len(mixtures) == 1000 and len(clean) == 100 and len(calls) == 2200
    assert len({row["noise_group"] for row in mixtures}) == 29
    assert all(row["samples"] == 48000 and row["source_split"] == "test" for row in mixtures + clean)
    provenance = json.loads(paths["provenance"].read_text())
    assert provenance["model_freeze_sha256"] == _hash(sealed["freeze"])
    assert provenance["inference_performed"] is False and provenance["test_used_for_selection"] is False
    assert (output / "model_freeze.json").read_bytes() == sealed["freeze"].read_bytes()
    with pytest.raises(FileExistsError, match="new directory"):
        final.prepare_final_evaluation(plan_path=sealed["plan"], freeze_path=sealed["freeze"],
                                       speech_archive=sealed["archive"], output_dir=output)


def test_artifact_mutation_during_render_never_publishes_final_output(sealed, monkeypatch):
    def mutate_artifact(*_arguments):
        Path(sealed["freeze_data"]["artifacts"][0]["path"]).write_bytes(b"changed during preparation")
        return {"mixtures": [], "clean": []}

    monkeypatch.setattr(final, "_render_suites", mutate_artifact)
    output = sealed["root"] / "unpublished"
    with pytest.raises(ValueError, match="Frozen file hash"):
        final.prepare_final_evaluation(plan_path=sealed["plan"], freeze_path=sealed["freeze"],
                                       speech_archive=sealed["archive"], output_dir=output)
    assert not output.exists()
