"""Tiny local archives verify source boundaries and aligned dynamic mixtures."""

from dataclasses import replace
import gzip
import hashlib
import io
import json
import tarfile
import zipfile

import numpy as np
import pytest
import soundfile as sf
import torch

from esp32_denoiser import extra_data as extra
from esp32_denoiser.data import pad_collate


def audio_bytes(samples, fmt="WAV"):
    stream = io.BytesIO()
    sf.write(stream, samples, 16000, format=fmt, subtype="PCM_16")
    return stream.getvalue()


def tar_archive(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, contents in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    archives = tmp_path / "archives"
    archives.mkdir()
    speech, noise, rirs = {}, {}, {}
    for number in range(1, 5):
        clean = np.concatenate([np.zeros(320), .2 * np.sin(np.arange(1280) * (.04 * number))])
        for utterance in range(2):
            # Distinct encodings keep duplicate-content detection meaningful.
            speech[f"LibriSpeech/train-clean-100/{number}/10/{number}-10-{utterance:04d}.flac"] = audio_bytes(clean * (1 - utterance * .01), "FLAC")
        noise[f"musan/noise/free-sound/noise-{number:04d}.wav"] = audio_bytes(np.random.default_rng(number).uniform(-.1, .1, 400))
        for position in range(2):
            rirs[f"RIRS_NOISES/simulated_rirs/smallroom/Room{number:03d}/rir-{position}.wav"] = audio_bytes(np.exp(-np.arange(400) * .1) * (.1 * number + position * .01))
    noise["musan/music/music.wav"] = audio_bytes(np.ones(500) * .1)
    noise["musan/speech/speech.wav"] = audio_bytes(np.ones(500) * .1)
    noise["musan/noise/free-sound/LICENSE"] = b"Keep attribution."
    speech["LibriSpeech/LICENSE.TXT"] = b"Keep original attribution."
    rirs["RIRS_NOISES/pointsource_noises/noise.wav"] = audio_bytes(np.ones(500) * .1)
    rirs["RIRS_NOISES/real_rirs_isotropic_noises/rir.wav"] = audio_bytes(np.ones(500) * .1)
    tar_archive(archives / "train-clean-100.tar.gz", speech)
    tar_archive(archives / "musan.tar.gz", noise)
    with zipfile.ZipFile(archives / "rirs_noises.zip", "w") as archive:
        for name, contents in rirs.items():
            archive.writestr(name, contents)
    local_source = replace(extra.SOURCES["speech"], expected_md5=hashlib.md5((archives / "train-clean-100.tar.gz").read_bytes()).hexdigest())
    monkeypatch.setitem(extra.SOURCES, "speech", local_source)
    return tmp_path, extra.prepare_extra_data(tmp_path, include_rir=True, validation_fraction=.25)


def test_prepare_preserves_sources_holds_out_groups_and_hashes(prepared):
    root, outputs = prepared
    for kind in ("speech", "noise", "rir"):
        train = extra.read_extra_manifest(outputs[f"{kind}_train"], kind)
        val = extra.read_extra_manifest(outputs[f"{kind}_val"], kind)
        for key in ("id", "group", "path", "sha256"):
            assert {row[key] for row in train}.isdisjoint({row[key] for row in val})
        assert len({row["group"] for row in train}) == 3
        assert len({row["group"] for row in val}) == 1
        assert all(extra._hash_file(row["path"])["sha256"] == row["sha256"] for row in train + val)
        assert all("mtime_ns" not in row for row in train + val)
    assert not list((root / "sources" / "noise").rglob("music.wav"))
    assert not list((root / "sources" / "noise").rglob("speech.wav"))
    assert not (root / "sources/rir/RIRS_NOISES/pointsource_noises").exists()
    assert (root / "sources/noise/musan/noise/free-sound/LICENSE").read_text() == "Keep attribution."
    assert len(list((root / "sources/speech").rglob("*.flac"))) == 8
    assert not list((root / "sources/speech").rglob("*.wav"))
    provenance = json.loads(outputs["provenance"].read_text())
    assert provenance["sources"]["speech"]["publisher_checksum_verified"]
    assert not provenance["sources"]["noise"]["publisher_checksum_verified"]
    for name, details in provenance["splits"].items():
        assert details["manifest_sha256"] == extra._hash_file(outputs[name])["sha256"]
    # Repreparation reuses intact audio while preserving all manifest bytes.
    original = {key: path.read_bytes() for key, path in outputs.items()}
    extra.prepare_extra_data(root, include_rir=True, validation_fraction=.25)
    assert original == {key: path.read_bytes() for key, path in outputs.items()}


def test_preparation_requires_explicit_download_and_checks_cached_changes(prepared, tmp_path):
    with pytest.raises(FileNotFoundError, match="download=True"):
        extra.prepare_extra_data(tmp_path / "missing")
    root, outputs = prepared
    record = extra.read_extra_manifest(outputs["speech_train"], "speech")[0]
    path = __import__("pathlib").Path(record["path"])
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="checksum"):
        extra.prepare_extra_data(root)


def test_unsafe_archive_member_cannot_write_outside_destination(tmp_path):
    archive = tmp_path / "musan.tar.gz"
    tar_archive(archive, {"../escape.txt": b"bad"})
    with pytest.raises(ValueError, match="Unsafe archive member"):
        extra._extract_source(archive, extra.SOURCES["noise"], "noise", tmp_path / "extracted")
    assert not (tmp_path / "escape.txt").exists()


def test_duplicate_content_across_holdout_is_rejected():
    records = [{"id": str(i), "group": str(i), "path": str(i), "sha256": "same"} for i in range(3)]
    with pytest.raises(ValueError, match="sha256 overlap"):
        extra._group_split(records, .33, 2026)


def test_mixture_active_snr_alignment_padding_and_worker_rng(prepared):
    _, paths = prepared
    dataset = extra.DynamicMixtureDataset(paths["speech_train"], paths["noise_train"], crop_seconds=.2,
                                          snr_db=(10, 10), gain_db=(12, 12), clean_identity_prob=0,
                                          samples_per_epoch=12, peak_limit=.3)
    assert len(dataset) == 12 and dataset.sample_rate == 16000 and dataset.split == "train"
    torch.manual_seed(132)
    first = dataset[0]
    torch.manual_seed(132)
    repeat = dataset[0]
    assert first["mixture"] == repeat["mixture"]
    torch.testing.assert_close(first["noisy"], repeat["noisy"], rtol=0, atol=0)
    clean = first["clean"].numpy()[:first["length"]]
    noise = (first["noisy"] - first["clean"]).numpy()[:first["length"]]
    active = extra._active_mask(clean)
    assert 10 * np.log10(np.mean(clean[active] ** 2) / np.mean(noise[active] ** 2)) == pytest.approx(10, abs=1e-5)
    source = next(row for row in dataset.speech_records if row["id"] == first["mixture"]["speech_id"])
    original, _ = sf.read(source["path"], dtype="float32")
    np.testing.assert_allclose(clean, original * first["mixture"]["common_gain"], rtol=1e-6, atol=1e-8)
    assert first["length"] == 1600 and first["noisy"].shape == (3200,)
    assert first["noisy"].abs().max() <= .300001
    assert torch.count_nonzero(first["clean"][1600:]) == torch.count_nonzero(first["noisy"][1600:]) == 0
    batch = pad_collate([first, repeat])
    assert batch["noisy"].shape == (2, 3200)
    assert batch["length"].tolist() == [1600, 1600]


def test_identity_and_bounded_partial_reads(prepared, monkeypatch):
    _, paths = prepared
    calls, actual_read = [], sf.read

    def read(*args, **kwargs):
        calls.append(kwargs)
        return actual_read(*args, **kwargs)

    monkeypatch.setattr(extra.sf, "read", read)
    dataset = extra.DynamicMixtureDataset(paths["speech_train"], paths["noise_train"], crop_seconds=.05,
                                          gain_db=(0, 0), clean_identity_prob=1)
    item = dataset[0]
    assert torch.equal(item["noisy"], item["clean"])
    assert item["mixture"]["noise_id"] is None and item["length"] == 800
    assert len(calls) == 1 and calls[0]["frames"] == 800 and calls[0]["start"] >= 0


def test_mixed_or_unapproved_source_partitions_rejected(prepared):
    _, paths = prepared
    with pytest.raises(ValueError, match="same train/validation"):
        extra.DynamicMixtureDataset(paths["speech_train"], paths["noise_val"])
    rows = [json.loads(line) for line in paths["noise_train"].read_text().splitlines()]
    rows[0]["source"] = "https://www.openslr.org/12/"
    paths["noise_train"].write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="unapproved"):
        extra.read_extra_manifest(paths["noise_train"], "noise")


def test_ignored_download_range_restarts_instead_of_appending(tmp_path, monkeypatch):
    contents = gzip.compress(b"a source archive")
    url = "https://www.openslr.org/resources/17/test.tar.gz"
    source = extra.SourceArchive("test.tar.gz", (url,), extra.SOURCES["noise"].source, "CC-BY-4.0",
                                 hashlib.md5(contents).hexdigest())
    (tmp_path / "test.tar.gz.part").write_bytes(contents[:5])
    (tmp_path / "test.tar.gz.part.url").write_text(url)
    requests = []

    def respond(request, timeout):
        requests.append(request)
        response = io.BytesIO(contents)
        response.status = 200
        response.headers = {"Content-Length": str(len(contents))}
        return response

    monkeypatch.setattr(extra.urllib.request, "urlopen", respond)
    path = extra.download_source(source, tmp_path)
    assert requests[0].headers["Range"] == "bytes=5-"
    assert path.read_bytes() == contents
    assert not (tmp_path / "test.tar.gz.part").exists()


def test_html_archive_download_fails_cleanly(tmp_path, monkeypatch):
    source = extra.SourceArchive("bad.tar.gz", ("https://www.openslr.org/bad",), "test", "test")

    def respond(*args, **kwargs):
        response = io.BytesIO(b"<!doctype html>")
        response.status = 200
        response.headers = {"Content-Type": "text/html"}
        return response

    monkeypatch.setattr(extra.urllib.request, "urlopen", respond)
    with pytest.raises(RuntimeError, match="returned HTML"):
        extra.download_source(source, tmp_path)
    assert not (tmp_path / "bad.tar.gz").exists()
