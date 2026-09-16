"""High-value data alignment, split isolation, and metric regression checks."""

import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from esp32_denoiser.data import (
    ARCHIVES,
    HF_REVISION,
    PairedAudioDataset,
    discover_pairs,
    pad_collate,
    prepare_voicebank,
    prepare_voicebank_parquet,
    read_manifest,
    split_training_pairs,
)
from esp32_denoiser.metrics import evaluate_utterances, si_sdr, si_sdri, summarize_utterances


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def raw_pair(self, utterance, clean, noisy=None, rate=48000, split="train"):
        for role, audio in (("clean", clean), ("noisy", clean if noisy is None else noisy)):
            directory = self.root / "raw" / ARCHIVES[f"{split}_{role}"].filename.removesuffix(".zip")
            directory.mkdir(parents=True, exist_ok=True)
            sf.write(directory / f"{utterance}.wav", audio, rate, subtype="FLOAT")

    def prepare_fixture(self):
        waveform = np.sin(np.arange(4800) * 2 * np.pi * 440 / 48000).astype(np.float32) * 0.7
        for utterance in ("p226_001", "p287_001", "p225_001"):
            self.raw_pair(utterance, waveform, waveform * 2)
        return prepare_voicebank(self.root, workers=2)

    def test_resampling_alignment_validation_and_untouched_test(self):
        manifests = self.prepare_fixture()
        self.assertEqual(set(manifests), {"train", "val"})
        training = read_manifest(manifests["train"])
        validation = read_manifest(manifests["val"])
        self.assertEqual({r["speaker"] for r in training}, {"p225"})
        self.assertEqual({r["speaker"] for r in validation}, {"p226", "p287"})
        self.assertEqual(training[0]["samples"], 1600)
        clean, sr = sf.read(training[0]["clean"])
        noisy, _ = sf.read(training[0]["noisy"])
        self.assertEqual(sr, 16000)
        np.testing.assert_allclose(noisy, clean * 2, atol=1e-7)
        self.assertGreater(abs(noisy).max(), 1.0)  # no clipping introduced by cache
        self.assertEqual(sf.info(training[0]["noisy"]).subtype, "FLOAT")
        self.assertFalse((self.root / "manifests" / "test.jsonl").exists())
        stamp = Path(training[0]["clean"]).stat().st_mtime_ns
        prepare_voicebank(self.root, workers=1)
        self.assertEqual(Path(training[0]["clean"]).stat().st_mtime_ns, stamp)

    def test_no_download_without_explicit_request(self):
        with patch("urllib.request.urlopen") as urlopen:
            with self.assertRaises(FileNotFoundError):
                prepare_voicebank(self.root)
        urlopen.assert_not_called()

    def test_missing_pairs_and_missing_validation_speakers_fail(self):
        self.raw_pair("p225_001", np.ones(100, dtype=np.float32))
        with self.assertRaises(ValueError):
            split_training_pairs([dict(id="p225_001", speaker="p225")])
        noisy = self.root / "raw" / "noisy_trainset_28spk_wav" / "p225_001.wav"
        noisy.rename(noisy.with_name("p225_002.wav"))
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            discover_pairs(self.root / "raw" / "clean_trainset_28spk_wav", noisy.parent)

    def test_misaligned_pair_fails_instead_of_truncating(self):
        for utterance in ("p226_001", "p287_001", "p225_001"):
            self.raw_pair(utterance, np.ones(100, dtype=np.float32), np.ones(99, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "lengths must match"):
            prepare_voicebank(self.root)

    def test_shared_gain_crop_noise_scaling_and_valid_lengths(self):
        manifests = self.prepare_fixture()
        dataset = PairedAudioDataset(manifests["train"], crop_seconds=0.05, gain_db=(6, 6))
        torch.manual_seed(10)
        first = dataset[0]
        torch.manual_seed(10)
        repeated = dataset[0]
        torch.testing.assert_close(first["clean"], repeated["clean"])
        torch.testing.assert_close(first["noisy"], first["clean"] * 2)
        self.assertEqual(first["length"], 800)
        padded = PairedAudioDataset(manifests["train"], crop_seconds=0.2, random_crop=False)[0]
        self.assertEqual(padded["length"], 1600)
        self.assertEqual(padded["clean"].numel(), 3200)
        self.assertEqual(padded["clean"][1600:].abs().sum(), 0)
        batch = pad_collate([first, padded])
        self.assertEqual(batch["clean"].shape, (2, 3200))
        self.assertEqual(batch["length"].tolist(), [800, 1600])
        scaled = PairedAudioDataset(manifests["train"], crop_seconds=None, noise_scale_db=(20, 20))[0]
        torch.testing.assert_close(scaled["noisy"], scaled["clean"] * 11)
        identity = PairedAudioDataset(manifests["train"], clean_identity_prob=1)[0]
        torch.testing.assert_close(identity["noisy"], identity["clean"])

    def test_evaluation_refuses_augmented_or_cropped_audio(self):
        manifests = self.prepare_fixture()
        for kwargs in ({"crop_seconds": 0.05}, {"crop_seconds": None, "gain_db": (1, 1)},
                       {"crop_seconds": None, "noise_scale_db": (-6, 6)},
                       {"crop_seconds": None, "clean_identity_prob": 0.1}):
            with self.assertRaises(ValueError):
                evaluate_utterances(lambda x: x, PairedAudioDataset(manifests["train"], **kwargs))

    def test_parquet_mirror_is_streamed_and_provenance_is_explicit(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("Optional pyarrow is not installed")
        target = self.root / "archives" / "hf16k" / "data"
        target.mkdir(parents=True)
        blob = io.BytesIO()
        waveform = np.sin(np.arange(160) * 0.1).astype(np.float32)
        sf.write(blob, waveform, 16000, format="WAV", subtype="FLOAT")
        ids = ("p226_001", "p287_001", "p225_001", "p225_002", "p225_003")
        for shard, utterance in enumerate(ids):
            table = pa.Table.from_pylist([dict(id=utterance, clean=dict(bytes=blob.getvalue(), path="ignored.wav"),
                                             noisy=dict(bytes=blob.getvalue(), path="ignored.wav"))])
            pq.write_table(table, target / f"train-{shard:05d}-of-00005.parquet")
        manifests = prepare_voicebank_parquet(self.root, download=False, workers=2)
        self.assertEqual(len(read_manifest(manifests["train"])), 3)
        self.assertEqual(len(read_manifest(manifests["val"])), 2)
        provenance = json.loads((manifests["train"].parent / "provenance.json").read_text())
        self.assertEqual(provenance["mirror_revision"], HF_REVISION)
        self.assertIn("unverified", provenance["resampling"])
        self.assertFalse(provenance["official_test_prepared"])


class MetricTests(unittest.TestCase):
    def test_known_ratio_zero_mean_and_scale_invariance(self):
        reference = torch.tensor([1, -1, 1, -1], dtype=torch.float64)
        orthogonal = torch.tensor([1, 1, -1, -1], dtype=torch.float64)
        estimate = reference + orthogonal * 0.1
        self.assertAlmostEqual(si_sdr(estimate, reference).item(), 20.0, places=8)
        self.assertAlmostEqual(si_sdr(estimate * -3 + 20, reference + 7).item(), 20.0, places=8)
        self.assertAlmostEqual(si_sdri(estimate, reference + orthogonal, reference).item(), 20.0, places=8)

    def test_padding_never_changes_score_even_nonfinite_padding(self):
        reference = torch.tensor([[1, -1, 1, -1, math.nan, math.inf]], dtype=torch.float32)
        estimate = torch.tensor([[1.1, -0.9, 0.9, -1.1, math.nan, math.inf]], dtype=torch.float32)
        torch.testing.assert_close(si_sdr(estimate, reference, torch.tensor([4])),
                                   si_sdr(estimate[:, :4], reference[:, :4]))

    def test_silence_nonfinite_zero_estimate_and_invalid_lengths(self):
        reference = torch.tensor([1.0, -1.0, 1.0, -1.0])
        self.assertTrue(torch.isnan(si_sdr(reference, torch.ones(4))).item())
        self.assertTrue(torch.isnan(si_sdr(torch.tensor([math.nan, 1, 2, 3]), reference)).item())
        self.assertEqual(si_sdr(torch.zeros(4), reference).item(), -80.0)
        self.assertEqual(si_sdr(reference, reference).item(), 80.0)
        with self.assertRaises(ValueError):
            si_sdr(reference, reference, torch.tensor(0))

    def test_summary_is_equal_per_utterance_and_reports_invalids(self):
        records = [dict(id="short", samples=10, si_sdr_noisy=0, si_sdr_enhanced=2),
                   dict(id="long", samples=10000, si_sdr_noisy=10, si_sdr_enhanced=20),
                   dict(id="silent", samples=100, si_sdr_noisy=math.nan, si_sdr_enhanced=math.nan)]
        result = summarize_utterances(records)
        self.assertEqual(result["si_sdri"], 6.0)
        self.assertEqual(result["valid_utterances"], 2)
        self.assertEqual(result["invalid_utterances"], 1)
        self.assertIsNone(summarize_utterances(records[-1:])["si_sdri"])
        with self.assertRaises(ValueError):
            summarize_utterances(records * 2)

    def test_evaluation_restores_mode_and_fails_on_bad_model_output(self):
        audio = torch.tensor([1.0, -1.0, 1.0, -1.0])
        dataset = [dict(id="one", length=4, noisy=audio, clean=audio)]
        module = torch.nn.Identity().train()
        result = evaluate_utterances(module, dataset)
        self.assertTrue(module.training)
        self.assertEqual(result["summary"]["si_sdri"], 0)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            evaluate_utterances(lambda x: x * math.nan, dataset)
        with self.assertRaisesRegex(ValueError, "shape"):
            evaluate_utterances(lambda x: x[:, :-1], dataset)


if __name__ == "__main__":
    unittest.main()
