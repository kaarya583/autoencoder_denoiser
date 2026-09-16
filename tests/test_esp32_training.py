"""Tiny real optimization passes exercise float, QAT, resume, and split guards."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from esp32_denoiser.train import TrainConfig, speech_loss, train


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        manifests = {}
        for split, speakers in (("train", ("p225", "p227")), ("val", ("p226", "p287"))):
            rows = []
            for index, speaker in enumerate(speakers):
                samples = 768 + index * 256
                time = np.arange(samples) / 16000
                clean = (0.15 * np.sin(2 * np.pi * 420 * time)).astype(np.float32)
                noisy = clean + (0.08 * np.cos(2 * np.pi * 1200 * time)).astype(np.float32)
                row = dict(id=f"{speaker}_001", speaker=speaker, samples=samples, sample_rate=16000, source_split="train")
                for role, audio in (("clean", clean), ("noisy", noisy)):
                    path = self.root / f"{speaker}_{role}.wav"
                    sf.write(path, audio, 16000, subtype="FLOAT")
                    row[role] = str(path)
                rows.append(row)
            path = self.root / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifests[split] = str(path)
        self.config = TrainConfig(
            train_manifest=manifests["train"], val_manifest=manifests["val"],
            output_dir=str(self.root / "float"), epochs=1, batch_size=2,
            crop_seconds=0.048, learning_rate=3e-4, workers=0,
            max_hours=0.01, eval_batch_size=2, max_steps_per_epoch=1,
            width=4, dilations=(1, 2), amp=False,
        )

    def test_float_to_qat_and_same_phase_resume(self):
        summary = train(self.config)
        self.assertEqual(summary["epoch"], 1)
        self.assertTrue(math.isfinite(summary["best_si_sdri"]))
        float_dir = Path(self.config.output_dir)
        float_checkpoint = torch.load(float_dir / "last.pt", weights_only=False)
        self.assertFalse(any("input_exponent" in key for key in float_checkpoint["model"]))
        history = [json.loads(row) for row in (float_dir / "history.jsonl").read_text().splitlines()]
        self.assertTrue(math.isfinite(history[0]["loss"]))
        qat_config = replace(self.config, phase="qat", resume=str(float_dir / "best.pt"), output_dir=str(self.root / "qat"))
        qat_summary = train(qat_config)
        self.assertEqual(qat_summary["epoch"], 1)
        qat_dir = Path(qat_config.output_dir)
        checkpoint = torch.load(qat_dir / "last.pt", weights_only=False)
        self.assertEqual(checkpoint["phase"], "qat")
        self.assertIn("input_proj.input_exponent", checkpoint["model"])
        self.assertEqual(checkpoint["calibration"]["source"], "training manifest only")
        resumed = train(replace(qat_config, epochs=2, resume=str(qat_dir / "last.pt")))
        self.assertEqual(resumed["epoch"], 2)
        self.assertTrue(math.isfinite(resumed["best_si_sdri"]))
        checkpoint = torch.load(qat_dir / "last.pt", weights_only=False)
        self.assertEqual(checkpoint["epoch"], 2)
        for value in checkpoint["model"].values():
            self.assertTrue(torch.isfinite(value).all())
        provenance = json.loads((qat_dir / "provenance.json").read_text())
        self.assertFalse(provenance["test_used_for_selection"])
        self.assertFalse(set(provenance["train_speakers"]) & set(provenance["validation_speakers"]))
        with self.assertRaisesRegex(ValueError, "QAT checkpoint"):
            train(replace(qat_config, phase="float", resume=str(qat_dir / "last.pt")))

    def test_resumed_epoch_matches_uninterrupted_float_training(self):
        train(replace(self.config, epochs=2, output_dir=str(self.root / "uninterrupted")))
        train(self.config)
        train(replace(self.config, epochs=2, resume=str(Path(self.config.output_dir) / "last.pt")))
        expected = torch.load(self.root / "uninterrupted" / "last.pt", weights_only=False)["model"]
        actual = torch.load(Path(self.config.output_dir) / "last.pt", weights_only=False)["model"]
        for key in expected:
            # CUDA reductions may differ by an ulp despite identical epoch RNGs.
            tolerance = dict(rtol=1e-6, atol=1e-7) if expected[key].is_cuda else dict(rtol=0, atol=0)
            torch.testing.assert_close(actual[key], expected[key], **tolerance)

    def test_silent_and_padded_loss_has_finite_gradients(self):
        estimate = torch.zeros(2, 768, requires_grad=True)
        clean = torch.zeros_like(estimate)
        loss = speech_loss(estimate, clean, torch.tensor([768, 512]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(estimate.grad).all())
        self.assertEqual(estimate.grad[1, 512:].abs().sum(), 0)

    def test_overlapping_speakers_and_official_test_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "speaker overlap"):
            train(replace(self.config, val_manifest=self.config.train_manifest))
        path = Path(self.config.val_manifest)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["source_split"] = "test"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "training-origin"):
            train(self.config)

    def test_zero_step_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "limits must be positive"):
            train(replace(self.config, max_steps_per_epoch=0))

    def test_fine_tuning_restarts_optimizer_and_records_lineage(self):
        train(self.config)
        fine = replace(self.config, resume=str(Path(self.config.output_dir) / "best.pt"),
                       output_dir=str(self.root / "fine"), resume_optimizer=False,
                       learning_rate=1e-5, spectral_loss_weight=0.1)
        result = train(fine)
        self.assertEqual(result["epoch"], 1)
        checkpoint = torch.load(self.root / "fine/last.pt", weights_only=False)
        self.assertEqual(checkpoint["model_kind"], "spectral_tcn")
        self.assertEqual(checkpoint["optimizer"]["param_groups"][0]["lr"], 1e-5)
        self.assertEqual(len(checkpoint["provenance"]["manifest_sha256"]["val"]), 64)

    def test_frequency_training_checkpoint_and_resume(self):
        from esp32_denoiser.evaluate import load_checkpoint
        config = replace(self.config, model_kind="frequency_unet",
                         model_options={"encoder_channels": [2, 3, 4], "global_width": 4,
                                        "local_dilations": [1], "global_dilations": [1, 2]})
        train(config)
        path = Path(config.output_dir) / "last.pt"
        model, metadata = load_checkpoint(path)
        self.assertEqual(metadata["model_kind"], "frequency_unet")
        self.assertEqual(model.config.encoder_channels, (2, 3, 4))
        result = train(replace(config, epochs=2, resume=str(path)))
        self.assertEqual(result["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
