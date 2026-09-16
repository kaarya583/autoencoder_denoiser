"""High-value checks for the causal audio and neural-state contracts."""

import unittest

import torch
from torch.nn import functional as F

from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig


class SpectralTCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_residual_path_preserves_negative_state_and_gradient(self):
        model = SpectralTCN()
        block = model.blocks[0]
        with torch.no_grad():
            block.pointwise.weight.zero_()
            block.pointwise.bias.zero_()
        state = torch.full((1, 64, 12), -0.25, requires_grad=True)
        result = block(state)
        torch.testing.assert_close(result, state, atol=0, rtol=0)
        result.sum().backward()
        torch.testing.assert_close(state.grad, torch.ones_like(state), atol=0, rtol=0)

    def test_default_budget(self):
        stats = SpectralTCN().model_stats()
        self.assertEqual(stats["learned_parameters"], 84738)
        self.assertEqual(stats["convolution_weights"], 83392)
        self.assertEqual(stats["macs_per_second"], 5212000)
        self.assertEqual(stats["neural_state_bytes_int8"], 8064)

    def test_identity_initialization_preserves_length_and_silence(self):
        model = SpectralTCN().eval()
        with torch.no_grad():
            for length in (1, 255, 256, 257, 1301):
                audio = torch.randn(2, length) * 0.1
                self.assertEqual(model(audio).shape, audio.shape)
                torch.testing.assert_close(model(audio), audio, atol=2e-6, rtol=2e-5)
            silence = model(torch.zeros(2, 1024))
            self.assertTrue(torch.isfinite(silence).all())
            self.assertEqual(silence.abs().max().item(), 0)

    def test_streaming_matches_vectorized_nontrivial_model(self):
        torch.manual_seed(2)
        model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2, 4))).eval()
        with torch.no_grad():
            model.head.weight.normal_(std=0.04)
            audio = torch.randn(2, 1357) * 0.1
            expected = model(audio)
            padded = F.pad(audio, (0, (-audio.shape[-1]) % 256 + 256))
            state = model.init_stream_state(2)
            outputs = []
            for chunk in padded.split(256, dim=-1):
                output, state = model.stream_step(chunk, state)
                outputs.append(output)
            actual = torch.cat(outputs, dim=-1)[:, 256:256 + audio.shape[-1]]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_future_neural_features_cannot_change_past_output(self):
        model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2))).eval()
        with torch.no_grad():
            model.head.weight.normal_(std=0.1)
            first = torch.randn(1, 387, 20)
            second = first.clone()
            second[..., 10:] = torch.randn_like(second[..., 10:])
            torch.testing.assert_close(model.forward_features(first)[..., :10],
                                       model.forward_features(second)[..., :10])

    def test_differentiable_waveform_reconstruction(self):
        model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2)))
        audio = torch.randn(2, 1027) * 0.1
        prediction = model(audio)
        loss = (prediction - audio * 0.8).square().mean()
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)

    def test_bfloat16_neural_autocast_keeps_dsp_valid(self):
        model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2)))
        audio = torch.randn(2, 1024) * 0.1
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(audio)
            loss = (output - audio * 0.8).square().mean()
        loss.backward()
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())

    def test_complex_head_can_correct_opposite_phase(self):
        model = SpectralTCN()
        spectrum = torch.ones(1, 1, 257, dtype=torch.complex64)
        deltas = torch.zeros(1, 514, 1)
        deltas[:, :257] = -1
        torch.testing.assert_close(model.apply_mask(spectrum, deltas), -spectrum)

    def test_frame_features_are_bounded_and_independent_of_other_frames(self):
        model = SpectralTCN().eval()
        frames = torch.randn(2, 5, 512) * 0.1
        _, features = model.frame_features(frames)
        _, one = model.frame_features(frames[:, 2:3])
        torch.testing.assert_close(features[..., 2:3], one)
        self.assertLessEqual(features.abs().max().item(), 1)


if __name__ == "__main__":
    unittest.main()
