import pytest
import torch

from esp32_denoiser.losses import compressed_spectral_loss


def test_spectral_loss_padding_matches_individual_clips_and_gradients():
    torch.manual_seed(17)
    target = torch.randn(2, 1300) * 0.1
    estimate = (target + torch.randn_like(target) * 0.03).requires_grad_()
    lengths = torch.tensor([513, 1300])
    batch = compressed_spectral_loss(estimate, target, lengths)
    individual = torch.stack([
        compressed_spectral_loss(estimate[i:i+1, :n], target[i:i+1, :n], torch.tensor([n]))
        for i, n in enumerate(lengths.tolist())
    ]).mean()
    torch.testing.assert_close(batch, individual, atol=1e-7, rtol=1e-6)
    batch.backward()
    assert torch.isfinite(estimate.grad).all()
    assert estimate.grad[0, 513:].abs().sum() == 0


def test_spectral_loss_identity_silence_and_short_clips():
    audio = torch.zeros(2, 31, requires_grad=True)
    loss = compressed_spectral_loss(audio, audio.detach(), torch.tensor([1, 31]))
    assert loss.item() == pytest.approx(0, abs=1e-8)
    loss.backward()
    assert torch.isfinite(audio.grad).all()
    with pytest.raises(ValueError, match="length"):
        compressed_spectral_loss(audio, audio, torch.tensor([0, 31]))
