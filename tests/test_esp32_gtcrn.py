"""Fresh-weight checks for the attributed GTCRN waveform reference."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json

import pytest
import torch

from esp32_denoiser.gtcrn_model import GTCRNConfig, GTCRNDenoiser


@pytest.fixture
def model():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(73)
        result = GTCRNDenoiser().eval()
    return result


def test_pinned_source_accounting_and_checkpoint_roundtrip(model):
    root = Path(__file__).parents[1] / "esp32_denoiser/vendor/gtcrn"
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    for row in provenance["files"]:
        assert hashlib.sha256((root / row["local_path"]).read_bytes()).hexdigest() == row["vendored_sha256"]
    assert provenance["pretrained_weights_included"] is False
    stats = model.model_stats()
    assert stats["learned_parameters"] == 23_669
    assert stats["total_parameters_including_fixed"] == 48_245
    assert stats["fixed_erb_bytes_float32"] == 98_304
    assert stats["fixed_erb_nonzero_values_one_direction"] == 382
    model.requires_grad_(False)
    assert model.model_stats()["learned_parameters"] == 23_669
    assert model.model_stats()["fixed_erb_parameters"] == 24_576
    rebuilt = GTCRNDenoiser(GTCRNConfig.from_checkpoint(asdict(model.config))).eval()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    legacy_config = asdict(model.config)
    legacy_config.pop("normalize_input")
    legacy_config.pop("rms_floor")
    legacy = GTCRNDenoiser(GTCRNConfig.from_checkpoint(legacy_config)).eval()
    legacy.load_state_dict(model.state_dict(), strict=True)
    assert legacy.config.normalize_input is False
    audio = torch.randn(1, 1043) * .05
    with torch.no_grad():
        torch.testing.assert_close(model(audio), rebuilt(audio), rtol=0, atol=0)
        torch.testing.assert_close(model(audio), legacy(audio), rtol=0, atol=0)
    with pytest.raises(ValueError, match="requires"):
        GTCRNConfig(n_fft=1024)
    with pytest.raises(ValueError, match="revision"):
        GTCRNConfig(upstream_revision="unknown")


@pytest.mark.parametrize("length", [1, 255, 256, 257, 1541])
def test_exact_length_and_complete_streaming_parity(model, length):
    # Nontrivial running statistics check real BatchNorm conversion, not only identity BN.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-.15, .15)
                module.running_var.uniform_(.8, 1.3)
    audio = torch.randn(2, length) * .07
    state_before = torch.random.get_rng_state().clone()
    stream = model.make_streaming()
    assert torch.equal(state_before, torch.random.get_rng_state())
    with torch.no_grad():
        offline = model(audio)
    streamed = stream.denoise(audio)
    assert offline.shape == streamed.shape == audio.shape
    torch.testing.assert_close(offline, streamed, rtol=3e-4, atol=2e-6)
    torch.testing.assert_close(streamed, stream.denoise(audio), rtol=0, atol=0)
    assert all(not p.requires_grad for p in stream.network.parameters())
    assert sum(p.numel() for p in model.parameters()) == 48_245


def test_framing_reconstructs_identity_without_normalization(model):
    # Isolate boundary alignment from enhancement; identity complex spectra must reconstruct.
    model.core = torch.nn.Identity()
    audio = torch.randn(2, 1103) * .13
    torch.testing.assert_close(model(audio), audio, rtol=2e-5, atol=1e-7)


def test_causality_and_length_independent_normalization(model):
    prefix = torch.randn(1, 2048) * .05
    first = torch.cat((prefix, torch.randn(1, 1024) * .08), -1)
    second = torch.cat((prefix, torch.randn(1, 2048) * .3), -1)
    with torch.no_grad():
        y1, y2 = model(first), model(second)
    # One hop of analysis overlap is legitimately influenced by the suffix.
    torch.testing.assert_close(y1[:, :1792], y2[:, :1792], rtol=2e-4, atol=2e-6)
    assert float((y1[:, 2304:] - y2[:, 2304:3072]).abs().max()) > 1e-3


def test_training_autocast_gradients_and_eval_only_streaming(model):
    model.train()
    with pytest.raises(ValueError, match="eval"):
        model.make_streaming()
    noisy = torch.randn(2, 2048) * .1
    target = noisy * .7 + torch.randn_like(noisy) * .01
    with torch.autocast("cpu", dtype=torch.bfloat16):
        estimate = model(noisy)
        loss = (estimate - target).square().mean()
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
               for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.core.erb.parameters())
    assert any(float(p.grad.abs().max()) > 0 for p in model.parameters() if p.requires_grad)
