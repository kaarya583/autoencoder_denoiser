"""Feature-layout ablations must preserve causality and checkpoint meaning."""

from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from esp32_denoiser.evaluate import load_checkpoint
from esp32_denoiser.export import export_model
from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig
from esp32_denoiser.train import TrainConfig, train


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_full_magnitudes_and_low_phase_have_exact_order_and_same_neural_budget():
    model = SpectralTCN(SpectralTCNConfig(feature_layout="fullmag_lowphase"))
    torch.manual_seed(31)
    frames = torch.randn(2, 3, 512) * 0.1
    spectrum, features = model.frame_features(frames)
    normalized = spectrum / (512 * frames.square().mean(-1, keepdim=True).sqrt())
    root = normalized.abs().clamp_min(1e-8).sqrt()
    torch.testing.assert_close(features[:, :257], root.transpose(1, 2))
    torch.testing.assert_close(features[:, 257:322],
                               (normalized.real[..., :65] / root[..., :65]).transpose(1, 2))
    torch.testing.assert_close(features[:, 322:],
                               (normalized.imag[..., :65] / root[..., :65]).transpose(1, 2))
    _, individual = model.frame_features(frames[:, 1:2])
    torch.testing.assert_close(individual, features[..., 1:2])
    assert features.abs().max() <= 0.841
    assert model.model_stats() == SpectralTCN().model_stats()
    assert not hasattr(model, "erb_lower")


def test_fullmag_streaming_and_autocast_match_nontrivial_offline_model():
    torch.manual_seed(38)
    model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2, 4),
                                          feature_layout="fullmag_lowphase")).eval()
    with torch.no_grad():
        model.head.weight.normal_(std=0.15)
        model.head.bias.normal_(std=0.03)
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
        assert (expected - audio).abs().max() > 0.01
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = (model(audio) - audio * 0.8).square().mean()
    loss.backward()
    assert torch.isfinite(model.head.weight.grad).all()


@pytest.mark.parametrize("layout, activation, omitted", [
    ("erb_complex", "relu", ("activation_mode", "feature_layout")),
    ("erb_complex", "signed", ("feature_layout",)),
    ("fullmag_lowphase", "signed", ()),
])
def test_checkpoint_loading_preserves_historical_and_current_semantics(tmp_path, layout, activation, omitted):
    config = SpectralTCNConfig(width=8, dilations=(1, 2), activation_mode=activation, feature_layout=layout)
    model = SpectralTCN(config).eval()
    with torch.no_grad():
        model.head.weight.normal_(std=0.1)
    saved_config = asdict(config)
    for name in omitted:
        del saved_config[name]
    saved_config["dilations"] = list(saved_config["dilations"])
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "model_config": saved_config, "phase": "float"}, checkpoint)
    loaded, _ = load_checkpoint(checkpoint)
    assert loaded.config == config
    audio = torch.randn(1, 701) * 0.1
    with torch.inference_mode():
        torch.testing.assert_close(loaded(audio), model(audio), atol=0, rtol=0)


def test_resume_normalizes_existing_signed_config_and_rejects_feature_mismatch(tmp_path, monkeypatch):
    model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2)))
    saved_config = asdict(model.config)
    del saved_config["feature_layout"]  # Existing signed v2 checkpoint.
    saved_config["dilations"] = list(saved_config["dilations"])
    checkpoint = tmp_path / "v2.pt"
    torch.save({"model": model.state_dict(), "model_config": saved_config, "phase": "float"}, checkpoint)

    class ReachedDataset(Exception):
        pass

    def dataset_after_successful_resume(*args, **kwargs):
        raise ReachedDataset

    monkeypatch.setattr("esp32_denoiser.train.PairedAudioDataset", dataset_after_successful_resume)
    config = TrainConfig(train_manifest="unused", val_manifest="unused",
                         output_dir=str(tmp_path / "run"), resume=str(checkpoint),
                         width=8, dilations=(1, 2))
    with pytest.raises(ReachedDataset):
        train(config)  # The architecture comparison and state loading succeed.
    config.feature_layout = "fullmag_lowphase"
    with pytest.raises(ValueError, match="Checkpoint architecture differs"):
        train(config)


def test_experimental_layout_cannot_be_exported_as_existing_firmware_format(tmp_path):
    model = SpectralTCN(SpectralTCNConfig(feature_layout="fullmag_lowphase"))
    with pytest.raises(ValueError, match="only the erb_complex feature layout"):
        export_model(model, tmp_path / "unsupported.bin")
    assert not (tmp_path / "unsupported.bin").exists()
