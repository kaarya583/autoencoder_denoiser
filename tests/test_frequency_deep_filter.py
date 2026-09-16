"""Past-only deep filtering preserves the baseline and streaming DSP contract."""

from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from esp32_denoiser.frequency_deep_filter import FrequencyDeepFilter, FrequencyDeepFilterConfig
from esp32_denoiser.frequency_model import FrequencyUNet


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(**kwargs):
    return FrequencyDeepFilterConfig(encoder_channels=(4, 6, 8), global_width=8,
                                      local_dilations=(1, 2), global_dilations=(1, 2), **kwargs)


def test_default_cost_and_exact_spectral_history_allocation():
    model = FrequencyDeepFilter()
    stats = model.model_stats()
    assert stats["learned_parameters"] == 83_564
    assert stats["deployed_parameters"] == 83_340
    assert stats["training_normalization_parameters"] == 224
    assert stats["deep_filter_added_parameters"] == 170
    assert stats["macs_per_second"] == 28_211_000
    assert stats["deep_filter_added_neural_macs_per_frame"] == 10_400
    assert stats["deep_filter_real_products_per_second"] == 81_250
    assert stats["deep_filter_coefficient_scale_products_per_second"] == 40_625
    assert stats["deep_filter_spectral_history_bytes_float32"] == 2_080
    assert stats["additional_lookahead_frames"] == 0
    assert stats["neural_output_values_per_frame"] == 1_164
    assert "float-only" in stats["precision"]
    history = model.init_stream_state().spectral_history
    assert history.shape == (1, 4, 65) and history.dtype == torch.complex64
    assert history.untyped_storage().nbytes() == 2_080
    with torch.inference_mode():
        _, history = model.apply_filter(torch.randn(1, 2, 257, dtype=torch.complex64),
                                        torch.zeros(1, 1164, 2), history)
    assert history.untyped_storage().nbytes() == 2_080
    values = asdict(model.config)
    for key in ("encoder_channels", "local_dilations", "global_dilations"):
        values[key] = list(values[key])
    assert FrequencyDeepFilterConfig.from_checkpoint(values) == model.config


@pytest.mark.parametrize("length", (1, 257, 1037))
def test_initial_model_is_waveform_identity(length):
    model = FrequencyDeepFilter(_config()).eval()
    audio = torch.randn(2, length) * 0.1
    with torch.inference_mode():
        torch.testing.assert_close(model(audio), audio, atol=3e-7, rtol=3e-6)
        torch.testing.assert_close(model(torch.zeros_like(audio)), torch.zeros_like(audio), atol=0, rtol=0)


def test_zero_filter_head_exactly_matches_copied_nontrivial_bn_baseline():
    torch.manual_seed(77)
    config = _config()
    baseline = FrequencyUNet(config.convolution_config())
    with torch.no_grad():
        baseline.head.weight.normal_(std=0.25)
        baseline.head.bias.normal_(std=0.05)
        for _ in range(3):
            baseline(torch.randn(3, 1792) * 0.1)
    baseline.eval()
    enhanced = FrequencyDeepFilter(config).eval()
    missing = enhanced.load_state_dict(baseline.state_dict(), strict=False)
    assert missing.missing_keys == ["filter_head.weight", "filter_head.bias"]
    assert not missing.unexpected_keys
    audio = torch.randn(2, 2199) * 0.1
    with torch.inference_mode():
        assert (baseline(audio) - audio).abs().max() > 0.01
        torch.testing.assert_close(enhanced(audio), baseline(audio), atol=0, rtol=0)
    enhanced.train()
    (enhanced(audio) - audio * 0.7).square().mean().backward()
    assert torch.isfinite(enhanced.filter_head.weight.grad).all()
    assert enhanced.filter_head.weight.grad.abs().sum() > 0


def test_filter_taps_use_real_past_spectra_and_preserve_high_bins():
    model = FrequencyDeepFilter(_config()).double()
    spectra = torch.randn(2, 13, 257, dtype=torch.complex128)
    predictions = torch.zeros(2, model.output_size, 13, dtype=torch.float64)
    # Select lag1's real coefficient and lag4's imaginary coefficient. Explicit
    # shifts detect reversed tap order or accidental future-frame access.
    bins, order = model.config.filter_bins, model.config.filter_order
    predictions[:, 514 + bins:514 + 2 * bins] = 0.4
    predictions[:, 514 + (order + 4) * bins:514 + (order + 5) * bins] = -0.2
    expected = spectra.clone()
    expected[:, 1:, :bins] += 0.2 * spectra[:, :-1, :bins]
    expected[:, 4:, :bins] += -0.1j * spectra[:, :-4, :bins]
    actual, state = model.apply_filter(spectra, predictions)
    torch.testing.assert_close(actual, expected, atol=3e-16, rtol=3e-15)
    torch.testing.assert_close(actual[..., bins:], spectra[..., bins:], atol=0, rtol=0)
    torch.testing.assert_close(state, spectra[:, -4:, :bins], atol=0, rtol=0)
    first, state = model.apply_filter(spectra[:, :3], predictions[:, :, :3])
    second, state = model.apply_filter(spectra[:, 3:8], predictions[:, :, 3:8], state)
    third, state = model.apply_filter(spectra[:, 8:], predictions[:, :, 8:], state)
    torch.testing.assert_close(torch.cat((first, second, third), 1), actual, atol=0, rtol=0)
    changed = spectra.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:])
    torch.testing.assert_close(model.apply_mask(changed, predictions)[:, :7], actual[:, :7], atol=0, rtol=0)


def test_nonzero_filter_waveform_streaming_reset_and_bfloat16_gradients():
    torch.manual_seed(223)
    model = FrequencyDeepFilter(_config()).eval()
    with torch.no_grad():
        model.head.weight.normal_(std=0.2)
        model.filter_head.weight.normal_(std=0.15)
        model.filter_head.bias.normal_(std=0.1)
    audio = torch.randn(2, 2311) * 0.1
    with torch.inference_mode():
        expected = model(audio)
        assert (expected - audio).abs().max() > 0.01
        state = model.init_stream_state(batch_size=2)
        padded = F.pad(audio, (0, (-audio.shape[-1]) % 256 + 256))
        outputs = []
        for chunk in padded.split(256, -1):
            output, state = model.stream_step(chunk, state)
            outputs.append(output)
        actual = torch.cat(outputs, -1)[:, 256:256 + audio.shape[-1]]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        first, reset_state = model.stream_step(padded[:, :256], model.init_stream_state(batch_size=2))
        torch.testing.assert_close(first, outputs[0], atol=0, rtol=0)
        assert reset_state.spectral_history.shape == (2, 4, 65)
        feature_input = torch.randn(2, 3, 15, 257) * 0.1
        changed = feature_input.clone()
        changed[:, :, 8:] = torch.randn_like(changed[:, :, 8:])
        torch.testing.assert_close(model.forward_features(changed)[:, :, :8],
                                   model.forward_features(feature_input)[:, :, :8], atol=0, rtol=0)
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = (model(audio) - audio * 0.7).square().mean()
    loss.backward()
    for parameter in (model.stem.weight, model.filter_head.weight, model.head.weight):
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_single_tap_filter_requires_no_history_and_invalid_configs_fail():
    model = FrequencyDeepFilter(_config(filter_order=1, filter_bins=17))
    spectra = torch.randn(1, 2, 257, dtype=torch.complex64)
    _, history = model.apply_filter(spectra, torch.zeros(1, model.output_size, 2))
    assert history.shape == (1, 0, 17) and history.numel() == 0
    with pytest.raises(ValueError, match="positive integer"):
        FrequencyDeepFilterConfig(filter_order=0)
    with pytest.raises(ValueError, match="filter_bins"):
        FrequencyDeepFilterConfig(filter_bins=258)
    with pytest.raises(ValueError, match="filter_scale"):
        FrequencyDeepFilterConfig(filter_scale=float("nan"))
