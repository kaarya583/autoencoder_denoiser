"""Shared temporal recurrence: independent band states and causal audio parity."""

from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from esp32_denoiser.frequency_gru import FrequencyGRU, FrequencyGRUConfig
from esp32_denoiser.frequency_model import FrequencyUNet


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _small(**kwargs):
    return FrequencyGRU(FrequencyGRUConfig(encoder_channels=(4, 6, 8), recurrent_hidden=4,
                                          global_width=8, global_dilations=(1, 2), **kwargs))


def _nontrivial():
    torch.manual_seed(208)
    model = _small().eval()
    with torch.no_grad():
        model.head.weight.normal_(std=0.3)
        model.head.bias.normal_(std=0.05)
        model.local_projection.weight.mul_(3)
    return model


def test_default_budget_and_explicit_float_only_status():
    model = FrequencyGRU()
    stats = model.model_stats()
    assert stats["learned_parameters"] == stats["deployed_parameters"] == 81_986
    assert stats["macs_per_second"] == 25_251_000
    assert stats["neural_state_bytes_int8"] == 4_560
    assert stats["neural_state_bytes_float32"] == 18_240
    assert stats["recurrent_state_bytes_int8"] == 528
    assert "float-only" in stats["precision"]
    assert "forecast" in stats["neural_state_bytes_int8_status"]
    assert not hasattr(model, "local_blocks")
    state = model.init_stream_state()
    actual = state.recurrent.numel() + sum(t.numel() for t in state.global_temporal)
    assert actual == stats["neural_state_bytes_int8"]
    checkpoint_config = asdict(model.config)
    checkpoint_config["global_dilations"] = list(checkpoint_config["global_dilations"])
    checkpoint_config["encoder_channels"] = list(checkpoint_config["encoder_channels"])
    assert FrequencyGRUConfig.from_checkpoint(checkpoint_config) == model.config
    with pytest.raises(ValueError, match="positive integer"):
        FrequencyGRUConfig(recurrent_hidden=0)


def test_shared_modules_keep_matched_seed_weights_and_optional_encoder_bn():
    torch.manual_seed(14)
    baseline = FrequencyUNet()
    torch.manual_seed(14)
    recurrent = FrequencyGRU()
    for name, parameter in baseline.named_parameters():
        if name.startswith("local_blocks."):
            continue
        torch.testing.assert_close(parameter, recurrent.get_parameter(name), atol=0, rtol=0)
    normalized = _small(encoder_batch_norm=True)
    assert normalized.model_stats()["training_normalization_parameters"] == 56
    restored = FrequencyGRU(FrequencyGRUConfig.from_checkpoint(asdict(normalized.config)))
    restored.load_state_dict(normalized.state_dict())


@pytest.mark.parametrize("length", [1, 257, 1037])
def test_zero_head_is_waveform_identity(length):
    model = _small().eval()
    audio = torch.randn(2, length) * 0.1
    with torch.inference_mode():
        torch.testing.assert_close(model(audio), audio, atol=3e-7, rtol=3e-6)
        assert torch.isfinite(model(torch.zeros_like(audio))).all()


def test_nontrivial_waveform_streaming_causality_and_gradients():
    model = _nontrivial()
    audio = torch.randn(2, 2819) * 0.1
    with torch.inference_mode():
        expected = model(audio)
        assert (expected - audio).abs().max() > 0.01
        state = model.init_stream_state(batch_size=2)
        outputs = []
        chunks = F.pad(audio, (0, (-audio.shape[-1]) % 256 + 256)).split(256, dim=-1)
        for chunk in chunks:
            output, state = model.stream_step(chunk, state)
            outputs.append(output)
        actual = torch.cat(outputs, -1)[:, 256:256 + audio.shape[-1]]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        features = torch.randn(2, 3, 17, 257) * 0.1
        changed = features.clone()
        changed[:, :, 9:] = torch.randn_like(changed[:, :, 9:])
        torch.testing.assert_close(model.forward_features(changed)[:, :, :9],
                                   model.forward_features(features)[:, :, :9], atol=0, rtol=0)
        first, _ = model.stream_step(chunks[0], model.init_stream_state(batch_size=2))
        torch.testing.assert_close(first, outputs[0], atol=0, rtol=0)
    model.train()
    loss = (model(audio) - audio * 0.7).square().mean()
    loss.backward()
    for parameter in (model.stem.weight, model.local_gru.weight_ih_l0,
                      model.local_gru.weight_hh_l0, model.local_projection.weight):
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_recurrence_chunking_preserves_independent_frequency_states():
    model = _nontrivial()
    features = torch.randn(2, 8, 23, 33) * 0.3
    with torch.inference_mode():
        expected, final_state = model._recur(features)
        first, state = model._recur(features[:, :, :7])
        second, state = model._recur(features[:, :, 7:19], state)
        third, state = model._recur(features[:, :, 19:], state)
        torch.testing.assert_close(torch.cat((first, second, third), dim=2), expected, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(state, final_state, atol=1e-7, rtol=1e-6)
        changed = features.clone()
        changed[0, :, :, 5] += 5
        actual, _ = model._recur(changed)
        keep = torch.ones(33, dtype=torch.bool)
        keep[5] = False
        torch.testing.assert_close(actual[0, :, :, keep], expected[0, :, :, keep], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


def test_gru_matches_explicit_pytorch_reset_after_equations():
    model = _nontrivial().double()
    cell = model.local_gru
    sequence = torch.randn(3, 11, 8, dtype=torch.double)
    initial = torch.randn(1, 3, 4, dtype=torch.double) * 0.2
    with torch.inference_mode():
        expected, expected_final = cell(sequence, initial)
        state = initial[0]
        outputs = []
        for current in sequence.unbind(1):
            ir, iz, inn = F.linear(current, cell.weight_ih_l0, cell.bias_ih_l0).chunk(3, dim=-1)
            hr, hz, hn = F.linear(state, cell.weight_hh_l0, cell.bias_hh_l0).chunk(3, dim=-1)
            reset = torch.sigmoid(ir + hr)
            update = torch.sigmoid(iz + hz)
            # PyTorch applies the reset gate after the recurrent affine map.
            candidate = torch.tanh(inn + reset * hn)
            state = (1 - update) * candidate + update * state
            outputs.append(state)
        torch.testing.assert_close(torch.stack(outputs, 1), expected, atol=2e-15, rtol=2e-14)
        torch.testing.assert_close(state, expected_final[0], atol=2e-15, rtol=2e-14)
