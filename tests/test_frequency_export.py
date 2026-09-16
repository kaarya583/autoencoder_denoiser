"""Actual frequency-model bytes and persistent integer reference semantics."""

import struct

import numpy as np
import pytest
import torch

from esp32_denoiser.frequency_export import (
    DSP_HEADER, HEADER, LAYER, IntegerFrequencyDenoiser, export_frequency_model,
)
from esp32_denoiser.frequency_model import FrequencyUNet, FrequencyUNetConfig
from esp32_denoiser.frequency_quantization import configure_frequency_qat
from esp32_denoiser.quantization import round_away


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _nontrivial():
    torch.manual_seed(73)
    model = FrequencyUNet(FrequencyUNetConfig(encoder_channels=(4, 6, 8), global_width=8,
                                              local_dilations=(1, 2), global_dilations=(1, 2, 4)))
    with torch.no_grad():
        model.head.weight.normal_(std=0.3)
        model.head.bias.normal_(std=0.08)
        model.global_out.weight.mul_(4)
        for block in (*model.local_blocks, *model.global_blocks):
            block.pointwise.weight.mul_(4)
    return configure_frequency_qat(model, hidden_exponent=-6).eval()


def test_actual_default_payload_and_all_weight_guards_fit_budget(tmp_path):
    model = configure_frequency_qat(FrequencyUNet())
    binary = tmp_path / "frequency.bin"
    report = export_frequency_model(model, binary)
    data = binary.read_bytes()
    assert len(data) == report["model_bytes"] <= 99_000
    assert report["weight_bytes"] == 81_296
    assert report["bias_bytes"] == 7_496
    assert report["neural_history_bytes"] == 18_816
    assert report["macs_per_second"] == 27_561_000
    assert report["parameter_payload_reduction"] > 119
    oracle = IntegerFrequencyDenoiser(data)
    for index, layer in enumerate(oracle.layers):
        fields = LAYER.unpack_from(data, HEADER.size + LAYER.size * index)
        weights_offset, bias_offset = fields[-3:-1]
        assert weights_offset % 16 == 0
        assert weights_offset + layer.weights.size + 16 <= bias_offset
        assert data[weights_offset + layer.weights.size:weights_offset + layer.weights.size + 16] == bytes(16)
    with pytest.raises(ValueError, match="exceeding"):
        export_frequency_model(model, tmp_path / "oversize.bin", max_bytes=100)
    assert not (tmp_path / "oversize.bin").exists()


def test_integer_reference_matches_nontrivial_qat_and_chunked_history(tmp_path):
    model = _nontrivial()
    binary = tmp_path / "frequency.bin"
    export_frequency_model(model, binary)
    oracle = IntegerFrequencyDenoiser(binary)
    rng = np.random.default_rng(915)
    features = rng.integers(-127, 128, (39, 3, 257), dtype=np.int8)
    with torch.inference_mode():
        tensor = torch.from_numpy(features.transpose(1, 0, 2).copy())[None].float() / 128
        expected = round_away(model.forward_features(tensor)[0].T * 128).numpy()
    actual = oracle.process(features)
    assert np.count_nonzero(actual) > actual.size // 3
    np.testing.assert_allclose(actual, expected, atol=1, rtol=0)
    assert sum(history.nbytes for history in oracle.histories.values()) == model.model_stats()["neural_state_bytes_int8"]
    oracle.reset()
    chunked = np.concatenate([oracle.process(features[:7]), oracle.process(features[7:22]), oracle.process(features[22:])])
    np.testing.assert_array_equal(chunked, actual)
    oracle.reset()
    np.testing.assert_array_equal(oracle.step(features[0]), actual[0])
    assert oracle.process(np.empty((0, 3, 257), dtype=np.int8)).shape == (0, 514)


def test_serialization_preserves_dsp_constants_and_rejects_malformed_blobs(tmp_path):
    model = _nontrivial()
    with torch.no_grad():
        model.window.mul_(0.9)
    binary = tmp_path / "frequency.bin"
    export_frequency_model(model, binary)
    data = binary.read_bytes()
    oracle = IntegerFrequencyDenoiser(data)
    np.testing.assert_array_equal(oracle.window, model.window.detach().numpy())
    malformed = [data[:15], b"BADMAGIC" + data[8:], data[:-1]]
    for header_index, value in ((1, 999), (6, 0), (8, 1), (13, len(data) + 1), (15, 0)):
        changed = bytearray(data)
        values = list(HEADER.unpack_from(changed))
        values[header_index] = value
        HEADER.pack_into(changed, 0, *values)
        malformed.append(bytes(changed))
    for layer_index, value in ((1, 2), (11, 0), (12, len(data)), (13, len(data))):
        changed = bytearray(data)
        values = list(LAYER.unpack_from(changed, HEADER.size))
        values[layer_index] = value
        LAYER.pack_into(changed, HEADER.size, *values)
        malformed.append(bytes(changed))
    changed = bytearray(data)
    dsp_offset = HEADER.unpack_from(changed)[14]
    struct.pack_into("<f", changed, dsp_offset + DSP_HEADER.size, float("nan"))
    malformed.append(bytes(changed))
    for blob in malformed:
        with pytest.raises(ValueError):
            IntegerFrequencyDenoiser(blob)


def test_export_requires_every_quantization_boundary(tmp_path):
    model = _nontrivial()
    model.up1.skip_quant.enabled = False
    with pytest.raises(ValueError, match="boundary"):
        export_frequency_model(model, tmp_path / "bad.bin")
