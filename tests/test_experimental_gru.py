"""Recurrent contract tests; no GTCRN or audio-quality claim."""
import math
import numpy as np
import pytest
import torch
from torch import nn

from esp32_denoiser.experimental_gru import (
    FakeQuantGRUCell, GRUQuantizationConfig, IntegerGRUCell,
)


@pytest.mark.parametrize("hidden,bidirectional", [(4, True), (8, False), (16, False)])
def test_float_control_matches_pytorch_gru_each_direction(hidden, bidirectional):
    torch.manual_seed(123)
    source = nn.GRU(8, hidden, batch_first=True, bidirectional=bidirectional).double()
    x = torch.randn(2, 17, 8, dtype=torch.float64)
    initial = torch.randn(2 if bidirectional else 1, 2, hidden, dtype=torch.float64)
    expected, final = source(x, initial)
    for index, direction in enumerate(("forward", "reverse") if bidirectional else ("forward",)):
        cell = FakeQuantGRUCell(source, direction=direction)
        state, outputs = initial[index], []
        ordered = x.flip(1) if direction == "reverse" else x
        for frame in ordered.unbind(1):
            state = cell.forward_float(frame, state)
            outputs.append(state)
        actual = torch.stack(outputs, 1)
        if direction == "reverse":
            actual = actual.flip(1)
        torch.testing.assert_close(actual, expected[..., index * hidden:(index + 1) * hidden], atol=2e-15, rtol=2e-14)
        torch.testing.assert_close(state, final[index], atol=2e-15, rtol=2e-14)


@pytest.mark.parametrize("inputs,hidden", [(8, 4), (8, 8), (8, 16), (32, 16)])
def test_integer_sequence_matches_fakequant_and_chunk_states_exactly(inputs, hidden):
    torch.manual_seed(192)
    rng = np.random.default_rng(82)
    source = nn.GRUCell(inputs, hidden)
    config = GRUQuantizationConfig(input_exponent=-4)
    qat = FakeQuantGRUCell(source, config)
    integer = IntegerGRUCell.from_torch(qat)
    assert integer.config == config
    x = rng.normal(0, .6, (2, 73, inputs)).astype(np.float32)
    x[:, 15:31] = 0
    x[:, 36] = 100  # Explicitly exercise the saturating input boundary.
    h0 = rng.integers(-100, 101, size=(2, hidden), dtype=np.int8)
    codes = integer.quantize_input(x)
    expected, last = integer.process(codes, h0)
    with torch.no_grad():
        actual, actual_last = qat.process(torch.from_numpy(x), torch.from_numpy(h0.astype(np.float32) / 128))
    np.testing.assert_array_equal(actual.numpy() * 128, expected)
    np.testing.assert_array_equal(actual_last.numpy() * 128, last)
    outputs, state = [], h0.copy()
    for start, end in ((0, 1), (1, 31), (31, 37), (37, 73)):
        chunk, state = integer.process(codes[:, start:end], state)
        outputs.append(chunk)
    np.testing.assert_array_equal(np.concatenate(outputs, axis=1), expected)
    np.testing.assert_array_equal(state, last)
    assert integer.statistics["input_clipped"] == 2 * inputs
    # Caller state and parameter snapshots cannot be changed by stepping.
    assert h0.dtype == np.int8 and not integer.weight_ih.flags.writeable
    baseline, _ = integer.process(codes)
    restarted, _ = integer.process(codes, integer.initial_state(2))
    np.testing.assert_array_equal(baseline, restarted)


def test_reset_after_candidate_includes_recurrent_bias_inside_gate():
    source = nn.GRUCell(1, 1)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.zero_()
        source.weight_hh[2, 0] = 2
        source.bias_hh[2] = 1
    integer = IntegerGRUCell.from_torch(source)
    updated, trace = integer.step(np.zeros((1, 1), np.int8), np.array([[64]], np.int8), return_trace=True)
    # r=z=128/255, recurrent candidate projection=2.0, gated value rounds
    # to logit1.0. tanh(1)*128 rounds97; blending with old state64 rounds80.
    assert trace["reset"].item() == trace["update"].item() == 0
    assert trace["candidate_logit"].item() == 16
    assert trace["candidate"].item() == 97 and updated.item() == 80
    assert all(value.dtype == np.int8 for value in trace.values())
    control = FakeQuantGRUCell(source)
    torch.testing.assert_close(control.forward_float(torch.zeros(1, 1), torch.full((1, 1), .5)),
                               source(torch.zeros(1, 1), torch.full((1, 1), .5)))


def test_endpoint_probability_preserves_state_and_saturates_candidate_explicitly():
    source = nn.GRUCell(1, 1)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.zero_()
        source.bias_ih[1] = 20
        source.bias_ih[2] = 20
    keep = IntegerGRUCell.from_torch(source)
    original = np.array([[-128], [127], [1]], np.int8)
    result, trace = keep.step(np.zeros((3, 1), np.int8), original, return_trace=True)
    assert (trace["update"] == 127).all()  # signed127 means probability255/255.
    assert (trace["candidate"] == 127).all()
    np.testing.assert_array_equal(result, original)
    assert keep.statistics["logit_clipped"] == 6
    with torch.no_grad():
        source.bias_ih[1] = -20
    overwrite = IntegerGRUCell.from_torch(source)
    result, trace = overwrite.step(np.zeros((3, 1), np.int8), original, return_trace=True)
    assert (trace["update"] == -128).all()  # probability0.
    assert (result == 127).all()
    assert overwrite.statistics["state_saturated"] == 3


def test_long_stream_state_is_int8_bounded_and_statistics_are_reproducible():
    torch.manual_seed(66)
    source = nn.GRU(8, 16, batch_first=True)
    integer = IntegerGRUCell.from_torch(source)
    rng = np.random.default_rng(44)
    inputs = rng.integers(-128, 128, (1, 2048, 8), dtype=np.int8)
    inputs[:, 700:1500] = 0
    first, state = integer.process(inputs)
    recorded = integer.statistics.copy()
    assert first.dtype == state.dtype == np.int8
    assert np.isfinite(first.astype(np.float32) / 128).all()
    assert recorded["state_values"] == 2048 * 16 and recorded["frames"] == 2048
    assert recorded["logit_values"] == 2048 * 3 * 16
    assert 0 <= recorded["state_saturated"] <= recorded["state_values"]
    assert 0 < recorded["state_unchanged"] < recorded["state_values"]
    integer.reset_statistics()
    second, second_state = integer.process(inputs)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(state, second_state)
    assert integer.statistics == recorded
    assert integer.storage_stats()["state_bytes_per_stream"] == 16
    assert integer.storage_stats()["matrix_macs_per_step"] == 1152


def test_fakequant_gradients_update_weights_and_export_stays_exact_under_amp():
    torch.manual_seed(17)
    qat = FakeQuantGRUCell(nn.GRUCell(8, 8))
    inputs = (torch.randn(2, 9, 8) * .2).requires_grad_()
    before = qat.weight_ih.detach().clone()
    optimizer = torch.optim.SGD(qat.parameters(), lr=.03)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, _ = qat.process(inputs)
        loss = (output - .25).square().mean()
    loss.backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0
    for parameter in qat.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(qat.weight_ih, before)
    integer = IntegerGRUCell.from_torch(qat)
    with torch.no_grad():
        expected, _ = qat.process(inputs.detach())
    actual, _ = integer.process(integer.quantize_input(inputs.detach().numpy()))
    np.testing.assert_array_equal(expected.numpy() * 128, actual)


def test_parameter_snapshot_rejects_nonfinite_overflow_and_unsupported_shapes():
    with pytest.raises(ValueError, match="explicit input/recurrent biases"):
        IntegerGRUCell.from_torch(nn.GRUCell(8, 4, bias=False))
    with pytest.raises(ValueError, match="one GRU layer"):
        IntegerGRUCell.from_torch(nn.GRU(8, 4, num_layers=2))
    with pytest.raises(ValueError, match="no reverse"):
        IntegerGRUCell.from_torch(nn.GRU(8, 4), direction="reverse")
    source = nn.GRUCell(1, 1)
    with torch.no_grad():
        source.bias_ih[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        IntegerGRUCell.from_torch(source)
    with torch.no_grad():
        source.bias_ih[0] = 1e20
    with pytest.raises(OverflowError, match="bias"):
        IntegerGRUCell.from_torch(source)
    with pytest.raises(ValueError, match="Q0.7"):
        GRUQuantizationConfig(state_exponent=-15)
    source = nn.GRUCell(512, 1)
    with torch.no_grad():
        source.weight_ih.fill_(2000)
        source.bias_ih.zero_()
        source.bias_hh.zero_()
    with pytest.raises(OverflowError, match="accumulator"):
        IntegerGRUCell.from_torch(source, GRUQuantizationConfig(input_exponent=0))


def test_integer_boundary_rejects_floating_state_and_invalid_inputs():
    integer = IntegerGRUCell.from_torch(nn.GRUCell(8, 4))
    with pytest.raises(ValueError, match="input must be INT8"):
        integer.step(np.zeros((1, 8), np.float32))
    with pytest.raises(ValueError, match="state must be INT8"):
        integer.step(np.zeros((1, 8), np.int8), np.zeros((1, 4), np.int16))
    with pytest.raises(ValueError, match="finite floating"):
        integer.quantize_input(np.array([np.nan], np.float32))
    with pytest.raises(ValueError, match="sequence must be INT8"):
        integer.process(np.zeros((1, 0, 8), np.int8))


def test_cloning_qat_cell_preserves_nondefault_grids_and_modified_tables_cannot_export():
    config = GRUQuantizationConfig(input_exponent=-8, logit_exponent=-3)
    original = FakeQuantGRUCell(nn.GRUCell(8, 4), config)
    cloned = FakeQuantGRUCell(original)
    assert cloned.config == config
    inputs = torch.randn(2, 8) * .2
    with torch.no_grad():
        torch.testing.assert_close(cloned(inputs), original(inputs), rtol=0, atol=0)
        original.sigmoid_lut[128] = 127  # Zero logit no longer means quantized 1/2.
    with pytest.raises(ValueError, match="lookup tables differ"):
        IntegerGRUCell.from_torch(original)


@pytest.mark.parametrize("logit_exponent,probability_range", [
    (-6, [30, 224]), (-5, [5, 250]), (-4, [0, 255]), (-3, [0, 255]), (-2, [0, 255]),
])
def test_luts_match_independent_scalar_math_and_expose_unreachable_gate_endpoints(logit_exponent, probability_range):
    integer = IntegerGRUCell.from_torch(nn.GRUCell(1, 1), GRUQuantizationConfig(logit_exponent=logit_exponent))
    # Scalar libm evaluation and Python integer rounding are independent of
    # the production NumPy table generator shared by QAT and integer paths.
    for code in range(-128, 128):
        logit = code * 2.0**logit_exponent
        probability = math.floor(255 / (1 + math.exp(-logit)) + .5)
        candidate = math.tanh(logit) * 128
        rounded = math.floor(abs(candidate) + .5) * (-1 if candidate < 0 else 1)
        assert int(integer.sigmoid_lut[code + 128]) + 128 == probability
        assert int(integer.tanh_lut[code + 128]) == max(-128, min(127, rounded))
    assert integer.storage_stats()["sigmoid_probability_code_range"] == probability_range
    assert integer.storage_stats()["sigmoid_reaches_both_endpoints"] == (logit_exponent >= -4)


@pytest.mark.parametrize("sign", [-1, 1])
def test_reset_product_near_int32_limit_uses_wide_intermediate(sign, monkeypatch):
    source = nn.GRUCell(1, 1)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.zero_()
        source.bias_ih[0], source.bias_ih[1] = 20, -20  # Fully reset-open and update-overwrite.
        source.weight_hh[2, 0] = sign * 16
        source.bias_hh[2] = sign * 520000
    integer = IntegerGRUCell.from_torch(source)
    captured, original_logit = [], integer._logit

    def observe(accumulator):
        captured.append(accumulator.copy())
        return original_logit(accumulator)

    monkeypatch.setattr(integer, "_logit", observe)
    previous = np.array([[127]], dtype=np.int8)
    updated = integer.step(np.zeros((1, 1), dtype=np.int8), previous)
    # Both terms are exact on the source/accumulator grids. Multiplication by
    # probability255 exceeds INT32 before division restores this accumulator.
    expected_candidate = sign * (520000 * 4096 + 16 * 127 * 32)
    assert abs(expected_candidate * 255) > 2**31
    assert captured[2].dtype == np.int32 and captured[2].item() == expected_candidate
    assert updated.item() == (127 if sign > 0 else -128)
    np.testing.assert_array_equal(previous, [[127]])


def test_export_rejects_sum_of_separately_valid_aligned_affines():
    source = nn.GRUCell(1, 1)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.zero_()
        source.weight_ih[0, 0] = source.weight_hh[0, 0] = 16
        source.bias_ih[0] = source.bias_hh[0] = 300000
    # Each aligned path is ~1.23e9, but their reset-logit sum exceeds INT32.
    with pytest.raises(OverflowError, match="accumulator exceeds INT32"):
        IntegerGRUCell.from_torch(source, GRUQuantizationConfig(input_exponent=0))
