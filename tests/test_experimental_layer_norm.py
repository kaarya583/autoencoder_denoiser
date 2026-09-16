"""Framewise integer LayerNorm feasibility: exact arithmetic, not speech quality."""
import math

import numpy as np
import pytest
import torch

from esp32_denoiser.experimental_layer_norm import (
    FakeQuantLayerNorm, IntegerLayerNormParameters, INT64_MAX, _round_divide,
    _torch_isqrt, integer_layer_norm, layer_norm_error_report,
    quantize_layer_norm, torch_integer_layer_norm,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _codes():
    rng = np.random.default_rng(103)
    codes = rng.integers(-128, 128, (12, 33, 16), dtype=np.int8)
    codes[0] = 0
    codes[1] = -128
    codes[2] = 127
    codes[3].reshape(-1)[::2] = -128
    codes[3].reshape(-1)[1::2] = 127
    codes[4] = 0
    codes[4, 0, 0] = 1
    codes[5] = 127
    codes[5, 0, 0] = -128
    return codes


@pytest.mark.parametrize("input_exp,output_exp,gamma_exp", [(-4, -4, -6), (-16, -7, -6), (8, -2, -6), (-4, 8, -16)])
def test_exact_reference_torch_and_fakequant_forward(input_exp, output_exp, gamma_exp):
    rng = np.random.default_rng(30)
    gamma = rng.integers(-128, 128, (33, 16), dtype=np.int8)
    beta = rng.integers(-16, 17, (33, 16), dtype=np.int32)
    parameters = IntegerLayerNormParameters(gamma, beta, input_exp, gamma_exp, output_exp)
    codes = _codes()
    expected = integer_layer_norm(codes, parameters)
    actual = torch_integer_layer_norm(torch.from_numpy(codes), parameters)
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert actual.dtype == torch.int8
    fake = FakeQuantLayerNorm(input_exponent=input_exp, output_exponent=output_exp, gamma_exponent=gamma_exp)
    with torch.no_grad():
        fake.weight.copy_(torch.from_numpy(gamma.astype(np.float32)) * 2.0**gamma_exp)
        fake.bias.copy_(torch.from_numpy(beta.astype(np.float32)) * 2.0**output_exp)
        result = fake(torch.from_numpy(codes.astype(np.float32)) * 2.0**input_exp)
    np.testing.assert_array_equal((result.numpy() / 2.0**output_exp).astype(np.int8), expected)
    assert parameters.bounds()["max_rounded_numerator"] <= INT64_MAX


def test_integer_sqrt_and_ties_near_exact_boundaries():
    roots = (1, 2, 37, 2**20, 2**26 + 1, 2**31 - 2)
    values = [root**2 + delta for root in roots for delta in (-1, 0, 1)]
    actual = _torch_isqrt(torch.tensor(values, dtype=torch.int64)).tolist()
    assert actual == [math.isqrt(value) for value in values]
    for denominator in (2, 3, 8, 527, 528):
        for numerator in range(-2 * denominator, 2 * denominator + 1):
            expected = math.floor(abs(numerator) / denominator + .5) * (-1 if numerator < 0 else 1)
            assert _round_divide(numerator, denominator) == expected


def test_integer_paths_use_preencoded_epsilon_and_no_float_sqrt(monkeypatch):
    parameters = quantize_layer_norm(np.ones((33, 16)), np.zeros((33, 16)))
    codes = _codes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Floating conversion/normalization reached the integer path")

    monkeypatch.setattr("esp32_denoiser.experimental_layer_norm.Fraction", forbidden)
    monkeypatch.setattr(np, "sqrt", forbidden)
    monkeypatch.setattr(torch, "sqrt", forbidden)
    np.testing.assert_array_equal(integer_layer_norm(codes, parameters),
                                  torch_integer_layer_norm(torch.from_numpy(codes), parameters).numpy())


def test_constant_frames_singleton_and_beta_saturation():
    parameters = quantize_layer_norm(np.ones((33, 16)), np.linspace(-10, 10, 528).reshape(33, 16))
    codes = np.stack([np.full((33, 16), value, dtype=np.int8) for value in (-128, 0, 127)])
    actual, diagnostics = integer_layer_norm(codes, parameters, return_diagnostics=True)
    expected = np.clip(parameters.beta, -128, 127).astype(np.int8)
    for frame in actual:
        np.testing.assert_array_equal(frame, expected)
    assert diagnostics["zero_variance_frames"] == 3
    assert diagnostics["saturated_outputs"] > 0
    singleton = IntegerLayerNormParameters(np.array([-128], np.int8), np.array([-7], np.int32))
    np.testing.assert_array_equal(integer_layer_norm(np.array([[-128], [127]], np.int8), singleton), [[-7], [-7]])


def test_epsilon_is_encoded_on_the_input_variance_grid():
    # A one-code input difference at a very fine input scale is epsilon dominated.
    codes = np.zeros((33, 16), dtype=np.int8)
    codes.reshape(-1)[::2] = 1
    small = quantize_layer_norm(np.ones((33, 16)), np.zeros((33, 16)),
                                input_exponent=-16, output_exponent=-7, epsilon=1e-8)
    large = quantize_layer_norm(np.ones((33, 16)), np.zeros((33, 16)),
                                input_exponent=-16, output_exponent=-7, epsilon=1e-4)
    assert small.epsilon_code > 0
    assert small.effective_epsilon == pytest.approx(1e-8, rel=1e-12)
    assert np.max(np.abs(integer_layer_norm(codes, small))) > np.max(np.abs(integer_layer_norm(codes, large)))
    assert layer_norm_error_report(codes, small)["maximum_absolute_error"] <= 2**-8 + 1e-6
    coarse = quantize_layer_norm(np.ones((33, 16)), np.zeros((33, 16)), input_exponent=8)
    assert coarse.epsilon_code == 1 and coarse.effective_epsilon > 0


def test_shape_wise_error_report_and_memory_accounting():
    parameters = quantize_layer_norm(np.ones((33, 16)), np.zeros((33, 16)), output_exponent=-2)
    report = layer_norm_error_report(_codes(), parameters)
    assert report["normalized_shape"] == [33, 16] and report["frames"] == 12
    assert len(report["per_frame_mean_absolute_error"]) == 12
    assert report["maximum_absolute_error"] <= .126
    assert report["integer_diagnostics"]["saturated_outputs"] == 0
    memory = report["memory"]
    assert memory["gamma_bytes_int8"] == 528
    assert memory["beta_bytes_int32"] == 2112
    assert memory["parameter_array_bytes"] == 2640
    assert memory["four_gtcrn_affine_array_bytes"] == 10560
    assert memory["persistent_neural_state_bytes"] == 0
    # The smaller output grid clips rare normalized outliers; expose that loss.
    clipped = layer_norm_error_report(_codes(), quantize_layer_norm(np.ones((33, 16)), np.zeros((33, 16))))
    assert clipped["integer_diagnostics"]["saturated_outputs"] >= 2
    assert clipped["maximum_absolute_error"] > 10


def test_snapshots_are_immutable_and_overflow_is_rejected():
    gamma = np.ones((33, 16), np.int8)
    beta = np.zeros((33, 16), np.int32)
    parameters = IntegerLayerNormParameters(gamma, beta)
    gamma[:] = 0
    assert np.all(parameters.gamma == 1) and not parameters.gamma.flags.writeable
    with pytest.raises(ValueError, match="INT32"):
        quantize_layer_norm(np.ones(2), np.ones(2) * 1e20)
    with pytest.raises(ValueError, match="square-root range"):
        IntegerLayerNormParameters(np.ones(2, np.int8), np.zeros(2, np.int32), epsilon=1e20)
    with pytest.raises(ValueError, match="requantization"):
        IntegerLayerNormParameters(np.ones((33, 16), np.int8),
                                   np.full((33, 16), 2**31 - 1, np.int32),
                                   input_exponent=-16, gamma_exponent=-16, output_exponent=8)
    with pytest.raises(ValueError, match="INT8"):
        integer_layer_norm(np.zeros((33, 16), np.int16), parameters)
    with pytest.raises(ValueError, match="shape"):
        integer_layer_norm(np.zeros((16, 33), np.int8), parameters)
    with pytest.raises(ValueError, match="finite"):
        quantize_layer_norm(np.array([np.nan]), np.zeros(1))


def test_fakequant_gradients_and_updated_snapshot_remain_exact():
    source = torch.nn.LayerNorm((33, 16), eps=1e-8).double()
    fake = FakeQuantLayerNorm.from_float(source, input_exponent=-5, output_exponent=-4)
    assert fake.weight.dtype == source.weight.dtype
    inputs = (torch.randn(2, 33, 16) * .3).requires_grad_()
    weighting = torch.linspace(-.8, .8, 528).reshape(1, 33, 16)
    before = fake.snapshot()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = fake(inputs)
        loss = (output * weighting).sum() / 2
    loss.backward()
    for gradient in (inputs.grad, fake.weight.grad, fake.bias.grad):
        assert gradient is not None and bool(torch.isfinite(gradient).all())
        assert float(gradient.abs().max()) > 0
    torch.optim.SGD(fake.parameters(), lr=.25).step()
    after = fake.snapshot()
    assert np.any(before.beta != after.beta)
    scaled = inputs.detach().numpy() / 2**-5
    codes = np.clip(np.sign(scaled) * np.floor(np.abs(scaled) + .5), -128, 127).astype(np.int8)
    with torch.no_grad():
        actual = fake(inputs).numpy() / 2**-4
    np.testing.assert_array_equal(actual.astype(np.int8), integer_layer_norm(codes, after))


@pytest.mark.parametrize("input_exp,output_exp", [(-4, -2), (-16, -7), (8, -2)])
def test_independent_pytorch_layer_norm_semantics(input_exp, output_exp):
    # This reference does not call the prototype's variance, epsilon encoding,
    # square root, affine arithmetic, or fake-quant exact-forward wrapper.
    rng = np.random.default_rng(104)
    codes = rng.integers(-128, 128, (2, 12, 33, 16), dtype=np.int8)
    frames = codes.reshape(-1, 33, 16)
    frames[:4] = 0
    frames[0, 0, 0] = 1  # A nonintegral mean, dominated by eps on the fine grid.
    frames[1, 0, 0] = 127
    frames[2].reshape(-1)[::3] = 1
    frames[3] = 127
    frames[3, 0, 0] = -128
    gamma = rng.integers(-64, 65, (33, 16), dtype=np.int8)
    beta = rng.integers(-4, 5, (33, 16), dtype=np.int32)
    parameters = IntegerLayerNormParameters(gamma, beta, input_exp, -6, output_exp)
    reference = torch.nn.functional.layer_norm(
        torch.from_numpy(codes.astype(np.float64)) * 2.0**input_exp,
        (33, 16), torch.from_numpy(gamma.astype(np.float64)) * 2.0**-6,
        torch.from_numpy(beta.astype(np.float64)) * 2.0**output_exp,
        eps=1e-8,
    ).numpy()
    reference_codes = np.clip(
        np.sign(reference) * np.floor(np.abs(reference) / 2.0**output_exp + .5), -128, 127
    ).astype(np.int8)
    actual = integer_layer_norm(codes, parameters)
    # Exact for this finite diagnostic corpus; the floor-sqrt approximation is
    # not claimed to preserve every possible float rounding threshold.
    np.testing.assert_array_equal(actual, reference_codes)
    for index in range(12):
        np.testing.assert_array_equal(actual[:, index], integer_layer_norm(codes[:, index], parameters))


@pytest.mark.parametrize("size", [527, 528, 1024])
def test_reduction_bound_is_attained_by_signed_rail_inputs(size):
    codes = np.full(size, -128, dtype=np.int8)
    codes[:size // 2] = 127
    parameters = IntegerLayerNormParameters(np.full(size, -128, np.int8), np.zeros(size, np.int32))
    actual, diagnostics = integer_layer_norm(codes, parameters, return_diagnostics=True)
    np.testing.assert_array_equal(actual, torch_integer_layer_norm(torch.from_numpy(codes), parameters).numpy())
    expected_variance = (size // 2) * (size - size // 2) * 255**2
    assert parameters.bounds()["max_variance_numerator"] == expected_variance
    assert diagnostics["max_squared_denominator"] == (expected_variance << 24) + parameters.epsilon_code
    assert diagnostics["max_abs_affine_numerator"] <= parameters.bounds()["max_abs_affine_numerator"]
    if size >= 528:
        assert expected_variance > 2**32 - 1  # Even unsigned INT32 is insufficient.


def test_affine_snapshot_accepts_bfloat16_and_subnormal_finite_weights():
    source = torch.nn.LayerNorm((33, 16), eps=1e-8).to(torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.linspace(-1, 1, 528).reshape(33, 16))
        source.bias.copy_(torch.linspace(-.5, .5, 528).reshape(33, 16))
    fake = FakeQuantLayerNorm.from_float(source)
    assert fake.weight.dtype == torch.bfloat16
    expected = quantize_layer_norm(source.weight.double(), source.bias.double())
    np.testing.assert_array_equal(fake.snapshot().gamma, expected.gamma)
    np.testing.assert_array_equal(fake.snapshot().beta, expected.beta)
    codes = _codes()[:2]
    with torch.no_grad():
        result = fake(torch.from_numpy(codes).to(torch.bfloat16) * 2**-4)
    np.testing.assert_array_equal((result.float().numpy() / 2**-4).astype(np.int8),
                                  integer_layer_norm(codes, expected))
    tiny = quantize_layer_norm(np.array([np.nextafter(0., 1.)]), np.zeros(1))
    assert tiny.gamma_exponent == -16 and tiny.gamma.tolist() == [0]
