"""Independent operator semantics for the future GTCRN integer graph."""
import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from esp32_denoiser.experimental_gtcrn_ops import (
    ActivationGrid, ActivationObserver, IntegerAffine, IntegerPReLU, IntegerStreamConv,
    attention_energy, attention_product, folded_parameters, requantize_activation, residual_add, shuffle_pair,
    subband_features,
)
from esp32_denoiser.vendor.gtcrn.convolution import StreamConv2d, StreamConvTranspose2d
from esp32_denoiser.vendor.gtcrn.streaming import ConvBlock, DPGRNN, SFE, StreamGTCRN, StreamGTConvBlock, StreamTRA


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _source(kind):
    if kind == "linear":
        return nn.Linear(8, 6), (2, 8)
    if kind == "grouped":
        return nn.Conv2d(4, 6, (2, 3), stride=(1, 2), padding=(1, 2), dilation=(2, 1), groups=2), (2, 4, 6, 9)
    if kind == "depthwise":
        return nn.Conv2d(3, 3, (3, 3), padding=(0, 1), dilation=(2, 1), groups=3), (2, 3, 6, 7)
    if kind == "transpose":
        return nn.ConvTranspose2d(4, 6, (2, 3), stride=2, padding=1, output_padding=1, dilation=(1, 2), groups=2), (2, 4, 3, 5)
    if kind == "transpose_point":
        return nn.ConvTranspose2d(4, 6, 1, groups=2), (2, 4, 3, 5)
    raise AssertionError(kind)


def _torch_layout(weight, layer):
    value = torch.from_numpy(np.array(weight, dtype=np.float64))
    if isinstance(layer, nn.ConvTranspose2d):
        value = value.reshape(layer.groups, layer.out_channels // layer.groups,
                              layer.in_channels // layer.groups, *layer.kernel_size).permute(0, 2, 1, 3, 4).reshape_as(layer.weight)
    return value


def _torch_affine(layer, inputs, canonical_weight, bias):
    weight, bias = _torch_layout(canonical_weight, layer), torch.from_numpy(np.array(bias, dtype=np.float64))
    if isinstance(layer, nn.Linear):
        return F.linear(inputs, weight, bias)
    kwargs = dict(stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
    if isinstance(layer, nn.ConvTranspose2d):
        return F.conv_transpose2d(inputs, weight, bias, output_padding=layer.output_padding, **kwargs)
    return F.conv2d(inputs, weight, bias, **kwargs)


def _round_ratio(numerator, denominator=1):
    value = (abs(int(numerator)) + denominator // 2) // denominator
    return max(-128, min(127, -value if numerator < 0 else value))


def _requantize_oracle(accumulator, exponents, channel_axis):
    result = np.empty(accumulator.shape, dtype=np.int8)
    for index in np.ndindex(accumulator.shape):
        exponent = int(exponents[index[channel_axis]])
        value = int(accumulator[index])
        result[index] = _round_ratio(value * (1 << max(0, exponent)), 1 << max(0, -exponent))
    return result


@pytest.mark.parametrize("kind", ["linear", "grouped", "depthwise", "transpose", "transpose_point"])
def test_eval_batch_norm_fold_preserves_float_function_and_source(kind):
    torch.manual_seed(114)
    source, shape = _source(kind)
    source = source.double().eval()
    outputs = source.out_features if kind == "linear" else source.out_channels
    bn = (nn.BatchNorm1d(outputs) if kind == "linear" else nn.BatchNorm2d(outputs)).double().eval()
    with torch.no_grad():
        bn.running_mean.copy_(torch.linspace(-1, 1, outputs))
        bn.running_var.copy_(torch.linspace(.03, 2, outputs))
        bn.weight.copy_(torch.linspace(-1.3, .7, outputs))
        bn.bias.copy_(torch.linspace(.3, -.4, outputs))
    before = {name: value.clone() for name, value in source.state_dict().items()}
    x = torch.randn(shape, dtype=torch.float64)
    weight, bias = folded_parameters(source, bn)
    torch.testing.assert_close(_torch_affine(source, x, weight, bias), bn(source(x)), rtol=1e-12, atol=2e-12)
    for name, value in before.items():
        torch.testing.assert_close(source.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["linear", "grouped", "depthwise", "transpose", "transpose_point"])
def test_exact_int32_dots_groups_padding_transpose_and_output_grids(kind):
    torch.manual_seed(151)
    source, shape = _source(kind)
    source = source.eval()
    with torch.no_grad():
        source.weight.mul_(3.2)
        source.bias.copy_(torch.linspace(-.3, .4, len(source.bias)))
    integer = IntegerAffine.from_torch(source, ActivationGrid(-5), ActivationGrid(-3))
    rng = np.random.default_rng(813)
    codes = rng.integers(-128, 128, shape, dtype=np.int8)
    actual, accumulator = integer(codes, return_accumulator=True)
    # Ordinary PyTorch operators perform these small integer dot products
    # exactly in float64, independently testing all indexing/group conventions.
    expected_accumulator = _torch_affine(source, torch.from_numpy(codes.astype(np.float64)), integer.weights, integer.bias).numpy()
    assert accumulator.dtype == np.int32
    np.testing.assert_array_equal(accumulator, expected_accumulator)
    expected = _requantize_oracle(expected_accumulator, integer.shifts, -1 if kind == "linear" else 1)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int8 and not integer.weights.flags.writeable
    assert integer.metadata()["maximum_abs_accumulator_bound"] <= 2**31 - 1


@pytest.mark.parametrize("transpose,frequency_stride", [(False, 1), (True, 1), (True, 2)])
def test_streaming_convolution_preserves_vendor_cache_and_converted_kernel_semantics(transpose, frequency_stride):
    torch.manual_seed(151)
    wrapper_type = StreamConvTranspose2d if transpose else StreamConv2d
    wrapper = wrapper_type(2, 2, (3, 3), stride=(1, frequency_stride), padding=(0, 1),
                           dilation=(2, 2), groups=2).eval()
    integer = IntegerStreamConv.from_torch(wrapper, ActivationGrid(-4), ActivationGrid(-2))
    oracle = copy.deepcopy(wrapper).double()
    convolution = oracle.ConvTranspose2d if transpose else oracle.Conv2d
    with torch.no_grad():
        convolution.weight.copy_(torch.from_numpy(integer.affine.weights.copy()).double())
        convolution.bias.copy_(torch.from_numpy(integer.affine.bias.copy()).double())
    rng = np.random.default_rng(134)
    codes = rng.integers(-128, 128, (2, 2, 17, 9), dtype=np.int8)
    state = integer.initial_state(2, 9)
    torch_state = torch.from_numpy(state.astype(np.float64))
    original = state.copy()
    for time in range(17):
        frame = codes[:, :, time:time+1]
        actual, next_state = integer.step(frame, state)
        expected_accumulator, torch_state = oracle(torch.from_numpy(frame.astype(np.float64)), torch_state)
        expected = _requantize_oracle(expected_accumulator.detach().numpy(), integer.affine.shifts, 1)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(next_state, torch_state.numpy())
        assert next_state.dtype == np.int8 and next_state.nbytes == 2*2*4*9
        state = next_state
    np.testing.assert_array_equal(original, integer.initial_state(2, 9))
    restarted, _ = integer.step(codes[:, :, :1])
    explicit, _ = integer.step(codes[:, :, :1], integer.initial_state(2, 9))
    np.testing.assert_array_equal(restarted, explicit)


@pytest.mark.parametrize("slopes", [[.25], [-.5, 0., 1., 2.25]])
def test_prelu_handles_negative_zero_and_greater_than_one_learned_slopes(slopes):
    layer = nn.PReLU(len(slopes))
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(slopes))
    integer = IntegerPReLU(layer, ActivationGrid(-4), ActivationGrid(-6))
    codes = np.array([-128, -3, -1, 0, 1, 3, 127], np.int8)[None, None, None].repeat(4, axis=1)
    expected = np.empty_like(codes)
    for index in np.ndindex(codes.shape):
        x = int(codes[index])
        slope = int(integer.slopes[0 if len(slopes) == 1 else index[1]])
        shift = 2 + (integer.exponent if x < 0 else 0)
        expected[index] = _round_ratio(x * (slope if x < 0 else 1) * (1 << max(0, shift)), 1 << max(0, -shift))
    np.testing.assert_array_equal(integer(codes), expected)


def test_residual_aligns_without_premature_rounding_or_branch_clipping():
    fine, coarse = ActivationGrid(-5), ActivationGrid(-4)
    # Each +.5-code contribution must sum to1 before the only rounding.
    a = np.array([1, -1, 127, -128, 127], np.int8)
    b = np.array([1, -1, -128, 127, 127], np.int8)
    expected = np.array([1, -1, -1, -1, 127], np.int8)
    np.testing.assert_array_equal(residual_add(a, b, fine, fine, coarse), expected)
    # Large intermediate cancellation must survive the INT8 output boundary.
    np.testing.assert_array_equal(residual_add(np.array([127], np.int8), np.array([-127], np.int8),
                                               ActivationGrid(8), ActivationGrid(8), ActivationGrid(-16)), [0])


@pytest.mark.parametrize("input_exp,output_exp", [(-16, 8), (8, -16), (-4, -5), (-5, -4)])
def test_explicit_activation_requantization_covers_every_signed_code(input_exp, output_exp):
    values = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    shift = input_exp-output_exp
    expected = [_round_ratio(int(value)*(1 << max(0, shift)), 1 << max(0, -shift)) for value in values]
    np.testing.assert_array_equal(requantize_activation(values, ActivationGrid(input_exp), ActivationGrid(output_exp)), expected)


@pytest.mark.parametrize("input_exp,output_exp", [(-4, -7), (-16, 8), (8, -16)])
def test_attention_energy_and_endpoint_product_have_single_exact_rational_round(input_exp, output_exp):
    rng = np.random.default_rng(411)
    codes = rng.integers(-128, 128, (2, 8, 3, 33), dtype=np.int8)
    codes[0, 0] = 0
    codes[0, 1] = -128
    input_grid, output_grid = ActivationGrid(input_exp), ActivationGrid(output_exp)
    actual = attention_energy(codes, input_grid, output_grid)
    expected = np.empty(codes.shape[:-1], dtype=np.int8)
    shift = 2*input_exp-output_exp
    for index in np.ndindex(expected.shape):
        total = sum(int(x)**2 for x in codes[index])
        expected[index] = _round_ratio(total*(1 << max(0, shift)), 33*(1 << max(0, -shift)))
    np.testing.assert_array_equal(actual, expected)
    gates = rng.integers(-128, 128, (*codes.shape[:-1], 1), dtype=np.int8)
    gates[:, 0] = -128
    gates[:, 1] = 127
    product = attention_product(codes, gates, input_grid, output_grid)
    expected_product = np.empty_like(codes)
    shift = input_exp-output_exp
    for index in np.ndindex(codes.shape):
        value = int(codes[index]) * (int(gates[index[:-1]+(0,)])+128)
        expected_product[index] = _round_ratio(value*(1 << max(0, shift)), 255*(1 << max(0, -shift)))
    np.testing.assert_array_equal(product, expected_product)
    assert not product[:, 0].any()


def test_sfe_and_shuffle_retain_exact_vendored_channel_order():
    codes = np.arange(2*4*3*7, dtype=np.int16).astype(np.int8).reshape(2, 4, 3, 7)
    expected = SFE()(torch.from_numpy(codes.astype(np.float32))).numpy().astype(np.int8)
    np.testing.assert_array_equal(subband_features(codes), expected)
    left, right = codes[:, :2], codes[:, 2:]
    block = StreamGTConvBlock(4, 4, (3, 3), (1, 1), (0, 1), (1, 1))
    expected = block.shuffle(torch.from_numpy(left), torch.from_numpy(right)).numpy()
    np.testing.assert_array_equal(shuffle_pair(left, right), expected)


def test_calibration_records_training_declaration_and_covers_asymmetric_int8_rails():
    observer = ActivationObserver("dpgrnns.0.intra_ln.input")
    observer.observe(torch.tensor([-128., 127.], dtype=torch.bfloat16))
    observer.observe(np.array([-.001, .003], dtype=np.float32))
    metadata = observer.finish("a"*64, checkpoint_sha256="b"*64)
    assert metadata["grid"] == {"exponent": 0}
    assert metadata["observed_values"] == 4 and metadata["split"] == "train"
    assert metadata["checkpoint_sha256"] == "b"*64
    assert metadata["observed_minimum"] == -128 and metadata["observed_maximum"] == 127
    assert ActivationGrid(0).quantize(np.array([-128., 127.])).tolist() == [-128, 127]
    with pytest.raises(ValueError, match="train"):
        observer.finish("a"*64, checkpoint_sha256="b"*64, split="val")
    with pytest.raises(ValueError, match="finite"):
        observer.observe(np.array([np.nan]))


def test_fold_and_integer_contract_reject_unsafe_parameters():
    source = nn.Conv2d(2, 2, 1).eval()
    with pytest.raises(ValueError, match="eval"):
        folded_parameters(source, nn.BatchNorm2d(2))
    bn = nn.BatchNorm2d(2).eval()
    bn.running_var[0] = -1
    with pytest.raises(ValueError, match="variance"):
        folded_parameters(source, bn)
    with torch.no_grad():
        source.weight.fill_(1)
        source.bias.fill_(1e20)
    with pytest.raises(OverflowError, match="bias"):
        IntegerAffine.from_torch(source, ActivationGrid(-16), ActivationGrid(0))
    source = nn.Linear(1, 1).double()
    with torch.no_grad():
        source.weight.fill_(1)
        source.bias.fill_((2**31-1) / 64 - .5)
    with pytest.raises(OverflowError, match="dot"):
        IntegerAffine.from_torch(source, ActivationGrid(0), ActivationGrid(0))
    with pytest.raises(ValueError, match="INT8"):
        attention_energy(np.zeros((1, 33), np.int16), ActivationGrid(-4), ActivationGrid(-4))
    with pytest.raises(TypeError, match="ActivationGrid"):
        residual_add(np.ones(1, np.int8), np.ones(1, np.int8), 0, ActivationGrid(0), ActivationGrid(0))


def test_all_vendored_streaming_affine_and_prelu_shapes_have_explicit_snapshots():
    core = StreamGTCRN().eval()
    grid = ActivationGrid(-4)  # Shape/contract smoke test; this is not calibration.
    affines, prelus, histories = [], [], []
    for module in core.modules():
        if isinstance(module, ConvBlock):
            affines.append(IntegerAffine.from_torch(module.conv, grid, grid, batch_norm=module.bn))
        elif isinstance(module, StreamGTConvBlock):
            affines.append(IntegerAffine.from_torch(module.point_conv1, grid, grid, batch_norm=module.point_bn1))
            stream = IntegerStreamConv.from_torch(module.depth_conv, grid, grid, batch_norm=module.depth_bn)
            affines.append(stream.affine)
            histories.append(stream.metadata()["history_frames"])
            affines.append(IntegerAffine.from_torch(module.point_conv2, grid, grid, batch_norm=module.point_bn2))
        elif isinstance(module, StreamTRA):
            affines.append(IntegerAffine.from_torch(module.att_fc, ActivationGrid(-7), grid))
        elif isinstance(module, DPGRNN):
            affines.extend(IntegerAffine.from_torch(layer, ActivationGrid(-7), grid)
                           for layer in (module.intra_fc, module.inter_fc))
        if isinstance(module, nn.PReLU):
            prelus.append(IntegerPReLU(module, grid, grid).metadata())
    assert len(affines) == 32 and sum(affine.batch_norm_folded for affine in affines) == 22
    assert len(prelus) == 15 and all(item["slope_array_bytes"] == 1 for item in prelus)
    assert histories == [2, 4, 10, 10, 4, 2]
    assert sum(histories)*16*33 == 16896  # Exact INT8 convolution history only.
