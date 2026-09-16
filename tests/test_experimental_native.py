"""Native numerical parity and arithmetic safety, not full GTCRN performance."""
import ctypes
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from esp32_denoiser.experimental_gru import GRUQuantizationConfig, IntegerGRUCell
from esp32_denoiser.experimental_layer_norm import (
    IntegerLayerNormParameters, _round_divide, integer_layer_norm, quantize_layer_norm,
)
from esp32_denoiser.experimental_native import (
    CIntegerGRUCell, CIntegerLayerNorm, _Buffer, _GRUSpec, _LayerNormSpec,
    _buffer, _handle, _native_runtime,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("inputs,hidden,input_exp,logit_exp", [
    (1, 1, -12, -6), (8, 4, -8, -5), (8, 8, -4, -4),
    (8, 16, 0, -3), (32, 16, -4, -2),
])
def test_native_gru_stream_and_every_gate_match_numpy(inputs, hidden, input_exp, logit_exp):
    torch.manual_seed(183)
    rng = np.random.default_rng(185)
    source = torch.nn.GRUCell(inputs, hidden)
    # Different row scales exercise per-row exponents rather than one matrix grid.
    with torch.no_grad():
        source.weight_ih.mul_(torch.logspace(-2, 1, 3 * hidden)[:, None])
    reference = IntegerGRUCell.from_torch(source, GRUQuantizationConfig(
        input_exponent=input_exp, logit_exponent=logit_exp))
    native = CIntegerGRUCell(reference)
    codes = rng.integers(-128, 128, (2, 95, inputs), dtype=np.int8)
    codes[:, 17:41] = 0
    codes[:, 56] = -128
    codes[:, 57] = 127
    state = rng.integers(-128, 128, (2, hidden), dtype=np.int8)
    original_state = state.copy()
    expected, final = reference.process(codes, state)
    actual, native_final = native.process(codes, state)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(native_final, final)
    np.testing.assert_array_equal(state, original_state)
    assert actual.dtype == native_final.dtype == np.int8
    for index in (0, 17, 56, 57, 94):
        before = state if index == 0 else expected[:, index - 1]
        py_out, py_trace = reference.step(codes[:, index], before, return_trace=True)
        c_out, c_trace = native.step(codes[:, index], before, return_trace=True)
        np.testing.assert_array_equal(c_out, py_out)
        for name, value in py_trace.items():
            np.testing.assert_array_equal(c_trace[name], value)
    chunks, state = [], original_state
    for start, end in ((0, 1), (1, 26), (26, 59), (59, 95)):
        chunk, state = native.process(codes[:, start:end], state)
        chunks.append(chunk)
    np.testing.assert_array_equal(np.concatenate(chunks, 1), expected)
    restarted, _ = native.process(codes)
    reset, _ = native.process(codes, native.initial_state(2))
    np.testing.assert_array_equal(restarted, reset)
    accounting = native.memory_accounting()
    assert accounting["persistent_state_bytes_per_stream"] == accounting["explicit_scratch_bytes"] == hidden
    assert accounting["parameter_and_table_bytes"] == reference.storage_stats()["parameter_and_table_bytes"]


@pytest.mark.parametrize("sign", [-1, 1])
def test_native_reset_after_uses_wide_product_and_bias(sign):
    source = torch.nn.GRUCell(1, 1)
    with torch.no_grad():
        for value in source.parameters():
            value.zero_()
        source.bias_ih[0], source.bias_ih[1] = 20, -20
        source.weight_hh[2, 0] = sign * 16
        source.bias_hh[2] = sign * 520000
    reference = IntegerGRUCell.from_torch(source)
    native = CIntegerGRUCell(reference)
    x, state = np.zeros((1, 1), np.int8), np.array([[127]], np.int8)
    actual, trace = native.step(x, state, return_trace=True)
    expected = sign * (520000 * 4096 + 16 * 127 * 32)
    assert trace["candidate_accumulator"].item() == expected
    assert abs(expected * 255) > 2**31
    np.testing.assert_array_equal(actual, reference.step(x, state))
    assert actual.item() == (127 if sign > 0 else -128)
    # Supported native aliasing must retain every old state value until all
    # recurrent projections have been computed.
    assert native.library.ednx_gru_step(native.handle, x.ctypes.data, state.ctypes.data,
                                       state.ctypes.data, native.scratch.ctypes.data,
                                       native.scratch_bytes, None) == 0
    np.testing.assert_array_equal(state, actual)


def test_native_in_place_state_preserves_cross_channel_recurrence():
    torch.manual_seed(113)
    reference = IntegerGRUCell.from_torch(torch.nn.GRUCell(8, 16))
    native = CIntegerGRUCell(reference)
    rng = np.random.default_rng(188)
    codes = rng.integers(-128, 128, (1, 2048, 8), dtype=np.int8)
    codes[:, 700:1500] = 0
    expected, final = reference.process(codes)
    state = np.zeros(16, dtype=np.int8)
    actual = np.empty_like(expected)
    for index, frame in enumerate(codes[0]):
        assert native.library.ednx_gru_step(native.handle, frame.ctypes.data, state.ctypes.data,
                                           state.ctypes.data, native.scratch.ctypes.data,
                                           native.scratch_bytes, None) == 0
        actual[0, index] = state
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(state, final[0])
    assert state.nbytes == 16


@pytest.mark.parametrize("input_exp,gamma_exp,output_exp,bits", [
    (-4, -6, -4, 24), (-16, -6, -7, 24), (8, -6, -2, 24),
    (-4, -16, 8, 24), (-4, -6, -4, 0), (-8, -3, -6, 12),
])
def test_native_layer_norm_matches_numpy_exactly(input_exp, gamma_exp, output_exp, bits):
    rng = np.random.default_rng(134)
    gamma = rng.integers(-128, 128, (33, 16), dtype=np.int8)
    beta = rng.integers(-16, 17, (33, 16), dtype=np.int32)
    parameters = IntegerLayerNormParameters(gamma, beta, input_exp, gamma_exp,
                                           output_exp, variance_fractional_bits=bits)
    codes = rng.integers(-128, 128, (2, 9, 33, 16), dtype=np.int8)
    frames = codes.reshape(-1, 33, 16)
    frames[0], frames[1], frames[2] = 0, -128, 127
    frames[3].reshape(-1)[::2] = -128
    frames[3].reshape(-1)[1::2] = 127
    frames[4] = 0
    frames[4, 0, 0] = 1
    frames[5] = 127
    frames[5, 0, 0] = -128
    native = CIntegerLayerNorm(parameters)
    expected = integer_layer_norm(codes, parameters)
    np.testing.assert_array_equal(native(codes), expected)
    original = frames[3].copy()
    assert native.library.ednx_layer_norm_frame(native.handle, original.ctypes.data, original.ctypes.data) == 0
    np.testing.assert_array_equal(original, expected.reshape(-1, 33, 16)[3])
    for frame in range(9):
        np.testing.assert_array_equal(native(codes[:, frame]), expected[:, frame])
    accounting = native.memory_accounting()
    assert accounting["parameter_array_bytes"] == 2640
    assert accounting["persistent_neural_state_bytes"] == accounting["explicit_scratch_bytes"] == 0


def test_native_layer_norm_signed_ties_include_beta_before_rounding():
    parameters = IntegerLayerNormParameters(np.ones(2, np.int8), np.array([1, -1], np.int32),
                                            gamma_exponent=-1, output_exponent=0, variance_fractional_bits=0)
    codes = np.array([-1, 1], dtype=np.int8)
    # The normalized affine term is [-.5,+.5], then beta gives [+.5,-.5].
    # Rounding the first term before adding beta would incorrectly give [0,0].
    np.testing.assert_array_equal(integer_layer_norm(codes, parameters), [1, -1])
    np.testing.assert_array_equal(CIntegerLayerNorm(parameters)(codes), [1, -1])


@pytest.mark.parametrize("size", [1, 527, 1024])
def test_native_layer_norm_extreme_sizes_and_wide_bias(size):
    gamma = np.full(size, -128, dtype=np.int8)
    beta = np.full(size, 2**31 - 1, dtype=np.int32)
    beta[::2] = -(2**31)
    parameters = IntegerLayerNormParameters(gamma, beta, variance_fractional_bits=0)
    codes = np.full(size, -128, dtype=np.int8)
    codes[:size // 2] = 127
    np.testing.assert_array_equal(CIntegerLayerNorm(parameters)(codes), integer_layer_norm(codes, parameters))


def test_native_reciprocal_and_sqrt_match_independent_python_arithmetic():
    library, _ = _native_runtime()
    rng = np.random.default_rng(151)
    limit = 2**63 - 1
    denominators = [1, 2, 3, 255, 528, 2**31 - 1, 2**31, 2**32 - 1, 2**32 + 1, 2**55 - 1, limit]
    denominators += [int(rng.integers(1, limit)) for _ in range(700)]
    for denominator in denominators:
        maximum = limit - denominator // 2
        values = [0, maximum, -maximum, maximum // 2, -(maximum // 2)]
        for quotient in (0, 1, 2, 126, 127, 128):
            middle = quotient * denominator + denominator // 2
            values.extend(value for value in (middle - 1, middle, middle + 1) if 0 <= value <= maximum)
        for numerator in values:
            result = ctypes.c_int64()
            assert library.ednx_test_round_divide(numerator, denominator, ctypes.byref(result)) == 0
            assert result.value == _round_divide(numerator, denominator)
            assert library.ednx_test_round_divide(-numerator, denominator, ctypes.byref(result)) == 0
            assert result.value == _round_divide(-numerator, denominator)
    values = [0, 1, 2**64 - 1]
    for root in (2, 37, 2**20, 2**26 + 1, 2**31 - 1, 2**32 - 1):
        values.extend([root**2 - 1, root**2, root**2 + 1])
    values += [int(rng.integers(0, 2**64 - 1, dtype=np.uint64)) for _ in range(1000)]
    assert [library.ednx_test_isqrt(value) for value in values] == [math.isqrt(value) for value in values]


def test_native_init_rejects_lengths_grids_and_combined_arithmetic_overflow():
    reference = IntegerGRUCell.from_torch(torch.nn.GRUCell(8, 4))
    native = CIntegerGRUCell(reference)
    for field, value in (("input_size", 0), ("hidden_size", 129), ("input_exponent", 1),
                         ("state_exponent", -8), ("logit_exponent", -7), ("accumulator_exponent", -11)):
        spec = _GRUSpec.from_buffer_copy(native.spec)
        setattr(spec, field, value)
        target = _handle(native.handle_bytes)
        assert native.library.ednx_gru_init(target, ctypes.byref(spec)) == -1
        assert native.library.ednx_gru_scratch_bytes(target) == 0
    spec = _GRUSpec.from_buffer_copy(native.spec)
    spec.sigmoid_lut.bytes = 255
    assert native.library.ednx_gru_init(_handle(native.handle_bytes), ctypes.byref(spec)) == -1
    rows = 3 * native.hidden_size
    zero_i, zero_h = np.zeros((rows, 8), np.int8), np.zeros((rows, 4), np.int8)
    bias = np.full(rows, 1100000000, np.int32)
    ei, eh = np.full(rows, -7, np.int8), np.full(rows, -5, np.int8)
    spec = _GRUSpec.from_buffer_copy(native.spec)
    spec.weight_ih, spec.weight_hh = _buffer(zero_i), _buffer(zero_h)
    spec.bias_ih = spec.bias_hh = _buffer(bias)
    spec.exponent_ih, spec.exponent_hh = _buffer(ei), _buffer(eh)
    # Each raw/aligned path fits INT32; adding the two does not.
    assert native.library.ednx_gru_init(_handle(native.handle_bytes), ctypes.byref(spec)) == -1
    x, state, output = np.zeros(8, np.int8), np.zeros(4, np.int8), np.zeros(4, np.int8)
    assert native.library.ednx_gru_step(native.handle, x.ctypes.data, state.ctypes.data,
                                       output.ctypes.data, native.scratch.ctypes.data, 3, None) == -1
    ln = CIntegerLayerNorm(quantize_layer_norm(np.ones((33, 16)), np.zeros((33, 16))))
    for field, value in (("size", 0), ("epsilon_code", 0), ("epsilon_code", 2**64 - 1),
                         ("variance_fractional_bits", 25), ("gamma_exponent", -17)):
        spec = _LayerNormSpec.from_buffer_copy(ln.spec)
        setattr(spec, field, value)
        assert ln.library.ednx_layer_norm_init(_handle(ln.handle_bytes), ctypes.byref(spec)) == -1
    spec = _LayerNormSpec.from_buffer_copy(ln.spec)
    huge_beta = np.full(528, 2**31 - 1, np.int32)
    spec.beta = _buffer(huge_beta)
    spec.gamma_exponent, spec.output_exponent = -16, 8
    assert ln.library.ednx_layer_norm_init(_handle(ln.handle_bytes), ctypes.byref(spec)) == -1
    spec = _LayerNormSpec.from_buffer_copy(ln.spec)
    spec.beta = _Buffer(ln.beta.ctypes.data + 1, ln.beta.nbytes)
    assert ln.library.ednx_layer_norm_init(_handle(ln.handle_bytes), ctypes.byref(spec)) == -1
    with pytest.raises(ValueError, match="state must be INT8"):
        native.step(np.zeros((1, 8), np.int8), np.zeros((1, 4), np.int16))
    with pytest.raises(ValueError, match="INT8"):
        ln(np.zeros((33, 16), np.float32))


def test_portable_c_under_undefined_behavior_and_address_sanitizers(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("No C compiler")
    source = Path(__file__).resolve().parents[1] / "firmware/experimental_int8"
    harness = tmp_path / "sanitize.c"
    harness.write_text(r'''
#include "primitives.h"
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
int main(void) {
    int8_t wi[3] = {0,0,0}, wh[3] = {0,0,64}, ei[3] = {-20,-20,-20}, eh[3] = {-20,-20,-2};
    int32_t bi[3] = {671088640,-671088640,0}, bh[3] = {0,0,266240000};
    int8_t sigmoid[256], tanh[256];
    for (int j=0;j<256;++j) { sigmoid[j]=(int8_t)(j-128); tanh[j]=(int8_t)(j-128); }
    ednx_gru_spec gs = {1,1,-5,-7,-4,-12,
        {wi,sizeof(wi)},{wh,sizeof(wh)},{bi,sizeof(bi)},{bh,sizeof(bh)},
        {ei,sizeof(ei)},{eh,sizeof(eh)},{sigmoid,sizeof(sigmoid)},{tanh,sizeof(tanh)}};
    ednx_gru *g = malloc(ednx_gru_handle_bytes());
    assert(g && ednx_gru_init(g,&gs)==0);
    int8_t input=-128, state=127, scratch=0;
    for(int j=0;j<4096;++j) assert(ednx_gru_step(g,&input,&state,&state,&scratch,1,NULL)==0);
    int8_t gamma[1024], data[1024]; int32_t beta[1024];
    for(int j=0;j<1024;++j) { gamma[j]=-128; beta[j]=(j%2)?INT32_MAX:INT32_MIN; data[j]=(j%2)?127:-128; }
    ednx_layer_norm_spec ls = {1024,-4,-6,-4,0,1,{gamma,sizeof(gamma)},{beta,sizeof(beta)}};
    ednx_layer_norm *ln = malloc(ednx_layer_norm_handle_bytes());
    assert(ln && ednx_layer_norm_init(ln,&ls)==0);
    for(int j=0;j<100;++j) assert(ednx_layer_norm_frame(ln,data,data)==0);
    int64_t result;
    for(uint64_t d=1;d<10000;++d) {
        int64_t n=INT64_MAX-(int64_t)(d/2);
        assert(ednx_test_round_divide(n,d,&result)==0);
        assert(ednx_test_round_divide(-n,d,&result)==0);
    }
    free(g); free(ln); return 0;
}
''')
    executable = tmp_path / "sanitize"
    subprocess.run([compiler, "-std=c99", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                    "-DEDNX_TESTING", "-fsanitize=undefined,address", "-fno-sanitize-recover=all",
                    "-I", str(source), str(source / "primitives.c"), str(harness), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
