"""Exact recurrent QAT forward and useful gradients; no MCU timing claim."""
import copy

import numpy as np
import pytest
import torch

from esp32_denoiser.experimental_gru import GRUQuantizationConfig, IntegerGRUCell
from esp32_denoiser.experimental_native import CIntegerGRUCell
from esp32_denoiser.gtcrn_qat_gru import BatchedQATGRU


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _codes(value, exponent):
    value = value.detach().cpu().numpy().astype(np.float64) / 2.0**exponent
    return (np.sign(value) * np.floor(np.abs(value) + .5)).clip(-128, 127).astype(np.int8)


def _oracle(cells, inputs, state, native=False):
    sequences, finals = [], []
    for direction, cell in enumerate(cells):
        codes = _codes(inputs, cell.config.input_exponent)
        initial = _codes(state[direction], -7)
        runtime = CIntegerGRUCell(cell) if native else cell
        output, final = runtime.process(codes[:, ::-1].copy() if direction else codes, initial)
        sequences.append(output[:, ::-1] if direction else output)
        finals.append(final)
    return np.concatenate(sequences, axis=-1), np.stack(finals)


def _assert_codes(actual, expected):
    sequence, state = actual
    assert sequence.dtype == state.dtype == torch.float32
    np.testing.assert_array_equal(sequence.detach().cpu().numpy() * 128, expected[0])
    np.testing.assert_array_equal(state.detach().cpu().numpy() * 128, expected[1])


@pytest.mark.parametrize("inputs,hidden,input_exp,logit_exp,bidirectional", [
    (1, 1, -12, -6, False), (8, 16, -5, -4, False),
    (8, 8, 0, -2, True), (16, 8, -8, -5, True),
])
def test_initial_exact_forward_matches_independent_numpy_and_c(
        inputs, hidden, input_exp, logit_exp, bidirectional):
    torch.manual_seed(241)
    source = torch.nn.GRU(inputs, hidden, batch_first=True, bidirectional=bidirectional)
    config = GRUQuantizationConfig(input_exponent=input_exp, logit_exponent=logit_exp)
    model = BatchedQATGRU.from_float(source, config)
    # This independent preparation uses the pre-existing NumPy/double oracle,
    # not the new QAT module's integer_snapshots implementation.
    cells = tuple(IntegerGRUCell.from_torch(source, config, direction=direction)
                  for direction in (("forward", "reverse") if bidirectional else ("forward",)))
    for expected, actual in zip(cells, model.integer_snapshots()):
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh", "exponent_ih", "exponent_hh"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    x = torch.randn(3, 33, inputs) * (50 * 2.0**input_exp)
    x[:, 9:14] = 0
    x[:, 15], x[:, 16] = -128 * 2.0**input_exp, 127 * 2.0**input_exp
    # Exact positive/negative input ties and clipped initial states.
    x[0, 17:19] = torch.tensor([-.5, .5])[:, None] * 2.0**input_exp
    initial = torch.randn(model.num_directions, 3, hidden)
    actual = model(x, initial)
    _assert_codes(actual, _oracle(cells, x, initial))
    _assert_codes(actual, _oracle(cells, x, initial, native=True))


def test_unidirectional_chunking_keeps_true_q7_state():
    torch.manual_seed(248)
    model = BatchedQATGRU.from_float(torch.nn.GRU(8, 16, batch_first=True))
    x = torch.randn(4, 96, 8) * .2
    x[:, 19:68] = 0
    complete, final = model(x)
    state, outputs = None, []
    for start, end in ((0, 1), (1, 19), (19, 68), (68, 96)):
        chunk, state = model(x[:, start:end], state)
        assert torch.equal(state * 128, (state * 128).to(torch.int8).float())
        outputs.append(chunk)
    torch.testing.assert_close(torch.cat(outputs, 1), complete, rtol=0, atol=0)
    torch.testing.assert_close(state, final, rtol=0, atol=0)


def test_optimizer_updates_share_masters_and_preserve_frozen_grids():
    torch.manual_seed(252)
    source = torch.nn.GRU(8, 16, batch_first=True, bidirectional=True)
    model = BatchedQATGRU.from_float(source)
    assert model.weight_ih_l0 is source.weight_ih_l0
    assert model.weight_hh_l0_reverse is source.weight_hh_l0_reverse
    saved_grids = {name: value.clone() for name, value in model.named_buffers() if name.startswith("exponent_")}
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    x, target = torch.randn(4, 17, 8) * .3, torch.randn(4, 17, 32) * .25
    original = model.weight_ih_l0.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        output, state = model(x)
        ((output - target).square().mean() + .1 * state.square().mean()).backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(original, model.weight_ih_l0)
    for name, previous in saved_grids.items():
        assert torch.equal(getattr(model, name), previous)
    snapshots = model.integer_snapshots()
    # Independently encode updated float masters using the *saved* exponents.
    for direction, snapshot in enumerate(snapshots):
        suffix = "_l0" + ("_reverse" if direction else "")
        for side, input_exp in (("ih", model.config.input_exponent), ("hh", -7)):
            scales = np.exp2(saved_grids["exponent_" + side + suffix].numpy().astype(np.float64))
            weight = getattr(source, "weight_" + side + suffix).detach().numpy().astype(np.float64) / scales[:, None]
            bias = getattr(source, "bias_" + side + suffix).detach().numpy().astype(np.float64) / (scales * 2.0**input_exp)
            rounded = lambda value: np.sign(value) * np.floor(np.abs(value) + .5)
            np.testing.assert_array_equal(getattr(snapshot, "weight_" + side), rounded(weight).clip(-128, 127).astype(np.int8))
            np.testing.assert_array_equal(getattr(snapshot, "bias_" + side), rounded(bias).astype(np.int32))
    initial = torch.randn(2, 4, 16) * .2
    _assert_codes(model(x, initial), _oracle(snapshots, x, initial, native=True))


def test_saved_snapshot_scales_and_metadata_survive_restore():
    torch.manual_seed(257)
    source = torch.nn.GRU(8, 8, batch_first=True)
    config = GRUQuantizationConfig(input_exponent=-8, logit_exponent=-5)
    snapshot = IntegerGRUCell.from_torch(source, config)
    # Force a later masterweight maximum across several dynamic-grid choices.
    with torch.no_grad():
        source.weight_ih_l0.mul_(.125)
    dynamic = IntegerGRUCell.from_torch(source, config)
    assert not np.array_equal(dynamic.exponent_ih, snapshot.exponent_ih)
    model = BatchedQATGRU.from_float(source, snapshots=(snapshot,))
    np.testing.assert_array_equal(model.integer_snapshots()[0].exponent_ih, snapshot.exponent_ih)
    saved = copy.deepcopy(model.state_dict())
    restored = BatchedQATGRU.from_float(torch.nn.GRU(8, 8, batch_first=True))
    restored.load_state_dict(saved, strict=True)
    assert restored.config == config
    x = torch.randn(2, 8, 8) * .1
    for expected, actual in zip(model(x), restored(x)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(restored.integer_snapshots()[0].exponent_ih, snapshot.exponent_ih)


def test_forward_batches_input_projection_and_never_takes_cpu_snapshot(monkeypatch):
    model = BatchedQATGRU.from_float(torch.nn.GRU(8, 16, batch_first=True, bidirectional=True))
    linear = torch.nn.functional.linear
    calls = []
    def traced(input, weight, bias=None):
        calls.append(input.shape)
        return linear(input, weight, bias)
    def forbidden(*args, **kwargs):
        raise AssertionError("No offline snapshot/CPU array conversion is allowed in forward")
    monkeypatch.setattr(torch.nn.functional, "linear", traced)
    monkeypatch.setattr(model, "integer_snapshots", forbidden)
    monkeypatch.setattr(IntegerGRUCell, "from_torch", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    x = torch.randn(3, 8, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, state = model(x)
        (output.square().mean() + state.square().mean()).backward()
    assert output.dtype == state.dtype == torch.float32
    assert calls.count(torch.Size([3, 8, 8])) == 2
    assert len(calls) == 2 * (8 + 1)


@pytest.mark.parametrize("sign", [-1, 1])
def test_large_aligned_candidate_uses_checked_wide_reset_product(sign):
    source = torch.nn.GRU(1, 1, batch_first=True)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.zero_()
    model = BatchedQATGRU.from_float(source)
    with torch.no_grad():
        model.exponent_ih_l0.fill_(0)
        model.exponent_hh_l0.fill_(4)
        model.bias_ih_l0[:2] = torch.tensor([20., -20.])
        model.weight_hh_l0[2, 0] = sign * 16
        model.bias_hh_l0[2] = sign * 520000
    cell = model.integer_snapshots()[0]
    x, state = torch.zeros(1, 3, 1), torch.tensor([[[127 / 128]]])
    _assert_codes(model(x, state), _oracle((cell,), x, state, native=True))
    native = CIntegerGRUCell(cell)
    _, trace = native.step(np.zeros((1, 1), np.int8), np.array([[127]], np.int8), return_trace=True)
    assert abs(trace["candidate_accumulator"].item() * 255) > 2**31
    assert model(x, state)[0][0, 0, 0].item() == (127 / 128 if sign > 0 else -1)


@pytest.mark.parametrize("defect", ["raw_precision", "aligned_overflow", "exponent", "nan_weight", "nan_input", "double_master"])
def test_invalid_precision_and_numerical_contract_fail_before_gemm(defect, monkeypatch):
    source = torch.nn.GRU(1, 1, batch_first=True)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.zero_()
    model = BatchedQATGRU.from_float(source)
    x = torch.zeros(1, 2, 1)
    with torch.no_grad():
        if defect == "raw_precision":
            model.exponent_ih_l0.fill_(0)
            model.bias_ih_l0[0] = 2**19  # raw code=2**24, while aligned still fits INT32
        elif defect == "aligned_overflow":
            model.exponent_ih_l0.fill_(4)
            model.bias_ih_l0[0] = 2**19  # raw code=2**20, aligned=2**31
        elif defect == "exponent":
            model.exponent_ih_l0[0] = 127
        elif defect == "nan_weight":
            model.weight_ih_l0[0, 0] = float("nan")
        elif defect == "nan_input":
            x[0, 0, 0] = float("nan")
        elif defect == "double_master":
            model.double()
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid arithmetic must fail before any matrix multiply")
    monkeypatch.setattr(torch.nn.functional, "linear", forbidden)
    with pytest.raises(ValueError, match="GRU|masterweights"):
        model(x)


def test_lookup_tampering_is_rejected_at_explicit_snapshot_boundary():
    model = BatchedQATGRU.from_float(torch.nn.GRU(8, 8, batch_first=True))
    model.sigmoid_lut[128] += 1
    with pytest.raises(ValueError, match="lookup table"):
        model.integer_snapshots()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA hardware")
def test_cuda_eight_steps_exact_before_and_after_optimizer_with_autocast():
    torch.manual_seed(271)
    source = torch.nn.GRU(8, 16, batch_first=True, bidirectional=True)
    model = BatchedQATGRU.from_float(source).cuda()
    x, state = torch.randn(33, 8, 8, device="cuda") * .2, torch.randn(2, 33, 16, device="cuda") * .2
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    previous = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.float16):
                actual = model(x, state)
            assert torch.backends.cuda.matmul.fp32_precision == "tf32"
            _assert_codes(actual, _oracle(model.integer_snapshots(), x, state))
            optimizer.zero_grad()
            (actual[0].square().mean() + actual[1].square().mean()).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous
