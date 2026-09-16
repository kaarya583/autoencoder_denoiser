"""Fixed-grid training must export the exact graph it simulated."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from esp32_denoiser.gtcrn_integer import GTCRNIntegerDenoiser, calibrate_gtcrn_integer
from esp32_denoiser.gtcrn_integer_export import pack_gtcrn_integer, load_gtcrn_integer
from esp32_denoiser.gtcrn_native import CIntegerGTCRN
from esp32_denoiser.gtcrn_qat import GTCRNQAT
from esp32_denoiser.gtcrn_qat_ops import round_away
from test_gtcrn_integer import _source, _audio


@pytest.fixture(scope="module")
def prepared():
    torch.set_num_threads(1)
    source = _source()
    _, calibration = calibrate_gtcrn_integer(source, [_audio()], max_batches=1)
    return source, pack_gtcrn_integer(GTCRNIntegerDenoiser(source, calibration)), calibration


def _model(prepared):
    return GTCRNQAT.from_float(prepared[0], prepared[1], calibration=prepared[2])


def _features(model, frames=4, batch=2):
    # Integer-grid extremes exercise saturation and residual rounding, not
    # merely the narrow distribution of an untrained random network.
    codes = np.random.default_rng(870).integers(-128, 128, (batch, 3, frames, 129), dtype=np.int16).astype(np.int8)
    return torch.from_numpy(codes.astype(np.float32))*2.0**model.grids["erb.output"]


def _assert_native(model, features):
    snapshot = model.integer_snapshot(checkpoint_sha256="a"*64)
    blob = pack_gtcrn_integer(snapshot)
    loaded = load_gtcrn_integer(blob, calibration=snapshot.calibration)
    native = CIntegerGTCRN(blob)
    actual = model.forward_features(features).detach().numpy()
    for batch in range(features.shape[0]):
        state = None
        for frame in range(features.shape[2]):
            codes = (features[batch, :, frame].numpy()/2.0**model.grids["erb.output"]).astype(np.int8)
            mask, state = native.neural_step(codes, state)
            np.testing.assert_array_equal(actual[batch, :, frame], mask.astype(np.float32)/128)
    assert loaded.source_sha256 == "a"*64
    assert loaded.calibration["grids"] == model.grids
    assert len(blob) < 99_000
    return blob


def test_exact_c_graph_before_and_after_optimizer_update(prepared):
    model = _model(prepared).train()
    original = {name: value.clone() for name, value in prepared[0].state_dict().items()}
    frozen_bn = {name: value.clone() for name, value in model.master.named_buffers() if "running_" in name or "num_batches" in name}
    features = _features(model)
    before = _assert_native(model, features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    prediction = model.forward_features(features)
    loss = (prediction-torch.linspace(-.5, .5, 129)).square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert len(gradients) > 150 and all(torch.isfinite(value).all() for value in gradients)
    assert any(torch.count_nonzero(value) for value in gradients)
    optimizer.step()
    after = _assert_native(model, features)
    assert before != after
    restored = GTCRNQAT.from_checkpoint(model.checkpoint_payload())
    torch.testing.assert_close(restored.forward_features(features), model.forward_features(features), rtol=0, atol=0)
    assert all(torch.equal(value, dict(model.master.named_buffers())[name]) for name, value in frozen_bn.items())
    assert all(torch.equal(value, prepared[0].state_dict()[name]) for name, value in original.items())
    assert not any(layer.training for layer in model.modules() if isinstance(layer, nn.BatchNorm2d))


def test_batch_prefix_and_checkpoint_restore(prepared):
    model = _model(prepared)
    features = _features(model)
    complete, state = model.forward_features(features, return_state=True)
    separate = torch.cat([model.forward_features(row[None]) for row in features])
    torch.testing.assert_close(complete, separate, rtol=0, atol=0)
    torch.testing.assert_close(model.forward_features(features[:, :, :2]), complete[:, :, :2], rtol=0, atol=0)
    assert len(state) == 14
    payload = model.checkpoint_payload()
    restored = GTCRNQAT.from_checkpoint(payload)
    torch.testing.assert_close(restored.forward_features(features), complete, rtol=0, atol=0)
    assert pack_gtcrn_integer(restored.integer_snapshot()) == pack_gtcrn_integer(model.integer_snapshot())
    broken = deepcopy(payload)
    broken["current_master_sha256"] = "f"*64
    with pytest.raises(ValueError, match="fingerprint"):
        GTCRNQAT.from_checkpoint(broken)
    model.grids["erb.output"] += 1
    with pytest.raises(ValueError, match="recipe"):
        model.integer_snapshot()


def test_waveform_silence_finite_backward_and_shared_source_guard(prepared):
    model = _model(prepared).train()
    audio = torch.cat((_audio(513), torch.zeros(1, 513))).requires_grad_()
    output = model(audio)
    assert output.shape == audio.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(audio.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    changed = deepcopy(prepared[0])
    with torch.no_grad():
        next(changed.core.encoder.parameters()).add_(.01)
    with pytest.raises(ValueError, match="differs"):
        GTCRNQAT.from_float(changed, prepared[1], calibration=prepared[2])
    with pytest.raises(ValueError, match="features"):
        model.forward_features(torch.full((1, 3, 1, 129), float("nan")))


def test_rounding_immediately_below_half_is_not_tie():
    half = torch.tensor(.5)
    below = torch.nextafter(half, torch.tensor(0.))
    values = torch.stack((below, half, -below, -half))
    torch.testing.assert_close(round_away(values), torch.tensor([0., 1., 0., -1.]), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA acceptance runs in Colab")
def test_cuda_codes_match_cpu_and_backward(prepared):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        cpu = _model(prepared)
        cuda = GTCRNQAT.from_checkpoint(cpu.checkpoint_payload()).cuda()
        features = _features(cpu, frames=3)
        actual = cuda.forward_features(features.cuda())
        torch.testing.assert_close(actual.cpu(), cpu.forward_features(features), rtol=0, atol=0)
        actual.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in cuda.parameters() if p.grad is not None)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
