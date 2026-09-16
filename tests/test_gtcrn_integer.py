"""Independent graph gate, strict neural precision, and causal waveform framing."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from esp32_denoiser.experimental_gtcrn_ops import IntegerAffine, IntegerPReLU, IntegerStreamConv
from esp32_denoiser.gtcrn_integer import (
    GTCRNFloatShadow, GTCRNIntegerDenoiser, calibrate_gtcrn_integer,
    from_checkpoint_training, main,
)
from esp32_denoiser.gtcrn_model import GTCRNConfig, GTCRNDenoiser

# The existing audited hybrid fixture supplies genuine source/preparation
# hashes and disjoint tiny audio files; reuse it instead of inventing lineage.
from test_gtcrn_recurrent_probe import broad_checkpoint


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def _source(normalized=True, *, constant_mask=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8147)
        source = GTCRNDenoiser(GTCRNConfig(normalize_input=normalized)).eval()
        # Nondefault BN means, variances and signed scales exercise all folds
        # and decoder orientation without relying on an identity network.
        with torch.no_grad():
            for module in source.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean.uniform_(-.08, .08)
                    module.running_var.uniform_(.7, 1.3)
                    module.weight.uniform_(.7, 1.3)
                    module.bias.uniform_(-.05, .05)
            if constant_mask:
                head = source.core.decoder.de_convs[4].bn
                head.weight.zero_()
                head.bias.copy_(torch.tensor([10., 0.]))
    return source


def _audio(samples=769, batch=1):
    return torch.from_numpy(np.random.default_rng(314).normal(0, .04, (batch, samples)).astype(np.float32))


@pytest.fixture(scope="module")
def prepared():
    torch.set_num_threads(1)
    source = _source()
    _, calibration = calibrate_gtcrn_integer(source, [_audio()], max_batches=1)
    return source, calibration


@pytest.mark.parametrize("normalized", [False, True])
def test_float_shadow_matches_untouched_output_and_every_history(normalized):
    source = _source(normalized)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(157)
        spectra = torch.randn(1, 257, 18, 2) * .3
    spectra[:, (0, -1), :, 1] = 0
    shadow = GTCRNFloatShadow(source)
    result = shadow.verify(spectra.split(1, dim=2))
    assert result["frames"] == 18 and result["histories_checked_per_frame"] == 14
    assert result["max_abs_spectral_error"] < 2e-6
    assert result["max_abs_history_error"] < 2e-6
    complete, final = shadow.sequence(spectra)
    first, state = shadow.sequence(spectra[:, :, :7])
    second, continued = shadow.sequence(spectra[:, :, 7:], state)
    torch.testing.assert_close(torch.cat((first, second), 2), complete, rtol=2e-5, atol=2e-6)
    assert sum(value.numel() for value in final.values()) == 18_048
    for name in final:
        torch.testing.assert_close(final[name], continued[name], rtol=2e-5, atol=2e-6)
    # A later spectrum may alter later frames only, including frequency GRUs.
    changed = spectra.clone()
    changed[:, :, 7:] *= -3
    altered, _ = shadow.sequence(changed)
    torch.testing.assert_close(altered[:, :, :7], complete[:, :, :7], rtol=2e-5, atol=2e-6)


def test_neural_graph_has_only_integer_parameters_edges_and_state(prepared, monkeypatch):
    source, calibration = prepared
    snapshot = {key: value.clone() for key, value in source.state_dict().items()}
    integer = GTCRNIntegerDenoiser(source, calibration)
    # No learned Torch operation may be called after construction. FFT/ERB
    # are explicitly NumPy DSP; primitive preparation happened beforehand.
    def forbidden(*args, **kwargs):
        raise AssertionError("Floating learned operation called during integer inference")
    for kind in (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.GRU, nn.LayerNorm, nn.PReLU, nn.BatchNorm2d):
        monkeypatch.setattr(kind, "forward", forbidden)
    output, state = integer.stream_step(_audio(256).numpy()[0])
    assert np.isfinite(output).all() and np.any(output != 0)
    assert len(state.neural) == 14 and sum(value.nbytes for value in state.neural.values()) == 18_048
    assert all(value.dtype == np.int8 for value in state.neural.values())
    assert len(integer.backend.edges) == 188
    for operation in integer.backend.ops.values():
        if isinstance(operation, IntegerStreamConv):
            operation = operation.affine
        if isinstance(operation, IntegerAffine):
            assert operation.weights.dtype == operation.exponents.dtype == np.int8
            assert operation.bias.dtype == np.int32
        elif isinstance(operation, IntegerPReLU):
            assert operation.slopes.dtype == np.int8
        elif isinstance(operation, tuple):
            for cell in operation:
                assert cell.weight_ih.dtype == cell.weight_hh.dtype == np.int8
                assert cell.bias_ih.dtype == cell.bias_hh.dtype == np.int32
                assert cell.sigmoid_lut.dtype == cell.tanh_lut.dtype == np.int8
        else:
            assert operation.gamma.dtype == np.int8 and operation.beta.dtype == np.int32
    assert not any(isinstance(value, nn.Module) for value in vars(integer.backend).values())
    assert all(torch.equal(value, source.state_dict()[key]) for key, value in snapshot.items())
    stats = integer.model_stats()
    assert stats["neural_state_bytes_int8"] == 18_048
    assert stats["erb_payload_bytes"] == 2040
    assert stats["parameter_and_lut_array_bytes"] + 2040 + stats["window_bytes"] < 99_000
    assert "excludes graph/grid descriptors" in stats["scope"]
    assert "NumPy reference only" in stats["deployment_status"]


@pytest.mark.parametrize("samples", [1, 255, 256, 257, 1031])
def test_whole_audio_exact_streaming_reset_and_flush(prepared, samples):
    integer = GTCRNIntegerDenoiser(*prepared)
    audio = _audio(samples).numpy()
    expected = integer(audio)
    state, output = integer.initial_state(), []
    for chunk in np.pad(audio[0], (0, (-samples) % 256 + 256)).reshape(-1, 256):
        value, state = integer.stream_step(chunk, state)
        output.append(value)
    actual = np.concatenate(output)[256:256+samples][None]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(integer(audio), expected)
    torch.testing.assert_close(integer(torch.from_numpy(audio)), torch.from_numpy(expected), rtol=0, atol=0)
    assert expected.shape == (1, samples)


@pytest.mark.parametrize("normalized", [False, True])
def test_constant_real_mask_proves_dsp_alignment_dc_nyquist_and_q7_endpoint(normalized):
    source = _source(normalized, constant_mask=True)
    waveform = _audio(1031)
    waveform[:, 0] += .3
    waveform[:, -1] -= .2
    waveform += torch.arange(1031).remainder(2) * .02 + .03
    _, calibration = calibrate_gtcrn_integer(source, [waveform], max_batches=1)
    integer = GTCRNIntegerDenoiser(source, calibration)
    # Tanh's positive endpoint is 127/128 in strict Q7. Dense and sparse ERB
    # transpose weights form a partition of unity, including first/last bins.
    torch.testing.assert_close(integer(waveform), waveform * (127/128), rtol=3e-6, atol=7e-8)
    torch.testing.assert_close(source(waveform), waveform, rtol=3e-6, atol=7e-8)
    silence = np.zeros((1, 513), np.float32)
    np.testing.assert_array_equal(integer(silence), silence)


def test_state_is_caller_owned_and_future_audio_does_not_change_emitted_prefix(prepared):
    integer = GTCRNIntegerDenoiser(*prepared)
    _, state = integer.stream_step(_audio(256).numpy()[0])
    original = deepcopy(state)
    chunk = _audio(256).numpy()[0] * 2
    first, next_state = integer.stream_step(chunk, state)
    repeated, repeated_state = integer.stream_step(chunk, state)
    np.testing.assert_array_equal(first, repeated)
    for name in state.neural:
        np.testing.assert_array_equal(state.neural[name], original.neural[name])
        np.testing.assert_array_equal(next_state.neural[name], repeated_state.neural[name])
    for key in ("analysis", "synthesis", "synthesis_weight"):
        np.testing.assert_array_equal(getattr(state, key), getattr(original, key))
    audio = _audio(1024)
    changed = torch.cat([audio[:, :768], _audio(519) * -3], 1)
    torch.testing.assert_close(integer(audio)[:, :512], integer(changed)[:, :512], rtol=0, atol=0)


def test_invalid_inputs_and_malformed_histories_rejected(prepared):
    integer = GTCRNIntegerDenoiser(*prepared)
    for value in (np.zeros(256, np.float64), np.zeros(255, np.float32), np.full(256, np.nan, np.float32)):
        with pytest.raises(ValueError, match="256 finite float32"):
            integer.stream_step(value)
    for value in (np.zeros(257, np.float32), np.zeros(256, np.complex64), np.full(257, np.nan, np.complex64)):
        with pytest.raises(ValueError, match="257 bins"):
            integer.spectrum_frame(value)
    spectrum = np.zeros(257, np.complex64)
    _, valid = integer.spectrum_frame(spectrum)
    for state in ([1], {"unknown": np.zeros(1, np.int8)}, {next(iter(valid)): next(iter(valid.values()))}):
        with pytest.raises(ValueError, match="complete named"):
            integer.spectrum_frame(spectrum, state)
    for invalid in (np.zeros((1,), np.int8), next(iter(valid.values())).astype(np.float32)):
        state = dict(valid)
        state[next(iter(state))] = invalid
        with pytest.raises(ValueError, match="Invalid INT8"):
            integer.spectrum_frame(spectrum, state)
    for wave in (np.zeros((1, 0), np.float32), np.zeros((2,), np.float32), np.full((1, 8), np.inf, np.float32)):
        with pytest.raises(ValueError, match="waveforms"):
            integer(wave)


def test_calibration_is_deterministic_bounded_and_source_bound(prepared):
    source, calibration = prepared
    consumed = []
    def batches():
        consumed.append(1)
        yield _audio()
        raise AssertionError("Consumed more than max_batches")
    state = torch.random.get_rng_state().clone()
    grids, repeated = calibrate_gtcrn_integer(source, batches(), max_batches=1)
    assert consumed == [1] and repeated == calibration and grids == calibration["grids"]
    assert torch.equal(state, torch.random.get_rng_state())
    assert repeated["frames"] == 5 and repeated["batches"] == repeated["utterances"] == 1
    assert len(repeated["recipes"]) == 84 and len(repeated["grids"]) == 182
    assert len(repeated["probability_encodings"]) == 6
    integer = GTCRNIntegerDenoiser(source, repeated)
    for name in repeated["probability_encodings"]:
        assert name not in grids
        with pytest.raises(ValueError, match="Probability edge"):
            integer.backend.grid(name)
    for name, recipe in repeated["recipes"].items():
        if recipe["kind"] == "gru":
            assert -12 <= grids[name + ".input"] <= 0 and grids[name + ".output"] == -7
    changed = deepcopy(source)
    with torch.no_grad():
        changed.core.encoder.en_convs[0].conv.bias[0] += .001
    with pytest.raises(ValueError, match="differs from the frozen source"):
        GTCRNIntegerDenoiser(changed, repeated)
    bad = deepcopy(repeated)
    bad["grids"].pop("encoder.en_convs.0.conv.input")
    with pytest.raises(ValueError, match="Missing calibrated"):
        GTCRNIntegerDenoiser(source, bad)
    bad = deepcopy(repeated)
    bad["grids"]["dpgrnn1.intra_rnn.rnn1.output"] = -6
    with pytest.raises(ValueError, match="fixed Q7"):
        GTCRNIntegerDenoiser(source, bad)
    for invalid in (source.train(), deepcopy(source).eval().double()):
        with pytest.raises(ValueError, match="CPU float32 eval"):
            GTCRNFloatShadow(invalid)
    source.eval()
    with pytest.raises(ValueError, match="At least one"):
        calibrate_gtcrn_integer(source, [])
    with pytest.raises(ValueError, match="finite CPU"):
        calibrate_gtcrn_integer(source, [torch.full((1, 256), float("nan"))])


def test_attention_gru_grid_overflow_fails_instead_of_silent_narrowing():
    source = _source()
    with torch.no_grad():
        layer = source.core.encoder.en_convs[2].point_bn2
        layer.weight.zero_()
        layer.bias.fill_(20)  # Mean-square energy400 needs exponent2, unsupported.
    with pytest.raises(ValueError, match="Calibrated GRU edge exceeds supported input grid"):
        calibrate_gtcrn_integer(source, [_audio(256)], max_batches=1)


def test_checkpoint_derived_calibration_replays_real_hybrid_and_binds_file(broad_checkpoint, tmp_path):
    source, saved, validation = broad_checkpoint
    saved["model_config"]["normalize_input"] = True
    checkpoint = tmp_path / "frozen.pt"
    torch.save(saved, checkpoint)
    integer, control, report = from_checkpoint_training(checkpoint, validation, crops=4, seed=483)
    assert report["recipe"] == "checkpoint training policy"
    assert report["paired_crops"] > 0 and report["synthetic_crops"] > 0
    assert report["paired_crops"] + report["synthetic_crops"] == 4
    assert report["source_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert integer.source_sha256 == report["source_checkpoint_sha256"]
    assert control.config.normalize_input and not control.training
    for crop in report["crops"]:
        assert len(crop["noisy_crop_sha256"]) == 64
        assert all(len(asset["sha256"]) == 64 for asset in crop["sources"])
    assert bool(torch.isfinite(integer(_audio(257))).all())
    saved["phase"] = "qat"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="frozen float"):
        from_checkpoint_training(checkpoint, validation, crops=1)


def test_real_cli_reports_common_waveforms_and_sealed_test_rejection(broad_checkpoint, tmp_path, monkeypatch):
    _, saved, validation = broad_checkpoint
    saved["model_config"]["normalize_input"] = True
    checkpoint, output = tmp_path / "frozen.pt", tmp_path / "whole_integer.json"
    torch.save(saved, checkpoint)
    monkeypatch.setattr("sys.argv", ["integer", "--checkpoint", str(checkpoint), "--manifest", str(validation),
                                    "--output", str(output), "--calibration-crops", "2", "--max-utterances", "1"])
    main()
    result = json.loads(output.read_text())
    assert result["float"]["utterances"][0]["audio_sha256"] == result["integer"]["utterances"][0]["audio_sha256"]
    assert result["float"]["utterances"][0]["samples"] == result["integer"]["utterances"][0]["samples"]
    assert result["model_stats"]["neural_state_bytes_int8"] == 18_048
    assert "no QAT recovery" in result["scope"]
    rows = [dict(json.loads(line), source_split="test") for line in validation.read_text().splitlines()]
    validation.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output.unlink()
    with pytest.raises(ValueError, match="test remains sealed"):
        main()
    assert not output.exists()
