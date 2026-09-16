"""Direction/reset parity and auditable GRU-only waveform sensitivity."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from esp32_denoiser.experimental_gru import GRUQuantizationConfig
from esp32_denoiser.gtcrn_model import GTCRNConfig, GTCRNDenoiser
from esp32_denoiser.gtcrn_recurrent_probe import (
    GRUSequenceProbe, GTCRNRecurrentProbe, _audit_calibration,
    calibrate_checkpoint_training, calibrate_gru_input_configs, evaluate_recurrent_probe, main,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize("bidirectional,batch_first", [(False, True), (True, True), (True, False)])
def test_direction_order_matches_float_and_fake_integer_are_exact(bidirectional, batch_first):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(739)
        source = nn.GRU(8, 4, bidirectional=bidirectional, batch_first=batch_first).eval()
        inputs = torch.randn(3, 9, 8) * .3
        hx = torch.randn(1 + int(bidirectional), 3, 4) * .2
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    integer = GRUSequenceProbe(source)
    fake = GRUSequenceProbe(source, backend="fake_quant")
    actual, final = integer(inputs, hx)
    simulated, simulated_final = fake(inputs, hx)
    torch.testing.assert_close(actual, simulated, rtol=0, atol=0)
    torch.testing.assert_close(final, simulated_final, rtol=0, atol=0)
    for dtype in (torch.bfloat16, torch.float64):
        # The integer boundary accepts finite floating CPU tensors regardless
        # of storage dtype, including leaves that require gradients.
        encoded_input = inputs.to(dtype).requires_grad_()
        typed, typed_final = integer(encoded_input, hx.to(dtype))
        typed_fake, typed_fake_final = fake(encoded_input, hx.to(dtype))
        torch.testing.assert_close(typed, typed_fake, rtol=0, atol=0)
        torch.testing.assert_close(typed_final, typed_fake_final, rtol=0, atol=0)
        assert not typed.requires_grad
    # Reuse the same sequence traversal with each cell's unquantized formula.
    # This detects reverse ordering, grouped state, and reset-after mistakes.
    for cell in fake.fake_cells:
        cell.forward = cell.forward_float
    with torch.no_grad():
        expected, expected_final = source(inputs, hx)
    control, control_final = fake(inputs, hx)
    torch.testing.assert_close(control, expected, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(control_final, expected_final, rtol=2e-6, atol=1e-7)
    assert actual.shape == expected.shape and final.shape == expected_final.shape
    assert torch.equal(final * 128, (final * 128).round())


def test_integer_chunk_state_continuity_reset_and_gate_counters():
    source = nn.GRU(8, 8, batch_first=True).eval()
    probe = GRUSequenceProbe(source)
    inputs = torch.randn(2, 27, 8) * .5
    inputs[:, 10:17] = 300  # Deliberately clips the input grid.
    complete, final = probe(inputs)
    probe.reset_statistics()
    first, state = probe(inputs[:, :11])
    second, last = probe(inputs[:, 11:], state)
    torch.testing.assert_close(torch.cat((first, second), 1), complete, rtol=0, atol=0)
    torch.testing.assert_close(last, final, rtol=0, atol=0)
    statistics = probe.statistics()["directions"]["forward"]
    assert statistics["counts"]["sequence_steps"] == 27
    assert statistics["counts"]["input_clipped"] == 2 * 7 * 8
    assert statistics["counts"]["state_values"] == 2 * 27 * 8
    for counter in statistics["gates"].values():
        assert counter["values"] == 2 * 27 * 8
        assert 0 <= counter["low_endpoint"] + counter["high_endpoint"] <= counter["values"]
    probe.reset_statistics()
    replay, replay_final = probe(inputs)
    torch.testing.assert_close(replay, complete, rtol=0, atol=0)
    torch.testing.assert_close(replay_final, final, rtol=0, atol=0)
    with pytest.raises(ValueError, match="hidden state"):
        probe(inputs, torch.zeros(2, 2, 8))
    with pytest.raises(ValueError, match="input must"):
        probe(torch.full((2, 1, 8), float("nan")))
    with pytest.raises(ValueError, match="single GRU"):
        GRUSequenceProbe(nn.GRU(8, 8, num_layers=2))


@pytest.mark.parametrize("normalized", [False, True])
def test_full_probe_replaces_only_grus_preserves_source_and_causality(normalized):
    source = GTCRNDenoiser(GTCRNConfig(normalize_input=normalized)).train()
    original = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    probe = GTCRNRecurrentProbe(source)
    assert source.training and all(p.requires_grad for p in source.core.encoder.parameters())
    assert not probe.training and all(not p.requires_grad for p in probe.parameters())
    assert probe.config.normalize_input is normalized
    assert not any(isinstance(module, nn.GRU) for module in probe.modules())
    assert len(probe.replacements) == 14
    assert probe.statistics()["gru_directions"] == 18
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, original[name])
    prefix = torch.randn(1, 768) * .04
    noisy = torch.cat((prefix, torch.randn(1, 256) * .1), -1)
    changed = torch.cat((prefix, torch.randn(1, 513) * .2), -1)
    output = probe(noisy)
    assert output.shape == noisy.shape and bool(torch.isfinite(output).all())
    torch.testing.assert_close(probe(noisy), output, rtol=0, atol=0)
    torch.testing.assert_close(probe(changed)[:, :512], output[:, :512], rtol=2e-4, atol=2e-6)
    inter = GTCRNRecurrentProbe(source, groups=("inter",))
    assert inter.statistics()["gru_modules"] == 4
    assert sum(isinstance(m, nn.GRU) for m in inter.modules()) == 10
    with pytest.raises(ValueError, match="groups"):
        GTCRNRecurrentProbe(source, groups=("unknown",))


def test_calibration_restores_mode_and_hooks_and_reports_scale_limits():
    source = GTCRNDenoiser().train()
    original = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    configs, report = calibrate_gru_input_configs(source, [torch.randn(1, 1024) * 100], max_batches=1)
    assert source.training and report["batches"] == 1 and len(configs) == 14
    assert all(not module._forward_pre_hooks for module in source.modules())
    assert all(torch.equal(original[name], tensor) for name, tensor in source.state_dict().items())
    assert all(-12 <= config.input_exponent <= 0 for config in configs.values())
    assert any(row["observed_peak_exceeds_grid"] for row in report["modules"].values())
    with pytest.raises(ValueError, match="at least one"):
        calibrate_gru_input_configs(source, [])
    assert source.training and all(not module._forward_pre_hooks for module in source.modules())
    with pytest.raises(ValueError, match="Nonfinite"):
        calibrate_gru_input_configs(source, [torch.full((1, 768), float("nan"))])
    assert source.training and all(not module._forward_pre_hooks for module in source.modules())


def _manifest(tmp_path, name, speaker, *, silent=False, split="train"):
    records = []
    for index in range(2):
        samples = 512 + index * 256
        time = np.arange(samples) / 16000
        clean = np.zeros(samples) if silent and index == 1 else .06 * np.sin(2 * np.pi * 330 * time)
        noisy = clean + .013 * np.cos(2 * np.pi * 1420 * time)
        row = dict(id=f"{speaker}_{index}", speaker=speaker, source_split=split,
                   sample_rate=16000, samples=samples)
        for role, values in (("clean", clean), ("noisy", noisy)):
            path = tmp_path / f"{name}_{index}_{role}.wav"
            sf.write(path, values.astype(np.float32), 16000, subtype="FLOAT")
            row[role] = str(path)
        records.append(row)
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return path, records


def test_paired_cohort_same_valid_denominator_and_calibration_guards(tmp_path):
    manifest, evaluation = _manifest(tmp_path, "val", "p226", silent=True)
    _, training = _manifest(tmp_path, "train", "p225")
    source = GTCRNDenoiser().eval()
    result = evaluate_recurrent_probe(source, manifest, max_utterances=2, groups=("inter",))
    assert result["paired_summary"]["valid_utterances"] == 1
    assert result["paired_summary"]["invalid_utterances"] == 1
    assert result["paired_utterances"][1]["probe_minus_float_db"] is None
    first = result["paired_utterances"][0]
    assert result["paired_summary"]["probe_minus_float_db"] == first["recurrent_int8_si_sdri"] - first["float_si_sdri"]
    assert result["float"]["manifest_sha256"] == result["recurrent_int8"]["manifest_sha256"]
    _audit_calibration(training, evaluation)
    for key in ("id", "speaker", "noisy"):
        bad = [dict(row) for row in training]
        bad[0][key] = evaluation[0][key]
        with pytest.raises(ValueError, match="overlap"):
            _audit_calibration(bad, evaluation)
    bad = [dict(row, source_split="development") for row in training]
    with pytest.raises(ValueError, match="training-origin"):
        _audit_calibration(bad, evaluation)
    test_manifest, _ = _manifest(tmp_path, "official_test", "p232", split="test")
    with pytest.raises(ValueError, match="test remains sealed"):
        evaluate_recurrent_probe(source, test_manifest)


def test_real_cli_checkpoint_calibration_and_json_provenance(tmp_path, monkeypatch):
    calibration, _ = _manifest(tmp_path, "train", "p225")
    evaluation, _ = _manifest(tmp_path, "val", "p226")
    source = GTCRNDenoiser(GTCRNConfig(normalize_input=True)).eval()
    checkpoint = tmp_path / "fresh.pt"
    torch.save({"model_kind": "gtcrn", "model_config": asdict(source.config),
                "model": source.state_dict(), "phase": "float", "epoch": 1}, checkpoint)
    output = tmp_path / "reports" / "probe.json"
    monkeypatch.setattr("sys.argv", ["probe", "--checkpoint", str(checkpoint), "--manifest", str(evaluation),
                                    "--output", str(output), "--calibration-manifest", str(calibration),
                                    "--calibration-utterances", "1", "--calibration-crop-seconds", ".032",
                                    "--max-utterances", "1", "--groups", "attention", "--threads", "1"])
    main()
    result = json.loads(output.read_text())
    assert len(result["checkpoint"]["sha256"]) == len(result["calibration"]["manifest_sha256"]) == 64
    assert result["model_config"]["normalize_input"] is True
    assert result["paired_summary"]["valid_utterances"] == 1
    assert len(result["calibration"]["ids"]) == 1
    assert result["recurrent_statistics"]["gru_modules"] == 6
    assert all(row["directions"]["forward"]["counts"]["state_values"] > 0
               for row in result["recurrent_statistics"]["modules"].values())


@pytest.fixture
def broad_checkpoint(tmp_path):
    from esp32_denoiser.extra_data import SOURCES, VERSION, _asset_identity
    from esp32_denoiser.train import TrainConfig

    paired, _ = _manifest(tmp_path, "paired_train", "p225")
    validation, _ = _manifest(tmp_path, "paired_val", "p226")
    directory = tmp_path / "extra" / "manifests"
    directory.mkdir(parents=True)
    preparation = {"version": VERSION, "sample_rate": 16000, "splits": {}}
    for kind in ("speech", "noise"):
        for split, index in (("train", 101), ("val", 202)):
            member = (f"LibriSpeech/train-clean-100/{index}/10/{index}-10-0000.flac" if kind == "speech" else
                      f"musan/noise/free-sound/noise-{index}.wav")
            identifier, group, speaker = _asset_identity(kind, member)
            path = directory / f"{kind}_{split}{'.flac' if kind == 'speech' else '.wav'}"
            audio = np.sin(np.arange(1100) * ((.09 if kind == "speech" else .42) + index / 10000)) * .1
            sf.write(path, audio, 16000)
            row = {"id": identifier, "group": group, "speaker": speaker, "kind": kind, "split": split,
                   "source": SOURCES[kind].source, "license": SOURCES[kind].license, "member": member,
                   "path": path.name, "samples": len(audio), "sample_rate": 16000,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest = directory / f"{kind}_{split}.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            preparation["splits"][f"{kind}_{split}"] = {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "records": 1, "groups": [group]}
    (directory / "provenance.json").write_text(json.dumps(preparation))
    source = GTCRNDenoiser().train()
    config = TrainConfig(str(paired), str(validation), str(tmp_path / "run"), model_kind="gtcrn",
                         crop_seconds=.048, synthetic_speech_manifest=str(directory / "speech_train.jsonl"),
                         synthetic_noise_manifest=str(directory / "noise_train.jsonl"), synthetic_probability=.5,
                         epoch_samples=10)
    module_root = Path(__file__).parents[1] / "esp32_denoiser"
    checkpoint = {"model_kind": "gtcrn", "model_config": asdict(source.config), "model": source.state_dict(),
                  "phase": "float", "train_config": asdict(config), "provenance": {
                      "test_used_for_selection": False,
                      "clean_identity_probability": config.clean_identity_probability,
                      "manifest_sha256": {"train": hashlib.sha256(paired.read_bytes()).hexdigest(),
                                          "val": hashlib.sha256(validation.read_bytes()).hexdigest()},
                      "source_sha256": {name: hashlib.sha256((module_root / name).read_bytes()).hexdigest()
                                        for name in ("data.py", "extra_data.py", "mixtures.py", "train.py")},
                      "added_training_sources": {
                          "speech_manifest_sha256": preparation["splits"]["speech_train"]["manifest_sha256"],
                          "noise_manifest_sha256": preparation["splits"]["noise_train"]["manifest_sha256"],
                          "speech_recordings": 1, "noise_recordings": 1,
                          "synthetic_probability": .5, "samples_per_epoch": 10}}}
    return source, checkpoint, validation


def test_checkpoint_hybrid_calibration_reproduces_draws_hashes_and_rng(broad_checkpoint):
    source, checkpoint, validation = broad_checkpoint
    state = torch.random.get_rng_state().clone()
    configs, report = calibrate_checkpoint_training(source, checkpoint, validation, crops=32, seed=71)
    assert source.training and torch.equal(state, torch.random.get_rng_state())
    assert report["paired_crops"] + report["synthetic_crops"] == 32
    assert report["paired_crops"] > 0 and report["synthetic_crops"] > 0
    assert report["synthetic_probability"] == .5
    assert report["source_audit"]["partitions"]["speech"]["train"]["recordings"] == 1
    assert len({r["seed"] for r in report["crops"]}) == 32
    for row in report["crops"]:
        assert len(row["noisy_crop_sha256"]) == len(row["clean_crop_sha256"]) == 64
        for asset in row["sources"]:
            assert asset["sha256"] == hashlib.sha256(Path(asset["path"]).read_bytes()).hexdigest()
            assert asset["matches_preparation_hash"] is (row["kind"] == "synthetic")
    repeated_configs, repeated = calibrate_checkpoint_training(source, checkpoint, validation, crops=32, seed=71)
    assert repeated_configs == configs and repeated == report
    assert torch.equal(state, torch.random.get_rng_state())


def test_checkpoint_calibration_legacy_identity_recipe_replays_explicit_default(broad_checkpoint):
    source, checkpoint, validation = broad_checkpoint
    configs, report = calibrate_checkpoint_training(source, checkpoint, validation, crops=32, seed=71)
    del checkpoint["train_config"]["clean_identity_probability"]
    del checkpoint["provenance"]["clean_identity_probability"]
    legacy_configs, legacy = calibrate_checkpoint_training(source, checkpoint, validation, crops=32, seed=71)
    assert report["clean_identity_probability"] == .03
    assert legacy_configs == configs and legacy == report


@pytest.mark.parametrize("probability", [0., .15, 1.])
def test_checkpoint_calibration_identity_recipe_controls_both_sources(broad_checkpoint, probability):
    source, checkpoint, validation = broad_checkpoint
    checkpoint["train_config"]["clean_identity_probability"] = probability
    checkpoint["provenance"]["clean_identity_probability"] = probability
    _, report = calibrate_checkpoint_training(source, checkpoint, validation, crops=32, seed=71)
    assert report["clean_identity_probability"] == probability
    assert report["source_audit"]["clean_identity_probability"] == probability
    for kind in ("paired", "synthetic"):
        crops = [row for row in report["crops"] if row["kind"] == kind]
        assert crops
        identities = [row["clean_crop_sha256"] == row["noisy_crop_sha256"] for row in crops]
        if probability in (0., 1.):
            assert all(identity is bool(probability) for identity in identities)
        else:
            assert any(identities) and not all(identities)


@pytest.mark.parametrize("recorded", [None, .15])
def test_checkpoint_calibration_rejects_unverified_identity_recipe(broad_checkpoint, recorded):
    source, checkpoint, validation = broad_checkpoint
    if recorded is None:
        del checkpoint["provenance"]["clean_identity_probability"]
    else:
        checkpoint["provenance"]["clean_identity_probability"] = recorded
    with pytest.raises(ValueError, match="clean_identity_probability"):
        calibrate_checkpoint_training(source, checkpoint, validation, crops=1)
    assert source.training


def test_checkpoint_calibration_rejects_changed_sources_and_development_noise(broad_checkpoint):
    source, checkpoint, validation = broad_checkpoint
    original = validation.read_text()
    noise_manifest = Path(checkpoint["train_config"]["synthetic_noise_manifest"])
    noise = json.loads(noise_manifest.read_text())
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]["noise_id"] = noise["id"]
    development = validation.with_name("development.jsonl")
    development.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="underlying development id"):
        calibrate_checkpoint_training(source, checkpoint, development, crops=1)
    paired = Path(checkpoint["train_config"]["train_manifest"])
    paired.write_text(paired.read_text() + "\n")
    with pytest.raises(ValueError, match="train manifest hash"):
        calibrate_checkpoint_training(source, checkpoint, validation, crops=1)
    paired.write_text(paired.read_text().removesuffix("\n"))
    noise_path = noise_manifest.parent / noise["path"]
    sf.write(noise_path, np.cos(np.arange(1100) * .15) * .1, 16000)
    with pytest.raises(ValueError, match="source bytes changed"):
        calibrate_checkpoint_training(source, checkpoint, validation, crops=32, seed=71)
    assert source.training and all(not m._forward_pre_hooks for m in source.modules())


def test_checkpoint_recipe_cli_uses_hybrid_calibration(broad_checkpoint, tmp_path, monkeypatch):
    _, checkpoint, validation = broad_checkpoint
    path = tmp_path / "hybrid.pt"
    torch.save(checkpoint, path)
    output = tmp_path / "hybrid_probe.json"
    monkeypatch.setattr("sys.argv", ["probe", "--checkpoint", str(path), "--manifest", str(validation),
                                    "--output", str(output), "--calibration-from-checkpoint",
                                    "--calibration-utterances", "4", "--max-utterances", "1",
                                    "--groups", "inter", "--threads", "1"])
    main()
    result = json.loads(output.read_text())
    assert result["calibration"]["recipe"] == "checkpoint training policy"
    assert result["calibration"]["paired_crops"] + result["calibration"]["synthetic_crops"] == 4
    assert result["calibration"]["crop_seconds"] == .048
    assert result["recurrent_statistics"]["gru_modules"] == 4


def test_layer_norm_probe_matches_integer_and_fake_with_collapse_and_clipping():
    from esp32_denoiser.experimental_layer_norm import FakeQuantLayerNorm, integer_layer_norm
    from esp32_denoiser.gtcrn_quantization_probe import LayerNormGrid, LayerNormProbe

    source = nn.LayerNorm((33, 16), eps=1e-8)
    with torch.no_grad():
        source.weight.fill_(1.5)
        source.weight[12:].neg_()
        source.bias.fill_(.25)
    probe = LayerNormProbe(source, LayerNormGrid(-4, -4))
    values = torch.zeros(1, 4, 33, 16)
    values[:, 1] = torch.randn(33, 16) * 1e-7  # Quantized variance collapses.
    values[:, 2, 0, 0] = 4.0  # Affine normalized outlier exceeds output grid.
    values[:, 3, ::2] = -100
    values[:, 3, 1::2] = 100
    encoded = torch.sign(values) * torch.floor((values * 16).abs() + .5)
    encoded = encoded.clamp(-128, 127).to(torch.int8).numpy()
    expected = torch.from_numpy(integer_layer_norm(encoded, probe.snapshot).astype(np.float32)) / 16
    actual = probe(values.requires_grad_())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad
    fake = FakeQuantLayerNorm.from_float(source, input_exponent=-4, output_exponent=-4)
    torch.testing.assert_close(actual, fake(values), rtol=0, atol=0)
    statistics = probe.statistics()
    assert statistics["counts"]["frames"] == 4
    assert statistics["counts"]["zero_variance_frames"] == 2
    assert statistics["counts"]["float_zero_variance_frames"] == 1
    assert statistics["counts"]["quantization_collapsed_frames"] == 1
    assert statistics["counts"]["input_clipped"] == 528
    assert statistics["counts"]["output_clipped"] > 0
    assert statistics["counts"]["output_at_rail"] >= statistics["counts"]["output_clipped"]
    assert statistics["errors"]["output_mean_absolute_error"] > 0
    assert statistics["integer_maxima"]["max_squared_denominator"] <= statistics["integer_bounds"]["max_squared_denominator"]
    probe.reset_statistics()
    torch.testing.assert_close(probe(values), actual, rtol=0, atol=0)
    assert probe.statistics() == statistics
    with pytest.raises(ValueError, match="ending in"):
        probe(torch.zeros(1, 16, 33))


def test_joint_calibration_observes_same_crops_and_cleans_up_hooks():
    from esp32_denoiser.gtcrn_quantization_probe import calibrate_operator_configs

    source = GTCRNDenoiser().train()
    batches = [torch.randn(1, 768) * .05, torch.randn(1, 1024) * .08]
    configs, report = calibrate_operator_configs(source, batches, max_batches=2)
    standalone, _ = calibrate_gru_input_configs(source, batches, max_batches=2)
    assert source.training and configs["gru"] == standalone
    assert len(configs["layer_norm"]) == 4 and report["batches"] == 2
    for name, grid in configs["layer_norm"].items():
        assert report["layer_norm"][name]["input_peak"] <= 127 * 2.0 ** grid.input_exponent
        assert report["layer_norm"][name]["output_peak"] <= 127 * 2.0 ** grid.output_exponent
    with pytest.raises(ValueError, match="Nonfinite"):
        calibrate_operator_configs(source, [torch.full((1, 768), float("nan"))])
    assert source.training
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in source.modules())


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("seed", [0, 9, 23])
def test_partial_operator_graphs_reset_and_preserve_framewise_causality(normalized, seed):
    from esp32_denoiser.gtcrn_quantization_probe import GTCRNOperatorProbe, calibrate_operator_configs

    torch.manual_seed(seed)
    source = GTCRNDenoiser(GTCRNConfig(normalize_input=normalized)).eval()
    original = {name: value.clone() for name, value in source.state_dict().items()}
    audio = torch.randn(1, 1024) * .04
    configs, _ = calibrate_operator_configs(source, [audio], max_batches=1)
    for combined in (False, True):
        probe = GTCRNOperatorProbe(source, configs["layer_norm"], gru_configs=configs["gru"], combined=combined)
        assert len(probe.norms) == 4
        assert not any(isinstance(m, nn.LayerNorm) for m in probe.modules())
        assert sum(isinstance(m, nn.GRU) for m in probe.modules()) == (0 if combined else 14)
        assert not any(p.requires_grad for p in probe.parameters())
        result = probe(audio)
        assert result.shape == audio.shape and bool(torch.isfinite(result).all())
        statistics = probe.statistics()
        probe.reset_statistics()
        torch.testing.assert_close(probe(audio), result, rtol=0, atol=0)
        assert probe.statistics() == statistics
        # Isolate future-content dependence at a fixed tensor shape. Changing
        # sequence length can select different float kernels in this partial
        # probe; their rounding differences can cross INT8 thresholds.
        changed = torch.cat((audio[:, :768], torch.randn_like(audio[:, 768:]) * .2), -1)
        torch.testing.assert_close(probe(changed)[:, :512], result[:, :512], rtol=2e-4, atol=2e-6)
        assert all(row["counts"]["frames"] > 0 for row in statistics["layer_norm"].values())
        assert (statistics["gru"] is not None) is combined
    assert all(torch.equal(original[name], value) for name, value in source.state_dict().items())


def test_four_way_comparison_has_identical_audio_and_common_denominator(tmp_path):
    from esp32_denoiser.gtcrn_quantization_probe import calibrate_operator_configs, evaluate_operator_probes

    source = GTCRNDenoiser().eval()
    configs, _ = calibrate_operator_configs(source, [torch.randn(1, 1024) * .04], max_batches=1)
    manifest, _ = _manifest(tmp_path, "four_way", "p226", silent=True)
    result = evaluate_operator_probes(source, manifest, configs, max_utterances=2)
    assert result["common_summary"]["valid_utterances"] == 1
    assert result["common_summary"]["invalid_utterances"] == 1
    assert set(result["evaluations"]) == {"float", "gru_only", "layer_norm_only", "combined"}
    control = result["evaluations"]["float"]["utterances"]
    for name, report in result["evaluations"].items():
        assert [row["id"] for row in report["utterances"]] == result["cohort"]["ids"]
        assert [row["audio_sha256"] for row in report["utterances"]] == [row["audio_sha256"] for row in control]
        assert result["common_summary"]["si_sdri"][name] == report["utterances"][0]["si_sdri"]


def test_four_way_cli_reuses_actual_training_hybrid_calibration(broad_checkpoint, tmp_path, monkeypatch):
    from esp32_denoiser.gtcrn_quantization_probe import main as operator_main

    _, checkpoint, validation = broad_checkpoint
    path = tmp_path / "joint.pt"
    torch.save(checkpoint, path)
    output = tmp_path / "operator_probe.json"
    monkeypatch.setattr("sys.argv", ["probe", "--checkpoint", str(path), "--manifest", str(validation),
                                    "--output", str(output), "--calibration-crops", "4",
                                    "--max-utterances", "1", "--threads", "1"])
    operator_main()
    result = json.loads(output.read_text())
    assert result["calibration"]["paired_crops"] + result["calibration"]["synthetic_crops"] == 4
    assert result["calibration"]["synthetic_probability"] == .5
    assert len(result["calibration"]["layer_norm"]) == 4
    assert result["common_summary"]["valid_utterances"] == 1
    assert set(result["common_summary"]["minus_float_db"]) == {"gru_only", "layer_norm_only", "combined"}
    assert all(len(value) == 64 for value in result["probe_source_sha256"].values())
