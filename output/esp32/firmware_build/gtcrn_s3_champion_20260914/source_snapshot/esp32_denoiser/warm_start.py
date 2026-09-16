"""Verified zero-head initialization for the float deep-filter ablation."""

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile

import torch

from .frequency_deep_filter import FrequencyDeepFilter, FrequencyDeepFilterConfig
from .frequency_model import FrequencyUNet, FrequencyUNetConfig


@torch.inference_mode()
def _verify_waveform_parity(baseline, converted) -> dict:
    baseline.eval()
    converted.eval()
    generator = torch.Generator().manual_seed(481)
    timeline = torch.arange(4097, dtype=torch.float32) / 16000
    probes = (
        torch.zeros(1, 1),
        torch.randn(2, 1031, generator=generator) * 0.07,
        (0.1 * torch.sin(2 * torch.pi * 217 * timeline) + 0.03
         + 0.025 * torch.randn(4097, generator=generator))[None],
    )
    for probe in probes:
        expected, actual = baseline(probe), converted(probe)
        if not torch.isfinite(expected).all() or not torch.equal(actual, expected):
            raise ValueError("Zero-head conversion did not preserve exact baseline waveform output")
    return {"exact_float32_waveform_equality": True, "probe_batches": len(probes),
            "probe_samples": sum(probe.numel() for probe in probes),
            "scope": "deterministic identity check; not an audio-quality evaluation"}


def prepare_deep_filter_warm_start(source, destination, *, filter_order=5, filter_bins=65,
                                  filter_scale=0.5, expected_source_sha256=None) -> dict:
    """Copy a frozen float BN baseline and add only zero filter-head parameters.

    Source bytes are read once for loading, then re-read to ensure the source
    did not change during conversion. The artifact contains no optimizer or
    scheduler and is marked initialization_only; training must set
    resume_optimizer=False. Existing destinations are never overwritten.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Initialization destination must differ from the source checkpoint")
    if destination.exists():
        raise FileExistsError(f"Initialization already exists: {destination}")
    source_bytes = source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if expected_source_sha256 is not None and source_sha != expected_source_sha256:
        raise ValueError("Source checkpoint SHA-256 differs from the expected frozen source")
    checkpoint = torch.load(io.BytesIO(source_bytes), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("phase") != "float":
        raise ValueError("Deep-filter initialization requires a float checkpoint")
    if checkpoint.get("model_kind") != "frequency_unet":
        raise ValueError("Deep-filter initialization requires model_kind=frequency_unet")
    source_config = FrequencyUNetConfig.from_checkpoint(checkpoint["model_config"])
    if not source_config.encoder_batch_norm:
        raise ValueError("Matched deep-filter initialization requires an encoder BatchNorm baseline")
    config = FrequencyDeepFilterConfig.from_checkpoint({
        **asdict(source_config), "filter_order": filter_order,
        "filter_bins": filter_bins, "filter_scale": filter_scale,
    })
    # Conversion must not perturb the caller's training RNG state.
    with torch.random.fork_rng(devices=[]):
        baseline = FrequencyUNet(source_config)
        baseline.load_state_dict(checkpoint["model"], strict=True)
        converted = FrequencyDeepFilter(config)
        incompatible = converted.load_state_dict(baseline.state_dict(), strict=False)
    new_parameters = ["filter_head.weight", "filter_head.bias"]
    if set(incompatible.missing_keys) != set(new_parameters) or incompatible.unexpected_keys:
        raise ValueError("Conversion changed parameters outside the new filter head")
    if any(torch.count_nonzero(converted.get_parameter(name)) for name in new_parameters):
        raise ValueError("New filter-head parameters must remain exactly zero")
    parity = _verify_waveform_parity(baseline, converted)
    provenance = {
        "operation": "frequency_bn_to_zero_head_causal_deep_filter",
        "source_checkpoint": str(source), "source_checkpoint_sha256": source_sha,
        "source_model_kind": "frequency_unet", "source_model_config": asdict(source_config),
        "source_checkpoint_epoch": checkpoint.get("epoch"),
        "source_recorded_best_si_sdri": checkpoint.get("best_si_sdri"),
        "source_provenance": checkpoint.get("provenance"),
        "new_zero_parameters": new_parameters,
        "optimizer_state": "not included; resume_optimizer=False is required",
        "validation_performed": False, "waveform_parity": parity,
    }
    payload = {"model": converted.state_dict(), "model_config": asdict(config),
               "model_kind": "frequency_deep_filter", "phase": "float", "epoch": 0,
               "initialization_only": True, "provenance": provenance}
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
        raise ValueError("Source checkpoint changed during conversion; initialization was not written")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=destination.name + ".", suffix=".tmp",
                                         dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"source_checkpoint_sha256": source_sha, "destination": str(destination),
            "initialization_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "initialization_only": True, "resume_optimizer": False,
            "model_kind": "frequency_deep_filter", "model_config": asdict(config),
            "model_stats": converted.model_stats(), "waveform_parity": parity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--filter-order", type=int, default=5)
    parser.add_argument("--filter-bins", type=int, default=65)
    parser.add_argument("--filter-scale", type=float, default=0.5)
    args = parser.parse_args()
    report = prepare_deep_filter_warm_start(args.source, args.output,
        filter_order=args.filter_order, filter_bins=args.filter_bins, filter_scale=args.filter_scale,
        expected_source_sha256=args.expected_source_sha256)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
