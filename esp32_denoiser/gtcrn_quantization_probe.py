"""Matched float / GRU / LayerNorm / combined GTCRN sensitivity on CPU.

All grids are calibrated on the checkpoint's training-only hybrid recipe.
Four fixed development evaluations use exactly the same full utterances.
Replaced operators use audited INT8 arithmetic, while all other operators
and waveform DSP remain float32. This is not a complete INT8 graph, export,
QAT recovery result or MCU performance measurement.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .comparison import compare_evaluations
from .development import common_valid_ids
from .evaluate import _json_finite, evaluate_manifest, load_checkpoint
from .experimental_layer_norm import integer_layer_norm, quantize_layer_norm
from .gtcrn_model import GTCRNDenoiser
from .gtcrn_recurrent_probe import (
    GTCRNRecurrentProbe, _development_records,
    calibrate_checkpoint_training, calibrate_gru_input_configs,
)


@dataclass(frozen=True)
class LayerNormGrid:
    input_exponent: int
    output_exponent: int

    def __post_init__(self):
        for value in (self.input_exponent, self.output_exponent):
            if isinstance(value, bool) or not isinstance(value, int) or not -16 <= value <= 8:
                raise ValueError("LayerNorm grid exponents must be integers in [-16,8]")


def _source_norms(model):
    if not isinstance(model, GTCRNDenoiser):
        raise TypeError("LayerNorm sensitivity requires a GTCRNDenoiser")
    result = {name: module for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)}
    expected = {f"core.dpgrnn{block}.{direction}_ln" for block in (1, 2) for direction in ("intra", "inter")}
    if set(result) != expected or any(module.normalized_shape != (33, 16) for module in result.values()):
        raise ValueError("Expected the four framewise GTCRN LayerNorm operators over (33,16)")
    return result


class LayerNormProbe(nn.Module):
    """Actual integer normalization with float module boundaries and diagnostics.

    The Python integer reference computes the returned INT8 output. Original
    float gamma/beta are retained only for an error diagnostic; they do not
    contribute to that output. No recurrent state is stored. Input/output
    quantization uses nearest ties away from zero, matching the primitive.
    """

    def __init__(self, source: nn.LayerNorm, grid: LayerNormGrid):
        super().__init__()
        if (not isinstance(source, nn.LayerNorm) or source.normalized_shape != (33, 16)
                or not source.elementwise_affine or source.bias is None):
            raise ValueError("Require GTCRN LayerNorm over (33,16) with affine parameters")
        self.grid = grid
        self.snapshot = quantize_layer_norm(source.weight, source.bias, **asdict(grid), epsilon=source.eps)
        self.register_buffer("reference_gamma", source.weight.detach().cpu().float().clone())
        self.register_buffer("reference_beta", source.bias.detach().cpu().float().clone())
        self.requires_grad_(False).eval()
        self.reset_statistics()

    def reset_statistics(self):
        self.counts = {name: 0 for name in ("frames", "input_values", "input_clipped", "output_values",
                                           "output_clipped", "output_at_rail", "zero_variance_frames",
                                           "float_zero_variance_frames", "quantization_collapsed_frames")}
        self.errors = {name: 0.0 for name in ("input_absolute_error_sum", "input_maximum_absolute_error",
                                             "output_absolute_error_sum", "output_maximum_absolute_error",
                                             "maximum_frame_output_mae", "input_peak", "float_output_peak")}
        self.integer_maxima = {"max_squared_denominator": 0, "max_abs_affine_numerator": 0}

    @torch.no_grad()
    def forward(self, value):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not value.is_floating_point()
                or value.ndim < 2 or value.shape[-2:] != (33, 16) or not value.numel()
                or not bool(torch.isfinite(value).all())):
            raise ValueError("LayerNorm probe requires finite floating CPU input ending in (33,16)")
        original = value.detach().double().numpy()
        rounded = original / 2.0 ** self.grid.input_exponent
        rounded = np.copysign(np.floor(np.abs(rounded) + .5), rounded)
        codes = rounded.clip(-128, 127).astype(np.int8)
        result, diagnostics = integer_layer_norm(codes, self.snapshot, return_diagnostics=True)
        decoded = torch.from_numpy(result.astype(np.float32)) * 2.0 ** self.grid.output_exponent
        with torch.autocast("cpu", enabled=False):
            reference = F.layer_norm(value.float(), (33, 16), self.reference_gamma, self.reference_beta,
                                     self.snapshot.epsilon).double().numpy()
        frames = original.reshape(-1, 528)
        coded_frames = codes.reshape(-1, 528)
        float_constant = frames.max(-1) == frames.min(-1)
        coded_constant = coded_frames.max(-1) == coded_frames.min(-1)
        input_error = np.abs(codes.astype(np.float64) * 2.0 ** self.grid.input_exponent - original)
        output_error = np.abs(decoded.double().numpy() - reference)
        self.counts["frames"] += diagnostics["frames"]
        self.counts["input_values"] += codes.size
        self.counts["input_clipped"] += int(((rounded < -128) | (rounded > 127)).sum())
        self.counts["output_values"] += result.size
        self.counts["output_clipped"] += diagnostics["saturated_outputs"]
        self.counts["output_at_rail"] += int(((result == -128) | (result == 127)).sum())
        self.counts["zero_variance_frames"] += diagnostics["zero_variance_frames"]
        self.counts["float_zero_variance_frames"] += int(float_constant.sum())
        self.counts["quantization_collapsed_frames"] += int((coded_constant & ~float_constant).sum())
        self.errors["input_absolute_error_sum"] += float(input_error.sum())
        self.errors["output_absolute_error_sum"] += float(output_error.sum())
        for name, peak_value in (("input_maximum_absolute_error", input_error.max()),
                            ("output_maximum_absolute_error", output_error.max()),
                            ("maximum_frame_output_mae", output_error.reshape(-1, 528).mean(-1).max()),
                            ("input_peak", np.abs(original).max()), ("float_output_peak", np.abs(reference).max())):
            self.errors[name] = max(self.errors[name], float(peak_value))
        for name in self.integer_maxima:
            self.integer_maxima[name] = max(self.integer_maxima[name], diagnostics[name])
        return decoded.to(value.dtype)

    def statistics(self):
        counts = dict(self.counts)
        divide = lambda value, count: value / count if count else None
        fractions = {"input_clipped": divide(counts["input_clipped"], counts["input_values"]),
                     "output_clipped": divide(counts["output_clipped"], counts["output_values"]),
                     "output_at_rail": divide(counts["output_at_rail"], counts["output_values"]),
                     "zero_variance": divide(counts["zero_variance_frames"], counts["frames"]),
                     "quantization_collapsed": divide(counts["quantization_collapsed_frames"], counts["frames"])}
        errors = {key: value for key, value in self.errors.items() if not key.endswith("_sum")}
        errors.update(input_mean_absolute_error=divide(self.errors["input_absolute_error_sum"], counts["input_values"]),
                      output_mean_absolute_error=divide(self.errors["output_absolute_error_sum"], counts["output_values"]))
        return {"grid": asdict(self.grid), "gamma_exponent": self.snapshot.gamma_exponent,
                "requested_epsilon": self.snapshot.epsilon, "effective_epsilon": self.snapshot.effective_epsilon,
                "counts": counts, "fractions": fractions, "errors": errors,
                "error_reference": "Original float32 LayerNorm on this operator's actual incoming features; diagnostic only",
                "integer_maxima": dict(self.integer_maxima), "integer_bounds": self.snapshot.bounds(),
                "memory": self.snapshot.memory_accounting(),
                "memory_scope": "Integer affine arrays; retained float diagnostic buffers and Python overhead are excluded",
                "backend": "audited Python integer reference; no float arithmetic inside normalization",
                "diagnostic_scope": "Output clipping counts pre-clamp integer codes; rail contacts include valid endpoints; collapsed variance excludes truly constant float frames"}


class GTCRNOperatorProbe(nn.Module):
    """LayerNorm-only or combined GRU+LayerNorm frozen CPU copy."""

    def __init__(self, source, norm_grids, *, gru_configs=None, combined=False):
        super().__init__()
        if set(norm_grids) != set(_source_norms(source)):
            raise ValueError("Provide one calibrated grid for each of the four GTCRN LayerNorms")
        self.config = source.config
        self.combined = combined
        self.network = GTCRNRecurrentProbe(source, gru_configs) if combined else copy.deepcopy(source).cpu().float().eval()
        target = self.network.model if combined else self.network
        self.norms = {}
        for name, module in _source_norms(target).items():
            replacement = LayerNormProbe(module, norm_grids[name])
            parent, _, attribute = name.rpartition(".")
            setattr(target.get_submodule(parent), attribute, replacement)
            self.norms[name] = replacement
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, noisy):
        if self.training or noisy.device.type != "cpu":
            raise ValueError("Operator sensitivity is eval-only CPU inference")
        with torch.autocast("cpu", enabled=False):
            return self.network(noisy.float())

    def reset_statistics(self):
        for module in self.norms.values():
            module.reset_statistics()
        if self.combined:
            self.network.reset_statistics()

    def statistics(self):
        return {"scope": "GRU+LayerNorm INT8" if self.combined else "LayerNorm INT8; GRUs remain float32",
                "remaining_precision": "Convolution, linear, PReLU, residual addition, attention energy/products, ERB, mask and waveform DSP remain float32",
                "layer_norm": {name: module.statistics() for name, module in self.norms.items()},
                "gru": self.network.statistics() if self.combined else None}


@torch.inference_mode()
def calibrate_operator_configs(model, batches, *, max_batches=32, base_config=None):
    """Observe GRU and LayerNorm grids together on identical training crops.

    LayerNorm input/output peaks come from the original float graph. Exactly
    the same frozen grids are used in LN-only and combined probes; statistics
    expose any range drift after upstream operators become quantized.
    """
    modules = _source_norms(model)
    peaks = {name: {"input_peak": 0.0, "output_peak": 0.0, "frames": 0} for name in modules}
    handles = []
    try:
        for name, module in modules.items():
            def record(_module, arguments, output, name=name):
                for label, values in (("input_peak", arguments[0]), ("output_peak", output)):
                    if not bool(torch.isfinite(values).all()):
                        raise ValueError(f"Nonfinite LayerNorm calibration values: {name}")
                    peaks[name][label] = max(peaks[name][label], float(values.abs().max()))
                peaks[name]["frames"] += output.numel() // 528
            handles.append(module.register_forward_hook(record))
        gru_configs, gru_report = calibrate_gru_input_configs(model, batches, max_batches=max_batches, base_config=base_config)
    finally:
        for handle in handles:
            handle.remove()
    grids = {}
    for name, peak in peaks.items():
        def exponent(value):
            return -16 if value <= 127 * 2.0 ** -16 else min(8, math.ceil(math.log2(value / 127)))
        grid = LayerNormGrid(exponent(peak["input_peak"]), exponent(peak["output_peak"]))
        module = modules[name]
        parameters = quantize_layer_norm(module.weight, module.bias, **asdict(grid), epsilon=module.eps)
        grids[name] = grid
        peak.update(grid=asdict(grid), gamma_exponent=parameters.gamma_exponent,
                    requested_epsilon=parameters.epsilon, effective_epsilon=parameters.effective_epsilon,
                    input_peak_exceeds_grid=peak["input_peak"] > 127 * 2.0 ** grid.input_exponent,
                    output_peak_exceeds_grid=peak["output_peak"] > 127 * 2.0 ** grid.output_exponent)
    return {"gru": gru_configs, "layer_norm": grids}, {
        "batches": gru_report["batches"], "gru": gru_report, "layer_norm": peaks,
        "method": "Original float graph absolute peaks; per-GRU input and per-LayerNorm input/output power-of-two grids",
        "scope": "One training-only pass; identical LayerNorm grids for isolated and combined sensitivity"}


def evaluate_operator_probes(source, manifest, configs, *, max_utterances=16, seed=482, perceptual=False):
    """Evaluate four paths on one fixed, hash-checked development cohort."""
    if isinstance(max_utterances, bool) or not isinstance(max_utterances, int) or max_utterances < 1:
        raise ValueError("max_utterances must be a positive integer")
    if source.window.device.type != "cpu":
        raise ValueError("Sensitivity evaluation requires a CPU source")
    manifest_bytes = Path(manifest).read_bytes()
    records = _development_records(manifest)
    if Path(manifest).read_bytes() != manifest_bytes:
        raise ValueError("Development manifest changed while loading")
    indices = sorted(random.Random(seed).sample(range(len(records)), min(max_utterances, len(records))))
    cohort = [records[index] for index in indices]
    models = {"float": source, "gru_only": GTCRNRecurrentProbe(source, configs["gru"]),
              "layer_norm_only": GTCRNOperatorProbe(source, configs["layer_norm"]),
              "combined": GTCRNOperatorProbe(source, configs["layer_norm"], gru_configs=configs["gru"], combined=True)}
    reports = {}
    with tempfile.TemporaryDirectory(prefix="gtcrn-operator-probes-") as directory:
        path = Path(directory) / "cohort.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cohort))
        for name, model in models.items():
            print(json.dumps({"event": "operator_probe", "variant": name}), flush=True)
            reports[name] = evaluate_manifest(model, path, perceptual=perceptual)
    if Path(manifest).read_bytes() != manifest_bytes:
        raise ValueError("Development manifest changed during sensitivity evaluation")
    # Existing paired comparator verifies IDs, lengths, baselines and audio
    # content hashes. The small development intervals are ancillary diagnostics.
    comparisons = {name: compare_evaluations(report, reports["float"], manifest=cohort,
                                              cluster_key="base_crop" if all("base_crop" in r for r in cohort) else None,
                                              bootstrap_samples=1000, seed=seed)
                   for name, report in reports.items() if name != "float"}
    common = common_valid_ids(*reports.values())
    means = {name: math.fsum(row["si_sdri"] for row in report["utterances"] if row["id"] in common) / len(common)
             if common else None for name, report in reports.items()}
    return {"scope": "Partial-operator INT8 sensitivity; no complete graph, QAT, official test or MCU result",
            "timing_caveat": "Probe timings include Python integer loops and LayerNorm float-error diagnostics, not an optimized inference implementation",
            "model_config": asdict(source.config), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "cohort": {"seed": seed, "indices": indices, "ids": [row["id"] for row in cohort],
                       "available_utterances": len(records), "selected_utterances": len(cohort)},
            "common_summary": {"valid_utterances": len(common), "invalid_utterances": len(cohort) - len(common),
                               "weighting": "equal common valid full utterances", "si_sdri": means,
                               "minus_float_db": {name: value - means["float"] if value is not None else None
                                                  for name, value in means.items() if name != "float"}},
            "comparisons_vs_float": comparisons, "evaluations": reports,
            "statistics": {name: model.statistics() for name, model in models.items() if name != "float"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-crops", type=int, default=32,
                        help="Actual checkpoint-policy training crops; development audio is excluded")
    parser.add_argument("--max-utterances", type=int, default=16)
    parser.add_argument("--seed", type=int, default=482)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--perceptual", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    model, metadata = load_checkpoint(args.checkpoint)
    checkpoint_bytes = args.checkpoint.read_bytes()
    if metadata.get("model_sha256") != hashlib.sha256(checkpoint_bytes).hexdigest():
        raise ValueError("Checkpoint changed after model loading")
    import io
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    configs, calibration = calibrate_checkpoint_training(
        model, checkpoint, args.manifest, crops=args.calibration_crops, seed=args.seed + 1,
        calibrator=calibrate_operator_configs)
    result = evaluate_operator_probes(model, args.manifest, configs, max_utterances=args.max_utterances,
                                      seed=args.seed, perceptual=args.perceptual)
    result.update(checkpoint={**metadata, "path": str(args.checkpoint.resolve())}, calibration=calibration,
                  probe_source_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                       for name in ("gtcrn_quantization_probe.py", "gtcrn_recurrent_probe.py",
                                                    "experimental_layer_norm.py", "experimental_gru.py")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_finite(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "operator_probe_complete", "output": str(args.output), **result["common_summary"]}), flush=True)


if __name__ == "__main__":
    main()
