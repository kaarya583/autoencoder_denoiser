"""CPU GRU-only quantization sensitivity, never a full INT8 GTCRN export.

Replaced GRUs use the audited integer cell: INT8 inputs/weights/nonlinear
outputs/hidden state, INT32 biases/checked accumulators, and safe INT64
requantization products. Other operators and waveform DSP remain float32.
Module inputs are quantized; module outputs/final states are dequantized for
the unchanged surrounding graph. Recurrence itself retains INT8 state.

This frozen development probe is deliberately separate from training,
checkpoint formats and the vendored network. It does not implement an MCU
runtime. Optional input-scale calibration must use training audio only.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn

from .data import PairedAudioDataset, read_manifest
from .evaluate import _json_finite, evaluate_manifest, load_checkpoint
from .experimental_gru import FakeQuantGRUCell, GRUQuantizationConfig, IntegerGRUCell
from .gtcrn_model import GTCRNDenoiser


GROUPS = ("attention", "intra", "inter")


def _group(name):
    if name.endswith(".tra.att_gru"):
        return "attention"
    if ".intra_rnn." in name:
        return "intra"
    if ".inter_rnn." in name:
        return "inter"
    raise ValueError(f"Unrecognized GTCRN GRU location: {name}")


def _source_grus(model):
    if not isinstance(model, GTCRNDenoiser):
        raise TypeError("The recurrent probe requires a GTCRNDenoiser")
    result = {name: module for name, module in model.named_modules() if isinstance(module, nn.GRU)}
    if len(result) != 14 or sum(1 + int(m.bidirectional) for m in result.values()) != 18:
        raise ValueError("Unexpected GTCRN GRU layout; expected 14 modules / 18 directions")
    for name in result:
        _group(name)
    return result


class GRUSequenceProbe(nn.Module):
    """Frozen single-layer nn.GRU-compatible CPU sequence wrapper.

    Forward/reverse outputs retain original sequence order; h_n retains
    PyTorch direction order. Bidirectionality is used only within a frame's
    frequency axis in GTCRN. No hidden state survives between forward calls.
    ``fake_quant`` exists for contract parity tests, not fast GPU training.
    Gate endpoint/rail counters describe occupancy, not necessarily overflow.
    """

    def __init__(self, source: nn.GRU, config: GRUQuantizationConfig | None = None,
                 *, backend: str = "integer"):
        super().__init__()
        if not isinstance(source, nn.GRU) or source.num_layers != 1 or not source.bias or source.dropout:
            raise ValueError("Probe supports a single GRU layer with explicit biases and no dropout")
        if backend not in {"integer", "fake_quant"}:
            raise ValueError("backend must be integer or fake_quant")
        self.config = config or GRUQuantizationConfig()
        self.backend = backend
        self.input_size, self.hidden_size = source.input_size, source.hidden_size
        self.batch_first, self.bidirectional, self.num_layers = source.batch_first, source.bidirectional, 1
        self.directions = ("forward", "reverse") if source.bidirectional else ("forward",)
        self.cells = [IntegerGRUCell.from_torch(source, self.config, direction=d) for d in self.directions]
        self.fake_cells = nn.ModuleList(
            FakeQuantGRUCell(source, self.config, direction=d).cpu().float() for d in self.directions
        ) if backend == "fake_quant" else nn.ModuleList()
        self.requires_grad_(False).eval()
        self.reset_statistics()

    def reset_statistics(self):
        for cell in self.cells:
            cell.reset_statistics()
        self.gates = [{name: {"values": 0, "low_endpoint": 0, "high_endpoint": 0}
                       for name in ("reset", "update", "candidate")} for _ in self.cells]
        self.initial_state_values = self.initial_state_clipped = 0

    @torch.no_grad()
    def forward(self, inputs, hx=None):
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 3 or min(inputs.shape) < 1
                or inputs.shape[-1] != self.input_size or inputs.device.type != "cpu"
                or not inputs.is_floating_point() or not bool(torch.isfinite(inputs).all())):
            raise ValueError("Probe input must be finite floating CPU [batch,time,input] or [time,batch,input]")
        sequence = inputs if self.batch_first else inputs.transpose(0, 1)
        batch, steps, _ = sequence.shape
        if hx is None:
            hx = sequence.new_zeros((len(self.cells), batch, self.hidden_size))
        if (not isinstance(hx, torch.Tensor) or hx.shape != (len(self.cells), batch, self.hidden_size)
                or hx.device.type != "cpu" or not hx.is_floating_point() or not bool(torch.isfinite(hx).all())):
            raise ValueError("Probe hidden state must be finite floating CPU [directions,batch,hidden]")
        outputs, finals = [], []
        for direction, cell in enumerate(self.cells):
            indices = range(steps) if direction == 0 else range(steps - 1, -1, -1)
            rows = [None] * steps
            if self.backend == "integer":
                codes = cell.quantize_input(sequence.detach().double().numpy())
                initial = hx[direction].detach().double().numpy() / 2.0 ** self.config.state_exponent
                initial = np.copysign(np.floor(np.abs(initial) + .5), initial)
                self.initial_state_values += initial.size
                self.initial_state_clipped += int(((initial < -128) | (initial > 127)).sum())
                state = initial.clip(-128, 127).astype(np.int8)
                for index in indices:
                    state, trace = cell.step(codes[:, index], state, return_trace=True)
                    rows[index] = state
                    for name, counter in self.gates[direction].items():
                        values = trace[name]
                        counter["values"] += values.size
                        counter["low_endpoint"] += int((values == -128).sum())
                        counter["high_endpoint"] += int((values == 127).sum())
                output = torch.from_numpy(np.stack(rows, axis=1).astype(np.float32))
                final = torch.from_numpy(state.astype(np.float32))
                output *= 2.0 ** self.config.state_exponent
                final *= 2.0 ** self.config.state_exponent
            else:
                state = hx[direction]
                for index in indices:
                    state = self.fake_cells[direction](sequence[:, index], state)
                    rows[index] = state
                output, final = torch.stack(rows, dim=1), state
            outputs.append(output.to(inputs.dtype))
            finals.append(final.to(inputs.dtype))
        output = torch.cat(outputs, dim=-1)
        return (output if self.batch_first else output.transpose(0, 1)), torch.stack(finals)

    def statistics(self):
        if self.backend != "integer":
            return {"backend": self.backend, "statistics_available": False}
        directions = {}
        for name, cell, gates in zip(self.directions, self.cells, self.gates):
            counters = dict(cell.statistics)
            counters["sequence_steps"] = counters.pop("frames")
            fractions = {}
            for numerator, denominator in (("input_clipped", "input_values"), ("logit_clipped", "logit_values"),
                                           ("state_saturated", "state_values"), ("state_zero", "state_values"),
                                           ("state_unchanged", "state_values")):
                fractions[numerator + "_fraction"] = counters[numerator] / counters[denominator] if counters[denominator] else None
            directions[name] = {"counts": counters, "fractions": fractions,
                                "gates": copy.deepcopy(gates), "storage": cell.storage_stats()}
        return {"backend": self.backend, "config": asdict(self.config), "directions": directions,
                "initial_state_values": self.initial_state_values, "initial_state_clipped": self.initial_state_clipped,
                "counter_scope": "sequence steps include frequency traversal; endpoint occupancy is not overflow"}


class GTCRNRecurrentProbe(nn.Module):
    """Independent frozen CPU copy with only selected GRU groups quantized.

    This adapter is for offline development cohorts. It has no checkpoint
    exporter or streaming-runtime API; no full-model size/latency claim follows
    from these per-cell integer snapshots. Source weights/mode stay unchanged.
    """

    def __init__(self, source: GTCRNDenoiser, configs: dict[str, GRUQuantizationConfig] | None = None,
                 *, groups=GROUPS, default_config: GRUQuantizationConfig | None = None):
        super().__init__()
        original = _source_grus(source)
        groups = tuple(groups)
        if not groups or len(set(groups)) != len(groups) or not set(groups).issubset(GROUPS):
            raise ValueError(f"groups must be distinct choices from {GROUPS}")
        configs = {} if configs is None else dict(configs)
        if set(configs) - set(original):
            raise ValueError("Quantization configuration names do not match GTCRN GRUs")
        self.model = copy.deepcopy(source).cpu().float().eval()
        self.config = self.model.config
        self.groups = groups
        self.replacements = {}
        for name, module in _source_grus(self.model).items():
            if _group(name) not in groups:
                continue
            replacement = GRUSequenceProbe(module, configs.get(name, default_config))
            parent, _, attribute = name.rpartition(".")
            setattr(self.model.get_submodule(parent), attribute, replacement)
            self.replacements[name] = replacement
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, noisy):
        if self.training:
            raise ValueError("The recurrent sensitivity probe is eval-only")
        if noisy.device.type != "cpu":
            raise ValueError("The accurate recurrent sensitivity probe runs on CPU")
        with torch.autocast("cpu", enabled=False):
            return self.model(noisy.float())

    def reset_statistics(self):
        for module in self.replacements.values():
            module.reset_statistics()

    def statistics(self):
        return {"scope": "GRU-only INT8; all other operators and waveform DSP remain float32",
                "state_contract": "INT8 inside recurrence; dequantized module-boundary hidden/output tensors",
                "groups": list(self.groups), "gru_modules": len(self.replacements),
                "gru_directions": sum(len(m.cells) for m in self.replacements.values()),
                "modules": {name: {"group": _group(name), **module.statistics()}
                            for name, module in self.replacements.items()}}


@torch.inference_mode()
def calibrate_gru_input_configs(model, batches, *, max_batches=8, base_config=None):
    """Fit one power-of-two input grid per GRU using training representatives.

    Uses the observed absolute peak, capped to the primitive's [-12,0] input
    exponents. A peak outside the resulting grid is explicitly reported. The
    caller supplies CPU waveforms [B,N] and owns training-only provenance.
    Scales observe the original float graph, not quantized upstream drift.
    """
    if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1:
        raise ValueError("max_batches must be a positive integer")
    if model.window.device.type != "cpu":
        raise ValueError("GRU calibration requires a CPU source model")
    modules = _source_grus(model)
    base_config = base_config or GRUQuantizationConfig()
    peaks = {name: {"max_abs": 0.0, "values": 0} for name in modules}
    handles = []
    was_training = model.training
    try:
        for name, module in modules.items():
            def record(_module, arguments, name=name):
                values = arguments[0]
                if not bool(torch.isfinite(values).all()):
                    raise ValueError(f"Nonfinite float calibration features: {name}")
                peaks[name]["max_abs"] = max(peaks[name]["max_abs"], float(values.abs().max()))
                peaks[name]["values"] += values.numel()
            handles.append(module.register_forward_pre_hook(record))
        model.eval()
        count = 0
        for batch in islice(batches, max_batches):
            if not isinstance(batch, torch.Tensor) or batch.device.type != "cpu":
                raise ValueError("Calibration batches must be CPU waveform tensors")
            model(batch.float())
            count += 1
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if not count:
        raise ValueError("Calibration needs at least one training batch")
    configs = {}
    for name, statistics in peaks.items():
        peak = statistics["max_abs"]
        exponent = max(-12, min(0, math.ceil(math.log2(peak / 127)))) if peak else -12
        configs[name] = replace(base_config, input_exponent=exponent)
        statistics.update(input_exponent=exponent, symmetric_positive_range=127 * 2.0 ** exponent,
                          observed_peak_exceeds_grid=peak > 127 * 2.0 ** exponent)
    return configs, {"batches": count, "method": "per-GRU float input absolute peak, power-of-two scale",
                     "scope": "training representatives only; evaluate actual clipping after replacement",
                     "modules": peaks}


def _development_records(manifest):
    records = read_manifest(manifest)
    if any(r.get("source_split") not in {"train", "development"} for r in records):
        raise ValueError("This sensitivity probe accepts train-origin validation or development only; test remains sealed")
    return records


def _audit_calibration(records, evaluation_records):
    if any(r.get("source_split") != "train" for r in records):
        raise ValueError("Calibration requires training-origin records")
    for key in ("id", "speaker"):
        if {r[key] for r in records} & {r[key] for r in evaluation_records}:
            raise ValueError(f"Calibration overlaps evaluation {key}s")
    paths = lambda rows: {str(Path(r[role]).resolve()) for r in rows for role in ("clean", "noisy")}
    if paths(records) & paths(evaluation_records):
        raise ValueError("Calibration audio paths overlap evaluation")


def calibrate_checkpoint_training(model, checkpoint, evaluation_manifest, *, crops=32, seed=483,
                                  base_config=None, calibrator=None):
    """Reconstruct the checkpoint's audited training sampling policy.

    The broader recipe uses the actual HybridTrainingDataset Bernoulli draw,
    not a forced half-and-half cohort. Each draw has an independent recorded
    CPU seed; report counts expose its realized mixture. Data augmentation is
    the current recorded training recipe: paired gain/noise +/-6 dB;
    synthetic active-frame SNR [-5,20], gain +/-6 dB. Both sources use the
    saved clean-identity probability (historically 3% when omitted).
    Selected original assets and final float32 crop bytes are hashed. Original
    paired manifests do not commit waveform hashes, so those hashes are
    recorded now; selected synthetic bytes must match preparation hashes.
    An optional calibrator can observe additional operators on these same
    representatives; it follows calibrate_gru_input_configs' call/return API.
    """
    from .distillation import _audit_synthetic_sources, _sha256, checkpoint_clean_identity_probability
    from .extra_data import DynamicMixtureDataset
    from .mixtures import HybridTrainingDataset
    from .gtcrn_model import GTCRNConfig

    if isinstance(crops, bool) or not isinstance(crops, int) or crops < 1:
        raise ValueError("Calibration crops must be a positive integer")
    config, provenance = checkpoint.get("train_config"), checkpoint.get("provenance")
    if (checkpoint.get("model_kind") != "gtcrn" or not isinstance(config, dict)
            or not isinstance(provenance, dict) or provenance.get("test_used_for_selection") is not False):
        raise ValueError("Checkpoint-derived calibration requires known GTCRN training provenance without test selection")
    clean_identity_probability = checkpoint_clean_identity_probability(config, provenance)
    if GTCRNConfig.from_checkpoint(checkpoint["model_config"]) != model.config:
        raise ValueError("Calibration model configuration differs from checkpoint")
    source_root = Path(__file__).parent
    if provenance.get("source_sha256", {}).get("data.py") != _sha256(source_root / "data.py"):
        raise ValueError("Paired augmentation source differs from the recorded checkpoint recipe")
    crop_seconds = config.get("crop_seconds")
    if isinstance(crop_seconds, bool) or not isinstance(crop_seconds, (int, float)) or not math.isfinite(crop_seconds) or crop_seconds <= 0:
        raise ValueError("Checkpoint requires a valid training crop duration")
    evaluation = _development_records(evaluation_manifest)
    paired_path, validation_path = Path(config["train_manifest"]), Path(config["val_manifest"])
    paired_records, validation_records = read_manifest(paired_path), read_manifest(validation_path)
    for label, path in (("train", paired_path), ("val", validation_path)):
        if provenance.get("manifest_sha256", {}).get(label) != _sha256(path):
            raise ValueError(f"Checkpoint {label} manifest hash mismatch")
    if any(row.get("source_split") != "train" for row in validation_records):
        raise ValueError("Checkpoint validation must remain training-origin")
    _audit_calibration(paired_records, validation_records)
    _audit_calibration(paired_records, evaluation)
    paired = PairedAudioDataset(paired_records, crop_seconds=crop_seconds, random_crop=True,
                                gain_db=(-6, 6), noise_scale_db=(-6, 6),
                                clean_identity_prob=clean_identity_probability)
    synthetic = None
    source_audit = None
    synthetic_paths = [config.get(f"synthetic_{kind}_manifest") for kind in ("speech", "noise")]
    if any(synthetic_paths):
        if not all(isinstance(path, str) and path for path in synthetic_paths):
            raise ValueError("Broader calibration requires both synthetic training manifests")
        source_audit = _audit_synthetic_sources(config, provenance, len(paired),
                                                [("checkpoint validation", validation_records), ("probe evaluation", evaluation)])
        synthetic = DynamicMixtureDataset(*synthetic_paths, crop_seconds=crop_seconds, snr_db=(-5, 20),
                                           gain_db=(-6, 6), clean_identity_prob=clean_identity_probability)
        if synthetic.split != "train":
            raise ValueError("Synthetic calibration must use the training partition")
        all_extra = synthetic.speech_records + synthetic.noise_records
        # Rendered development paths/IDs differ from their underlying sources.
        for fields, source_key in ((("speech_id", "noise_id"), "id"),
                                   (("speech_source_sha256", "noise_source_sha256"), "sha256")):
            used = {row.get(field) for row in evaluation for field in fields} - {None}
            if used & {row[source_key] for row in all_extra}:
                raise ValueError(f"Synthetic calibration overlaps underlying development {source_key}")
        dataset = HybridTrainingDataset(paired, synthetic, synthetic_probability=config["synthetic_probability"],
                                         epoch_samples=config.get("epoch_samples"))
    else:
        if provenance.get("added_training_sources") is not None:
            raise ValueError("Checkpoint synthetic source provenance differs from its training config")
        dataset = paired
    paired_by_id = {row["id"]: row for row in paired_records}
    extra_by_id = {row["id"]: row for row in synthetic.speech_records + synthetic.noise_records} if synthetic else {}
    verified_files, crop_records = {}, []

    def source_hash(path, expected=None):
        path = str(Path(path).resolve())
        if path not in verified_files:
            verified_files[path] = _sha256(path)
        if expected is not None and verified_files[path] != expected:
            raise ValueError(f"Selected synthetic source bytes changed since preparation: {path}")
        return {"path": path, "sha256": verified_files[path], "matches_preparation_hash": expected is not None}

    def representatives():
        for index in range(crops):
            key = f"gtcrn-gru-calibration-v1:{seed}:{index}".encode()
            item_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(item_seed)
                dataset_index = index % len(dataset) if synthetic else int(torch.randint(len(dataset), ()).item())
                item = dataset[dataset_index]
            row = {"index": index, "seed": item_seed, "id": item["id"], "speaker": item["speaker"],
                   "length": item["length"], "crop_samples": item["noisy"].numel(), "sources": []}
            if "mixture" in item:
                row.update(kind="synthetic", mixture=item["mixture"])
                for role in ("speech", "noise"):
                    identifier = item["mixture"][role + "_id"]
                    if identifier is not None:
                        original = extra_by_id[identifier]
                        row["sources"].append({"id": identifier, "role": role,
                                               **source_hash(original["path"], original["sha256"])})
            else:
                row["kind"] = "paired"
                original = paired_by_id[item["id"]]
                for role in ("clean", "noisy"):
                    row["sources"].append({"id": item["id"], "role": role, **source_hash(original[role])})
            for role in ("clean", "noisy"):
                row[role + "_crop_sha256"] = hashlib.sha256(item[role].numpy().astype("<f4").tobytes()).hexdigest()
            crop_records.append(row)
            yield item["noisy"].unsqueeze(0)

    observer = calibrate_gru_input_configs if calibrator is None else calibrator
    configs, report = observer(model, representatives(), max_batches=crops, base_config=base_config)
    report.update(recipe="checkpoint training policy", seed=seed, crop_seconds=crop_seconds,
                  clean_identity_probability=clean_identity_probability,
                  paired_manifest_sha256=_sha256(paired_path), checkpoint_validation_manifest_sha256=_sha256(validation_path),
                  synthetic_probability=config["synthetic_probability"] if synthetic else 0,
                  paired_crops=sum(row["kind"] == "paired" for row in crop_records),
                  synthetic_crops=sum(row["kind"] == "synthetic" for row in crop_records),
                  source_audit=source_audit, crops=crop_records,
                  source_code_sha256={name: _sha256(source_root / name) for name in
                                      ("data.py", "extra_data.py", "mixtures.py", "gtcrn_recurrent_probe.py")},
                  hash_scope="Selected synthetic file hashes verified against preparation; paired file hashes recorded now; crop hashes cover little-endian float32 samples including padding")
    return configs, report


def evaluate_recurrent_probe(source, manifest, *, configs=None, default_config=None,
                             groups=GROUPS, max_utterances=16, seed=482, perceptual=False):
    """Compare identical full clips, with one vote per common valid utterance.

    A seeded sample avoids always taking the first validation speaker/SNR.
    Small-cohort differences diagnose sensitivity, not final audio quality.
    Both paths use float waveform I/O without independent normalization or
    PCM16 clipping, and fresh recurrent state for every utterance.
    """
    if isinstance(max_utterances, bool) or not isinstance(max_utterances, int) or max_utterances < 1:
        raise ValueError("max_utterances must be a positive integer")
    if source.window.device.type != "cpu":
        raise ValueError("Paired recurrent evaluation requires a CPU source")
    records = _development_records(manifest)
    indices = sorted(random.Random(seed).sample(range(len(records)), min(max_utterances, len(records))))
    cohort = [records[index] for index in indices]
    probe = GTCRNRecurrentProbe(source, configs, groups=groups, default_config=default_config)
    with tempfile.TemporaryDirectory(prefix="gtcrn-recurrent-probe-") as directory:
        cohort_path = Path(directory) / "cohort.jsonl"
        cohort_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in cohort))
        control = evaluate_manifest(source, cohort_path, perceptual=perceptual)
        quantized = evaluate_manifest(probe, cohort_path, perceptual=perceptual)
    paired = []
    for first, second in zip(control["utterances"], quantized["utterances"], strict=True):
        if first["id"] != second["id"] or first["samples"] != second["samples"]:
            raise RuntimeError("Float and recurrent probes evaluated different cohorts")
        valid = all(math.isfinite(r[k]) for r in (first, second) for k in ("si_sdr_noisy", "si_sdr_enhanced"))
        if valid and first["si_sdr_noisy"] != second["si_sdr_noisy"]:
            raise RuntimeError("Noisy baseline changed across the paired cohort")
        paired.append({"id": first["id"], "valid": valid,
                       "float_si_sdri": first["si_sdri"], "recurrent_int8_si_sdri": second["si_sdri"],
                       "probe_minus_float_db": second["si_sdri"] - first["si_sdri"] if valid else None})
    valid_rows = [row for row in paired if row["valid"]]
    mean = lambda key: math.fsum(r[key] for r in valid_rows) / len(valid_rows) if valid_rows else None
    return {"scope": "Frozen GRU-only CPU sensitivity; not full INT8, QAT recovery, test results or MCU performance",
            "model_config": asdict(source.config), "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            "cohort": {"seed": seed, "indices": indices, "ids": [r["id"] for r in cohort],
                       "available_utterances": len(records), "selected_utterances": len(cohort)},
            "paired_summary": {"valid_utterances": len(valid_rows), "invalid_utterances": len(paired) - len(valid_rows),
                               "float_si_sdri": mean("float_si_sdri"), "recurrent_int8_si_sdri": mean("recurrent_int8_si_sdri"),
                               "probe_minus_float_db": mean("probe_minus_float_db"), "weighting": "equal common valid utterances"},
            "paired_utterances": paired, "float": control, "recurrent_int8": quantized,
            "recurrent_statistics": probe.statistics()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-utterances", type=int, default=16)
    parser.add_argument("--seed", type=int, default=482)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--input-exponent", type=int, default=-5)
    parser.add_argument("--logit-exponent", type=int, default=-4)
    calibration_mode = parser.add_mutually_exclusive_group()
    calibration_mode.add_argument("--calibration-manifest", type=Path,
                                  help="Explicit paired-only calibration; use --calibration-from-checkpoint for the broader recipe")
    calibration_mode.add_argument("--calibration-from-checkpoint", action="store_true",
                                  help="Use recorded training manifests, augmentations and hybrid sampling probability")
    parser.add_argument("--calibration-utterances", type=int, default=32)
    parser.add_argument("--calibration-crop-seconds", type=float,
                        help="Paired-only center-crop duration (default 3 s); checkpoint mode uses its recorded duration")
    parser.add_argument("--perceptual", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.calibration_utterances < 1:
        parser.error("threads and calibration-utterances must be positive")
    if args.calibration_from_checkpoint and args.calibration_crop_seconds is not None:
        parser.error("Checkpoint calibration uses its recorded training crop duration")
    torch.set_num_threads(args.threads)
    source, metadata = load_checkpoint(args.checkpoint)
    _source_grus(source)
    evaluation_records = _development_records(args.manifest)
    base_config = GRUQuantizationConfig(input_exponent=args.input_exponent, logit_exponent=args.logit_exponent)
    configs, calibration = None, {"method": "fixed input grid; no calibration"}
    if args.calibration_from_checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        configs, calibration = calibrate_checkpoint_training(
            source, checkpoint, args.manifest, crops=args.calibration_utterances, seed=args.seed + 1,
            base_config=base_config)
    elif args.calibration_manifest:
        records = read_manifest(args.calibration_manifest)
        _audit_calibration(records, evaluation_records)
        indices = sorted(random.Random(args.seed + 1).sample(range(len(records)), min(args.calibration_utterances, len(records))))
        crop_seconds = 3.0 if args.calibration_crop_seconds is None else args.calibration_crop_seconds
        dataset = PairedAudioDataset([records[i] for i in indices], crop_seconds=crop_seconds,
                                     random_crop=False, gain_db=(0, 0))
        configs, calibration = calibrate_gru_input_configs(
            source, (dataset[i]["noisy"].unsqueeze(0) for i in range(len(dataset))),
            max_batches=len(dataset), base_config=base_config)
        calibration.update(recipe="explicit paired-only unaugmented center crops",
                           manifest_sha256=hashlib.sha256(args.calibration_manifest.read_bytes()).hexdigest(),
                           ids=[records[i]["id"] for i in indices], indices=indices,
                           crop_seconds=crop_seconds, crop="deterministic center")
    result = evaluate_recurrent_probe(source, args.manifest, configs=configs, default_config=base_config,
                                     groups=args.groups, max_utterances=args.max_utterances,
                                     seed=args.seed, perceptual=args.perceptual)
    result.update(checkpoint={**metadata, "path": str(args.checkpoint.resolve()),
                              "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}, calibration=calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_finite(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "recurrent_probe_complete", "output": str(args.output), **result["paired_summary"]}), flush=True)


if __name__ == "__main__":
    main()
