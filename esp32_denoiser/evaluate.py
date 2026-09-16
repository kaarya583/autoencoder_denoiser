"""Evaluate aligned full utterances using a checkpoint or actual integer binary.

The default binary backend compiles and invokes the portable C neural runtime.
Its weights, activations and persistent states are integers. FFT, spectral
features and overlap-add remain float32 DSP and use constants from the binary.
Reported processing RTF measures this host implementation, never an ESP32.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import ThreadpoolController

from .data import PairedAudioDataset
from .development import preservation_metrics
from .export import IntegerDenoiser
from .metrics import si_sdr, summarize_utterances
from .models import build_model, checkpoint_kind, configure_model_qat
from .quantization import round_away
from .runtime_cache import runtime_fingerprint


class _CModel(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("model_bytes", ctypes.c_size_t), ("state_bytes", ctypes.c_size_t),
        ("input_channels", ctypes.c_uint16), ("hidden_channels", ctypes.c_uint16),
        ("output_channels", ctypes.c_uint16), ("blocks", ctypes.c_uint16),
        ("input_exponent", ctypes.c_int8), ("hidden_exponent", ctypes.c_int8),
        ("output_exponent", ctypes.c_int8),
        ("dsp_data", ctypes.POINTER(ctypes.c_uint8)), ("dsp_bytes", ctypes.c_size_t),
    ]


def _compiled_runtime():
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("The C backend requires a C99 compiler (cc); use --backend numpy for the integer oracle")
    source = Path(__file__).resolve().parents[1] / "firmware/esp32_denoiser/denoiser.c"
    if not source.is_file():
        raise FileNotFoundError(f"Portable C runtime is missing: {source}")
    fingerprint = runtime_fingerprint(source.parent, ('denoiser.c', 'denoiser.h', 'integer_kernels.h'))
    return _compiled_runtime_for_source(source, compiler, fingerprint)


@lru_cache(maxsize=1)
def _compiled_runtime_for_source(source, compiler, fingerprint):
    # fingerprint is part of the cache key, including transitively included headers.
    directory = tempfile.TemporaryDirectory(prefix="esp32-denoiser-c-")
    library_path = Path(directory.name) / "denoiser.so"
    try:
        subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", str(source), "-o", str(library_path)],
                       check=True, capture_output=True, text=True)
        library = ctypes.CDLL(str(library_path))
    except Exception:
        directory.cleanup()
        raise
    library.edn_init.argtypes = [ctypes.POINTER(_CModel), ctypes.c_void_p, ctypes.c_size_t]
    library.edn_reset.argtypes = [ctypes.POINTER(_CModel), ctypes.c_void_p, ctypes.c_size_t]
    library.edn_process_frame.argtypes = [ctypes.POINTER(_CModel), ctypes.c_void_p,
                                        ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
    for name in ("edn_init", "edn_reset", "edn_process_frame"):
        getattr(library, name).restype = ctypes.c_int
    # Retain both the CDLL and its temporary directory for the process lifetime.
    return library, directory


class _CIntegerNetwork:
    def __init__(self, blob: bytes):
        self.library, self.directory = _compiled_runtime()
        self.blob = ctypes.create_string_buffer(blob)
        self.model = _CModel()
        if self.library.edn_init(ctypes.byref(self.model), self.blob, len(blob)):
            raise ValueError("The C runtime rejected the integer model")
        self.state = ctypes.create_string_buffer(self.model.state_bytes)
        self.reset()

    def reset(self):
        if self.library.edn_reset(ctypes.byref(self.model), self.state, len(self.state)):
            raise RuntimeError("The C runtime could not initialize its state")

    def process(self, features: np.ndarray) -> np.ndarray:
        if features.dtype != np.int8 or features.ndim != 2 or features.shape[1] != self.model.input_channels:
            raise ValueError("Expected an INT8 frame-by-feature matrix")
        features = np.ascontiguousarray(features)
        result = np.empty((len(features), self.model.output_channels), dtype=np.int8)
        for index in range(len(features)):
            status = self.library.edn_process_frame(
                ctypes.byref(self.model), self.state, len(self.state),
                features[index].ctypes.data, result[index].ctypes.data)
            if status:
                raise RuntimeError(f"Integer inference failed at frame {index}")
        return result


class IntegerWaveformEnhancer(nn.Module):
    """Waveform adapter for EDNSI8, with a C or NumPy integer neural core.

    Accepts CPU float32 [B,N] input and returns the same shape and alignment.
    State is reset per utterance, retained across every frame, and flushed at
    the end. This class contains no floating-point neural network parameters.
    """

    def __init__(self, source: str | Path | bytes, backend: str = "c"):
        super().__init__()
        if backend not in {"c", "numpy"}:
            raise ValueError("backend must be c or numpy")
        metadata = IntegerDenoiser(source)
        self.backend = backend
        self.network = _CIntegerNetwork(metadata.data) if backend == "c" else metadata
        self.sample_rate = metadata.sample_rate
        self.n_fft = metadata.n_fft
        self.hop_length = metadata.hop_length
        self.low_bins = metadata.low_bins
        self.high_bands = metadata.high_bands
        self.mask_scale = metadata.mask_scale
        self.output_gain = metadata.output_gain
        self.input_exponent = metadata.input_exponent
        self.output_exponent = metadata.output_exponent
        self.model_bytes = len(metadata.data)
        self.neural_state_bytes = metadata.state_bytes
        self.model_sha256 = hashlib.sha256(metadata.data).hexdigest()
        if (self.sample_rate, self.n_fft, self.hop_length,
            self.low_bins, self.high_bands, metadata.input_channels, metadata.output_channels) != (
                16000, 512, 256, 65, 64, 387, 514):
            raise ValueError("Unsupported DSP/feature contract in the integer model")
        if not math.isfinite(self.mask_scale) or self.mask_scale <= 0:
            raise ValueError("Invalid mask scale in the integer model")
        for name in ("window", "erb_lower", "erb_upper", "erb_lower_weight", "erb_upper_weight"):
            values = getattr(metadata, name)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite DSP constants: {name}")
            tensor = torch.from_numpy(values.copy())
            if name in {"erb_lower", "erb_upper"}:
                tensor = tensor.long()
                if bool((tensor >= self.high_bands).any()):
                    raise ValueError("Invalid sparse filterbank indices")
            self.register_buffer(name, tensor)

    def _merge_bands(self, values: torch.Tensor) -> torch.Tensor:
        high = values[..., self.low_bins:]
        bands = values.new_zeros((*values.shape[:-1], self.high_bands))
        bands.index_add_(-1, self.erb_lower, high * self.erb_lower_weight)
        bands.index_add_(-1, self.erb_upper, high * self.erb_upper_weight)
        return torch.cat((values[..., :self.low_bins], bands), dim=-1)

    def _features(self, frames: torch.Tensor):
        spectrum = torch.fft.rfft(frames * self.window, n=self.n_fft)
        rms = frames.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        normalized = spectrum / (self.n_fft * rms)
        root = self._merge_bands(normalized.abs()).clamp_min(1e-8).sqrt()
        real = self._merge_bands(normalized.real) / root
        imag = self._merge_bands(normalized.imag) / root
        return spectrum, torch.cat((root, real, imag), dim=-1)

    @torch.inference_mode()
    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 2 or noisy.shape[-1] == 0 or noisy.device.type != "cpu":
            raise ValueError("IntegerWaveformEnhancer expects nonempty CPU audio [B,N]")
        if not noisy.is_floating_point() or not bool(torch.isfinite(noisy).all()):
            raise ValueError("Audio must be finite floating-point samples")
        noisy = noisy.float()
        samples = noisy.shape[-1]
        hop = self.hop_length
        padded = F.pad(noisy, (hop, (-samples) % hop + hop))
        frames = padded.unfold(-1, self.n_fft, hop)
        spectrum, features = self._features(frames)
        encoded = round_away(features / (2.0 ** self.input_exponent)).clamp(-128, 127).to(torch.int8)
        outputs = []
        for sequence in encoded.numpy():
            self.network.reset()
            outputs.append(torch.from_numpy(self.network.process(sequence)))
        deltas = torch.stack(outputs).float() * (2.0 ** self.output_exponent)
        real, imag = deltas.split(self.n_fft // 2 + 1, dim=-1)
        enhanced = spectrum * torch.complex(1.0 + self.mask_scale * real, self.mask_scale * imag)
        synthesis = torch.fft.irfft(enhanced, n=self.n_fft) * self.window
        output = F.fold(synthesis.transpose(1, 2), (1, padded.shape[-1]),
                        kernel_size=(1, self.n_fft), stride=(1, hop))
        weights = self.window.square().view(1, -1, 1).expand(1, -1, frames.shape[1])
        denominator = F.fold(weights, (1, padded.shape[-1]),
                             kernel_size=(1, self.n_fft), stride=(1, hop))
        output = output[:, 0, 0] / denominator[0, 0, 0].clamp_min(1e-8)
        return output[:, hop:hop + samples] * self.output_gain


def load_checkpoint(source: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    # Hash the bytes actually loaded, even if a training job replaces the path.
    data = Path(source).read_bytes()
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    kind = checkpoint_kind(checkpoint)
    model = build_model(kind, checkpoint["model_config"], checkpoint=True)
    phase = checkpoint.get("phase", "float")
    if phase == "qat":
        configure_model_qat(model, kind)
    elif phase != "float":
        raise ValueError(f"Unknown checkpoint phase: {phase}")
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), {
        "phase": phase,
        "model_kind": kind,
        "precision": "fake-quantized PyTorch simulation" if phase == "qat" else "float32 PyTorch",
        "model_config": asdict(model.config),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_sha256": hashlib.sha256(data).hexdigest(),
    }


def _verify_audio_record(record: dict) -> tuple[dict, dict, int]:
    """Bind scored files to their declared length and, when present, hashes."""
    hashes, signatures, declared = {}, {}, 0
    for role in ("clean", "noisy"):
        path = Path(record[role])
        stat = path.stat()
        signatures[role] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        info = sf.info(path)
        if info.frames != record["samples"] or info.samplerate != record["sample_rate"] or info.channels != 1:
            raise ValueError(f"Full-file length/rate/channel mismatch for {record['id']}: {role}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[role] = digest.hexdigest()
        if role + "_sha256" in record:
            if record[role + "_sha256"] != hashes[role]:
                raise ValueError(f"Audio SHA256 mismatch for {record['id']}: {role}")
            declared += 1
    return hashes, signatures, declared


def _summarize_preservation(records: list[dict]) -> dict:
    """Keep level-sensitive diagnostics visible beside gain-invariant SI-SDR."""
    result = {"weighting": "equal per utterance for gains and errors; sample totals for PCM16 counts",
              "interpretation": "SI-SDR ignores global gain and polarity; rail contact alone does not prove clipping"}
    for role in ("noisy", "enhanced"):
        values = [record["preservation"][role] for record in records]
        summary = {}
        for name in ("projection_gain", "rms_ratio", "normalized_waveform_l1", "waveform_l1"):
            valid = [value[name] for value in values if value[name] is not None]
            summary[name] = {"valid_utterances": len(valid),
                             "mean": float(np.mean(valid)) if valid else None,
                             "min": min(valid) if valid else None, "max": max(valid) if valid else None}
        summary["peak_abs_max"] = max(value["peak_abs"] for value in values)
        for name in ("pcm16_out_of_range_samples", "pcm16_rail_or_exceeds_samples"):
            summary[name] = sum(value[name] for value in values)
            summary[name.removesuffix("_samples") + "_utterances"] = sum(value[name] > 0 for value in values)
        result[role] = summary
    return result


@torch.inference_mode()
def evaluate_manifest(enhancer, manifest: str | Path, *, device: str = "cpu",
                      audio_dir: str | Path | None = None, audio_examples: int = 3,
                      max_utterances: int | None = None, perceptual: bool = False) -> dict:
    """Evaluate complete unaugmented clips; averaging gives every clip one vote."""
    if audio_examples < 0 or (max_utterances is not None and max_utterances < 1):
        raise ValueError("audio_examples must be nonnegative and max_utterances positive")
    manifest = Path(manifest)
    manifest_data = manifest.read_bytes()
    dataset = PairedAudioDataset(manifest, crop_seconds=None, random_crop=False, gain_db=(0, 0))
    if manifest.read_bytes() != manifest_data:
        raise ValueError("Evaluation manifest changed while it was being loaded")
    quality = None
    if perceptual:
        from .quality import PerceptualMetrics
        quality = PerceptualMetrics(dataset.sample_rate)
    # Reuse the library inventory instead of rescanning shared libraries for
    # every utterance. Tiny metric dot products/third-octave projections can
    # spend more time scheduling BLAS workers than doing useful arithmetic.
    metric_threadpools = ThreadpoolController()
    count = len(dataset) if max_utterances is None else min(len(dataset), max_utterances)
    records = []
    declared_hashes = 0
    io_before = dict(getattr(enhancer, "io_statistics", {}))
    audio_root = Path(audio_dir) if audio_dir is not None else None
    if audio_root is not None:
        audio_root.mkdir(parents=True, exist_ok=True)
    was_training = enhancer.training if isinstance(enhancer, nn.Module) else None
    if was_training is not None:
        enhancer.eval()
    try:
        for index in range(count):
            source = dataset.records[index]
            audio_hashes, signatures, verified = _verify_audio_record(source)
            declared_hashes += verified
            item = dataset[index]
            for role, expected in signatures.items():
                stat = Path(source[role]).stat()
                actual = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if actual != expected:
                    raise ValueError(f"Audio changed while it was being loaded: {item['id']}: {role}")
            noisy, clean = item["noisy"].unsqueeze(0).to(device), item["clean"].unsqueeze(0).to(device)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            enhanced = enhancer(noisy)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            if enhanced.shape != noisy.shape or not bool(torch.isfinite(enhanced).all()):
                raise ValueError(f"Invalid enhanced waveform for {item['id']}")
            # Scope the BLAS limit to metrics; inference keeps the caller's
            # settings, and limits are restored even when a metric raises.
            with metric_threadpools.limit(limits=1, user_api="blas"):
                baseline = float(si_sdr(noisy, clean))
                score = float(si_sdr(enhanced, clean))
                records.append({"id": item["id"], "samples": item["length"],
                                "audio_sha256": audio_hashes,
                                "si_sdr_noisy": baseline, "si_sdr_enhanced": score,
                                "si_sdri": score - baseline, "processing_seconds": seconds})
                records[-1]["preservation"] = preservation_metrics(noisy[0], clean[0], enhanced[0])
                if quality is not None:
                    records[-1].update(quality(clean[0].cpu().numpy(), noisy[0].cpu().numpy(),
                                               enhanced[0].cpu().numpy()))
            if audio_root is not None and index < audio_examples:
                # Numeric filenames cannot escape the selected output directory.
                for role, waveform in (("noisy", noisy), ("clean", clean), ("enhanced", enhanced)):
                    sf.write(audio_root / f"{index:03d}_{role}.wav", waveform[0].cpu().numpy(),
                             dataset.sample_rate, subtype="FLOAT")
            if index % 50 == 0 or index + 1 == count:
                print(json.dumps({"event": "evaluation", "completed": index + 1, "total": count}), flush=True)
    finally:
        if was_training is not None:
            enhancer.train(was_training)
    if manifest.read_bytes() != manifest_data:
        raise ValueError("Evaluation manifest changed during scoring")
    audio_seconds = sum(r["samples"] for r in records) / dataset.sample_rate
    processing_seconds = sum(r["processing_seconds"] for r in records)
    result = {
        "summary": summarize_utterances(records),
        "preservation": _summarize_preservation(records),
        "timing": {
            "platform": platform.platform(), "device": str(device),
            "measurement": "Host offline throughput, including DSP and neural inference; not ESP32 performance or acoustic latency",
            "audio_seconds": audio_seconds, "processing_seconds": processing_seconds,
            "processing_rtf": processing_seconds / audio_seconds,
            "excludes": "compilation, model loading, disk I/O, metric computation",
            "torch_threads": torch.get_num_threads(),
        },
        "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "source_verification": {"manifest_unchanged": True, "whole_file_lengths_verified": True,
                                "audio_files_hashed": 2 * count, "declared_audio_hashes_verified": declared_hashes},
        "metric_execution": {"blas_threads": 1, "scope": "metrics only; prior limits restored before inference"},
        "limited_evaluation": count < len(dataset),
        "utterances": records,
    }
    if io_before:
        result["io_statistics"] = {key: enhancer.io_statistics[key] - value for key, value in io_before.items()}
    if quality is not None:
        from .quality import summarize_perceptual
        result["perceptual"] = {"summary": summarize_perceptual(records),
                                "implementation": quality.provenance}
    return result


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--integer-model", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("c", "numpy"), default="c")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--audio-examples", type=int, default=3)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--perceptual", action="store_true",
                        help="Also report CPU PESQ WB and STOI using optional pesq/pystoi packages")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.integer_model:
        if args.device != "cpu":
            parser.error("Integer evaluation runs the host C/NumPy runtime and requires --device cpu")
        with args.integer_model.open("rb") as handle:
            magic = handle.read(8)
        if magic == b"EDNFQ8\0\0":
            from .frequency_evaluate import FrequencyIntegerWaveformEnhancer
            enhancer = FrequencyIntegerWaveformEnhancer(args.integer_model, args.backend)
        else:
            enhancer = IntegerWaveformEnhancer(args.integer_model, args.backend)
        provenance = {"source": str(args.integer_model.resolve()), "backend": args.backend,
                      "precision": "INT8 neural weights, activations and persistent state; float32 external DSP",
                      "model_bytes": enhancer.model_bytes,
                      "neural_state_bytes": getattr(enhancer, "neural_state_bytes", None),
                      "neural_history_bytes": getattr(enhancer, "neural_history_bytes", None),
                      "model_sha256": enhancer.model_sha256}
    else:
        enhancer, provenance = load_checkpoint(args.checkpoint, args.device)
        provenance["source"] = str(args.checkpoint.resolve())
    result = evaluate_manifest(enhancer, args.manifest, device=args.device,
                               audio_dir=args.audio_dir, audio_examples=args.audio_examples,
                               max_utterances=args.max_utterances, perceptual=args.perceptual)
    result["model"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_finite(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "evaluation_complete", **result["summary"],
                      "output": str(args.output)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
