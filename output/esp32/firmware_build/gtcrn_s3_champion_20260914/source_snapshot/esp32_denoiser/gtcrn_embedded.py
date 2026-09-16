"""Complete C GTCRN audio path, with explicit float32 or PCM16 waveform I/O.

FFT/ERB/masking/OLA are conventional C float32 DSP; every learned operation
and recurrent history uses the packed integer contract. Host FFT arithmetic
differs from ESP-DSP, so host parity/latency cannot establish MCU throughput.
"""
from __future__ import annotations

import argparse
import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import torch
from torch import nn

from .gtcrn_integer_export import load_gtcrn_integer, render_packed_tables_header
from .gtcrn_native import _layout_header
from .runtime_cache import runtime_fingerprint


def _runtime():
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("Complete GTCRN C audio requires a C99 compiler")
    root = Path(__file__).resolve().parents[1] / "firmware"
    if (root / "experimental_gtcrn/layout.h").read_text() != _layout_header() or (root / "experimental_gtcrn/packed_tables.h").read_text() != render_packed_tables_header():
        raise ValueError("Generated C packed topology/constants are stale")
    sources = ("experimental_gtcrn/audio.c", "experimental_gtcrn/graph.c", "experimental_gtcrn/packed.c",
               "experimental_int8/operators.c", "experimental_int8/primitives.c", "experimental_dsp/gtcrn_erb.c")
    headers = ("experimental_gtcrn/audio.h", "experimental_gtcrn/graph.h", "experimental_gtcrn/internal.h",
               "experimental_gtcrn/layout.h", "experimental_gtcrn/packed_tables.h", "experimental_int8/operators.h",
               "experimental_int8/primitives.h", "experimental_dsp/gtcrn_erb.h", "esp32_denoiser/audio_dsp.h")
    return _compile(root, compiler, sources, runtime_fingerprint(root, sources+headers))


@lru_cache(maxsize=1)
def _compile(root, compiler, sources, fingerprint):
    directory = tempfile.TemporaryDirectory(prefix="gtcrn-complete-audio-")
    path = Path(directory.name) / "audio.so"
    try:
        subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                        *[str(root / name) for name in sources], "-lm", "-o", str(path)],
                       check=True, capture_output=True, text=True)
        library = ctypes.CDLL(str(path))
    except Exception:
        directory.cleanup()
        raise
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    for name in ("edng_handle_bytes", "edng_state_bytes", "edng_workspace_bytes", "edng_audio_bytes"):
        getattr(library, name).argtypes = []
        getattr(library, name).restype = size
    signatures = {"edng_init": [pointer, size, pointer, size],
                  "edng_audio_init": [pointer, size, pointer, pointer, size, pointer, size],
                  "edng_audio_reset": [pointer], "edng_audio_process": [pointer, pointer, pointer],
                  "edng_audio_process_pcm16": [pointer, pointer, pointer]}
    for name, signature in signatures.items():
        getattr(library, name).argtypes = signature
        getattr(library, name).restype = ctypes.c_int
    return library, directory


def _aligned(size):
    raw = np.zeros(size+15, dtype=np.uint8)
    start = (-raw.ctypes.data) % 16
    return raw[start:start+size]


class EmbeddedGTCRN(nn.Module):
    """Host-compiled full C audio, resetting each utterance independently.

    PCM16 rounds/clips input once, invokes the C PCM16 API, and returns decoded
    output. Float32 keeps the unclipped C output. Both remove the initial
    overlap hop and flush one final zero hop. Instances are not thread-safe.
    """
    def __init__(self, source, *, io_format="pcm16", calibration=None):
        super().__init__()
        if io_format not in {"pcm16", "float32"}:
            raise ValueError("io_format must be pcm16 or float32")
        snapshot = load_gtcrn_integer(source, calibration=calibration)
        self.config, self.sample_rate, self.hop_length = snapshot.config, 16000, 256
        self.io_format, self.metadata = io_format, snapshot.model_stats()
        self.model_sha256, self.source_sha256 = snapshot.packed_metadata["packed_sha256"], snapshot.source_sha256
        self.library, self.directory = _runtime()
        self.buffers = {name: _aligned(size) for name, size in (
            ("blob", len(snapshot.packed_data)), ("model", self.library.edng_handle_bytes()),
            ("audio", self.library.edng_audio_bytes()), ("neural", self.library.edng_state_bytes()),
            ("workspace", self.library.edng_workspace_bytes()))}
        self.buffers["blob"][:] = np.frombuffer(snapshot.packed_data, np.uint8)
        ptr = lambda name: self.buffers[name].ctypes.data
        count = lambda name: self.buffers[name].nbytes
        if self.library.edng_init(ptr("model"), count("model"), ptr("blob"), count("blob")):
            raise ValueError("C rejected the packed GTCRN model")
        if self.library.edng_audio_init(ptr("audio"), count("audio"), ptr("model"), ptr("neural"), count("neural"), ptr("workspace"), count("workspace")):
            raise ValueError("C rejected the GTCRN audio state")
        self.handle = ptr("audio")
        self.reset_statistics()
        self.eval()

    def reset_statistics(self):
        self.io_statistics = dict(samples=0, input_clipped_samples=0, output_at_rail_samples=0)

    def close(self):
        self.handle = None
        self.buffers.clear()

    @torch.inference_mode()
    def forward(self, noisy):
        if not self.handle:
            raise RuntimeError("This C audio enhancer is closed")
        if not isinstance(noisy, torch.Tensor) or noisy.device.type != "cpu" or noisy.ndim != 2 or min(noisy.shape) < 1 or not noisy.is_floating_point() or not bool(torch.isfinite(noisy).all()):
            raise ValueError("Expected finite nonempty CPU waveforms[B,N]")
        samples, outputs = noisy.shape[-1], []
        for row in noisy.detach().float().numpy():
            padded = np.pad(row, (0, (-samples)%256+256))
            if self.io_format == "pcm16":
                scaled = np.rint(padded.astype(np.float64)*32768)
                self.io_statistics["input_clipped_samples"] += int(np.count_nonzero((scaled[:samples] < -32768)|(scaled[:samples] > 32767)))
                padded = scaled.clip(-32768, 32767).astype(np.int16)
                process = self.library.edng_audio_process_pcm16
            else:
                process = self.library.edng_audio_process
            result = np.empty_like(padded)
            if self.library.edng_audio_reset(self.handle):
                raise RuntimeError("C audio state reset failed")
            for offset in range(0, len(padded), 256):
                if process(self.handle, padded[offset:].ctypes.data, result[offset:].ctypes.data):
                    raise RuntimeError(f"C GTCRN audio processing failed at sample{offset}")
            result = result[256:256+samples]
            if self.io_format == "pcm16":
                self.io_statistics["output_at_rail_samples"] += int(np.count_nonzero((result == -32768)|(result == 32767)))
                result = result.astype(np.float32)/32768
            self.io_statistics["samples"] += samples
            outputs.append(torch.from_numpy(result.copy()))
        return torch.stack(outputs)

    def model_stats(self):
        return {**self.metadata, "io_format": self.io_format,
                "native_buffer_bytes": {name: buffer.nbytes for name, buffer in self.buffers.items()},
                "native_ram_subtotal_bytes": sum(value.nbytes for name, value in self.buffers.items() if name != "blob"),
                "native_memory_scope": "Host sizeof explicit buffers; excludes stack, caller audio I/O, vendor FFT tables and Python objects; packed blob can reside in flash",
                "deployment_status": "Complete scalar C audio path on host; ESP32 build and sustained timing require separate verification"}


def main():
    from .evaluate import evaluate_manifest, _json_finite
    from .gtcrn_recurrent_probe import _development_records
    from .comparison import compare_evaluations
    from .gtcrn_native import CIntegerGTCRN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integer-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--io-format", choices=("pcm16", "float32"), default="pcm16")
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--perceptual", action="store_true")
    parser.add_argument("--compare-reference", action="store_true", help="Also evaluate the same packed C neural graph with NumPy DSP")
    args = parser.parse_args()
    if args.threads < 1 or args.max_utterances is not None and args.max_utterances < 1:
        parser.error("threads and max-utterances must be positive")
    sources = [args.integer_model, args.manifest] + ([args.calibration] if args.calibration else [])
    if args.output.resolve() in [path.resolve() for path in sources]:
        parser.error("Output must differ from immutable evaluation inputs")
    contents = {path: path.read_bytes() for path in sources}
    _development_records(args.manifest)  # Official test remains sealed.
    audit = json.loads(contents[args.calibration]) if args.calibration else None
    torch.set_num_threads(args.threads)
    model = EmbeddedGTCRN(contents[args.integer_model], io_format=args.io_format, calibration=audit)
    report = evaluate_manifest(model, args.manifest, max_utterances=args.max_utterances, perceptual=args.perceptual)
    report.update(model_stats=model.model_stats(), io_statistics=model.io_statistics,
                  scope="Complete C audio path on host; external DSP float32, learned graph INT8; no official test or MCU timing claim")
    if args.compare_reference:
        reference = CIntegerGTCRN(contents[args.integer_model], calibration=audit)
        control = evaluate_manifest(reference, args.manifest, max_utterances=args.max_utterances, perceptual=args.perceptual)
        report.update(numpy_dsp_reference=control, dsp_comparison=compare_evaluations(report, control))
    if any(path.read_bytes() != data for path, data in contents.items()):
        raise ValueError("A frozen evaluation input changed during inference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_finite(report), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "summary": _json_finite(report["summary"])}))


if __name__ == "__main__":
    main()
