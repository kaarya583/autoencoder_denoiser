"""Evaluate the complete deployable C frontend, including optional PCM16 I/O.

The host uses a portable FFT instead of ESP-DSP. This tests the same streaming
DSP formulas and integer neural core, but cannot establish ESP32 performance.
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

from .evaluate import IntegerWaveformEnhancer, _json_finite, evaluate_manifest
from .export import IntegerDenoiser
from .runtime_cache import runtime_fingerprint


_WRAPPER = r"""
#define _POSIX_C_SOURCE 200112L
#include "audio.h"
#include <stdlib.h>
typedef struct { edn_model model; edn_audio_state audio; void *neural; } host_context;
void *edn_host_create(const void *blob, size_t bytes) {
    host_context *context;
    if (posix_memalign((void **)&context, 16, sizeof(*context))) return NULL;
    context->neural = NULL;
    if (edn_init(&context->model, blob, bytes)) { free(context); return NULL; }
    context->neural = calloc(1, context->model.state_bytes);
    if (!context->neural || edn_audio_init(&context->audio, &context->model,
                                          context->neural, context->model.state_bytes)) {
        free(context->neural); free(context); return NULL;
    }
    return context;
}
void edn_host_destroy(void *handle) {
    host_context *context = handle;
    if (context) { free(context->neural); free(context); }
}
int edn_host_reset(void *handle) {
    return edn_audio_reset(&((host_context *)handle)->audio);
}
int edn_host_float(void *handle, const float *input, float *output) {
    return edn_audio_process(&((host_context *)handle)->audio, input, output);
}
int edn_host_pcm16(void *handle, const int16_t *input, int16_t *output) {
    return edn_audio_process_pcm16(&((host_context *)handle)->audio, input, output);
}
"""


def _compiled_audio_runtime():
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("Complete C audio evaluation requires a C99 compiler (cc)")
    source = Path(__file__).resolve().parents[1] / "firmware/esp32_denoiser"
    fingerprint = runtime_fingerprint(source, ('denoiser.c', 'denoiser.h', 'integer_kernels.h', 'audio.c', 'audio.h', 'audio_dsp.h'), _WRAPPER)
    return _compiled_audio_runtime_for_source(source, compiler, fingerprint)


@lru_cache(maxsize=1)
def _compiled_audio_runtime_for_source(source, compiler, fingerprint):
    # fingerprint is part of the cache key, including transitively included headers.
    directory = tempfile.TemporaryDirectory(prefix="esp32-denoiser-audio-")
    wrapper = Path(directory.name) / "host.c"
    library_path = Path(directory.name) / "audio.so"
    wrapper.write_text(_WRAPPER)
    try:
        subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", "-I", str(source), str(wrapper),
                        str(source / "denoiser.c"), str(source / "audio.c"),
                        "-lm", "-o", str(library_path)],
                       check=True, capture_output=True, text=True)
        library = ctypes.CDLL(str(library_path))
    except Exception:
        directory.cleanup()
        raise
    library.edn_host_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.edn_host_create.restype = ctypes.c_void_p
    library.edn_host_destroy.argtypes = [ctypes.c_void_p]
    library.edn_host_destroy.restype = None
    library.edn_host_reset.argtypes = [ctypes.c_void_p]
    library.edn_host_reset.restype = ctypes.c_int
    for name in ("edn_host_float", "edn_host_pcm16"):
        getattr(library, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        getattr(library, name).restype = ctypes.c_int
    return library, directory


class EmbeddedWaveformEnhancer(nn.Module):
    """Complete C inference on CPU audio [B,N], reset independently per row.

    ``pcm16`` rounds input to nearest even, clips to signed PCM16, invokes the
    firmware's PCM16 API, and decodes output back to float. ``float32`` preserves
    the firmware's unclipped float output. Both compensate one overlap hop and
    flush the final hop. A single instance must not be called concurrently.
    """

    def __init__(self, source: str | Path | bytes, io_format: str = "pcm16"):
        super().__init__()
        if io_format not in {"float32", "pcm16"}:
            raise ValueError("io_format must be float32 or pcm16")
        metadata = IntegerDenoiser(source)
        self.io_format = io_format
        self.sample_rate = metadata.sample_rate
        self.hop_length = metadata.hop_length
        self.model_bytes = len(metadata.data)
        self.neural_state_bytes = metadata.state_bytes
        self.model_sha256 = hashlib.sha256(metadata.data).hexdigest()
        self.library, self.directory = _compiled_audio_runtime()
        self.blob = ctypes.create_string_buffer(metadata.data)
        self.handle = self.library.edn_host_create(self.blob, len(metadata.data))
        if not self.handle:
            raise ValueError("The C audio frontend rejected the integer model")
        self.reset_statistics()

    def reset_statistics(self):
        self.io_statistics = {"samples": 0, "input_clipped_samples": 0, "output_at_rail_samples": 0}

    def close(self):
        if getattr(self, "handle", None):
            self.library.edn_host_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    @torch.inference_mode()
    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        if not self.handle:
            raise RuntimeError("This audio enhancer has been closed")
        if noisy.ndim != 2 or noisy.shape[-1] == 0 or noisy.device.type != "cpu":
            raise ValueError("EmbeddedWaveformEnhancer expects nonempty CPU audio [B,N]")
        if not noisy.is_floating_point() or not bool(torch.isfinite(noisy).all()):
            raise ValueError("Audio must be finite floating-point samples")
        count = noisy.shape[-1]
        outputs = []
        for row in noisy.detach().float().numpy():
            padded = np.pad(row, (0, (-count) % self.hop_length + self.hop_length))
            if self.io_format == "pcm16":
                scaled = np.rint(padded * 32768.0)
                self.io_statistics["input_clipped_samples"] += int(
                    np.count_nonzero((scaled[:count] < -32768) | (scaled[:count] > 32767)))
                padded = np.clip(scaled, -32768, 32767).astype(np.int16)
                process = self.library.edn_host_pcm16
            else:
                process = self.library.edn_host_float
            output = np.empty_like(padded)
            if self.library.edn_host_reset(self.handle):
                raise RuntimeError("The C audio frontend failed to reset")
            for offset in range(0, len(padded), self.hop_length):
                if process(self.handle, padded[offset:].ctypes.data, output[offset:].ctypes.data):
                    raise RuntimeError(f"C audio processing failed at sample {offset}")
            output = output[self.hop_length:self.hop_length + count]
            if self.io_format == "pcm16":
                self.io_statistics["output_at_rail_samples"] += int(
                    np.count_nonzero((output == -32768) | (output == 32767)))
                output = output.astype(np.float32) / 32768.0
            self.io_statistics["samples"] += count
            outputs.append(torch.from_numpy(output.copy()))
        return torch.stack(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integer-model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--io-format", choices=("pcm16", "float32"), default="pcm16")
    parser.add_argument("--compare-reference", action="store_true",
                        help="Also score C neural/Torch DSP and unclipped full C audio on the same clips")
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--perceptual", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    # Every reference pass must use the exact deployed bytes scored first.
    # A concurrently replaced model path must not change the comparison.
    model_data = args.integer_model.read_bytes()
    frequency = model_data[:8] == b"EDNFQ8\0\0"
    if frequency:
        from .frequency_embedded import FrequencyEmbeddedWaveformEnhancer
        from .frequency_evaluate import FrequencyIntegerWaveformEnhancer
        embedded_type, reference_type = FrequencyEmbeddedWaveformEnhancer, FrequencyIntegerWaveformEnhancer
    else:
        embedded_type, reference_type = EmbeddedWaveformEnhancer, IntegerWaveformEnhancer
    enhancer = embedded_type(model_data, args.io_format)
    try:
        result = evaluate_manifest(enhancer, args.manifest, max_utterances=args.max_utterances,
                                   audio_dir=args.audio_dir, perceptual=args.perceptual)
        result["model"] = {"source": str(args.integer_model.resolve()),
                           "model_sha256": enhancer.model_sha256, "model_bytes": enhancer.model_bytes,
                           "neural_state_bytes": enhancer.neural_state_bytes,
                           "precision": "INT8 C neural core, float32 C DSP, portable host FFT",
                           "io_format": enhancer.io_format, "io_statistics": result["io_statistics"]}
        if args.compare_reference:
            reference = evaluate_manifest(reference_type(model_data), args.manifest,
                                          max_utterances=args.max_utterances)
            comparison = {"torch_dsp_integer_neural_summary": reference["summary"],
                          "interpretation": "All passes use original aligned float inputs and the same model bytes. "
                          "PCM16 versus float32 includes BOTH input and output quantization/clipping; it does not isolate output clipping."}
            float_audio = embedded_type(model_data, "float32")
            try:
                unclipped = evaluate_manifest(float_audio, args.manifest, max_utterances=args.max_utterances)
            finally:
                float_audio.close()
            comparison["float32_full_c_summary"] = unclipped["summary"]
            comparison["float32_full_c_preservation"] = unclipped["preservation"]
            def identities(report):
                return [(row["id"], row["samples"], row["audio_sha256"]) for row in report["utterances"]]

            for name, other in (("versus_torch_dsp", reference), ("versus_unclipped_c", unclipped)):
                if (other["manifest_sha256"] != result["manifest_sha256"] or
                        identities(other) != identities(result)):
                    raise ValueError("Reference comparison requires identical manifest and audio bytes")
                score, other_score = result["summary"]["si_sdri"], other["summary"]["si_sdri"]
                comparison[name + "_si_sdri_difference_db"] = (
                    score - other_score if score is not None and other_score is not None else None)
            result["comparison"] = comparison
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_json_finite(result), indent=2, allow_nan=False) + "\n")
        print(json.dumps({"event": "embedded_evaluation_complete", **result["summary"],
                          "output": str(args.output)}, allow_nan=False), flush=True)
    finally:
        enhancer.close()


if __name__ == "__main__":
    main()
