"""Complete scalar C GTCRN neural graph, with separate NumPy waveform DSP.

The packed model is checked by both the Python loader and the native loader.
One native invocation executes the entire learned graph on INT8 arrays and
caller-owned INT8 state. No source floating learned network is retained.
FFT/ERB/masking/OLA remain the numerical reference DSP; this is not yet an
ESP32 firmware benchmark or a native audio frontend.
"""
from __future__ import annotations

import argparse
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import json
import shutil
import subprocess
import tempfile

import numpy as np

from . import gtcrn_integer_export as packed
from .gtcrn_integer import _Edge
from .runtime_cache import runtime_fingerprint


def _layout_header():
    """Render only pinned identifiers/dimensions, never model parameters."""
    symbol = lambda name: name.replace(".", "_").upper()
    lines = ["/* Generated from gtcrn_integer_export.SPECS/GRID_NAMES; checked by tests. */",
             "#ifndef EDNG_LAYOUT_H", "#define EDNG_LAYOUT_H", "#include <stdint.h>",
             f"#define EDNG_RECORDS {len(packed.SPECS)}", f"#define EDNG_GRIDS {len(packed.GRID_NAMES)}",
             f"#define EDNG_RECORD_OFFSET {packed.RECORD_OFFSET}", f"#define EDNG_ARRAY_OFFSET {packed.ARRAY_OFFSET}"]
    for index, spec in enumerate(packed.SPECS):
        suffix = f"_{spec.subtype}" if spec.kind == packed.GRU else ""
        lines.append(f"#define EDNG_OP_{symbol(spec.name)}{suffix} {index}")
    for index, name in enumerate(packed.GRID_NAMES):
        lines.append(f"#define EDNG_GRID_{symbol(name)} {index}")
    lines += ["typedef struct {", "    uint8_t kind, flags, subtype, references;",
              "    uint16_t dimensions[8];", "    int16_t input_grid, output_grid;",
              "} edng_layout_record;", "static const edng_layout_record edng_layout[EDNG_RECORDS] = {"]
    reference_counts = {packed.AFFINE: 3, packed.STREAM: 3, packed.PRELU: 1, packed.GRU: 8,
                        packed.LAYER_NORM: 2, packed.TANH: 1, packed.SIGMOID: 1, packed.ERB: 1, packed.WINDOW: 1}
    for spec in packed.SPECS:
        dimensions = ",".join(map(str, spec.dimensions))
        indices = ([-1, -1] if spec.kind in (packed.ERB, packed.WINDOW) else
                   [packed.GRID_NAMES.index(spec.name+suffix) if spec.name+suffix in packed.GRID_NAMES else -1
                    for suffix in (".input", ".output")])
        lines.append(f"    {{{spec.kind},{spec.flags},{spec.subtype},{reference_counts[spec.kind]},"
                     f"{{{dimensions}}},{indices[0]},{indices[1]}}},")
    lines += ["};", "#endif", ""]
    return "\n".join(lines)


def _runtime():
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("The GTCRN native graph requires a C99 compiler")
    directory = Path(__file__).resolve().parents[1] / "firmware"
    layout = directory / "experimental_gtcrn/layout.h"
    if layout.read_text() != _layout_header():
        raise ValueError("Generated C topology differs from the packed model schema")
    if (directory / "experimental_gtcrn/packed_tables.h").read_text() != packed.render_packed_tables_header():
        raise ValueError("Generated C nonlinear contracts differ from the packed model schema")
    sources = ("experimental_gtcrn/graph.c", "experimental_gtcrn/packed.c",
               "experimental_int8/operators.c", "experimental_int8/primitives.c")
    headers = ("experimental_gtcrn/graph.h", "experimental_gtcrn/internal.h", "experimental_gtcrn/layout.h",
               "experimental_gtcrn/packed_tables.h",
               "experimental_int8/operators.h", "experimental_int8/primitives.h")
    fingerprint = runtime_fingerprint(directory, sources + headers)
    return _runtime_for_source(directory, compiler, sources, fingerprint)


@lru_cache(maxsize=1)
def _runtime_for_source(source, compiler, sources, fingerprint):
    directory = tempfile.TemporaryDirectory(prefix="gtcrn-native-")
    path = Path(directory.name) / "gtcrn.so"
    try:
        subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", *[str(source/name) for name in sources], "-lm", "-o", str(path)],
                       check=True, capture_output=True, text=True)
        library = ctypes.CDLL(str(path))
    except Exception:
        directory.cleanup()
        raise
    for name in ("edng_handle_bytes", "edng_state_bytes", "edng_workspace_bytes"):
        getattr(library, name).argtypes = []
        getattr(library, name).restype = ctypes.c_size_t
    library.edng_init.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    library.edng_reset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.edng_input_exponent.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    library.edng_process_frame.argtypes = [ctypes.c_void_p] + [item for _ in range(4) for item in (ctypes.c_void_p, ctypes.c_size_t)]
    for name in ("edng_init", "edng_reset", "edng_input_exponent", "edng_process_frame"):
        getattr(library, name).restype = ctypes.c_int
    return library, directory


def _aligned_buffer(size):
    return (ctypes.c_uint64 * ((size+7)//8))()


def _state_layout():
    blocks = [(f"encoder.en_convs.{index}", history) for index, history in zip((2, 3, 4), (2, 4, 10))]
    blocks += [(f"decoder.de_convs.{index}", history) for index, history in zip((0, 1, 2), (10, 4, 2))]
    rows = [(name+".depth_conv", (1, 16, history, 33)) for name, history in blocks]
    rows += [(name+".tra.att_gru", (1, 1, 16)) for name, _ in blocks]
    rows += [(f"dpgrnn{index}.inter_rnn", (1, 33, 16)) for index in (1, 2)]
    offset, result = 0, []
    for name, shape in rows:
        count = int(np.prod(shape))
        result.append((name, shape, offset, offset+count))
        offset += count
    if offset != 18_048:
        raise AssertionError("Pinned native state layout changed")
    return tuple(result)


STATE_LAYOUT = _state_layout()


class _NativeGraph:
    """Bridge float external features to one complete native neural call."""
    def __init__(self, owner):
        self.owner = owner

    def frame(self, features, state=None):
        owner = self.owner
        if features.shape != (1, 3, 1, 129):
            raise ValueError("Native GTCRN needs one [1,3,1,129] ERB feature frame")
        codes = owner.backend.grid("erb.output").quantize(features).reshape(387)
        flat = np.zeros(owner.native_state_bytes, np.int8)
        if state:
            for name, shape, start, stop in STATE_LAYOUT:
                value = state[name]
                if value.shape != shape or value.dtype != np.int8:
                    raise ValueError("Native GTCRN history differs from its fixed layout")
                flat[start:stop] = value.reshape(-1)
        mask, flat = owner.neural_step(codes, flat)
        history = {name: flat[start:stop].reshape(shape) for name, shape, start, stop in STATE_LAYOUT}
        return _Edge(mask.reshape(1, 2, 1, 129), "decoder.de_convs.4.act.output"), history


class CIntegerGTCRN(packed.PackedGTCRNIntegerDenoiser):
    """Frozen packed C graph with reference DSP and caller-owned stream state.

    A model instance reuses one native workspace and must not be called
    concurrently. The C graph allocates no memory; this host convenience
    wrapper copies/allocates caller buffers to preserve branchable states.
    """
    def __init__(self, source, *, calibration=None):
        metadata = packed.load_gtcrn_integer(source, calibration=calibration)
        self.__dict__.update(metadata.__dict__)
        self.library, self._directory = _runtime()
        self.native_handle_bytes = self.library.edng_handle_bytes()
        self.native_state_bytes = self.library.edng_state_bytes()
        self.native_workspace_bytes = self.library.edng_workspace_bytes()
        if self.native_state_bytes != 18_048:
            raise ValueError("Native GTCRN state size differs from its packed topology")
        self._blob = _aligned_buffer(len(self.packed_data))
        ctypes.memmove(self._blob, self.packed_data, len(self.packed_data))
        self._handle = _aligned_buffer(self.native_handle_bytes)
        self._workspace = np.empty(self.native_workspace_bytes, np.int8)
        if self.library.edng_init(self._handle, self.native_handle_bytes, self._blob, len(self.packed_data)):
            raise ValueError("Native GTCRN rejected the packed model")
        exponent = ctypes.c_int()
        if self.library.edng_input_exponent(self._handle, ctypes.byref(exponent)) or exponent.value != self.backend.grid("erb.output").exponent:
            raise ValueError("Native GTCRN input grid differs from packed metadata")
        self.graph = _NativeGraph(self)
        self.native_frames = 0

    def neural_step(self, features, state=None):
        """One complete native neural frame; returns Q7 mask and fresh state."""
        features = np.asarray(features)
        if features.dtype != np.int8 or features.shape not in ((387,), (3, 129)):
            raise ValueError("Native features must be INT8 [387] or [3,129]")
        features = np.ascontiguousarray(features).reshape(387)
        if state is None:
            state = np.empty(self.native_state_bytes, np.int8)
            if self.library.edng_reset(self._handle, state.ctypes.data, state.nbytes):
                raise RuntimeError("Native GTCRN state reset failed")
        elif not isinstance(state, np.ndarray) or state.dtype != np.int8 or state.shape != (self.native_state_bytes,):
            raise ValueError("Native state must be one flat INT8 history of 18,048 bytes")
        else:
            state = state.copy()
        mask = np.empty((2, 129), np.int8)
        status = self.library.edng_process_frame(self._handle, state.ctypes.data, state.nbytes,
                                                features.ctypes.data, features.nbytes, mask.ctypes.data, mask.nbytes,
                                                self._workspace.ctypes.data, self._workspace.nbytes)
        if status:
            raise RuntimeError(f"Native GTCRN neural frame failed with status {status}")
        self.native_frames += 1
        return mask, state

    def statistics(self):
        return {"native_frames": self.native_frames,
                "scope": "Full C learned graph; detailed per-edge saturation counters are not instrumented"}

    def model_stats(self):
        return {**super().model_stats(), "native_handle_bytes": self.native_handle_bytes,
                "source_checkpoint_sha256": self.source_sha256,
                "native_neural_state_bytes": self.native_state_bytes,
                "native_workspace_bytes": self.native_workspace_bytes,
                "native_neural_buffer_subtotal": self.native_handle_bytes+self.native_state_bytes+self.native_workspace_bytes+387+258,
                "native_memory_scope": "Host sizeof(handle), explicit state/workspace and feature/mask buffers; excludes compiler stack, external DSP, Python objects and packed constants",
                "deployment_status": "Complete scalar C learned graph with NumPy external DSP; ESP32 build, native audio frontend and hardware timing remain unverified"}


def main():
    """Evaluate a frozen packed graph; no calibration or training is performed."""
    import torch
    from .evaluate import evaluate_manifest, _json_finite
    from .gtcrn_recurrent_probe import _development_records
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integer-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="Optional hash-bound calibration audit JSON")
    parser.add_argument("--checkpoint", type=Path, help="Optional frozen source checkpoint to verify by SHA256")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--perceptual", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or (args.max_utterances is not None and args.max_utterances < 1):
        parser.error("threads and max-utterances must be positive")
    sources = [path for path in (args.integer_model, args.manifest, args.calibration, args.checkpoint) if path is not None]
    if args.output.resolve() in [path.resolve() for path in sources]:
        parser.error("Evaluation output must differ from its frozen source paths")
    manifest_bytes = args.manifest.read_bytes()
    _development_records(args.manifest)  # Official test remains sealed.
    if args.manifest.read_bytes() != manifest_bytes:
        raise ValueError("Development manifest changed while its split was checked")
    audit_bytes = args.calibration.read_bytes() if args.calibration else None
    audit = json.loads(audit_bytes) if audit_bytes is not None else None
    torch.set_num_threads(args.threads)
    model = CIntegerGTCRN(args.integer_model, calibration=audit)
    def file_hash(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if args.checkpoint and file_hash(args.checkpoint) != model.source_sha256:
        raise ValueError("Source checkpoint SHA256 differs from the packed model")
    if args.manifest.read_bytes() != manifest_bytes:
        raise ValueError("Development manifest changed before inference")
    report = evaluate_manifest(model, args.manifest, max_utterances=args.max_utterances, perceptual=args.perceptual)
    if (args.manifest.read_bytes() != manifest_bytes or
            file_hash(args.integer_model) != model.packed_metadata["packed_sha256"] or
            args.calibration and args.calibration.read_bytes() != audit_bytes or
            args.checkpoint and file_hash(args.checkpoint) != model.source_sha256):
        raise ValueError("A frozen evaluation source changed during inference")
    report.update(model_stats=model.model_stats(), runtime_statistics=model.statistics(),
                  artifact_verification={"packed_file_sha256": model.packed_metadata["packed_sha256"],
                                         "source_checkpoint_sha256": model.source_sha256,
                                         "checkpoint_file_verified": args.checkpoint is not None,
                                         "calibration_sha256": model.packed_metadata["calibration_sha256"],
                                         "calibration_audit_file_verified": audit_bytes is not None,
                                         "scope": "Immutable file association/integrity; training lineage requires the separate calibration audit"},
                  io_contract="Floating waveform I/O with NumPy FFT/ERB/OLA; no PCM16 input rounding or output saturation",
                  scope="Complete scalar C learned graph; no native audio frontend, official test score or ESP32 timing claim")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_finite(report), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "summary": _json_finite(report["summary"])}))


if __name__ == "__main__":
    main()
