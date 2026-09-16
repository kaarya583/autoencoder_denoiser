"""One exact CSR table for GTCRN's fixed ERB transform and its transpose.

The pinned64x192 float32 matrix has382 nonzeros, including62 tiny epsilon
entries. They are retained bit-for-bit. Forward copies65 low bins and maps
192->64 high bins; inverse copies65 low bins and applies the same matrix's
transpose64->192. "Inverse" names the upstream synthesis map, not a matrix
inverse or a promise of lossless spectral reconstruction.

This isolated external float32 DSP block is not an INT8 neural graph, learned
pruning, or an MCU throughput measurement. No model/vendor file is changed.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .runtime_cache import runtime_fingerprint


LOW_BINS, BANDS, HIGH_BINS = 65, 64, 192
FFT_BINS, ERB_BINS = LOW_BINS + HIGH_BINS, LOW_BINS + BANDS
OFFSETS_BYTES = (BANDS + 1) * 2
UPSTREAM_REVISION = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"


def _features(value, bins):
    value = np.asarray(value)
    if value.dtype != np.float32 or value.ndim < 1 or value.shape[-1] != bins or value.size == 0:
        raise ValueError(f"Expected nonempty float32 features with last dimension{bins}")
    if not np.isfinite(value).all():
        raise ValueError("ERB input must be finite")
    return value


class SparseGTCRNERB:
    """Validated immutable CSR snapshot, with ordered float32 reference math.

    Headerless payload:65 uint16 little-endian offsets, nnz uint8 columns,
    then nnz float32 little-endian values. The final offset supplies nnz.
    Dimensions are the fixed FFT512 GTCRN contract, so no dimensions or
    duplicate transpose need to be stored in this DSP table.
    """
    def __init__(self, source: bytes | str | Path):
        self.data = bytes(source) if isinstance(source, bytes) else Path(source).read_bytes()
        if len(self.data) < OFFSETS_BYTES:
            raise ValueError("Truncated ERB row offsets")
        self.row_offsets = np.frombuffer(self.data, dtype="<u2", count=BANDS + 1)
        self.nonzero_count = int(self.row_offsets[-1])
        if (self.nonzero_count > BANDS * HIGH_BINS or
                len(self.data) != OFFSETS_BYTES + 5 * self.nonzero_count):
            raise ValueError("Invalid ERB payload size or nonzero count")
        offsets = self.row_offsets.astype(np.int32)
        if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError("ERB row offsets must start at zero and be monotone")
        self.column_indices = np.frombuffer(self.data, dtype=np.uint8, count=self.nonzero_count, offset=OFFSETS_BYTES)
        self.values = np.frombuffer(self.data, dtype="<f4", count=self.nonzero_count,
                                   offset=OFFSETS_BYTES + self.nonzero_count)
        if np.any(self.column_indices >= HIGH_BINS) or not np.isfinite(self.values).all() or np.any(self.values == 0):
            raise ValueError("ERB columns or finite nonzero coefficients are invalid")
        for row in range(BANDS):
            columns = self.column_indices[offsets[row]:offsets[row + 1]].astype(np.int32)
            if np.any(np.diff(columns) <= 0):
                raise ValueError("ERB columns must be strictly increasing within each row")

    @classmethod
    def from_dense(cls, matrix, inverse_matrix=None):
        matrix = np.asarray(matrix)
        if matrix.shape != (BANDS, HIGH_BINS) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
            raise ValueError("Expected the finite float32 GTCRN64x192 high-band matrix")
        if inverse_matrix is not None:
            inverse = np.asarray(inverse_matrix)
            if (inverse.shape != (HIGH_BINS, BANDS) or inverse.dtype != np.float32 or
                    inverse.tobytes(order="C") != matrix.T.tobytes(order="C")):
                raise ValueError("The inverse ERB matrix must be an exact bitwise transpose")
        offsets, columns, values = [0], [], []
        for row in matrix:
            # Exact comparison, never a threshold: upstream epsilon entries
            # are part of the fixed transform and must not be pruned.
            indices = np.flatnonzero(row != 0)
            columns.extend(indices.tolist())
            values.extend(row[indices].tolist())
            offsets.append(len(columns))
        return cls(np.asarray(offsets, dtype="<u2").tobytes() + np.asarray(columns, dtype=np.uint8).tobytes()
                   + np.asarray(values, dtype="<f4").tobytes())

    @classmethod
    def from_torch(cls, erb):
        """Snapshot the given frozen upstream ERB module; never build a model."""
        import torch
        if (getattr(erb, "erb_subband_1", None) != LOW_BINS or
                erb.erb_fc.bias is not None or erb.ierb_fc.bias is not None):
            raise ValueError("Unsupported GTCRN ERB shape/bias contract")
        weights = (erb.erb_fc.weight, erb.ierb_fc.weight)
        if any(weight.dtype != torch.float32 or weight.requires_grad for weight in weights):
            raise ValueError("ERB coefficients must be frozen float32 DSP constants")
        return cls.from_dense(*(weight.detach().cpu().numpy() for weight in weights))

    def to_bytes(self):
        return self.data

    def dense_matrix(self):
        """Materialize only for inspection/oracles; inference uses the CSR."""
        result = np.zeros((BANDS, HIGH_BINS), np.float32)
        for row in range(BANDS):
            start, end = int(self.row_offsets[row]), int(self.row_offsets[row + 1])
            result[row, self.column_indices[start:end]] = self.values[start:end]
        return result

    def storage_stats(self):
        return {"format": "GTCRN fixed65/64/192 CSR, little-endian, headerless",
                "coefficient_precision": "float32 external DSP; unchanged nonzero values",
                "nonzero_coefficients": self.nonzero_count,
                "row_offset_bytes": self.row_offsets.nbytes, "column_index_bytes": self.column_indices.nbytes,
                "coefficient_bytes": self.values.nbytes, "payload_bytes": len(self.data),
                "payload_sha256": hashlib.sha256(self.data).hexdigest(),
                "dense_one_direction_bytes": BANDS * HIGH_BINS * 4,
                "dense_two_direction_bytes": BANDS * HIGH_BINS * 8,
                "duplicate_transpose_bytes": 0, "persistent_audio_state_bytes": 0,
                "forward_three_channel_macs": 3 * self.nonzero_count,
                "inverse_two_channel_macs": 2 * self.nonzero_count,
                "scope": "Table bytes and scalar multiply-add counts only; executable code, handles and caller buffers excluded; no MCU timing claim"}

    def forward(self, features):
        features = _features(features, FFT_BINS)
        output = np.zeros((*features.shape[:-1], ERB_BINS), dtype=np.float32)
        output[..., :LOW_BINS] = features[..., :LOW_BINS]
        with np.errstate(over="ignore", invalid="ignore"):
            for row in range(BANDS):
                for index in range(int(self.row_offsets[row]), int(self.row_offsets[row + 1])):
                    output[..., LOW_BINS + row] += self.values[index] * features[..., LOW_BINS + int(self.column_indices[index])]
        if not np.isfinite(output).all():
            raise ValueError("ERB forward overflowed float32")
        return output

    def inverse(self, features):
        features = _features(features, ERB_BINS)
        output = np.zeros((*features.shape[:-1], FFT_BINS), dtype=np.float32)
        output[..., :LOW_BINS] = features[..., :LOW_BINS]
        with np.errstate(over="ignore", invalid="ignore"):
            for row in range(BANDS):
                for index in range(int(self.row_offsets[row]), int(self.row_offsets[row + 1])):
                    output[..., LOW_BINS + int(self.column_indices[index])] += self.values[index] * features[..., LOW_BINS + row]
        if not np.isfinite(output).all():
            raise ValueError("ERB inverse overflowed float32")
        return output


def _native_runtime():
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("The sparse ERB C oracle needs a C99 compiler")
    source = Path(__file__).resolve().parents[1] / "firmware/experimental_dsp"
    fingerprint = runtime_fingerprint(source, ("gtcrn_erb.c", "gtcrn_erb.h"))
    return _compile_native(source, compiler, fingerprint)


@lru_cache(maxsize=1)
def _compile_native(source, compiler, fingerprint):
    directory = tempfile.TemporaryDirectory(prefix="gtcrn-erb-")
    target = Path(directory.name) / "erb.so"
    try:
        subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                        "-shared", "-fPIC", str(source / "gtcrn_erb.c"), "-o", str(target)],
                       check=True, capture_output=True, text=True)
        library = ctypes.CDLL(str(target))
    except Exception:
        directory.cleanup()
        raise
    library.ednx_erb_handle_bytes.restype = ctypes.c_size_t
    library.ednx_erb_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.ednx_erb_init.restype = ctypes.c_int
    library.ednx_erb_nonzero_count.argtypes = [ctypes.c_void_p]
    library.ednx_erb_nonzero_count.restype = ctypes.c_size_t
    for name in ("ednx_erb_forward", "ednx_erb_inverse"):
        method = getattr(library, name)
        method.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                           ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        method.restype = ctypes.c_int
    library.source_sha256 = fingerprint
    return library, directory


class CSparseGTCRNERB:
    """Host wrapper for the same isolated portable C DSP building block."""
    def __init__(self, source: SparseGTCRNERB | bytes | str | Path):
        self.table = source if isinstance(source, SparseGTCRNERB) else SparseGTCRNERB(source)
        self.library, self.directory = _native_runtime()
        self.blob = ctypes.create_string_buffer(self.table.data)
        self.handle = ctypes.create_string_buffer(self.library.ednx_erb_handle_bytes())
        if self.library.ednx_erb_init(self.handle, self.blob, len(self.table.data)):
            raise ValueError("The C ERB parser rejected the payload")

    def _apply(self, features, inverse):
        in_bins, out_bins = (ERB_BINS, FFT_BINS) if inverse else (FFT_BINS, ERB_BINS)
        features = np.ascontiguousarray(_features(features, in_bins))
        output = np.empty((*features.shape[:-1], out_bins), dtype=np.float32)
        method = self.library.ednx_erb_inverse if inverse else self.library.ednx_erb_forward
        if method(self.handle, features.ctypes.data, features.size, output.ctypes.data, output.size,
                  features.size // in_bins):
            raise ValueError("C ERB processing rejected the input or overflowed float32")
        return output

    def forward(self, features):
        return self._apply(features, False)

    def inverse(self, features):
        return self._apply(features, True)
