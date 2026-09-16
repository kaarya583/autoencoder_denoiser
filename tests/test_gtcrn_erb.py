"""Pinned exact ERB coefficients, shared transpose and independent C parser/math."""
import ctypes
import hashlib
import shutil
import struct

import numpy as np
import pytest
import torch

from esp32_denoiser.gtcrn_erb import (
    BANDS, HIGH_BINS, LOW_BINS, OFFSETS_BYTES, CSparseGTCRNERB, SparseGTCRNERB, _native_runtime,
)
from esp32_denoiser.vendor.gtcrn.network import ERB


@pytest.fixture
def pinned():
    with torch.random.fork_rng():
        model = ERB(65, 64)
    return model, SparseGTCRNERB.from_torch(model)


def test_exact_2040_byte_payload_preserves_every_epsilon_and_removes_duplicate_transpose(pinned, tmp_path):
    model, sparse = pinned
    dense = model.erb_fc.weight.numpy()
    stats = sparse.storage_stats()
    assert stats["payload_bytes"] == 2040
    assert stats["row_offset_bytes"] == 65 * 2
    assert stats["column_index_bytes"] == 382
    assert stats["coefficient_bytes"] == 382 * 4
    assert stats["dense_two_direction_bytes"] == 98_304
    assert stats["duplicate_transpose_bytes"] == stats["persistent_audio_state_bytes"] == 0
    assert stats["forward_three_channel_macs"] + stats["inverse_two_channel_macs"] == 1910
    assert sparse.nonzero_count == 382
    assert np.count_nonzero(np.abs(sparse.values) < 1e-10) == 62
    assert sparse.dense_matrix().tobytes() == dense.tobytes()
    assert sparse.values.tobytes() == dense[dense != 0].tobytes()
    assert hashlib.sha256(sparse.data).hexdigest() == "112fe941ff95c9a8a28ab6a9fdd571bc6e354b89dfa17994f25bd1e14574a207"
    destination = tmp_path / "erb.csr"
    destination.write_bytes(sparse.to_bytes())
    assert SparseGTCRNERB(destination).data == sparse.data
    for values in (sparse.row_offsets, sparse.column_indices, sparse.values):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 0


@pytest.mark.skipif(shutil.which("cc") is None, reason="C99 compiler required")
@pytest.mark.parametrize("case", ("random", "asymmetric", "edges", "silence"))
def test_three_channel_forward_and_two_channel_inverse_match_dense_pinned_torch(pinned, case):
    model, sparse = pinned
    native = CSparseGTCRNERB(sparse)
    rng = np.random.default_rng(729)
    forward = rng.normal(size=(2, 7, 3, 257)).astype(np.float32)
    inverse = rng.normal(size=(2, 7, 2, 129)).astype(np.float32)
    if case == "asymmetric":
        # Unequal signed scales expose accidental normalization, transposition
        # and assumptions that all three channels are magnitudes/nonnegative.
        forward *= np.array([1024, -0.015625, 2.5], np.float32)[None, None, :, None]
        inverse *= np.array([-16, 0.125], np.float32)[None, None, :, None]
    elif case == "edges":
        forward.fill(0)
        inverse.fill(0)
        forward[..., [0, 64, 65, 255, 256]] = [1, -2, 3, -4, 5]
        inverse[..., [0, 64, 65, 127, 128]] = [-1, 2, -3, 4, -5]
    elif case == "silence":
        forward.fill(0)
        inverse.fill(0)
    matrix = model.erb_fc.weight.double().numpy()
    for inputs, method, c_method, torch_method, weights in (
        (forward, sparse.forward, native.forward, model.bm, matrix),
        (inverse, sparse.inverse, native.inverse, model.bs, matrix.T),
    ):
        expected = torch_method(torch.from_numpy(inputs).movedim(-2, 1)).movedim(1, -2).numpy()
        actual = method(inputs)
        compiled = c_method(inputs)
        np.testing.assert_array_equal(actual, compiled)  # Host wrapper disables FMA contraction.
        np.testing.assert_array_equal(actual[..., :65], inputs[..., :65])
        # Dense BLAS can sum in a different order. Bound its absolute roundoff
        # by the sum of absolute float64 products, not a large arbitrary atol.
        magnitude = np.abs(inputs[..., 65:].astype(np.float64)) @ np.abs(weights).T
        tolerance = 16 * np.finfo(np.float32).eps * magnitude
        assert np.all(np.abs(actual[..., 65:].astype(np.float64) - expected[..., 65:]) <= tolerance)
        np.testing.assert_array_equal(c_method(inputs), compiled)  # Stateless deterministic replay.


@pytest.mark.skipif(shutil.which("cc") is None, reason="C99 compiler required")
def test_basis_vectors_recover_every_coefficient_in_both_directions_and_share_one_csr(pinned):
    model, sparse = pinned
    native = CSparseGTCRNERB(sparse)
    inputs = np.eye(257, dtype=np.float32)
    expected = np.zeros((257, 129), np.float32)
    expected[:65, :65] = np.eye(65)
    expected[65:, 65:] = model.erb_fc.weight.numpy().T
    np.testing.assert_array_equal(native.forward(inputs), expected)
    np.testing.assert_array_equal(sparse.forward(inputs), expected)
    np.testing.assert_array_equal(native.inverse(np.eye(129, dtype=np.float32)), expected.T)
    # Synthesis is the transpose, not a pseudoinverse: no false perfect
    # reconstruction assumption or implicit per-row normalization.
    assert not np.allclose(sparse.inverse(sparse.forward(inputs)), inputs)


def _corrupt(data, defect):
    value = bytearray(data)
    if defect == "truncated_offsets":
        return bytes(value[:129])
    if defect == "truncated_values":
        return bytes(value[:-1])
    if defect == "trailing":
        return bytes(value + b"x")
    if defect == "first_offset":
        struct.pack_into("<H", value, 0, 1)
    elif defect == "decreasing_offsets":
        struct.pack_into("<H", value, 4, 0)
    elif defect == "impossible_count":
        struct.pack_into("<H", value, 128, 65535)
    elif defect == "column_outside":
        value[OFFSETS_BYTES] = 192
    elif defect in {"duplicate_column", "unsorted_columns"}:
        offsets = np.frombuffer(data, "<u2", 65)
        row = int(np.flatnonzero(np.diff(offsets.astype(np.int32)) >= 2)[0])
        start = OFFSETS_BYTES + int(offsets[row])
        if defect == "duplicate_column":
            value[start + 1] = value[start]
        else:
            value[start], value[start + 1] = value[start + 1], value[start]
    else:
        coefficient = {"nan": float("nan"), "infinity": float("inf"), "zero": 0.0}[defect]
        struct.pack_into("<f", value, OFFSETS_BYTES + 382, coefficient)
    return bytes(value)


@pytest.mark.skipif(shutil.which("cc") is None, reason="C99 compiler required")
@pytest.mark.parametrize("defect", ("truncated_offsets", "truncated_values", "trailing", "first_offset",
                                   "decreasing_offsets", "impossible_count", "column_outside", "duplicate_column",
                                   "unsorted_columns", "nan", "infinity", "zero"))
def test_numpy_and_independent_c_parser_reject_invalid_csr(pinned, defect):
    _, sparse = pinned
    damaged = _corrupt(sparse.data, defect)
    with pytest.raises(ValueError):
        SparseGTCRNERB(damaged)
    library, directory = _native_runtime()
    assert directory
    handle = ctypes.create_string_buffer(library.ednx_erb_handle_bytes())
    original = ctypes.create_string_buffer(sparse.data)
    assert library.ednx_erb_init(handle, original, len(sparse.data)) == 0
    blob = ctypes.create_string_buffer(damaged)
    assert library.ednx_erb_init(handle, blob, len(damaged)) == -1
    # A failed re-init invalidates the previous handle instead of retaining a
    # stale pointer/model that a caller might accidentally continue to use.
    features, output = np.zeros(257, np.float32), np.zeros(129, np.float32)
    assert library.ednx_erb_forward(handle, features.ctypes.data, 257, output.ctypes.data, 129, 1) == -1


@pytest.mark.skipif(shutil.which("cc") is None, reason="C99 compiler required")
def test_c_parser_supports_unaligned_payload_and_operations_reject_alias_and_bad_sizes(pinned):
    _, sparse = pinned
    library, _ = _native_runtime()
    raw = ctypes.create_string_buffer(b"x" + sparse.data)
    handle = ctypes.create_string_buffer(library.ednx_erb_handle_bytes())
    assert library.ednx_erb_init(handle, ctypes.byref(raw, 1), len(sparse.data)) == 0
    assert library.ednx_erb_nonzero_count(handle) == 382
    features, output = np.arange(257, dtype=np.float32), np.zeros(129, np.float32)
    assert library.ednx_erb_forward(handle, features.ctypes.data, 257, output.ctypes.data, 129, 1) == 0
    np.testing.assert_array_equal(output, sparse.forward(features))
    assert library.ednx_erb_forward(handle, features.ctypes.data, 256, output.ctypes.data, 129, 1) == -1
    assert library.ednx_erb_forward(handle, features.ctypes.data, 257, output.ctypes.data, 128, 1) == -1
    assert library.ednx_erb_forward(handle, features.ctypes.data, 257, features.ctypes.data, 129, 1) == -1
    assert library.ednx_erb_forward(handle, features.ctypes.data, 257, output.ctypes.data, 129, 0) == -1
    features[0] = np.nan
    assert library.ednx_erb_forward(handle, features.ctypes.data, 257, output.ctypes.data, 129, 1) == -1


def test_builder_refuses_approximate_transpose_or_learned_and_low_precision_constants(pinned):
    model, sparse = pinned
    dense = sparse.dense_matrix()
    wrong = dense.T.copy()
    wrong[0, 0] = np.nextafter(wrong[0, 0], np.float32(0))
    with pytest.raises(ValueError, match="exact bitwise transpose"):
        SparseGTCRNERB.from_dense(dense, wrong)
    for invalid in (dense.astype(np.float64), dense[:63], np.full_like(dense, np.nan)):
        with pytest.raises(ValueError, match="finite float32"):
            SparseGTCRNERB.from_dense(invalid)
    model.erb_fc.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen float32"):
        SparseGTCRNERB.from_torch(model)
    model.erb_fc.weight.requires_grad_(False)
    model.half()
    with pytest.raises(ValueError, match="frozen float32"):
        SparseGTCRNERB.from_torch(model)


@pytest.mark.skipif(shutil.which("cc") is None, reason="C99 compiler required")
def test_empty_rows_nonfourbyte_value_alignment_and_overflow_fail_explicitly():
    matrix = np.zeros((BANDS, HIGH_BINS), np.float32)
    matrix[3, 191] = 2
    sparse = SparseGTCRNERB.from_dense(matrix)
    native = CSparseGTCRNERB(sparse)
    assert len(sparse.data) == 135  # Values start at131, deliberately unaligned.
    forward = np.arange(257, dtype=np.float32)
    np.testing.assert_array_equal(native.forward(forward), sparse.forward(forward))
    np.testing.assert_array_equal(native.inverse(np.arange(129, dtype=np.float32)), sparse.inverse(np.arange(129, dtype=np.float32)))
    for implementation in (sparse, native):
        with pytest.raises(ValueError, match="overflow"):
            implementation.forward(np.full(257, np.finfo(np.float32).max, np.float32))
        with pytest.raises(ValueError, match="overflow"):
            implementation.inverse(np.full(129, np.finfo(np.float32).max, np.float32))
        with pytest.raises(ValueError):
            implementation.forward(np.zeros(257, np.float64))
        with pytest.raises(ValueError):
            implementation.forward(np.zeros((0, 257), np.float32))
