"""Exercise the actual C parser independently of the Python validation gate."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest

from esp32_denoiser.gtcrn_integer_export import render_packed_tables_header
from test_gtcrn_integer_export import packed, threads, _mutated_blob


@pytest.fixture(scope="module")
def parser(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("A C99 compiler is required")
    root = Path(__file__).parents[1] / "firmware"
    path = tmp_path_factory.mktemp("packed-parser") / "parser.so"
    subprocess.run([compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-shared", "-fPIC",
                    *[str(root / name) for name in ("experimental_gtcrn/packed.c", "experimental_int8/operators.c", "experimental_int8/primitives.c")],
                    "-lm", "-o", str(path)], check=True, capture_output=True, text=True)
    library = ctypes.CDLL(str(path))
    library.edng_handle_bytes.argtypes = []
    library.edng_handle_bytes.restype = ctypes.c_size_t
    library.edng_init.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    library.edng_init.restype = ctypes.c_int
    return library


def _buffer(size):
    return (ctypes.c_uint64 * ((size+7)//8))()


def _initialize(library, blob):
    data, handle = _buffer(len(blob)), _buffer(library.edng_handle_bytes())
    ctypes.memmove(data, blob, len(blob))
    status = library.edng_init(handle, ctypes.sizeof(handle), data, len(blob))
    assert bytes(data)[:len(blob)] == blob
    return status, data, handle


def test_native_parser_loads_both_dsp_modes_and_retains_immutable_blob(packed, parser):
    _, blob = packed
    status, _, handle = _initialize(parser, blob)
    assert status == 0
    assert ctypes.c_uint32.from_buffer(handle).value == 0x47544938
    assert parser.edng_handle_bytes() % 8 == 0


@pytest.mark.parametrize("mutation", [
    "truncated", "oversized", "corrupt", "version", "reserved", "shape", "grid", "record_grid",
    "unaligned", "outside", "unused_ref", "dot_overflow", "weight_exponent", "gru_table",
    "ln_epsilon", "ln_overflow", "window", "padding",
])
def test_native_parser_rejects_same_rehashed_unsafe_payloads(packed, parser, mutation):
    _, blob = packed
    status, _, handle = _initialize(parser, _mutated_blob(blob, mutation))
    assert status != 0
    assert ctypes.c_uint32.from_buffer(handle).value == 0


def test_native_parser_rejects_short_unaligned_or_overlapping_handles(packed, parser):
    _, blob = packed
    status, data, handle = _initialize(parser, blob)
    assert status == 0
    assert parser.edng_init(handle, parser.edng_handle_bytes()-1, data, len(blob)) != 0
    assert ctypes.c_uint32.from_buffer(handle).value == 0
    assert parser.edng_init(ctypes.byref(handle, 1), ctypes.sizeof(handle)-1, data, len(blob)) != 0
    assert parser.edng_init(handle, ctypes.sizeof(handle), ctypes.byref(data, 1), len(blob)-1) != 0
    assert parser.edng_init(data, ctypes.sizeof(data), data, len(blob)) != 0
    assert bytes(data)[:len(blob)] == blob


@pytest.mark.parametrize("invalid", ["null", "short", "oversized", "unaligned"])
def test_failed_reinitialization_invalidates_a_safe_nonoverlapping_handle(packed, parser, invalid):
    _, blob = packed
    status, data, handle = _initialize(parser, blob)
    assert status == 0
    pointer = None if invalid == "null" else ctypes.byref(data, 1) if invalid == "unaligned" else data
    size = 1 if invalid == "short" else 99001 if invalid == "oversized" else len(blob)
    assert parser.edng_init(handle, ctypes.sizeof(handle), pointer, size) != 0
    assert ctypes.c_uint32.from_buffer(handle).value == 0


def test_native_lookup_and_epsilon_constants_are_reproducible():
    path = Path(__file__).parents[1] / "firmware/experimental_gtcrn/packed_tables.h"
    assert path.read_text() == render_packed_tables_header()
