"""A long-running evaluator must not retain code from an edited shared header."""
import importlib
from pathlib import Path
import shutil

import pytest


@pytest.mark.parametrize("module_name,function,headers", [
    ("evaluate", "_compiled_runtime", ("integer_kernels.h",)),
    ("frequency_evaluate", "_frequency_runtime", ("integer_kernels.h",)),
    ("embedded", "_compiled_audio_runtime", ("integer_kernels.h", "audio_dsp.h")),
    ("frequency_embedded", "_compiled_frequency_audio", ("integer_kernels.h", "audio_dsp.h")),
])
def test_shared_header_and_wrapper_changes_recompile_without_invalidating_live_runtime(
        tmp_path, monkeypatch, module_name, function, headers):
    if shutil.which("cc") is None:
        pytest.skip("A C compiler is required to verify compiled cache invalidation")
    module = importlib.import_module("esp32_denoiser." + module_name)
    actual_source = Path(__file__).resolve().parents[1] / "firmware/esp32_denoiser"
    source = tmp_path / "firmware/esp32_denoiser"
    source.mkdir(parents=True)
    for file in actual_source.iterdir():
        if file.suffix in (".c", ".h"):
            shutil.copyfile(file, source / file.name)
    # Point only this module at the isolated copy; never edit shared repo sources.
    monkeypatch.setattr(module, "__file__", str(tmp_path / "esp32_denoiser" / (module_name + ".py")))
    get_runtime = getattr(module, function)
    previous = get_runtime()
    assert get_runtime() is previous
    retained = [previous]
    for header in headers:
        path = source / header
        path.write_text(path.read_text() + "\n/* Changed during this evaluation process. */\n")
        current = get_runtime()
        assert current is not previous
        assert current[0] is not previous[0]
        assert get_runtime() is current
        retained.append(current)
        previous = current
    if hasattr(module, "_WRAPPER"):
        monkeypatch.setattr(module, "_WRAPPER", module._WRAPPER + "\n/* Wrapper changed. */\n")
        current = get_runtime()
        assert current is not previous
        retained.append(current)
    # Existing enhancer instances retain their own library+directory references.
    assert all(Path(directory.name).is_dir() for _, directory in retained)
