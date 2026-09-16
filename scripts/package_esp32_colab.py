"""Build a small auditable source bundle, excluding checkpoints and personal files."""
from hashlib import sha256
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "esp32"
OUTPUT.mkdir(parents=True, exist_ok=True)
# ESP-IDF downloads dependencies and generates sources inside project folders.
# Ship only this project's source, never the toolchain or generated build tree.
excluded_firmware_parts = {"build", "managed_components", ".git", "__pycache__"}
firmware_paths = [
    p for p in ROOT.glob("firmware/**/*")
    if p.is_file() and not excluded_firmware_parts.intersection(p.relative_to(ROOT).parts)
    and (p.suffix in {".c", ".h", ".S", ".txt", ".yml", ".yaml", ".md"}
         or p.name in {"sdkconfig.defaults", "dependencies.lock", "Kconfig.projbuild", "Kconfig"})
]
package_paths = [p for p in (ROOT / "esp32_denoiser").rglob("*")
                 if p.is_file() and (p.suffix in {".py", ".json", ".md"}
                                     or p.name in {"LICENSE", "NOTICE"})]
paths = sorted([*package_paths,
                *ROOT.glob("tests/test_*.py"),
                *ROOT.glob("configs/esp32*.json"),
                *firmware_paths,
                ROOT / "requirements" / "esp32.txt"])
paths = [p for p in paths if p.is_file()]
# Hash and archive the same byte snapshots even if another experiment edits a
# source file during packaging. Stable ZIP metadata makes identical inputs
# produce identical bundle hashes.
# Keep the deployed Colab bundle layout stable after organizing the repository.
sources = {("requirements-esp32.txt" if p == ROOT / "requirements" / "esp32.txt"
            else str(p.relative_to(ROOT))): p.read_bytes() for p in paths}
manifest = {name: sha256(data).hexdigest() for name, data in sources.items()}
with zipfile.ZipFile(OUTPUT / "esp32_training_source.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
    for name, data in {**sources, "SOURCE_MANIFEST.json": json.dumps(manifest, indent=2).encode()}.items():
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        bundle.writestr(info, data)
(OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
bundle_bytes = (OUTPUT / "esp32_training_source.zip").read_bytes()
snapshot = OUTPUT / "source_bundles" / (sha256(bundle_bytes).hexdigest() + ".zip")
snapshot.parent.mkdir(parents=True, exist_ok=True)
if not snapshot.exists():
    snapshot.write_bytes(bundle_bytes)
print(OUTPUT / "esp32_training_source.zip")
print(f"{len(paths)} source files; {(OUTPUT / 'esp32_training_source.zip').stat().st_size:,} bytes")
