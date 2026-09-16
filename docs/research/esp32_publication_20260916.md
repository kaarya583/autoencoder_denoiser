# GitHub publication verification — September 16, 2026

This publication includes the ESP32 training/inference source, configurations, Colab notebook, recovery controllers, tests, dated research reports, locally preserved experiment checkpoints/snapshots, and native build evidence. Binary checkpoints, packed models, ZIPs, split backup archives, ELF files, and object files use Git LFS. A clone needs `git lfs pull` to materialize them.

Training audio, Python environments, downloaded toolchains, transfer scratch, macOS Finder metadata, and debugger symbol directories remain local/ignored. Dataset manifests and source provenance are included. The original `.gitignore` additions were preserved.

## Validation

- `.venv-esp32/bin/python -m pytest -q tests`: **697 passed, 2 skipped in 62.50 seconds** on the local Python 3.12 environment. Optional skipped checks are not claimed as passed GPU/device validation.
- An initial suite run exposed a nondeterministic causality assertion in a legacy mixed float/INT8 operator probe. The test previously changed both future samples and sequence length; sequence shape can change float-kernel rounding before quantization. The revised test keeps shape fixed while perturbing future samples and covers three reproducible seeds. The model and deployed inference implementation were not changed.
- The publication's project-owned source/documentation passes `git diff --cached --check`. Pinned upstream GTCRN files retain their original line endings/whitespace to preserve provenance.
- A path-only scan of publishable text files found no matches for private keys, common GitHub/Google/OpenAI/AWS credential formats, or literal long secret assignments. This is a targeted scan, not a claim of exhaustive secret detection.
- The winning model and its checkpoint, calibration, candidate evaluation, and three final development reports were hash-verified against the saved verification record.

Winning model: `output/esp32/champions/gtcrn_qat_20260915/model.bin`, **46,784 bytes**, SHA-256 `3239dde377b30a9fc59a9a436f8dfa70e93b1a0b71f67a846869470c2a8fbc10`.

The latest quality results and deployment limitations are in [the September 15 update](esp32_update_20260915.md). The new champion has not been timed on a physical ESP32. Historical cross-builds and development snapshots must not be mistaken for final board benchmarks or untouched-test results.
