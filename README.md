# Compact Audio Denoising

A causal 16 kHz speech denoiser targeting ESP32-S3, with PyTorch training and C inference.

## Final model

**[Download final_model.bin](final_model.bin)** — the selected INT8 development champion, **46,784 bytes** and **23,669 learned parameters**.

| Development set | SI-SDR improvement | PESQ | STOI |
|---|---:|---:|---:|
| External: 500 mixtures | **9.43 dB** | 2.033 | 0.869 |
| VoiceBank: 770 utterances | **7.62 dB** | 1.955 | 0.830 |

Scores use the complete C audio pipeline with PCM16 I/O. These development sets were used for selection; official tests remain reserved. Physical ESP32 real-time performance is **not yet measured**. “Final model” identifies this published checkpoint, not completion of hardware validation.

The model uses a GTCRN-based architecture with frame-local normalization. Neural weights, activations and persistent state are INT8; accumulators/biases use INT32 and bounded wider intermediates where required. Audio DSP remains float32. Upstream [GTCRN](https://github.com/Xiaobin-Rong/gtcrn/tree/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73) is MIT-licensed; its license and provenance are preserved in the source tree.

[Full results and limitations](docs/research/esp32_update_20260915.md) · [Checkpoint, calibration and evaluation reports](output/esp32/champions/gtcrn_qat_20260915/) · [Artifact verification](output/esp32/champions/gtcrn_qat_20260915/verification.json)

SHA-256: `3239dde377b30a9fc59a9a436f8dfa70e93b1a0b71f67a846869470c2a8fbc10`

## Repository folders

| Folder | Contents |
|---|---|
| [esp32_denoiser/](esp32_denoiser/) | Current model, training, quantization, export and host inference |
| [firmware/](firmware/) | ESP32 C runtime, benchmarks and build instructions |
| [configs/](configs/) | Training and experiment configurations |
| [notebooks/](notebooks/) | Colab training notebook; earlier demos in `legacy/` |
| [models/](models/) | Historical model archives |
| [moe_baseline/](moe_baseline/) | Earlier mixture-of-experts implementation |
| [scripts/](scripts/) | Training/demo entry points and source packaging |
| [requirements/](requirements/) | Current ESP32 and legacy Python dependencies |
| [tests/](tests/) | Model, numerical, export and runtime checks |
| [docs/](docs/) | Research, data protocols, results and deployment audits |
| [output/](output/) | Preserved checkpoints, evaluations, recovery scripts and build evidence |

The root model is byte-identical to the champion's `model.bin` under `output/esp32/champions/gtcrn_qat_20260915/`. Historical artifacts retain their original layout for provenance and recovery.

## Getting started

Install [Git LFS](https://git-lfs.com/) to retrieve model binaries and experiment archives:

```sh
git lfs install
git clone https://github.com/kaarya583/autoencoder_denoiser.git
cd autoencoder_denoiser
git lfs pull
python3 -m venv .venv-esp32
source .venv-esp32/bin/activate
python -m pip install -r requirements/esp32.txt
python -m pytest -q tests
```

PESQ/STOI evaluation additionally uses `pesq==0.0.4` and `pystoi==0.4.1`. Training audio, local environments and downloaded toolchains are excluded; [the training guide](docs/esp32_training.md) describes obtaining data and reproducing experiments.

For deployment, follow the [ESP32-S3 benchmark instructions](firmware/gtcrn_benchmark/README.md), using `final_model.bin` as the validated model input. Regenerate known answers and rebuild before flashing: existing historical target binaries contain an earlier champion.

Earlier demos are in [notebooks/legacy/](notebooks/legacy/), with dependencies in [requirements/legacy.txt](requirements/legacy.txt). Run the earlier training entry point from the repository root with `python scripts/run_moe_baseline.py`.
