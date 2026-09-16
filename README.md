# Compact Audio Denoising: Architecture and INT8 Quantization

Compact audio models through architectural compression and INT8 quantization. The current implementation targets causal 16 kHz speech denoising on ESP32-S3, with PyTorch training and C inference.

The best model fits in **46.8 KB (46,784 bytes)** with **23,669 learned parameters** and achieves **9.43 dB SI-SDR improvement** on 500 external development mixtures through full C PCM16 inference—less than half the 99,000-byte model budget. On the 770-utterance VoiceBank development holdout it achieves **7.62 dB SI-SDR improvement**. These are development results used for model selection; physical ESP32 real-time performance remains unmeasured.

The [September 15 results report](docs/research/esp32_update_20260915.md) records the completed 240-epoch runs, six-hour QAT continuation, quality tradeoffs, artifact hashes, and deployment limitations. The [winning checkpoint and packed model](output/esp32/champions/gtcrn_qat_20260915/) include all three final evaluation reports and a [verification record](output/esp32/champions/gtcrn_qat_20260915/verification.json).

## Teacher vs. quantized student

All three models below use the same **770-clip VoiceBank development holdout**. SI-SDR gain measures improvement over noisy input.

| Spectral TCN | Parameters | Model storage | SI-SDR gain |
|---|---:|---:|---:|
| FP32 teacher reference | 632,322 | 2,529,288 B¹ | **8.72 dB** |
| FP32 student | 84,738 | 338,952 B¹ | **7.94 dB** |
| INT8 student after QAT, full C PCM16 | 84,738 | **94,480 B²** | **7.78 dB** |

¹ Raw FP32 parameters. ² Packed model, including metadata and DSP constants.

Quantizing the student reduces storage **3.59×** with **0.16 dB** loss from float to C. Compared with the teacher, the packed student is **26.77× smaller** and **0.94 dB lower**; that difference includes architecture and training changes. These controls do not establish a distillation benefit.

## Architecture

- **Spectral TCN — 84,738 parameters, 5.21 MMAC/s.** Six dilated depthwise/pointwise blocks use signed residual paths and zero-initialized depthwise biases to address activation collapse. An identity-initialized head predicts complex spectral masks.
- **GTCRN — 23,669 learned parameters.** Frequency-sharing convolutions and grouped dual-path GRUs capture spectral and temporal context. Adaptations to [MIT-licensed GTCRN](https://github.com/Xiaobin-Rong/gtcrn/tree/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73) add causal framing and frame-RMS normalization with amplitude restoration.
- **Shared sparse filterbank.** A single ERB table serves analysis and synthesis, reducing constants from **98,304 to 2,040 bytes** while preserving every nonzero coefficient.

## Quantization

**INT8 weights, activations, and persistent state; INT32 biases and dot accumulators**, with wider intermediates where required. Training-only calibration selects power-of-two scales; BatchNorm folds into convolutions. Quantization-aware training includes recurrent state, and C outputs are checked against a NumPy integer reference. Audio DSP remains float32.

An earlier GTCRN float checkpoint achieved **10.08 dB**, from a different checkpoint, so its gap to the INT8 result is not a matched quantization-loss measurement.

Official tests remain sealed; ESP32-S3 builds pass, but board latency is unmeasured.

## Teacher–student training

A frozen **632,322-parameter FP32 spectral teacher** guides the **84,738-parameter TCN student** using compressed spectral distillation alongside clean-target losses. Training-only calibration sets the loss weight and corrects teacher gain. The teacher adds no inference cost. Distillation remains experimental; the GTCRN results use separate training.

## Demo

See [MoE_Demo.ipynb](MoE_Demo.ipynb) for the earlier mixture-of-experts audio demo.

## Embedded implementation and reproducibility

- [Training guide and data protocol](docs/esp32_training.md), [Colab notebook](notebooks/ESP32_S3_Training.ipynb), and [experiment configurations](configs/).
- [Training, integer inference, and export implementation](esp32_denoiser/), with the upstream GTCRN license and pinned provenance under `esp32_denoiser/vendor/gtcrn/`.
- [ESP32-S3 benchmark firmware and build instructions](firmware/gtcrn_benchmark/README.md). The existing target build uses the preceding 9.20 dB champion; rebuild and regenerate known answers for the new 9.43 dB model before flashing.
- [Experiment artifacts, recovery controllers, checkpoints, and build evidence](output/esp32/). Historical reports retain their original dates and status; consult the latest dated report for current results.
- [Untouched evaluation protocol](docs/research/esp32_final_evaluation_protocol.md).

Install [Git LFS](https://git-lfs.com/) before cloning to retrieve checkpoints, packed models, archives, and compiled artifacts:

```sh
git lfs install
git clone https://github.com/kaarya583/autoencoder_denoiser.git
cd autoencoder_denoiser
git lfs pull
python3 -m venv .venv-esp32
source .venv-esp32/bin/activate
python -m pip install -r requirements-esp32.txt
python -m pytest -q tests
```

PESQ/STOI evaluation additionally uses `pesq==0.0.4` and `pystoi==0.4.1`. Training audio, local environments, downloaded toolchains, and transfer scratch are excluded. Dataset manifests and provenance are retained; follow the training guide to obtain the audio. Recovery snapshots are historical deltas, not independently complete training datasets.

The winning packed model is `output/esp32/champions/gtcrn_qat_20260915/model.bin`, with SHA-256 `3239dde377b30a9fc59a9a436f8dfa70e93b1a0b71f67a846869470c2a8fbc10`.
