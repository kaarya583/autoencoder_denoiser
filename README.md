# Compact Audio Denoising: Architecture and INT8 Quantization

Compact audio models through architectural compression and INT8 quantization. The current implementation targets causal 16 kHz speech denoising on ESP32-S3, with PyTorch training and C inference.

The best model fits in **46.8 KB (46,768 bytes)** with **23,669 learned parameters** and achieves **9.20 dB SI-SDR improvement** on 500 external development mixtures through full C PCM16 inference—less than half the 99,000-byte model budget.

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

GTCRN's best float result is **10.08 dB**, from a different checkpoint, so its gap to the INT8 result is not a matched quantization-loss measurement.

Official tests remain sealed; ESP32-S3 builds pass, but board latency is unmeasured.

## Teacher–student training

A frozen **632,322-parameter FP32 spectral teacher** guides the **84,738-parameter TCN student** using compressed spectral distillation alongside clean-target losses. Training-only calibration sets the loss weight and corrects teacher gain. The teacher adds no inference cost. Distillation remains experimental; the GTCRN results use separate training.

## Demo

See [MoE_Demo.ipynb](MoE_Demo.ipynb) for the earlier mixture-of-experts audio demo. The embedded training code and supporting reports are not yet included in this repository.
