# Repository project audit: objective and demonstrated results

Evidence reviewed 12 September 2026 PDT. This distinguishes the tracked original project from the ongoing `esp32_denoiser` implementation. It uses source, saved notebook output, and the metadata/state tensors inside `robust_UNet_model.zip`; a saved metric is not automatically an independent benchmark. New ESP32 results below are validation results, with exact reports in [backup 02](../../output/esp32/development_backup_02). The original Git remote is [kaarya583/autoencoder_denoiser](https://github.com/kaarya583/autoencoder_denoiser); the new local work is not thereby established as published there.

## Exact objective

**Originally:** investigate robust versus adaptive audio denoising under changing noise distributions, including specialist autoencoders, routing and a stronger waveform denoiser. The tracked work emphasizes denoising quality and adaptation. It does not establish weight compression or real-time microcontroller deployment as completed original objectives.

**Now:** build the strongest useful causal speech model within **99,000 packed model bytes**, using architecture downsizing and INT8 neural inference, then establish real-time operation on ESP32-S3. Quality, compression and eventual edge viability are combined objectives. Quality is still being improved, particularly outside VoiceBank conditions; board viability remains unmeasured. There is no fixed 8 dB quality ceiling.

## 1. Final architecture

There are several implementations, rather than one consistent final architecture:

| Implementation | Structure and representation | Approximate learned parameters |
|---|---|---:|
| [Adaptive autoencoder notebook](../../Adaptive_Autoencoders_Project.ipynb) | Waveform frames of 512 samples; dense encoder 512→256→128→64 and mirrored decoder. Noise is injected into the latent code; robust and noise-specialist networks are compared. | **345,408 per network** |
| [Packaged MoE](../../moe_baseline/model.py) | 512-sample waveform frames; 512-hidden/128-latent encoder, six-way router, shared residual expert plus six specialist residual experts. | **2,630,022** |
| [Archived robust model](../../robust_UNet_model.zip), reconstructed by the [microphone notebook](../../live_voice_denoiser_demo.ipynb) | `GatedTCNWaveformResUNet`: waveform input/output, three downsampling stages with channels 24/48/96, bottleneck 192, six dilated TCN blocks, GroupNorm/SiLU, concatenated skips, and sigmoid-gated residual correction. | **2,963,714**, all FP32 |
| Current strongest completed integer control | Spectral features from 16 kHz audio, FFT 512/hop 256; sparse ERB input mapping, width-64 six-block causal depthwise/pointwise TCN, signed residual states and full-bin complex-gain output. Zero depthwise-bias initialization. | **84,738**, packed **94,480 bytes** |

The historical U-Net uses STFT losses during training, but its network input is waveform. Symmetric convolutions and time-spanning GroupNorm make its chunk inference noncausal. The current spectral TCN is causal at the neural frame level; analysis framing still incurs delay.

The MoE's hard route does **not** imply sparse execution: current forward paths evaluate every expert before selecting/mixing outputs. Frequency-sharing U-Net and GRU candidates are now under study. The frequency U-Net has a supported integer/C path; the GRU remains float-only. A faithful, fresh GTCRN float reference is also implemented with 23,669 learned parameters plus 24,576 fixed ERB coefficients; no complete training result or integer implementation is yet claimed. No final frontier architecture has been frozen. [Current architecture details](../esp32_training.md).

## 2. Compression actually achieved

**Original project:** no demonstrated parameter pruning, distillation, INT8/INT4 deployment or compressed weight artifact. The U-Net archive stores FP32 tensors. AMP in its training config is mixed-precision training, not evidence of FP16 deployment. The adaptive notebook's “2-bit” switching signals describe routing metadata, and its 512→64 latent bottleneck describes representation size; neither is weight quantization.

**Current work:** substantial architecture downsizing, training-only activation calibration, QAT and exported **INT8 neural weights/activations/state**, with INT32 biases/accumulators. The complete C audio frontend still uses float32 DSP. Pruning and INT4 have not been implemented; response distillation and gradient calibration are implemented but no completed distillation gain is established.

- Historical U-Net: **2,963,714 parameters / 11,854,856 raw FP32 parameter bytes**.
- Completed compact control: **84,738 parameters / 94,480 packed deployment bytes**, including integer biases, scales, graph metadata and model-specific DSP constants.
- This is **34.98× fewer parameters** and **125.47× smaller payload**, combining architecture changes and quantization. The requested roughly 119× size target is therefore met by an actual packed model, not by quantization alone.
- The compact model's own raw FP32 parameter payload is **338,952 bytes**, so its FP32→packed reduction is **3.59×**. Executable firmware, audio buffers and stacks are separate.

The original checkpoint files are about 35.7 MB because they also contain optimizer and training state; comparing those files directly with inference-only bytes would inflate the compression ratio.

## 3. Strongest audio-quality results supported by evidence

**Historical U-Net:** the archive records best `SI-SDRi = 9.003835918833909 dB` at epoch 37 of 48. However, its config explicitly sets `use_official_test_for_val = true`; the test split selected checkpoints. This is a **logged selection score, not an independent final-test result**. The tracked archive does not retain PESQ/STOI or the noisy SI-SDR baseline, and the exact original metric aggregation code is not included with the inference notebook. The final epoch scores 8.745 dB, so it should not be confused with the best checkpoint.

**Other original demonstrations:** the saved MoE demo reports **1.78 dB noise reduction** on one six-second example. That quantity is an input/output MSE ratio, not SI-SDRi or a corpus benchmark. Router classification accuracies in the adaptive/MoE notebooks measure noise recognition; they are not denoising-quality metrics. Some adaptive routing uses the known injected latent noise, which is privileged information unavailable in ordinary microphone enhancement.

**Current completed full-C control:** all 770 full VoiceBank validation utterances are valid:

| Metric | Noisy input | Zero-bias integer control |
|---|---:|---:|
| SI-SDR | 6.017 dB | 13.793 dB |
| SI-SDR improvement | 0 dB | **7.776 dB** |
| WB-PESQ | 1.570 | **1.889** |
| STOI | 0.8125 | **0.8275** |

The deeper control gives **7.755 dB SI-SDRi**, **1.920 PESQ**, and **0.8282 STOI**. These different metric rankings should remain visible. A fresh, larger spectral teacher's [verified CPU perceptual report](../../output/esp32/development_backup_03/teacher_validation_perceptual.json) gives **8.720548 dB float SI-SDRi**, **2.108895 PESQ** and **0.843421 STOI** on the same 770-clip validation set, all valid. Its training-loop selected SI-SDRi is **8.720491 dB**, a roughly 0.000057 dB numerical difference. The CPU float teacher exceeds the zero-bias full-C control by **0.94430 dB**, **0.21956 PESQ** and **0.01597 STOI**. These span architecture/precision/training budgets and do not establish a quantization-only loss or a distillation benefit. [Exact experiment reports](esp32_experiments.md).

The external-development evaluation exposes a limitation: on 500 unseen-source LibriSpeech/MUSAN mixtures, the **same zero-bias full-C PCM16 integer model** gains only **0.810 dB SI-SDRi**, PESQ **1.4696→1.4866**, and STOI decreases **0.8134→0.8039**. On 100 clean clips, PESQ decreases **4.6439→4.5201** and STOI **1→0.9968**. The exact [mixture](../../output/esp32/development_backup_03/zero_bias_external_mixtures.json) and [clean](../../output/esp32/development_backup_03/zero_bias_external_clean.json) reports are now preserved in verified local backup 03, and the model SHA matches the archived VoiceBank report. Later broader-data float results are separate experiments, pending quantization. These findings motivate broader training: the 7.776 dB score alone does not demonstrate strong general-purpose denoising. No final current official-test score exists.

## 4. Real-time performance

The original microphone notebook records an entire clip, waits for recording to finish, then denoises overlapping two-second chunks. It includes a timer but retains no measured model latency/RTF output. It is a **record-then-denoise demo**, not proof of continuous real-time streaming.

Current training uses an **NVIDIA L4** in Colab. The full-C zero-bias host evaluator processes **2,279.58 seconds of audio in 27.69 seconds**, **RTF 0.01215**, on Linux x86_64; the report does not identify a precise host CPU SKU. This demonstrates faster-than-real-time **offline host inference**, excluding compilation, model loading, file I/O and metrics. It is not an L4 inference result or ESP32 measurement.

Real **ESP32-S3** firmware compiles with ESP-IDF 5.4.2/ESP-DSP 1.8.2, 240 MHz configuration and no PSRAM. The captured trained-baseline app is **324,352 bytes** with **48,268 bytes static writable RAM**. That build contains the earlier signed baseline, not the later zero-bias checkpoint. A frequency identity prototype also builds. **No board inference, full-hop latency, deadline-miss count, memory peak or I2S end-to-end delay has been measured.** The 5.212 MMAC/s global model budget is an operation count, not proof of MCU speed. [Memory/build audit](esp32_memory_audit.md).

## 5. Quality versus efficiency

The most defensible matched claim is:

> On 770 complete held-out-speaker VoiceBank validation clips, an 84,738-parameter model went from 338,952 raw FP32 parameter bytes to a 94,480-byte integer deployment payload, with **0.163 dB SI-SDRi loss** from its own float checkpoint to complete C PCM16 inference.

The **125.47× smaller than the old U-Net** payload claim is also supported, but **its quality loss is unknown**: the old U-Net used a different, test-selected protocol. Do not combine “125× smaller” with “0.163 dB loss” into one old-versus-new claim.

The fresh 632,322-parameter spectral teacher is a more relevant same-split reference: **8.720 dB float versus 7.776 dB compact full-C**, a **0.944 dB difference**, and 2,529,288 raw FP32 parameter bytes versus 94,480 packed bytes (**26.77×**). This comparison spans architectures, feature layouts, precision and training budgets; it does not isolate compression or establish a distillation effect.

## 6. Datasets and evaluation credibility

Original adaptive/MoE experiments use LibriSpeech dev-clean/test-clean speech subsets and synthetic noise families. Notebook configurations vary, including 800/200 and 1,200/300 audio-file splits. Results from different tasks/settings are not directly comparable. The U-Net config requests full VoiceBank-DEMAND, but its metadata does not record exact loaded utterance counts, and its official test was used for selection.

Current paired training uses **10,802 clips / 26 speakers**; validation holds out **770 clips / two speakers**, p226/p287, from the original training split. IDs, speakers and waveform paths are disjoint. The 16 kHz third-party mirror is commit-pinned and recorded; its resampling procedure and byte identity to original archives are unverified. The evaluation is a credible held-out-speaker development comparison, while noise conditions still come from the original training distribution.

The broader pipeline now has **90.53 hours / 25,675 training clips / 226 LibriSpeech speakers**, plus **837 MUSAN noise recordings**. It holds out 25 whole speakers and 93 whole noise recordings for fixed external development mixtures: **500 noisy clips across five SNRs plus 100 clean clips**. Only training-origin sources are added, with provenance/hashes and no DNS/DEMAND extras. The current official VoiceBank test remains reserved until final selection. External development becomes selection data once consulted; it is not a replacement untouched test. [Preparation and protocol](../esp32_training.md).

## 7. Research novelty

The original work goes beyond one bare Res-U-Net implementation through adaptive-versus-robust comparisons, RF/neural routing experiments, noise specialists, gated residual enhancement and augmentation/loss combinations. These are meaningful implementation and experimental choices, but the repository does not establish a new compression algorithm or architectural invention.

The current contribution is a careful quality/efficiency study: strict split and teacher-lineage checks, actual packed formats, persistent INT8 state, QAT, full C waveform evaluation, controlled depth/initialization/features/normalization comparisons, and real S3 compiler/linker accounting. The external generalization failure is an actionable experimental finding. Signed residual processing corrected an observed inactive-ReLU path, but the comparison also changes initialization and uses one seed; it does not isolate a novel principle. Stronger known-architecture baselines, broader training and device measurements are still needed before asserting superiority.

## 8. End product and completion

The original project has a **GitHub repository, notebooks, a saved FP32 U-Net checkpoint archive and a microphone recording/demo notebook**. The microphone notebook's default archive path differs from the tracked archive filename and needs adjustment before use. No tracked paper, poster or final benchmark report was found.

The new local work adds reproducible Colab training, provenance-aware datasets, full-C evaluators, actual compact integer model files, research reports and ESP32-S3 firmware builds. It is **ongoing**: broader-data quality, architecture selection, final sealed-test evaluation and real-board audio integration remain unfinished. Current artifacts are useful research/deployment candidates; they do not yet constitute a validated real-time ESP32 audio product.
