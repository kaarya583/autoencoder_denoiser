# ESP32-S3 denoiser: training and deployment guide

Status: 12 September 2026 PDT / 13 September UTC. The objective is to maximize speech-enhancement quality under a **99,000-byte actual model-data limit**, with ESP32-S3 deployment as the systems target. There is no fixed 8 dB stopping threshold. Training and architecture comparisons remain active; no final model or official-test result has been selected.

The strongest completed compact control achieves **7.776 dB SI-SDR improvement on 770 VoiceBank validation utterances**, using **94,480 packed bytes** and complete C DSP, integer neural inference and PCM16 I/O. It loses **0.163 dB** relative to its own selected float checkpoint. This result does not establish broad generalization: the newly reported 500-mixture LibriSpeech/MUSAN development baseline gives only **0.810 dB improvement**, with a slight STOI decrease. Broader training is therefore a priority, alongside architecture comparisons.

Completed controls and exact hashes are preserved in [backup 02](../output/esp32/development_backup_02), with the detailed [experiment record](research/esp32_experiments.md). [Run status](../output/esp32/run_status.json) distinguishes active trials from completed controls. Later external-development and teacher-perceptual reports are now preserved in verified local [backup 03](../output/esp32/development_backup_03). The [external mixture report](../output/esp32/development_backup_03/zero_bias_external_mixtures.json) uses the same zero-bias integer binary, SHA-256 `3d58fa7c301a746ba226e6d25ff7b4ac4190bd29e7e97893e08d694128914907`, and complete C PCM16 inference.

## Completed controls and their limitsu

All three rows use the same complete, equally weighted 770-utterance validation set. Float and QAT columns are selected training-loop checkpoints; the final column uses the exported model and full C PCM16 path on a Linux x86_64 host.

| Candidate | Float SI-SDRi | QAT SI-SDRi | Full C SI-SDRi | Float → C loss | Packed bytes |
|---|---:|---:|---:|---:|---:|
| Signed width 64, six blocks | 7.575 dB | 7.391 dB | 7.390 dB | 0.184 dB | 94,480 |
| Signed width 56, ten blocks | 7.891 dB | 7.755 dB | 7.755 dB | 0.135 dB | 96,496 |
| Width 64, zero depthwise-bias initialization | 7.939 dB | 7.776 dB | **7.776 dB** | **0.163 dB** | **94,480** |

For the strongest SI-SDR control, noisy SI-SDR is **6.017 dB** and enhanced SI-SDR is **13.793 dB**. Wideband PESQ improves **1.570 → 1.889**, and STOI **0.8125 → 0.8275**. All 770 clips are valid for all three metrics. The ten-block model has slightly lower SI-SDRi but better PESQ (**1.920**) and STOI (**0.8282**); the 0.021 dB SI-SDR difference is not evidence of a universal winner. [Exact full-C report](../output/esp32/development_backup_02/zero_bias_embedded_validation.json).

The unseen-source development check is more demanding. On 500 fixed LibriSpeech/MUSAN mixtures, the same zero-bias full-C integer control gains **0.8104 dB**, with noisy/enhanced SI-SDR **3.5130 / 4.3234 dB**, PESQ **1.4696 / 1.4866**, and STOI **0.8134 / 0.8039**. On 100 separate clean-preservation clips, enhanced SI-SDR is **29.427 dB**, PESQ falls **4.6439 → 4.5201**, and STOI **1.0000 → 0.9968**. These development measurements motivate broader training; they are neither a final benchmark nor proof that the completed control is a strong general-purpose denoiser. Subsequent broader student/teacher float scores must remain separate until those checkpoints complete matched integer evaluation.

## Architecture and byte accounting

The completed zero-bias control is a causal spectral TCN, separate from the repository's historical waveform Res-U-Net:

- Mono 16 kHz; 512-sample square-root Hann analysis window, 256-sample hop, and no future neural frames beyond the analysis framing.
- Frame-local RMS normalization; compressed magnitude, real and imaginary features. The lowest 65 FFT bins remain separate and 192 upper bins are mapped to 64 sparse triangular ERB bands, giving 387 inputs.
- Width 64 and six causal depthwise/pointwise residual blocks, with temporal dilations 1, 2, 4, 8, 16 and 32. Input projections and residual sums preserve signed values with clipping to [-6, 6]; depthwise outputs use ReLU6. Residual pointwise branches start at reduced weight scale, and the completed stronger control initializes depthwise biases to zero.
- A 514-output head predicts bounded complex-gain corrections at every original FFT bin, starting from identity. The neural receptive field is 127 frames.
- **84,738 learned parameters**, including 83,392 convolution weights; **5.212 million neural MAC/s**. Persistent INT8 temporal history is 8,064 bytes, and the required C neural workspace is 8,280 bytes. Audio DSP and stacks are additional.

The actual **EDNSI8-v2** payload includes INT8 weights, INT32 biases, per-channel scale exponents, descriptors, alignment and model-specific DSP constants. Its 94,480 bytes are **125.47× smaller** than the historical U-Net's 11,854,856 raw FP32 parameter bytes. Architecture downsizing contributes about **34.98× fewer parameters**; INT8 alone would reduce an unchanged FP32 weight array by about 4×. The compact model's own 338,952 raw FP32 parameter bytes become 94,480 packed bytes, a **3.59×** reduction including deployment overhead. Only this same-model comparison supports the measured 0.163 dB float-to-C loss; that loss must not be attached to the 125× historical-model comparison.

All **neural** C computation uses INT8 weights, features, activations, outputs and persistent state, with INT32 biases/accumulators and fixed power-of-two shifts. QAT simulates this graph in floating point during training. FFT, normalization, feature construction, complex-gain application and overlap-add remain **float32 DSP**. This is not an entirely integer audio pipeline or a TFLite/ESP-DL file. Export checks actual bytes, supported operators/layouts and accumulator overflow. No INT4, pruning or learned sparsity has been demonstrated.

The implementation now also supports a frequency-sharing U-Net with full-bin features, causal local blocks, a compact global TCN and additive encoder/decoder skips. The default has **83,170 parameters**, **27.561 MMAC/s**, a **94,300-byte EDNFQ8-v1 prototype payload**, and **34,844 bytes required neural workspace**. Its full integer neural and C DSP paths are implemented, but its trained quality is still under study. Optional encoder BatchNorm folds into convolutions before calibration/QAT; it adds no separate deployed normalization operator. The smaller variant uses 55,572 packed bytes but its completed float score was only 6.202 dB on the VoiceBank validation set.

The shared per-frequency GRU variant is **float-only**: 81,986 parameters and 25.251 MMAC/s of matrix work, excluding nonlinearities/gates. QAT and export explicitly reject it. A hypothetical INT8 state estimate is not evidence of an integer GRU implementation. Likewise, the global `fullmag_lowphase` feature ablation is float-only because that DSP layout has no compatible global exporter.

A faithful **GTCRN float reference** is now implemented from the [official MIT source at a pinned revision](../esp32_denoiser/vendor/gtcrn/PROVENANCE.json): **23,669 learned parameters plus 24,576 fixed ERB coefficients**, with 72,192 bytes of float neural streaming state. It preserves the official spectral operators and raw features, using our causal zero-padded waveform framing instead of upstream's centered reflect-padding example. LayerNorm normalizes frequency/channels within each frame; bidirectional GRUs run across frequency, while temporal recurrence is causal. Fresh-weight and trained-tiny-fixture checks cover streaming parity, gradients, checkpoint loading and resume. `esp32_gtcrn_float.json` and `esp32_gtcrn_broad_float.json` define fresh paired/broader runs with matching 120-epoch, 10,802-example-per-epoch budgets. No full training score, integer export or MCU result is established by these checks.

## Data and evaluation protocol

VoiceBank-DEMAND preparation uses the [16 kHz mirror pinned to revision 4497db3](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k/tree/4497db342d7312978c45690591fda86117831940), because the official Edinburgh server returned HTTP 403 from Colab. The mirror's resampling method and byte identity to the [original dataset](https://doi.org/10.7488/ds/2117) are unverified and recorded as such. Decoded audio is cached as FLOAT WAV without another resampling pass.

Training uses **10,802 utterances from 26 speakers**, about 8.76 hours. Validation holds out **p226 and p287**, all **770 utterances / 37.99 minutes**. Both are derived from the original training split, with disjoint speakers, IDs and paths. This tests unseen speakers in training-origin noise conditions; it is not an unseen-noise test. The current experiment lineage has not downloaded/prepared the official test. The historical repository U-Net did use that test for selection, so it is excluded from strict teacher lineage and cannot supply an independent final-test comparison.

Broader data are now prepared and audited:

| Source | Training partition | Held-out development sources |
|---|---|---|
| LibriSpeech train-clean-100 | 25,675 clips, 226 speakers, **90.53 h** | 2,864 clips, 25 speakers, **10.06 h** |
| MUSAN noise subset | **837 recordings**, about 5.68 h | **93 recordings**, about 0.54 h |

Entire speakers and noise recordings are held out before sampling crops. The pipeline does not add DNS/DEMAND extras or MUSAN speech/music. Original 16 kHz LibriSpeech FLAC is retained; archive/file hashes and licenses are recorded. Optional OpenSLR simulated-RIR preparation exists, with room-level partitions, but no completed RIR-training result is claimed.

`DynamicMixtureDataset` samples aligned speech/noise crops, active-speech SNR from -5 to 20 dB, common gain from -6 to 6 dB, and occasional identity examples. A shared attenuation controls peaks without changing the mixture SNR. Hybrid configs mix paired VoiceBank and synthetic training examples with probability 0.5. Dataset preparation downloads only when explicitly invoked.

The deterministic external development suite contains **100 shared speech/noise crops at each of -5, 0, 5, 10 and 20 dB**, plus **100 separately sampled clean clips**. It uses only held-out source partitions, frozen seeds and waveform/manifest hashes. The 500 mixtures are correlated across SNR; uncertainty estimates should group by base crop. Once used for selection, this suite is development data, not a final test.

SI-SDR is zero-mean, masked to each true utterance length, and averaged equally per valid utterance; SI-SDRi subtracts that utterance's noisy baseline. Full evaluations preserve alignment and report invalid denominators. Nonfinite model output fails evaluation. Gain projection, RMS ratio, normalized waveform L1 and clipping counts complement scale-invariant SI-SDR, especially for clean preservation.

Optional [PESQ](https://github.com/ludlows/PESQ) (`pesq==0.0.4`, wideband P.862.2 MOS-LQO) and [STOI](https://github.com/mpariente/pystoi) (`pystoi==0.4.1`, ordinary STOI) reuse the same enhanced utterances. Preprocessing applies a shared attenuation to clean/noisy/enhanced signals when needed, without independent gain changes or extra clipping. PCM16 conversion occurs in the deployed path before metrics. Perceptual metrics are ancillary reports; they do not select checkpoints. The evaluator limits BLAS to one thread only during metrics and restores the caller's settings before inference. This avoids measured worker-scheduling overhead without changing metric definitions or the inference throughput measurement.

## Training, calibration and teacher controls

The default objective combines negative SI-SDR and a small normalized waveform L1 term, with aligned three-second crops and training-only augmentation. Validation always uses complete, unmodified clips. Float-to-QAT calibration uses training audio to choose the hidden quantization grid; QAT then fixes activation/state scales. Resuming a phase retains optimizer and RNG state; float-to-QAT deliberately begins a new optimization phase. Old metadata-free ReLU checkpoints load only with their historical activation behavior.

Optional multi-resolution compressed spectral loss is calibrated against primary-loss gradient norms before selecting its coefficient. The archived **11.2326** coefficient targets a 0.1 gradient ratio for the clean-target spectral loss; it is **not a distillation coefficient**.

A fresh 632,322-parameter spectral teacher trained on the same strict VoiceBank split reaches **8.720 dB validation SI-SDRi**. Its [separate full-utterance CPU perceptual report](../output/esp32/development_backup_03/teacher_validation_perceptual.json) gives **8.7205483819 dB**, **PESQ 2.1088951447** and **STOI 0.8434212923**, with all 770 utterances valid and no metric errors. The training-loop selection value is **8.7204909427 dB**; the roughly 0.000057 dB numerical difference is preserved rather than silently substituted. Against the zero-bias full-C control, the CPU float teacher is higher by **0.94430 dB SI-SDRi**, **0.21956 PESQ** and **0.01597 STOI**. This comparison spans architecture, precision and training budget; it is neither quantization-only loss nor a measured distillation gain.

`FrozenTeacher` validates explicit fresh-training lineage, training-origin manifests, disjoint teacher training/selection/student-validation assets, and checkpoint/provenance hashes. Its initial strict track accepts paired-only teachers with no upstream teacher; a narrowly checked sidecar supports the current fresh legacy-format checkpoint. The historical test-selected U-Net is rejected.

Response distillation uses detached FP32 teacher predictions and compressed spectral loss only in training. An optional teacher-quality gate uses training clean/noisy references and is disabled for the plain KD control. `calibrate_distillation_weight` measures primary/KD gradient norms on eight training batches, targeting a 0.1 ratio while restoring model modes and gradient buffers. The initial config's fixed 0.1 coefficient is an uncalibrated hypothesis; use a recorded measured coefficient for a budgeted experiment. Compare against a matched no-KD fine-tune. No completed KD benefit is claimed. Broader student/teacher training and verified broader-data teacher lineage require their own recorded comparison.

The [frontier experiment plan](research/esp32_frontier_experiment_plan.md) defines staged validation, preservation checks and compute budgeting. Known architectures are legitimate baselines; the objective does not require inventing a new network.

## Reproduction and final freeze

`notebooks/ESP32_S3_Training.ipynb` is a self-contained Colab entry point with a hash-checked source archive. Configuration files select architecture, manifests, phase, checkpoint and output directory. Paths beginning `/content` require adjustment locally. The notebook is rebuilt with `python scripts/build_esp32_notebook.py`; `python scripts/package_esp32_colab.py` packages current source while excluding datasets, checkpoints and personal files. Install `requirements-esp32.txt` without replacing a working Colab GPU Torch installation.

```bash
python -m esp32_denoiser.extra_data --root /content/extra_audio --download
python -m esp32_denoiser.development --root /content/extra_audio
python -m esp32_denoiser.train --config configs/esp32_zero_bias_float.json
python -m esp32_denoiser.train --config configs/esp32_zero_bias_qat.json
python -m esp32_denoiser.embedded --integer-model denoiser_int8.bin \
  --manifest /path/to/val.jsonl --output embedded_validation.json \
  --compare-reference --perceptual
```

`evaluate --checkpoint` evaluates float/QAT checkpoints. `evaluate --integer-model` uses the C neural core with Python/Torch DSP. `embedded --integer-model` evaluates the **complete C frontend**, defaulting to PCM16; `--compare-reference` also measures Python DSP and unclipped full-C float32. Reports include paired means, valid counts, hashes, input clipping and output-at-rail counts. Contact with a PCM rail is not automatically proof that clipping occurred.

Before downloading the official test, finish architecture/training comparisons, evaluate each selected float/integer pair on the same complete validation/development cohorts, inspect preservation and listening examples, and freeze model/manifest hashes and the reporting protocol. The provisional same-model quality-loss limit is 0.3 dB and the packed-data cap is 99,000 bytes. The three completed global controls meet these validation requirements; that does not finish model selection. A subsequently observed test score must not select another model while retaining an untouched-test claim.

## Hardware evidence and remaining work

Real ESP32-S3 builds pass with ESP-IDF 5.4.2, ESP-DSP 1.8.2, the Xtensa toolchain and no PSRAM. The captured trained **signed-baseline** build occupies 324,352 application bytes and 48,268 static writable RAM bytes; it is not the later zero-bias model. The **untrained frequency prototype** build occupies 327,776 application bytes and 77,956 static RAM bytes. Source-derived ML reservations fit the provisional 200 KiB ceiling, but are not measured heap/stack peaks. Application bytes already include model data. [Detailed memory audit](research/esp32_memory_audit.md).

The zero-bias full-C host evaluator processes 2,279.58 seconds of audio in 27.69 seconds, **RTF 0.01215**, excluding loading, file I/O, compilation and metric calculation. This proves faster-than-real-time **host offline throughput only**. No ESP32 board has executed the benchmark, and Xtensa SIMD has compiled but not run in the captured evidence. Board p99/max hop time, 16 ms deadline misses, I2S behavior, memory peaks and acoustic end-to-end latency remain unmeasured. ESP32-S3 estimates must not be attributed to the original ESP32.

Run `python -m pytest tests -q` for relevant implementation checks. Coverage includes causality/streaming equivalence, paired alignment and split guards, deterministic external mixtures, SI-SDR and preservation invariants, frozen teacher/calibration behavior, phase/resume correctness, integer parity, complete C frontend/PCM16 behavior and optional perceptual metrics. Host tests cannot replace board execution.
