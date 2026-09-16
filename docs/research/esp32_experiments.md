# ESP32 speech denoising: experiment record

Development record updated 12 September 2026 PDT. Completed results below come from the SHA-verified local archive `output/esp32/development_backup_02`. Ongoing experiments are identified separately. These are training-speaker-holdout validation results, not official test results or ESP32 board measurements.

The strongest completed full-C SI-SDR result in this archive is **7.776251759116493 dB improvement** from the zero-depthwise-bias global TCN, using a **94,480-byte** model. The deeper TCN gives **7.755303590865879 dB**, with better measured PESQ and STOI. That small SI-SDR difference does not establish a universal winner. Frequency-sharing and recurrent experiments remain development candidates.

## Fixed protocol and evidence

- Mono 16 kHz audio; 512-sample analysis window and 256-sample hop. Neural inference is causal; waveform alignment includes the overlap needed by the STFT.
- VoiceBank-DEMAND third-party 16 kHz mirror pinned to `4497db342d7312978c45690591fda86117831940`. The mirror's resampling method and byte identity to the original archives are unverified. Decoded samples are cached as FLOAT WAV without additional resampling.
- Training: **10,802 utterances from 26 speakers**, 504,461,572 samples. Validation: **770 complete utterances from p226 and p287**, 36,473,269 samples / 2,279.5793125 seconds. Validation uses training-origin mixtures with held-out speakers; it is not a separate unseen-noise stress test.
- The official test was not prepared in the archived provenance and remains reserved until model selection is complete.
- Selection uses equally weighted, zero-mean utterance SI-SDR improvement. The full-C evaluator reports noisy SI-SDR **6.016738904979768 dB**. All 770 clips are valid. Validation manifest SHA-256: `c1e0ccf95766f1542da3e8eebc1cf33107ca6ecf3ccf59c7a2d8078c171f5668`.
- Completed global controls use seed 2026, batch 32, aligned 3-second crops, AdamW starting at 0.001, plateau reduction, at most 100 float epochs and 15 stale epochs. QAT starts from each best float checkpoint at learning rate 0.0001, allowing 25 epochs / 8 stale epochs. Gain and noise-level augmentation apply only during training.
- Frequency controls use the same seed, batch, crop and starting learning rate, but allow **120 epochs / 20 stale epochs**. The fresh spectral teacher allows **140 / 20**. These are different training budgets, recorded in each configuration.
- The deployment cap is **99,000 actual packed bytes**. Neural weights, activations and temporal state use INT8; biases and accumulators use INT32. Audio DSP uses float32. Model data, executable flash and working RAM are separate quantities.

Evidence: [dataset provenance](../../output/esp32/development_backup_02/data_provenance/voicebank/provenance.json), [training manifest](../../output/esp32/development_backup_02/data_provenance/voicebank/train.jsonl), [validation manifest](../../output/esp32/development_backup_02/data_provenance/voicebank/val.jsonl). Run directories retain configurations, source/manifest hashes, histories, checkpoints and validation reports.

## Completed controls: float → QAT → full C

Float and QAT columns are the selected training-loop validation results. QAT executes fake quantization in PyTorch. Full C executes the exported integer neural graph and complete C audio frontend, including PCM16 input/output, using the portable host FFT. It is the deployment-path quality measurement available here.

| Candidate | Selected float SI-SDRi, dB | Selected QAT SI-SDRi, dB | Full C + PCM16 SI-SDRi, dB | Float → full-C loss, dB |
|---|---:|---:|---:|---:|
| Signed width 64 / 6 blocks | 7.574700970469532 | 7.390656732369905 | 7.390343794074977 | 0.1843571763945553 |
| Signed width 56 / 10 blocks | 7.890509218170143 | 7.755129766382848 | 7.755303590865879 | 0.13520562730426366 |
| Width64 / 6, zero depthwise bias | 7.9390668729788985 | 7.776490720966843 | 7.776251759116493 | 0.16281511386240588 |

Float best checkpoints occur at epochs **60, 67, 86**, while runs stop at **75, 82, 100**, respectively. QAT best checkpoints occur at **3, 19, 19**, with runs ending at **11, 25, 25**. Best-checkpoint epochs must not be confused with stopping epochs.

The QAT-to-C differences are −0.0003129382949280668, +0.00017382448303138176 and −0.00023896185034999462 dB, respectively. These include integer rounding, DSP implementation and PCM16 conversion. For the baseline, matched Torch-DSP/integer-neural evaluation differs from full C + PCM16 by −0.00036269923340270793 dB. PCM16 scores +0.00004044989947438182 dB above the unclipped float32 C path. These measurements support numerical fidelity, not board speed.

Evidence: [signed float](../../output/esp32/development_backup_02/float_signed/best_validation.json), [signed QAT](../../output/esp32/development_backup_02/qat/best_validation.json), [signed full C](../../output/esp32/development_backup_02/baseline_embedded_validation.json); [depth float](../../output/esp32/development_backup_02/float_depth10/best_validation.json), [depth QAT](../../output/esp32/development_backup_02/qat_depth10/best_validation.json), [depth full C](../../output/esp32/development_backup_02/depth10_embedded_validation.json); [zero-bias float](../../output/esp32/development_backup_02/float_zero_bias/best_validation.json), [zero-bias QAT](../../output/esp32/development_backup_02/qat_zero_bias/best_validation.json), [zero-bias full C](../../output/esp32/development_backup_02/zero_bias_embedded_validation.json).

## Perceptual checks on the same validation set

| Signal/model | WB-PESQ | STOI | SI-SDRi, dB |
|---|---:|---:|---:|
| Noisy input | 1.5700197656433303 | 0.8125170903782315 | 0 |
| Depth10, float checkpoint evaluated on CPU | 1.9302681874919247 | 0.830662278964323 | 7.890486309916527 |
| Depth10, full C + PCM16 | 1.9201628332014207 | 0.8281813818308604 | 7.755303590865879 |
| Zero-bias, full C + PCM16 | 1.8893398334453633 | 0.827451939751484 | 7.776251759116493 |

All perceptual evaluations have 770 valid clips and no metric errors. The archived signed-baseline report contains no PESQ/STOI evaluation, so none is inferred. The separate CPU float result differs from training-loop SI-SDRi by about 0.000023 dB; both source values are preserved rather than silently treated as identical.

Depth10 loses **0.010105354290504076 PESQ** and **0.0024808971334625562 STOI** between matched CPU float and full-C evaluations. Zero-bias has a **0.020948168250613186 dB** SI-SDRi advantage over depth10 but lower PESQ and STOI. Final selection should include this tradeoff and listening examples.

The implementation uses `pesq` 0.0.4 in wideband MOS-LQO mode and `pystoi` 0.4.1 with `extended=False`. Reports retain paired valid-subset averaging and amplitude policy. PESQ/STOI were sanity checks, not checkpoint-selection objectives. These noisy scores belong to this custom split and must not be substituted for published official-test baselines.

Evidence: [depth float perceptual report](../../output/esp32/development_backup_02/depth10_float_validation_perceptual.json), [depth C perceptual report](../../output/esp32/development_backup_02/depth10_embedded_validation.json), [zero-bias C perceptual report](../../output/esp32/development_backup_02/zero_bias_embedded_validation.json).

## Architecture and memory accounting

| Candidate | Learned parameters | Neural MMAC/s | INT8 temporal history | Required C neural workspace | Actual packed data / status |
|---|---:|---:|---:|---:|---|
| Signed global width 64 / 6 | 84,738 | 5.212 | 8,064 B | 8,280 B | 94,480 B |
| Signed global width 56 / 10 | 85,186 | 5.2185 | 6,944 B | 7,152 B | 96,496 B |
| Global width 64 / 6, zero depthwise bias | 84,738 | 5.212 | 8,064 B | 8,280 B | 94,480 B |
| Global width 64 / 6, full magnitude / low phase | 84,738 | 5.212 | 8,064 B forecast | No compatible DSP deployment | Float only |
| Frequency U-Net, channels 16/24/32, global 32 | 83,170 | 27.561 | 18,816 B | **34,844 B** | **94,300 B**, measured prototype format |
| Small frequency U-Net, channels 12/16/24, global 24 | 46,534 | 14.584 | 7,776 B | Not recorded in this build evidence | 55,572 B, measured prototype format |
| FrequencyGRU, hidden 16 + global 32 | 81,986 | 25.251, matrix work only | 4,560 B **forecast** | Not implemented | Float only; no INT8 exporter |
| Fresh global spectral teacher, width 256 / 6 | 632,322 | 39.28 | 32,256 B forecast | Not deployed | Float reference |

Temporal history is only part of workspace. The frequency allocation also contains position counters, encoder skips and reusable activation storage. Its 34,844-byte required neural workspace fits the provisional 200 KiB ML-RAM ceiling, but device memory also includes the audio frontend, stack, FFT tables and runtime allocation. The S3 benchmark reserves 49,152 neural bytes rather than exactly the minimum.

The frequency U-Net retains full 257-bin compressed magnitude/real/imaginary input, downsamples frequency 257→129→65→33, applies local causal time-frequency blocks and a small global TCN, and predicts full-bin complex-gain deltas. The small model reduces widths and local depth. Optional encoder BatchNorm adds 224 training parameters to the default model and folds into existing convolutions before calibration/QAT; deployed dimensions and parameter count remain unchanged.

FrequencyGRU replaces only the three local blocks with a shared per-frequency GRU(32→16) and projection(16→32). Current states are float32 and consume **18,240 bytes**; 4,560 bytes is a future INT8-state budget, not an implemented integer measurement. GRU nonlinearities and elementwise gates are excluded from its matrix MAC count. Factory QAT, calibration and frequency export explicitly reject this float-only graph.

Evidence: [baseline export](../../output/esp32/development_backup_02/deploy_baseline/denoiser_int8.bin.json), [depth export](../../output/esp32/development_backup_02/deploy_depth10/denoiser_int8.bin.json), [zero-bias export](../../output/esp32/development_backup_02/deploy_zero_bias/denoiser_int8.bin.json), [frequency build accounting](../../output/esp32/firmware_build/frequency_prototype/build_report.json). Export metadata's `audio_quality: unmeasured` describes export-time status; subsequent evaluation reports provide measured quality for the three global models.

## Compression and reference comparisons

Against the historical repository teacher's **11,854,856 FP32 parameter bytes**, the 94,480-byte global payload is **125.47476714648603× smaller**; the 96,496-byte depth payload is **122.85334107113248× smaller**. The 94,300-byte frequency prototype is **125.71427359490986× smaller**, but that byte ratio does not imply trained quality.

These ratios combine architecture downsizing and INT8 storage. They are not a 119× effect from quantization alone. Packed payloads include INT32 biases, scales, descriptors and model-specific DSP constants; executable firmware and vendor FFT tables are separate. The zero-bias model has 83,392 INT8 matrix weights and 1,346 INT32 biases inside its 94,480-byte payload. Pruning, INT4 and distillation are not established by these completed controls.

The historical teacher's saved validation score used the official test split for selection, so a matched quality-loss claim against it is unavailable. The **fresh** width 256 spectral teacher uses the same 770-utterance held-out validation split: **8.720490942651143 dB SI-SDRi**, best epoch 70, early stop epoch 90. Its 632,322 parameters equal 2,529,288 raw FP32 bytes. Relative to this separate reference, the zero-bias packed model is **26.77061812023709× smaller** and scores **0.9442391835346502 dB lower**. This compares a float reference with an integer candidate across architectures and training budgets; it is not quantization-only loss or evidence of distillation.

Evidence: [fresh teacher selection](../../output/esp32/development_backup_02/float_spectral_teacher/best_validation.json), [teacher stopping summary](../../output/esp32/development_backup_02/float_spectral_teacher/summary.json), [teacher configuration](../../output/esp32/development_backup_02/float_spectral_teacher/config.json).

## Other controls and ongoing experiments

The historical all-ReLU control peaked at **3.794249741699514 dB** and stopped at epoch 24. Later residual states were entirely inactive. The signed correction preserves negative residual information and also starts residual pointwise branches with smaller weights. It changes neither parameter count nor MAC count, but this comparison does not isolate those two changes. A trained signed-model probe still found 21, 19, 13, 12, 9, 6 inactive depthwise channels on 64 training clips, motivating the zero-bias comparison.

The full-magnitude/low-phase global control completed at **7.74424117473514 dB**, best epoch 70, early stop epoch 85. It preserves all 257 magnitudes plus real/imaginary components for bins 0–64 while retaining 387 input channels. That DSP path has not been ported to integer deployment. [Feature-control summary](../../output/esp32/development_backup_02/float_fullmag/summary.json).

The **small frequency run is completed in this archive**, despite earlier live updates describing it as ongoing: its summary explicitly records early stopping at epoch 59, with **6.201894988899279 dB** best at epoch 39. [Small-run summary](../../output/esp32/development_backup_02/float_frequency_small/summary.json).

The **main frequency run was still in progress when backup 02 captured it**. Its archived history ends at epoch 51; the best recorded value is **7.413387088151869 dB** at epoch 50. This is a frozen snapshot, not a final score. The archived history file timestamp is **13 September 2026, 06:19:48 UTC**. [Archived history](../../output/esp32/development_backup_02/float_frequency/history.jsonl), [snapshot best validation](../../output/esp32/development_backup_02/float_frequency/best_validation.json).

The encoder-BatchNorm experiment has started. FrequencyGRU configurations with and without BN are prepared for matched trials. No completed quality claim for those variants is included here. Initial amplitude/gradient probes support testing BN as an optimization aid; they are not substitute validation results. New synthetic-data recipes must be recorded separately from these completed controls.

## Quantization and full-path checks

An early epoch 38 probe scored **7.518225482 dB float → 7.234225929 dB simulated INT8**, a 0.283999552 dB decrease over 770 clips. It preceded the completed QAT runs. Training-only calibration selected hidden exponent −4, observing post-nonlinearity values and pre-add branches needed for cancellation while ignoring extrema that clipping discards. Completed deployment rows supersede this probe for final-path quality.

C/PyTorch tests cover nontrivial weights, persistent state, full-bin feature layout, signed extrema, reset, real-FFT DC/Nyquist behavior, nondefault DSP constants, partial-hop flush and PCM16 rounding/clipping. An independent ESP-DSP-compatible FFT stub exercises bit reversal and inverse-transform conventions. These establish numerical contracts; they do not execute real Xtensa assembly.

Host full-C processing of 2,279.5793125 seconds of validation audio took **31.680845404996944 s** for baseline, **31.703354620995924 s** for depth10 and **27.687939142999312 s** for zero-bias. Host RTFs are **0.013897671921865603**, **0.013907546206947745** and **0.012146074054617616**. They exclude compilation, loading, file I/O and metric computation, use two Torch threads, and describe **host offline throughput**, not ESP32 latency or controlled hardware speedup comparisons.

## Actual ESP32-S3 builds

The real S3 toolchain compiled trained-global and frequency-identity firmware using **ESP-IDF 5.4.2, ESP-DSP 1.8.2 and Xtensa GCC 14.2.0 (`esp-14.2.0_20241119`)**, with `-O2`, `-std=gnu17`, `-mlongcalls`, 240 MHz, flash-mapped weights and no PSRAM.

| Captured build | Model payload | Application binary | Static writable RAM | Status |
|---|---:|---:|---:|---|
| Trained global baseline, scalar | 94,480 B | 323,072 B | 48,268 B | Earlier compiled source revision |
| Trained global baseline, S3 SIMD | 94,480 B | 324,064 B | 48,268 B | Earlier compiled source revision |
| Trained global baseline, shared kernels + S3 SIMD | 94,480 B | **324,352 B** | **48,268 B** | Newest captured trained-global build |
| Frequency U-Net, shared kernels + S3 SIMD | 94,300 B | **327,776 B** | **77,956 B** | **Untrained identity prototype** |

The trained-global model hash matches the signed-baseline full-C validation: `bfd31b653e5924e8ee10fec0721f8e61ba49943817e5c74d58e59279aa798182`. It is not the later higher-scoring zero-bias or depth10 model. Frequency prototype hash: `541aca39e2b15976d0ddefda2ca0e3311d8c9e9868f154aea4ee42bbdd1a4c91`; its build is not a quality result.

The 32-bit S3 compiler gives frequency a **40-byte model handle**, **10,528-byte audio state**, **34,844-byte required neural workspace**, and **49,152-byte reserved neural buffer**. The benchmark reserves 8,192 bytes of main task stack; the dense gather buffer uses 1,088 bytes within stack. The global audio state is 13,616 bytes, with 8,280 required / 16,384 reserved neural bytes.

The separate memory audit accounts for **69,896 reserved ML-related bytes with flash weights**, or **164,196 bytes with a complete SRAM model copy**, including the frequency neural reservation, audio state, model handle, 1,024 bytes of PCM buffers, 960-byte FFT heap request and main stack. These source-derived subtotals exclude unrelated RTOS, I2S/radio and application requirements; they are not observed heap/stack peaks. Static linker RAM and this runtime subtotal overlap and must not be added together. [Memory audit](esp32_memory_audit.md), [machine-readable accounting](../../output/esp32/firmware_build/memory_audit.json).

**No ESP32 board has been flashed or timed; the actual Xtensa self-test has not executed.** Build reports establish compilation/linking and static footprint only. Full-hop mean/p99/max latency, 16 ms deadline misses, internal-memory peaks, I2S integration and end-to-end delay remain unmeasured. Sizes belong to captured source revisions; later source changes require rebuilding.

Evidence: [trained scalar/SIMD builds](../../output/esp32/firmware_build/trained_baseline/build_report.json), [newest trained shared-kernel build](../../output/esp32/firmware_build/trained_baseline_shared_kernels/build_report.json), [frequency identity build](../../output/esp32/firmware_build/frequency_prototype/build_report.json). Exact artifact/source hashes and source snapshots are retained. The original 322,432-byte identity global prototype is superseded by these more specific records.

## Claim boundaries

Completed work demonstrates compact trained speech enhancement, bounded integer neural state, real packed payloads, close C/PyTorch agreement and real S3 toolchain builds. It does not yet establish real-time ESP32 execution or final official-test quality. Experiments combine established spectral masking, frequency sharing, causal temporal processing, recurrent modeling and QAT. One seed's comparison is not proof of architectural superiority or a new compression method.

There is no fixed 8 dB ceiling for the ongoing search. A final claim must attach the selected checkpoint, exact integer payload, frozen evaluation protocol, perceptual checks and measured hardware configuration to the result.
