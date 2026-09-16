# Speech Enhancement Within a 99 KB INT8 Budget

## 1. Recommended target

**Build a streaming, single-microphone, 16 kHz speech enhancer for ESP32-S3 with a deployed model-data payload of at most 99,000 bytes, maximizing measured integer SI-SDR improvement within the memory and timing constraints.** Eight decibels was an initial milestone, not a ceiling or stopping rule. Wideband PESQ of 2.85, STOI of 0.940, average complete processing time of 8 ms per 16 ms hop, and microphone-to-output latency below 50 ms remain aspirational benchmark targets. They are not measured results, predicted outcomes, or confidence intervals.

The expanded implementation now has completed integer controls and continuing architecture/training experiments. Read the [measured experiment ledger](esp32_experiments.md) and [frontier contract](esp32_frontier_experiment_plan.md) for current evidence. The strongest completed control at this update is 7.7763 dB SI-SDR improvement through complete C PCM16 inference on a custom 770-clip development split, with a 94,480-byte payload. That development protocol is distinct from the published VoiceBank test benchmarks below. The official test remains sealed; physical-board timing is unmeasured.

The literature supports this as a worthwhile research objective. It does not establish that this exact combination has already been achieved on ESP32-S3. The recommendation assumes speech enhancement for communication or recording, rather than general music restoration, speaker extraction, acoustic echo cancellation, or hearing-aid latency. Those additional tasks would change the optimization problem.

| Constraint | Recommended specification |
|---|---|
| Hardware | ESP32-S3 at 240 MHz; one core budgeted for enhancement |
| Model-data ceiling | 99,000 bytes, including weights, biases, scales, model-specific tables and graph metadata |
| Neural precision | INT8 weights, activations and persistent neural state; integer accumulation and requantization |
| Model search | Actual packed payload controls admission; approximately 5–33 MMAC/s under investigation, pending board timing |
| Streaming | 512-sample analysis window, 256-sample hop; no additional future frames |
| Processing | Mean <=8 ms; p99 <=12 ms; no observed deadline misses in the acceptance run |
| Audio/ML internal RAM | <=200 KiB, including resident weights and all audio/ML buffers |
| Quality goal | Maximize actual integer SI-SDRi; retain perceptual and clean-speech checks; 8 dB is an initial milestone only |
| Quality stretch | WB-PESQ >=3.00, subject to the same integer, timing and memory constraints |

Recent evidence improves the outlook beyond a generic tiny band-gain network: modern spectral models allocate capacity selectively across frequency and time. However, compression ratio alone cannot identify an optimal architecture. The winning model must be selected from measured integer implementations under these constraints.

The S3 is the assumed primary chip because it has dedicated SIMD instructions and 512 KB on-chip SRAM. This specification is not a promise for the original ESP32. Research and source availability were assessed as of September 12, 2026.[^3]

## 2. Compression accounting and the existing model

The repository's saved GatedTCNWaveformResUNet contains **2,963,714 FP32 parameters**, or **11,854,856 bytes** of raw parameter data. Its best recorded validation SI-SDR improvement is 9.0038 dB at epoch 37. The training configuration uses the official test split for validation and checkpoint selection, so that score is not an untouched final test measurement.[^1]

The compression calculation must compare defined quantities:

**11,854,856 / 119 = 99,620.64 bytes.**

A ceiling of **99,000 bytes** therefore exceeds the requested reduction: approximately **119.75x** relative to the current raw FP32 parameter payload. By contrast, a 100,000-byte payload gives 118.55x. Neither ratio means the entire firmware image or working RAM shrinks by that amount.

INT8 representation supplies only the first factor of four. Approximately another **30x reduction in model coefficients** must come from a more efficient student architecture. The original model's ZIP archive includes multiple checkpoints and optimizer state; its archive size is not the correct denominator.

| Quantity | Meaning and accounting rule |
|---|---|
| Learned parameter count | Trainable coefficients only; useful for architecture comparison |
| Deployed model data | All packed weights, integer biases, scale data, graph descriptors, padding and model-specific constants |
| Runtime RAM | Live activations, recurrent or convolution state, scratch, audio buffers, plus weights copied into RAM |
| Firmware flash | Executable code, runtime, model data and other firmware assets; report separately |

A 100K-parameter network is therefore too close to the ceiling. Start around 60K-80K learned coefficients, measure the complete binary, and spend any remaining bytes where validation shows the largest quality gain. Fixed filterbanks need accounting too: two dense 192-by-64 tables contain 24,576 values before precision and alignment are considered. Sparse or analytic filterbank implementations may avoid that storage.

The current waveform model also has an activation of 24 by 32,000 values for a two-second chunk: 768,000 bytes even if stored in INT8. Counting its convolution operations gives approximately 13.98 billion MACs per two-second chunk, excluding nonlinearities and normalization. Its one-second overlap hop would repeat that work approximately once per second. These are architecture calculations, not hardware measurements.[^2]

This supports replacing the inference architecture. The existing model remains a historical comparison; its test-selected checkpoint is excluded from the new strict distillation lineage. A fresh 632,322-parameter teacher has now been trained using only the permitted training/selection sources.

## 3. Hardware and runtime evidence

ESP-DL distinguishes the S3's accelerated PIE instruction path from the original ESP32's C implementation. Its convolution support permits ordinary and depthwise convolution, rather than arbitrary channel grouping. An architecture called "grouped" in a paper may consequently require decomposition or custom kernels.[^4]

Espressif already distributes neural noise suppression for S3: NSNet2 uses quantized ERB spectral masking at 16 kHz, with a 64 ms analysis window and 32 ms hop. This establishes a product-level precedent for the spectral approach. The isolated **4.534 ms** timing on the NSNet documentation page is explicitly a **P4 at 400 MHz** result, despite appearing under an S3 URL; it must not be used as an S3 estimate.[^7]

The Chinese S3 benchmark reports an AEC + NSNet2 + VAD low-cost pipeline using about 50.3 KB internal RAM and 821.4 KB PSRAM, with separate feed/fetch CPU percentages of 60.0% and 8.2% of one core. Its NSNet2 DNSMOS result of 2.71 uses a proprietary recording set without enough public scoring detail for cross-paper comparison. The English page currently reports materially different resource figures. These are full-pipeline vendor reports, not a 99 KB neural-model benchmark.[^8][^9]

Kernel evidence supports feasibility without supplying an application speed guarantee. ESP-NN reports 331,008 cycles for an optimized 64-by-64 pointwise convolution over 100 positions on S3. Dividing its 409,600 MACs by the reported cycles at 240 MHz gives approximately 297 MMAC/s for that specific shape. Small recurrent operations, depthwise layers, memory movement and runtime dispatch can have very different efficiency.[^10]

ESP-DSP reports 7,294 cycles for a 512-point INT16 complex FFT on S3, versus 44,594 for its FP32 counterpart. At 240 MHz these are approximately 0.030 and 0.186 ms. They exclude surrounding feature extraction, buffering and reconstruction. FFT cost should be measured, but is not automatically the dominant problem.[^11]

**Planning judgment:** constrain the first architecture to 10-20 MMAC/s and benchmark on the selected board before enlarging it. At 20 MMAC/s and 62.5 hops/s, the network performs about 320,000 MACs per hop. An 8 ms budget provides 1.92 million cycles at 240 MHz for the complete processing path. This is an explicit budget to test, not a conversion from MACs to guaranteed latency.

Keep frequently accessed state and scratch in internal RAM. A PSRAM-equipped development board is useful for instrumentation, but the proposed enhancement path must demonstrate operation without depending on PSRAM. Firmware, task stacks, networking and instruction placement need separate memory headroom.

## 4. Relevant compact-model results

Published scores below describe different implementations and evaluation protocols. None of these rows alone establishes a fully INT8, 99 KB, ESP32-S3 deployment.

| Model | Reported size / compute | Quality evidence | Main limitation for this target |
|---|---|---|---|
| GTCRN | 23.7K learned; 48.2K including fixed ERB values; 33 MMAC/s in current code | VCTK-DEMAND: PESQ 2.87, STOI .940, SI-SNR 18.83 dB | Float reference with GRUs and normalization; reported RTF is desktop CPU |
| LiSenNet | 37K; 56 MMAC/s | VoiceBank-DEMAND: PESQ 3.07, STOI .939; noisy-phase ablation PESQ 3.02 | Full result includes iterative phase refinement; integer recurrence needs work |
| UL-UNAS, revised paper | 171K; 35 MMAC/s | VCTK-DEMAND: PESQ 3.09, STOI .941 | Exceeds model-data budget before metadata |
| CoFi-Lite | 83.12K; 12.87 MMAC/s | Simulated DNS3: PESQ 2.16, ESTOI .7610, SI-SNR 11.80 dB | No demonstrated INT8/S3 result; code unavailable at review date |
| muNet | 46K; 28 MMAC/s; reported 90 KB static memory | Full INT8 evaluation and embedded DSP execution reported | NXP HiFi4 result, not S3; static memory excludes workspace |

GTCRN is the best accessible reference for reproducing a strong tiny baseline. Its author repository explains the learned-versus-fixed parameter discrepancy and provides streaming code and checkpoints. Its noisy SI-SNR is 8.45 dB, making the reported improvement 10.38 dB on that evaluation, not directly comparable to the repository's validation SI-SDRi.[^12]

LiSenNet demonstrates additional quality at modest parameter count; removing Griffin-Lim costs only 0.05 PESQ in its ablation. That makes its streaming-friendly variant worth testing. UL-UNAS demonstrates architectural headroom but exceeds this storage limit; the latest revision reports 171K/35M, superseding the earlier 169K/34M figures.[^13][^14]

CoFi-Lite is the strongest recent architectural lead for the compute budget. On its matched simulated DNS3 comparison it improves PESQ from GTCRN's 2.07 to 2.16 while reducing compute from 31.97 to 12.87 MMAC/s. It uses complementary coarse full-band and detailed low-frequency processing. These are floating-point paper results. Its authors plan an academic inference/checkpoint release for January 2027; an implementation today requires reconstruction and independent validation.[^15][^16]

**Selection:** use GTCRN as the immediate reproducible quality anchor, while developing a quantization-aware student with selective frequency processing. Treat CoFi-Lite as evidence for that design direction, not a ready-to-flash model.

## 5. Full INT8: achievable, but not automatic

Define the neural contract as INT8 weights, input features, intermediate activations, output masks and persistent neural state. Integer biases, wider accumulators, shifts and fixed-point nonlinear calculations are allowed. Do not require 8-bit PCM: the audio path should preserve at least 16-bit samples. FFT, windowing and mask application may use INT16/INT32 fixed-point DSP outside the neural graph.

ESP-DL's S3 path uses symmetric, per-tensor, power-of-two quantization with round-half-up. Training must reproduce the chosen runtime's arithmetic. A per-channel or arbitrary-scale QAT model is not automatically interchangeable. Stock ESP-DL GRU code also uses floating-point gate/state arithmetic internally in its nominal INT8 path; an integer-only requirement needs an audited fixed-point GRU or a different temporal block.[^5][^6]

The recent muNet paper reports full INT8 PTQ and real-time NXP RT685 HiFi4 execution, with static memory excluding workspace. Its quantization comparison is especially useful:[^17]

| Reported algorithmic latency | Float SI-SDR improvement | INT8 SI-SDR improvement |
|---|---:|---:|
| 16 ms | 3.59 dB | 3.55 dB |
| 8 ms | 3.09 dB | 2.31 dB |
| 4 ms | 2.52 dB | 0.50 dB |

Those results establish feasibility and latency sensitivity, not a transferable S3 speed or VoiceBank score. The 90 KB figure is reported static memory and should not be assumed to equal every INT8 export's file size. The publication is a recent preprint.

FQSE shows that specialized full W8A8 training can preserve Conv-TasNet SI-SNR: 14.74 dB float versus 14.77 dB quantized on LibriMix. Its exported model is 5.20 MB, so this is a quantization-method result, not a microcontroller-sized model. It identifies high-SNR inputs and input/output quantization as important failure cases.[^18]

TinyLSTMs reports approximately 11.9x storage reduction with 0.52 dB SI-SDR loss on CHiME2, but keeps a 16-bit output mask. A GAP9 study reports approximately 0.3 PESQ average loss from uniform INT8 PTQ, with smaller losses using mixed precision. These demonstrate why full-INT8 claims must name exactly which tensors and operations are quantized.[^19][^20]

Public GTCRN ports reinforce the implementation risk. GTCRN-Micro reports stalled MCU quantization; another INT8 project deploys only partially quantized ONNX operations, while its recurrent QAT simulation quantizes weights without fully quantizing state. Neither is proof of the requested end-to-end neural precision.[^22][^23]

## 6. Architecture to optimize

The recommended starting architecture is a **causal spectral student with compact convolutional encoders, inexpensive temporal memory, and predicted frequency gains**. Prioritize speech preservation and reliable integer execution. Retain the current waveform teacher for comparisons, but do not preserve its layer structure merely to call the change compression.

Use a 512-point transform at 16 kHz, with 256 new samples every hop. Preserve detailed low-frequency information, compress higher-frequency features, and test whether a small low-frequency refinement path adds enough quality to justify its bytes and cycles. Use stable compressed spectral features with a defined noise floor and training-time level augmentation. This is a proposed design search, not a claim that any one feature layout is optimal.

Compare two temporal implementations before committing substantial training:

| Candidate | Reason to test | Main risk |
|---|---|---|
| Compact GRU or grouped recurrent block | Strong quality references; persistent state is small | Exact integer gate/state implementation, per-step rounding and long-stream stability |
| Causal convolutional block with cached history | Simpler integer operator path; predictable scheduling | Larger history buffers or weaker quality at matched compute |

For strict integer recurrence, state feedback must be present in both training and inference. Quantizing a GRU weight tensor while maintaining hidden states in float does not test this requirement. A custom recurrent kernel should be validated against an explicit integer reference before being used as the foundation of the model.

For convolutions, benchmark dense pointwise and depthwise choices at their actual small tensor shapes. Use only supported grouping patterns or include the cost of decomposing grouped operations. Replace temporal GroupNorm or LayerNorm where necessary through retraining; input-dependent normalization cannot simply be folded into preceding weights like fixed inference-time BatchNorm.

Start with magnitude gains and unchanged noisy phase. Add complex correction or a small deep-filter head only if it improves held-out listening and metrics within the same budget. A model that suppresses noise aggressively while damaging consonants should lose the comparison, even if its noise-only score improves.

Deploy one model initially. Multiple specialist models consume storage, and the current MoE implementation evaluates all experts before choosing one. If noise awareness helps, test a small conditioning signal or auxiliary training target rather than shipping the entire expert bank.

Use a modest search over widths, temporal depth, state size, spectral resolution and output head. The selection objective should maximize held-out quality subject to measured model bytes, RAM and frame-time constraints. It should not maximize parameter count or minimize MACs independently.

## 7. Realistic quality and latency expectations

**The defensible quality objective is a useful, near-GTCRN-class speech enhancer.** A VoiceBank-DEMAND WB-PESQ range around 2.7-3.0 is a reasonable planning range for successful candidates, not a statistical forecast. The central acceptance goal is 2.85, with 3.00 as a stretch. Fully quantized hardware outputs must earn those scores; floating-point simulations do not satisfy the target.

| Measurement | Initial acceptance goal | Interpretation |
|---|---:|---|
| WB-PESQ, VoiceBank-DEMAND | >=2.85 | Same evaluation implementation and processing policy for all models |
| STOI, same set | >=0.940 | Report clean-input and low-SNR subsets too |
| SI-SDR improvement, same set | >=8 dB | Recompute teacher and noisy baseline; no reuse of incompatible published scores |
| Quantization penalty | <=0.10 PESQ; <=.005 STOI; <=0.3 dB SI-SDRi | Compare integer student with the same floating-point student |
| Mean compute RTF | <=0.50 | <=8 ms complete processing per 16 ms of audio |
| p99 processing | <=12 ms | Leaves scheduling margin before the 16 ms deadline |
| End-to-end audio delay | <50 ms | Physical capture-to-output measurement, including buffering |

These values are deliberately separated. A 16 ms hop is not a 16 ms acoustic latency. Window accumulation, synthesis, scheduling, DMA and output buffering contribute. Likewise, a low average RTF can hide rare overruns. Require zero observed dropped frames or deadline misses during a representative acceptance run, then extend the stress test.

Architecture reduction and quantization are different losses. First compare the float teacher against the causal float student; then compare float student against integer student. A two-stage distillation study found a 60K student improving from 6.34 to 6.77 dB SDR improvement, while its 1.9M teacher achieved 8.65 dB. Simple output matching barely helped that student. This is evidence that distillation can help, but does not erase capacity limits; the metric there is SDR, not SI-SDR.[^21]

Do not promise that 119x smaller means only 1-2 dB worse than the current model. A redesigned student could match or exceed the current teacher on some metrics, or lose more under difficult conditions. The existing teacher was not evaluated with the proposed streaming and independent-test protocol.

At very low SNR, competing speech, wind or strong reverberation, expect larger degradation than the central benchmark suggests. Quality targets for those conditions should be defined through matched comparisons, rather than importing the VoiceBank PESQ target. Power consumption also remains unestimated until board current is measured under the actual radio and audio workload.

## 8. Training and quantization strategy

Use clean-reference supervision as the primary training signal. Reproduce the teacher and a tiny public baseline first, including their exact normalization and framing. Distillation should be an ablation: compare supervised-only training, output distillation, and feature-assisted initialization followed by supervised fine-tuning. Keep it only when it improves the selected student under the same deployment constraints.

Begin with VoiceBank-DEMAND for rapid, interpretable experiments. For robust microphone behavior, expand to approximately **200-1,000 hours of generated mixtures** as a planning budget, using disjoint clean speakers, noise recordings and room responses. This is generated audio duration, not a requirement to collect that much unique raw material. Microsoft's DNS resources include clean speech, noise, room responses and synthesis code suited to this workflow.[^24]

The training distribution should cover stationary and rapidly changing noise, multiple input levels, distant speech, mild reverberation, microphone coloration and realistic bandwidth limits. Include high-SNR and clean examples explicitly. Maintain a separate set for clipping, wind, handling noise and competing speech, where failure behavior is more informative than average performance.

Freeze the task definition: preserve all speech, or extract a particular speaker. A general denoiser has no reliable way to identify which of two equally prominent speakers should be removed without an additional cue. Similarly, acoustic echo cancellation requires a playback reference and its own evaluation; it should not be silently included in the denoising target.

Use compressed spectral reconstruction losses plus a waveform or scale-invariant term, and evaluate a perceptual term if it improves listening. Do not optimize PESQ alone. Add clean-input preservation examples and inspect consonants, quiet speech onsets and speaker timbre. Loss weights are hyperparameters to validate, not facts inherited from the teacher.

Quantize early, once the student learns useful behavior. Calibrate using complete sequential audio, then run QAT with the actual deployment quantizer, including state clipping, scales and nonlinear approximations. Calibration and QAT must not see final test audio. Train recurrent models on sufficiently long sequences and test much longer ones to expose state drift.

The verification ladder is: floating-point streaming reference; fake-quantized student; explicit integer reference; compiled board inference; microphone pipeline. Save intermediate tensors and known-answer audio fixtures. Any output mismatch must be resolved before comparing quality, because a conversion error can otherwise masquerade as unavoidable INT8 degradation.

## 9. Evaluation and resource accounting

Partition the experiment into three views. First, use a fixed public benchmark for reproducibility and comparison with compact-model literature. Second, use an independent synthetic set with disjoint speakers, noise files and room responses. Third, use real microphone recordings from the intended board and acoustic enclosure. Reserve the official benchmark test split for final scoring; choose checkpoints on a separate development split.

Evaluate the unprocessed input, a classical noise suppressor, Espressif NSNet2 where available, GTCRN, the current teacher, the float student and the integer student. Reuse the exact same noisy signals. Fix sample rate, latency compensation, normalization, trimming and metric versions. Do not compare one model with an extra post-filter against another without declaring it.

Compute utterance-level SI-SDR improvement, wideband PESQ and STOI or ESTOI, keeping STOI and ESTOI distinct. Report both absolute output scores and improvements over noisy input. For noise-only or silent signals, use appropriate artifact/noise measurements rather than unstable SI-SDR values. Bootstrap paired utterances or speakers for uncertainty and inspect the distribution, not only the mean.

Use a listening comparison that separately rates speech distortion and remaining noise. Include clean/high-SNR inputs, fricatives, quiet speakers, speech onset after silence, changing noise, sustained background music and long recordings. Automated MOS predictors can supplement these tests but must have a pinned model version and should not substitute for listening. DNS Challenge work explicitly identifies speech distortion and word-accuracy regressions as concerns.[^25]

For hardware acceptance, replay at least 30 minutes continuously with production audio input/output and intended Wi-Fi or Bluetooth activity. Log per-hop processing duration, deadline misses, DMA overruns, queue depth, stack high-water marks, minimum free internal heap, peak allocations and board current. Extend to a multi-hour soak for the release candidate. Never reset recurrent state at arbitrary short evaluation boundaries if deployment will run continuously.

Measure RAM with the complete enhancement path loaded. Include copied weights, graph descriptors, audio queues, FFT workspace, feature buffers, temporal history, output overlap, DMA buffers and any runtime scratch. Count the maximum live memory, not the sum of separately measured isolated modules. Report PSRAM usage independently, even if it is zero.

A practical initial allocation is 97 KiB for resident model data, 40 KiB for neural working state, 20 KiB for DSP/audio buffers, 25 KiB for kernel scratch and 18 KiB margin: **200 KiB total**. This is a proposed budget; actual tensor lifetimes and allocator alignment decide whether it is achievable.

## 10. Implementation milestones and decision rule

| Milestone | Deliverable | Continue only when |
|---|---|---|
| 1. Reproduce baselines | Fixed manifests, teacher and GTCRN scores, streaming reference | Scores and signal alignment are understood |
| 2. Validate runtime | Integer convolution and temporal-block microbenchmarks on S3 | Arithmetic matches reference; time and memory are plausible |
| 3. Train candidates | Float recurrent and causal-convolution students | At least one approaches the quality goal inside size/compute budgets |
| 4. Quantize | QAT checkpoint, integer reference, packed deployment binary | <=99,000 bytes; no floating neural fallback; quantization losses pass |
| 5. Integrate audio | Complete microphone-to-output firmware | Mean <=8 ms/hop; p99 <=12 ms; RAM <=200 KiB; delay <50 ms |
| 6. Select and validate | Held-out quality report, paired listening and long-run trace | All resource and quality gates pass together |

The first implementation should reproduce GTCRN and validate the integer temporal block, while the student explores selective frequency processing. A CoFi-inspired design is a strong research candidate; unavailable source code and unverified quantization make it unsuitable as the sole dependency. Retain a simpler causal-convolution candidate as a deployment comparison.

If INT8 quality fails, inspect feature scaling, saturation, output masks and state behavior before adding parameters. If timing fails, profile memory movement and operator dispatch before cutting useful speech capacity. If the model-data limit fails, remove dense fixed tables and graph overhead before reducing learned capacity. Each intervention should address a measured bottleneck.

If the original ESP32 is mandatory, keep the 99 KB storage target but reopen the compute and quality targets after microbenchmarks. S3 SIMD results cannot be carried over. If the S3 meets the timing budget comfortably, use the spare capacity to test a better head or temporal block without exceeding the fixed storage and RAM ceilings.

**Recommended project target:** maximize measured SI-SDR improvement in a streaming ESP32-S3 speech enhancer with <=99,000 bytes of model data, INT8 neural weights/activations/state, and 16 kHz mono audio. Select within measured memory and processing limits while checking perceptual quality, clean-speech level and clipping. The 2.85 WB-PESQ / .940 STOI figures and <50 ms total-delay target are aspirations tied to their stated evaluation conditions, not established feasibility or interchangeable thresholds across datasets. Eight decibels is neither a ceiling nor a stopping rule. The complete combination remains to be demonstrated on a physical board.

## Sources

The numbered notes link to original publications, author repositories and vendor documentation. Web resources were accessed September 12, 2026. Live documentation and repository contents may change; implementation work should pin commits and toolchain versions.

[^1]: Local project archive, `robust_UNet_model.zip`: `run_metadata.json`, `history.json`, and `best_si_sdri.pt`. Checkpoint run completed May 28, 2026; 48 epochs. Parameter payload audited from checkpoint tensors in the repository review. [Local archive](/Users/sidchat/Documents/GitHub/autoencoder_denoiser/robust_UNet_model.zip).
[^2]: Local project, `live_voice_denoiser_demo.ipynb`. GatedTCNWaveformResUNet definition and chunked inference. [Notebook](/Users/sidchat/Documents/GitHub/autoencoder_denoiser/live_voice_denoiser_demo.ipynb).
[^3]: Espressif Systems, *ESP32-S3 Series Datasheet*, version 2.2, 2026. CPU, SIMD and SRAM specifications. [Datasheet](https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf).
[^4]: Espressif Systems, *ESP-DL Operator Support State*, live source. Platform acceleration and operator restrictions. [Support matrix](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md).
[^5]: Espressif Systems, *How to Quantize Model*, ESP-DL documentation, current version accessed 2026. Quantization schemes and S3 arithmetic. [Guide](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html).
[^6]: Espressif Systems, `dl_module_gru.hpp`, live ESP-DL source. Nominal INT8 GRU uses floating gate/state arithmetic internally. [GRU implementation](https://github.com/espressif/esp-dl/blob/master/esp-dl/dl/module/include/dl_module_gru.hpp).
[^7]: Espressif Systems, *Noise Suppression Model (NSNet)*, ESP-SR documentation. Quantized NSNet2, spectral framing and explicitly P4 timing. [NSNet documentation](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/nsnet/README.html).
[^8]: Espressif Systems, *Performance Test Results*, Chinese ESP32-S3 ESP-SR documentation. Full AFE resource use and dataset-specific DNSMOS. [Chinese benchmark](https://docs.espressif.com/projects/esp-sr/zh_CN/latest/esp32s3/benchmark/README.html).
[^9]: Espressif Systems, *Benchmark*, English ESP32-S3 ESP-SR documentation. Resource figures differ from the Chinese page at access time. [English benchmark](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/benchmark/README.html).
[^10]: Espressif Systems, *ESP-NN*, official repository benchmark tables. Shape-specific optimized kernel cycle counts. [ESP-NN](https://github.com/espressif/esp-nn).
[^11]: Espressif Systems, *Espressif DSP Library Benchmarks*, live documentation. FFT cycle measurements by chip and precision. [ESP-DSP benchmarks](https://docs.espressif.com/projects/esp-dsp/en/latest/esp32/esp-dsp-benchmarks.html).
[^12]: Xiaobin Rong and collaborators, *GTCRN*, official implementation of the ICASSP 2024 paper, updated repository. Parameter-accounting correction, quality tables and streaming code. [GTCRN](https://github.com/Xiaobin-Rong/gtcrn).
[^13]: Haoyin Yan et al., *LiSenNet: Lightweight Sub-band and Dual-Path Modeling for Real-Time Speech Enhancement*, 2024. Tables I-III and phase refinement. [Paper](https://arxiv.org/html/2409.13285v1).
[^14]: Xiaobin Rong et al., *UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via Network Architecture Search*, revised 2026 manuscript; IEEE TASLP 2026. Tables VI-VII. [Revised paper](https://arxiv.org/html/2503.00340v2).
[^15]: Leyan Yang et al., *CoFi-Lite: Pushing the Limits of Ultra-Lightweight Speech Enhancement*, July 2026, IEEE SPL accepted manuscript. Tables I-III. [Paper](https://arxiv.org/html/2607.10142).
[^16]: CoFi-Lite authors, *CoFi-Lite official repository*, July 2026 announcements and code-release status. [Repository](https://github.com/Acceleration123/CoFi-Lite).
[^17]: Shrishti Saha Shetu et al., *muNet: Ultra-Low-Memory and Low-Complexity Speech Enhancement for Embedded Digital Signal Processors*, August 2026 preprint. Table 2 and sections 3.4-3.5. [Paper](https://arxiv.org/html/2608.21155v1).
[^18]: Elad Cohen, Hai Victor Habi and Arnon Netzer, *Towards Fully Quantized Neural Networks for Speech Enhancement*, Interspeech 2023. Full W8A8 experiments and high-SNR sensitivity. [Paper](https://www.isca-archive.org/interspeech_2023/cohen23_interspeech.pdf).
[^19]: Igor Fedorov et al., *TinyLSTMs: Efficient Neural Speech Enhancement for Hearing Aids*, Interspeech 2020. Pruning, precision and MCU measurements. [Paper](https://www.isca-archive.org/interspeech_2020/fedorov20_interspeech.pdf).
[^20]: Manuele Rusci et al., *Accelerating RNN-based Speech Enhancement on a Multi-Core MCU with Mixed FP16-INT8 Post-Training Quantization*, 2022. Uniform and mixed quantization comparisons on GAP9. [Paper](https://arxiv.org/abs/2210.07692).
[^21]: Rayan Daod Nathoo, Mikolaj Kegler and Marko Stamenovic, *Two-Step Knowledge Distillation for Tiny Speech Enhancement*, ICASSP 2024. Teacher/student SDR improvements and distillation ablations. [Author-hosted paper](https://mkegler.github.io/publication/daod-nathoo-2023/daod-nathoo-2023.pdf).
[^22]: bglid, *GTCRN-Micro*, author project report and archived roadmap, live repository. MCU quantization difficulties and degraded TFLite output. [Repository](https://github.com/bglid/GTCRN-Micro).
[^23]: aditya8086, *Speech Enhancement with INT8 Quantization*, author repository and `fake_quant.py`, live source. Partial deployed quantization and weight-only GRU simulation. [README](https://github.com/aditya8086/speech-enhancement-with-int8-quantization/blob/main/README.md); [quantization code](https://github.com/aditya8086/speech-enhancement-with-int8-quantization/blob/main/fake_quant.py).
[^24]: Microsoft, *Deep Noise Suppression Challenge 3*, 2021 dataset and synthesis documentation. [DNS3 resources](https://github.com/microsoft/DNS-Challenge/blob/master/README-DNS3.md).
[^25]: Chandan K. A. Reddy and collaborators, *ICASSP 2022 Deep Noise Suppression Challenge*, Microsoft Research. Speech distortion, word accuracy and perceptual evaluation. [Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/07/0009271.pdf).
