# Frequency-sharing architectures for the ESP32 denoiser frontier

Research snapshot: 12 September 2026. This report distinguishes published floating-point results, measured properties of this repository's implementation, and proposed experiments. It does not claim a new architecture, a predicted quality score, or measured ESP32 throughput.

The best immediate direction is a frequency-shared encoder/decoder with a small global temporal branch. It spends computation reusing filters across frequency, while reserving most weights for inexpensive full-spectrum context. The implemented candidate occupies **94,300 bytes**, including integer neural parameters, metadata, alignment, readable weight guards, and its analysis window. Its neural graph costs **27.561 million MAC/s**. A second, controlled candidate replaces its local temporal convolutions with one frequency-shared GRU: this should reduce arithmetic to **25.251 million MAC/s** and persistent neural state from **18,816 to 4,560 bytes**, but requires substantially more integer-runtime work. Neither candidate has a quality guarantee.

## What the constraints actually mean

The working cap is **99,000 decimal bytes for the packed model**, including the model-specific DSP constants. Neural weights, activations, and persistent neural state must use 8-bit integer representations; biases and accumulators use INT32. FFT, feature normalization, overlap-add, and applying the predicted complex gain may remain floating point. This is full INT8 **neural inference**, not an all-integer audio signal path.

The 99 KB cap is distinct from program flash, vendor FFT tables, stack, audio buffers, and working RAM. The provisional **10–30 MMAC/s** band and **200 KiB ML working-RAM ceiling** are engineering targets. Satisfying the arithmetic target does not establish real-time performance: kernel layout, memory access, nonlinear operations, and frequency-sequential recurrence still matter. A board measurement must include the complete audio pipeline and its worst-case frame time.

The repository's FP32 parameter baseline is 11,854,856 bytes. Dividing it by the new 94,300-byte packed payload gives **125.714×**. That ratio combines architectural downsizing with reduced precision, and compares explicitly defined parameter/payload quantities. Quantizing one unchanged FP32 model to INT8 alone gives approximately 4× weight compression, not 119×.

## Published evidence worth using

The following are author-reported floating-point results. Numbers from different datasets or evaluation recipes must not be placed on one common quality leaderboard.

| Model and primary source | Published size and computation | Quality and evaluation scope | Implication for this project |
|---|---:|---|---|
| [GTCRN, official implementation](https://github.com/Xiaobin-Rong/gtcrn) | Current README: 48.2K total stored parameters including fixed ERB transforms; 33.0 MMAC/s | VCTK-DEMAND: PESQ 2.87, STOI 0.940, SI-SNR 18.83 dB versus noisy 8.45 dB | Strong available pretrained reference; complex integer GRUs and LayerNorm still need implementation. |
| [UL-UNAS, v2, 1 February 2026](https://arxiv.org/html/2503.00340v2) | 171K parameters, 35 MMAC/s | VCTK-DEMAND PESQ 3.09 | Full model exceeds the byte cap before metadata; useful architecture and training evidence. |
| [LiSenNet, 20 September 2024](https://arxiv.org/html/2409.13285v1) | 37K parameters, 56 MMAC/s | VoiceBank+DEMAND PESQ 3.07, STOI 0.939 | Weight-efficient, but above the provisional computation band; original phase postprocessing needs streaming scrutiny. |
| [FastEnhancer-T, author results](https://github.com/aask1357/fastenhancer/blob/main/README.md) | 22K parameters, 60 MMAC/s | VoiceBank-DEMAND: PESQ 2.99, SI-SDR 18.6 dB, STOI 0.940 | Tiny weights do not imply tiny compute; an excellent example of carefully checking streaming recipes. |
| [CoFi-Lite, July 2026](https://arxiv.org/html/2607.10142) | 83.12K parameters, 12.87 MMAC/s | Simulated DNS3: PESQ 2.16, ESTOI 76.10%, SI-SNR 11.80 dB versus noisy 5.61 dB | The strongest direct motivation for combining compact local features with inexpensive global context; no released implementation to reproduce yet. |

GTCRN's apparent parameter discrepancy is resolved by its authors: the paper reports 23.7K learned parameters, while the updated repository includes fixed ERB matrices. Inspection of the official model gives 23,669 learned parameters and 48,245 total parameters. The updated implementation also changes accounting/implementation of the fixed low-band mapping. Its released code, checkpoints, and streaming implementation are available under MIT. A sparse implementation can avoid storing redundant dense mappings, but those constants must still be counted in an exported payload. The two grouped dual-path recurrent blocks provide frequency and temporal context; LayerNorm and recurrent gates remain real integer-porting costs. [GTCRN repository and accounting note](https://github.com/Xiaobin-Rong/gtcrn), [official model source](https://github.com/Xiaobin-Rong/gtcrn/blob/main/gtcrn.py).

UL-UNAS v2 is the version to cite: **171K/35M**, rather than the older 169K/34M figures. Its smaller depthwise-separable prototype reports 37.23K parameters and 23.72 MMAC/s, with PESQ 1.99 and SI-SNR 10.80 dB in its DNS3-subset ablation, not the final VoiceBank experiment. The paper directly tests efficient convolution blocks, batch normalization, and reparameterization. It also finds that a tiny complex-mask model can learn almost no imaginary correction, motivating its magnitude-mask objective. Frequency-dependent affine activations add parameters; replacing them with ordinary per-channel operations is a new ablation, not a faithful reproduction. [UL-UNAS v2, architecture and ablation tables](https://arxiv.org/html/2503.00340v2), [MIT code and checkpoints](https://github.com/Xiaobin-Rong/ul-unas).

LiSenNet's useful result is its dual-path ablation: zero, one, and two recurrent modules respectively give 15K/22M/PESQ 2.68, 26K/39M/2.97, and 37K/56M/3.07. Removing Griffin–Lim gives 3.02, not 3.07. Frequency-bidirectional and time-unidirectional recurrent paths are a concrete prior for sharing temporal weights across bands. LayerNorm, Mish/GLU, learned sigmoid, and phase features complicate a faithful INT8 implementation. [LiSenNet paper, Tables I and III](https://arxiv.org/html/2409.13285v1), [author code](https://github.com/hyyan2k/LiSenNet).

FastEnhancer's comparison makes the metric warning tangible: its LiSenNet run using the original recipe scores PESQ 3.08 but SI-SDR 13.5 dB; a streamable retraining scores PESQ 2.98 and SI-SDR 18.5 dB. The authors explain that the original input normalization and Griffin–Lim prevent streaming, and remove them in the streamable comparison. These are changed recipes, not a clean one-variable ablation, but they show why PESQ alone cannot select our SI-SDR-oriented model. Its desktop RTFs do not establish MCU timing. [FastEnhancer author results and footnotes](https://github.com/aask1357/fastenhancer/blob/main/README.md).

CoFi-Lite separates coarse full-band processing from fine low-frequency processing and fuses their representations through a compact global branch. Its custom simulated evaluation uses DNS3-derived mixtures; it is not the VoiceBank benchmark. Its 83.12K learned-parameter count is not a measured 83 KB deployable integer blob: biases, scales, fixed transforms, normalization, and metadata need separate accounting. The author repository currently contains a release-status notice, with inference code and a pretrained model planned for academic use by 1 January 2027. We cannot treat it as an available checkpoint or assume a permissive implementation license. [CoFi-Lite paper](https://arxiv.org/html/2607.10142), [author release status](https://github.com/Acceleration123/CoFi-Lite).

## Candidate A: the implemented frequency U-Net

The model is implemented in `esp32_denoiser/frequency_model.py`, with QAT in `frequency_quantization.py` and actual serialization/reference inference in `frequency_export.py`. These are original implementations of established convolutional, residual, and temporal components; their combination is a testable engineering hypothesis, not a novelty claim.

Audio is mono, 16 kHz. A 512-sample square-root Hann analysis window advances by 256 samples. Framing, left overlap, waveform alignment, and overlap-add reuse the existing tested DSP contract. The neural network consumes the current and past frames only.

For each current frame, let `Z = FFT(window × frame)/(512 × max(frame_RMS, 1e-4))` and `r = sqrt(max(abs(Z), 1e-8))`. The three input channels are `r`, `real(Z)/r`, and `imag(Z)/r` at all 257 bins. This avoids utterance-level normalization and retains fine spectral/phase information before learned frequency downsampling. The shared input grid is initially 2^-7; feature behavior at quiet signals and unusual gains remains part of deployment validation.

| Stage | Dimensions per frame | Operations and purpose |
|---|---|---|
| Input | 3 × 257 | Full-bin compressed complex features |
| Stem | 16 × 129 | Frequency kernel 5, stride 2 |
| Downsample 1 | 24 × 65 | Depthwise frequency kernel 5/stride 2, then pointwise projection |
| Downsample 2 | 32 × 33 | Same factorization |
| Local temporal stack | 32 × 33 | Three depthwise 3×3 time-frequency blocks, time dilations 1, 2, 4; pointwise residual branches |
| Global compression | 32 | Flatten 32×33 and project 1,056→32 |
| Global temporal stack | 32 | Six causal depthwise-time/pointwise residual blocks, dilations 1, 2, 4, 8, 16, 32 |
| Global restoration | 32 × 33 | Project 32→1,056 and add the local representation |
| Decoder | 24×65, then 16×129 | Repeat/crop frequency upsampling, pointwise/depthwise convolutions, additive encoder skips |
| Output | 2 × 257 | Final frequency refinement and complex-gain deltas |

Residual states use signed clipping at ±6; depthwise interior activations use ReLU6. The zero-initialized head gives an identity starting point. Output deltas are clipped to ±1 and quantized; the DSP constructs `gain = (1 + 2*delta_real) + j*(2*delta_imag)`. A complex head permits phase correction, but its usefulness must be measured: capacity can still be spent mainly on magnitude suppression.

The high parameter count in the two global projections is deliberate. Together, their 67,584 weights cost only 4.224 MMAC/s, since they run once per frame. Local shared filters use far fewer weights but apply at many frequencies. This allocation fills available model storage without spending all arithmetic at high spectral resolution.

Measured properties from the repository's exporter are:

| Property | Main candidate | Smaller fallback |
|---|---:|---:|
| Encoder channels | 16, 24, 32 | 12, 16, 24 |
| Global width | 32 | 24 |
| Local dilations | 1, 2, 4 | 1, 2 |
| Learned parameters, including biases | 83,170 | 46,534 |
| INT8 convolution weights | 81,296 bytes | 45,184 bytes |
| INT32 biases | 7,496 bytes | 5,400 bytes |
| Actual complete model blob | **94,300 bytes** | **55,572 bytes** |
| Neural MACs/frame | 440,976 | 233,344 |
| Neural MMAC/s at 62.5 frames/s | **27.561** | **14.584** |
| Persistent INT8 neural history | **18,816 bytes** | **7,776 bytes** |
| Neural temporal context | 141 frames | 133 frames |
| Full C neural workspace | 34,844 bytes | Not yet recorded |
| Physical-board frame time | Unmeasured | Unmeasured |

The main history budget is 14,784 bytes for local convolutions and 4,032 bytes for the global TCN. Individual feature planes are small: the largest explicit decoder plane contains 4,112 INT8 elements. The implemented C workspace reuses three scratch planes and explicitly reserves encoder skips and the global residual. Its measured 34,844-byte requirement includes those buffers, history and ring positions. The S3 build has an additional 10,528-byte audio state and 40-byte model handle; the complete accounting is in the [memory audit](esp32_memory_audit.md). Python tensor memory is not a valid MCU working-set measurement.

Export and runtime regression tests verify the actual byte cap, weight alignment and guard bytes, malformed-blob rejection, metadata preservation, reset/chunk equivalence, and agreement within one output integer level against nontrivial QAT weights. Tests now also verify nontrivial C/reference parity and the complete C audio frontend. They establish the numerical contract; training quality must still be measured for each candidate, and board timing remains unmeasured.

## Candidate B: replace local TCN blocks with a shared GRU

Keep Candidate A's encoder, decoder, global TCN, features, mask, and loss unchanged. Replace only the three local temporal blocks at the 32×33 bottleneck with one **GRU(input 32, hidden 16)** followed by **Linear(16→32)** and an additive residual. Apply the same GRU weights independently to each of the 33 frequency locations. Each location retains its own temporal state; no recurrent operation runs backward in time.

This is a controlled use of the temporal branch already established in dual-path speech enhancement, not an ungrounded architectural invention. DPCRN models within-frame frequency structure and across-frame time structure with recurrent paths. GTCRN and LiSenNet subsequently use related frequency/time sharing to reduce size. [DPCRN, July 2021](https://arxiv.org/abs/2107.05429), [GTCRN official source](https://github.com/Xiaobin-Rong/gtcrn/blob/main/gtcrn.py).

The budget follows directly from matrix dimensions:

| Replacement region | Three current local blocks | Shared GRU plus projection |
|---|---:|---:|
| Matrix weights | `3 × (32×9 + 32×32) = 3,936` | `3×16×(32+16) + 16×32 = 2,816` |
| Biases | 192 | 128, retaining both GRU bias vectors |
| Matrix MMAC/s | `3,936×33×62.5 = 8.118` | `2,816×33×62.5 = 5.808` |
| Persistent INT8 temporal state | 14,784 bytes | `33×16 = 528` bytes |

The resulting full model has **81,986 learned parameters**, **25.251 MMAC/s** of matrix work, and **4,560 bytes** of persistent neural state. Gate products add roughly 0.1 million scalar products/s; nonlinear lookup, requantization, and address work are additional. Removing convolution parameters should save approximately 1.4 KB of payload before adding recurrent descriptors and lookup tables. A complete binary must confirm the final size. Budgeting around 1 KB of 8-bit gate tables should leave the model below 99 KB, but this remains a design estimate.

The quality rationale is longer per-band memory with much smaller history. The drawback is loss of the explicit neighboring-frequency kernel in those three blocks; encoder/decoder filters and the global branch still provide frequency interaction, but they may not fully substitute for it. Long state can retain obsolete noise information. A GRU therefore need not beat the finite-context TCN even before quantization.

Integer recurrence also needs a different precision contract from convolution. The current shared hidden step of 1/16 is too coarse for a state bounded around ±1. State and sigmoid/tanh outputs need separate grids, exact gate-product rescaling, and saturation-aware QAT. Small updates can round away and freeze a state; gate rounding near one can change effective memory duration. PyTorch uses a particular reset-after arrangement for the candidate state, so the integer cell must match that arrangement rather than an interchangeable-looking GRU equation. [PyTorch GRU definition and implementation note](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html).

Before promoting this float-training alternative to deployment, implement an integer cell oracle and tests over long sequences, resets, silence, abrupt noise changes, and thousands of frames. Compare actual integer waveform scores, not just floating recurrent outputs. Retaining INT16 recurrent state would relax quantization, but would no longer satisfy the current strict INT8-state target and should be labeled as a separate experiment.

## Improvements to test without increasing deployed arithmetic

**Folded encoder BatchNorm is the first optimization ablation.** Train with channelwise BN immediately after the stem and four downsample convolutions, before their current activations. Freeze its running statistics and fold it into convolution weights and biases before QAT. The deployed graph can then retain exactly the same operators and dimensions. For output channel `c`, use `a[c]=gamma[c]/sqrt(var[c]+eps)`, `W'[c]=a[c]*W[c]`, and `b'[c]=a[c]*(b[c]-mean[c])+beta[c]`. This is standard inference fusion. [PyTorch convolution/BN fusion](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.fuse_conv_bn_eval.html), [original BatchNorm paper, 2015](https://arxiv.org/abs/1502.03167).

The test must preserve the identity head and residual initialization. Adding normalization immediately before a residual addition can change branch scale substantially; starting with the encoder alone isolates the optimization question. Training statistics pool over batch/time/frequency, whereas inference uses constants. Full-utterance eval-mode validation and exact post-fold parity are necessary. Small running variances may amplify folded weights and worsen quantization, so select using integer quality after recalibration. Inference remains causal after folding; test data must never recalibrate statistics.

**Low-frequency causal deep filtering is a later targeted head ablation.** DeepFilterNet motivates separating envelope suppression from reconstruction using neighboring time-frequency information. A proposed five-tap filter for only 65 low bins would need four past complex spectra: 2,080 bytes in float32. Predicting ten real coefficients from 16 channels costs 160 additional weights and about 0.65 MMAC/s; applying 65×5 complex products per frame adds about 0.081 million real multiplications/s. These are our proposed dimensions, not a claim about the published model's exact configuration. Past-only taps introduce no neural lookahead, but coefficient scaling, causality, and speech distortion still require testing. [DeepFilterNet, October 2021](https://arxiv.org/abs/2110.05588), [author implementation](https://github.com/Rikorose/DeepFilterNet).

## How to select a winner

Use the same training corpus, speaker-held-out validation manifest, waveform alignment, augmentation, and initial training budget for the current global TCN, Candidate A, the folded-BN ablation, and Candidate B. Match random seeds where practical, then repeat promising candidates to avoid selecting a lucky run. Existing pretrained references may have trained on speakers held out by this repository; use them as disclosed external references rather than leakage-free model-selection evidence.

Rank by equal-utterance **actual integer SI-SDR improvement** on validation, with PESQ, STOI, clean-speech preservation, silence behavior, and representative listening samples as parallel checks. Large SI-SDR gains with worsening intelligibility or audible speech removal are not a stronger audio product. When the C audio frontend is available, include PCM16 input/output rounding and clipping in the measured result.

The official test should remain sealed until the final architecture, training recipe, quantization, and selection rule are fixed. Report that final score once alongside its noisy baseline, actual byte count, and measured hardware timing. For real time, measure total mean and tail frame latency on the intended ESP32 variant with the intended memory placement and audio I/O. A 16 ms hop gives a deadline, not a measured runtime. Keep the smaller 14.584 MMAC/s model as a practical fallback if the main architecture misses the deadline.

There is no evidence-based fixed ceiling such as 8 dB for this search. The useful target is the strongest reproducible validation quality that survives exact integer inference and the board deadline, with enough held-out and perceptual evidence to support that conclusion.
