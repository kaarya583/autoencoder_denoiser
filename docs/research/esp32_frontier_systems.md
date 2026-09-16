# ESP32-S3 systems frontier for fully integer speech enhancement

The strongest next experiment should combine frequency-shared convolutions, short local temporal history, and a compact global context path. The current global spectral TCN remains a useful deployable baseline. An integer GRU is a credible research branch, but it requires an explicitly quantized recurrent cell and additional validation. Neither a parameter count nor a kernel benchmark establishes the best attainable speech quality or real-time performance.

This audit, completed September 12, 2026, uses the ESP32-S3 target and the project’s constraints: a complete model payload no larger than 99,000 bytes, at most 200 KiB of ML working memory, and INT8 neural weights, activations, and persistent state. INT32 accumulation and bias are allowed. The FFT and other conventional signal processing remain floating point. The expanded training objective is a quality–efficiency frontier; 8 dB SI-SDR improvement is not a ceiling.

## What is established in this repository

The signed-state global TCN has 83,392 matrix weights and 1,346 biases, uses about 5.212 million MACs per second at 62.5 frames/s, and exports a 94,480-byte version-2 model containing quantized weights, biases, scales, graph records, and DSP constants. Its temporal FIFO is 8,064 INT8 bytes; positions and working vectors bring the required neural workspace to 8,280 bytes. The benchmark reserves 16,384 bytes. These are model and source calculations, not board measurements.

The portable C implementation performs integer neural arithmetic and has exact parity tests against the integer Python reference. ESP-IDF 5.4.2 and ESP-DSP 1.8.2 compile the complete FFT/features/neural/inverse-FFT/overlap-add pipeline for S3. Prior identity-model firmware is explicitly preserved as prototype build evidence. Neither compilation nor the host DSP tests prove board speed. The existing optional internal-SRAM model copy changes storage placement only; it needs a same-board comparison against flash mapping.

## Which optimized operators are actually available

The source examined here is ESP-NN commit `2c222c5e02225177b44ebf21169bc66df3c8b573`, component version 1.3.2. This matters because its recent API includes a per-output-channel fully connected operation. Older claims that ESP-NN only supports a shared fully connected output scale are no longer correct.[^fcapi]

| Operation | Verified capability | Consequence for this project |
|---|---|---|
| Dense/fully connected | S3 signed INT8 paths, including per-channel multiplier/shift arrays | Can accelerate projections; library requantization must first be reconciled with our exact rounding contract. |
| Raw signed dot product | Separate aligned and unaligned S3 assembly routines | Allows acceleration while retaining our existing power-of-two requantization. |
| Pointwise convolution | Spatial reuse with per-output-channel quantization | Attractive for a frequency-shared network; query scratch for each concrete shape. |
| Depthwise convolution | Specialized S3 paths, including 3×3 and suitable channel multiples | Use explicit causal history gathering and frequency padding. |
| Logistic | A 256-byte table drives the integer evaluation routine | Prepare the table offline because the provided preparation routine uses floating point. |
| GRU | No ready ESP-NN GRU with the required complete integer contract was found | Build and test the cell explicitly; a quantized wrapper alone is insufficient. |

The fully connected dispatcher considers alignment, offsets, row length, and row tails. For zero input offset, its 8-bit path has a threshold `row_len >= 64 + 8*(row_len & 15)`; other layouts may dispatch differently. Its per-channel implementation also uses an output-channel-sized correction array on the stack. Blindly adding two 514-entry INT32 scale arrays and this correction buffer to an 8 KiB task stack would consume substantial space.[^fcimpl]

For pointwise convolution with at least eight output positions, the inspected scratch calculation includes `2 * 8 * round_up(input_channels, 8) + 64` bytes: 576 bytes for 32 channels. Other convolution shapes have different storage requirements. The scratch pointer is configured through shared library state, so reuse must be serialized. A dilation field in an API struct is not sufficient evidence that every optimized implementation honors arbitrary dilation. Our causal implementation should gather the three required temporal slices explicitly and invoke a unit-dilation operation.[^conv]

ESP-NN’s published S3 timing table is kernel evidence, not a denoiser benchmark. Its stated cache configuration also differs from our current firmware configuration. No table entry is converted here into an application real-time factor.[^readme]

## Accelerate the baseline without changing its numerical model

Dense projections account for approximately 98.6% of the baseline’s MACs. The current portable scalar loop does not explicitly use S3 vector instructions. A small adapter around the official raw signed dot routines is a more controlled first step than changing both the numerical format and kernel library at once.

The aligned routine requires two 16-byte-aligned operands and a byte count divisible by 16. The unaligned routine requires an aligned first operand and takes its length in **16-byte blocks**, not bytes. Its pipelined loads can read beyond the logical dot-product interval. An adapter must provide readable input guards and verify readable model storage around each unaligned row; scalar fallback is preferable to assuming adjacent memory is safe.[^dot]

The implementation plan is to copy a dense input once into an aligned, guarded temporary buffer, invoke the appropriate raw dot routine for full blocks, and calculate the remaining elements scalarly. Bias addition, saturation, per-channel scale exponents, and nearest-away-from-zero requantization stay unchanged. Inputs beyond the bounded scratch capacity use the scalar path. No weight repacking, model format change, or new scale arrays are required. A startup known-answer test should exercise both assembly entry points on the actual target before collecting timings. Host adapter parity and successful assembly compilation are separate evidence from executing that test on an S3.

At 240 MHz, one 16 ms hop has 3.84 million CPU cycles. Reserving half of that interval for inference gives 1.92 million cycles. The resulting arithmetic budgets are:

| Neural workload | MACs per hop | Cycles/MAC at full deadline | Cycles/MAC at half deadline |
|---|---:|---:|---:|
| 10 MMAC/s | 160,000 | 24 | 12 |
| 20 MMAC/s | 320,000 | 12 | 6 |
| 30 MMAC/s | 480,000 | 8 | 4 |

These figures are ceilings before accounting for FFTs, feature construction, memory movement, activation handling, and scheduling. They are not latency predictions. The board acceptance test remains complete-hop p99 below 16 ms with a useful margin, reported alongside mean, maximum, deadline misses, and actual allocated internal memory.

## Recurrent gates can be fully integer, but must be designed that way

A 64-unit GRU with a 64-dimensional projected input uses `3*64*(64+64) = 24,576` matrix weights. A 387→64 input projection and 64→514 head bring the matrix total to 82,240 weights and about 5.14 MMAC/s. Its persistent hidden state needs only 64 INT8 bytes. The compact state is appealing, but this is an untrained candidate calculation, not an improvement demonstrated against the TCN.

Use a fixed, explicit GRU variant. Reset-before and reset-after formulations differ, including where the recurrent candidate bias is multiplied by the reset gate. Export two activation tables: sigmoid with unsigned-equivalent values 0…255 at scale 1/256, and tanh on a signed INT8 grid. Those two 256-entry tables add 512 bytes to the model budget. The lookup index also requires an explicit fixed input grid and saturation rule. ESP-NN’s logistic preparation routine calls floating-point functions, so invoking it on the MCU would not meet a strict integer neural initialization requirement; freeze those tables during export.[^logistic]

For hidden and candidate values on the same signed scale, an integer update can use

`h_new = saturate_int8(round_away(((256-z) * candidate + z * h_old) / 256))`.

The reset product needs an equally explicit requantization step. INT32 intermediate products are adequate for these bounded INT8 quantities. Every affine, gate, product, and state update must be represented during QAT. A stock quantized GRU cannot be assumed equivalent: the inspected ESP-DL GRU implementation allocates floating gate storage, which does not satisfy this project’s neural arithmetic contract.[^dlgru]

A material risk is state sticking: changes smaller than half a state LSB round away repeatedly, especially when a gate heavily retains the previous state. INT16 persistent state would alter the stated constraint. Validate long silence followed by quiet speech, minutes of continuous audio, abrupt level changes, low-SNR onsets, and the float→fake-quant→actual-integer waveform differences. A one-gate recurrent alternative can reduce affine cost, but its quality here remains an experimental question.[^mgu]

## Frequency sharing moves the constraint from weights to state

A global matrix is used once per frame, so a 99 KB matrix budget naturally provides only several MMAC/s at 62.5 frames/s. To use 10–30 MMAC/s without increasing model storage, reuse kernels across frequency positions. The corresponding temporal state must be counted explicitly:

`FIFO bytes = frequency_positions * channels * (temporal_kernel - 1) * sum(dilations)`.

For 65 frequency positions and 32 INT8 channels with a three-frame kernel:

| Local dilations | FIFO bytes |
|---|---:|
| 1, 2, 4, 1 | 33,280 |
| 1, 2, 4, 8 | 62,400 |
| 1, 2, 4, 8, 16, 32 | 262,080 |

The last FIFO alone exceeds 200 KiB. At 129 frequency positions it would require 520,128 bytes. Long context therefore belongs in a compact global path, while frequency-shared layers use shorter local context or a more compressed frequency axis.

One illustrative hybrid combines a 64-unit global GRU, a 64→32 context projection, and a 65-position local convolution path with 32 channels and dilations 1,2,4,1. A shared 32→8 head produces four complex frequency bins per position and crops the resulting 260 bins to 257. Including a frequency-coordinate input channel yields approximately 57,536 matrix weights and 28.172 MMAC/s. Estimated complete storage is approximately 66–70 KB after biases, scales, two gate tables, DSP constants, and graph metadata. This estimate is not an exported artifact. A conservative allowance of 99,000 model-copy bytes, 33,280 FIFO bytes, 16 KiB DSP storage, 16 KiB kernel scratch, 8 KiB stack, and 16 KiB margin totals 189,624 bytes, or 185.2 KiB.

The repository’s parallel frequency U-Net experiment takes the convolution-only route: three encoder widths 16,24,32, short local temporal blocks, and a compact global TCN at the compressed frequency axis. Its current design accounting is 81,296 matrix weights, 1,874 biases, and 27.561 MMAC/s. Its persistent INT8 history is 18,816 bytes: 14,784 local and 4,032 global. Frequency planes are 257 input bins, 129×16 and 65×24 encoder skips, and a 33×32 bottleneck. Its prototype exports to 94,300 bytes, including weight read guards, alignment, and DSP data. Its validated C implementation requires 34,844 bytes of neural workspace, including temporal history, positions, encoder skips, three reusable planes, and the global residual vector. The S3 build confirms a 10,528-byte audio state and 40-byte model handle. Board stack and heap measurements remain outstanding. It is the simpler immediate candidate because it avoids recurrent gate rounding and can train in parallel across time. The GRU hybrid remains a second architecture branch, not a prerequisite for progress.

GTCRN provides primary evidence that frequency-aware enhancement can achieve useful quality with very few parameters, but its floating recurrent and normalization operations are not proof of an S3 integer implementation.[^gtcrn] Published integer MCU enhancement results on other DSPs similarly motivate experiments without transferring their hardware timings to S3.[^munet]

## Recommended sequence and acceptance evidence

1. Preserve the trained global TCN and its exact scalar C evaluation as the reference. Compile its trained model into both scalar and optional SIMD S3 firmware, recording model hash, compiler configuration, storage placement, and linker memory figures.
2. Train the frequency U-Net under the real storage and state constraints, with actual exported byte counts as a selection gate. Compare matched validation conditions and compute budgets; do not infer quality from parameter count.
3. Add frequency-shared optimized kernels with explicit tensor layout, padding, history, alignment, and scratch contracts. Require exact integer reference parity before accepting a kernel port.
4. Explore the integer GRU branch only with an explicit exported gate/state contract and long-stream tests. Compare gains against the convolution-only candidate at matched complete payload and working memory.
5. Select models by actual integer audio quality and measure complete-hop timing on the intended board, including flash versus internal-SRAM placement. Until that measurement exists, report “compiled for ESP32-S3,” not “real-time on ESP32-S3.”

## Sources

[^fcapi]: Espressif, pinned ESP-NN S3 API, including per-channel fully connected: https://github.com/espressif/esp-nn/blob/2c222c5e02225177b44ebf21169bc66df3c8b573/include/esp_nn_esp32s3.h
[^fcimpl]: Espressif, S3 fully connected implementation and dispatch: https://github.com/espressif/esp-nn/blob/2c222c5e02225177b44ebf21169bc66df3c8b573/src/fully_connected/esp_nn_fully_connected_esp32s3.c
[^conv]: Espressif, S3 convolution implementation and scratch calculation: https://github.com/espressif/esp-nn/blob/2c222c5e02225177b44ebf21169bc66df3c8b573/src/convolution/esp_nn_conv_esp32s3.c
[^readme]: Espressif, benchmark conditions and S3 kernel measurements: https://github.com/espressif/esp-nn/blob/2c222c5e02225177b44ebf21169bc66df3c8b573/README.md
[^dot]: Espressif, raw signed dot product assembly and calling contracts: https://github.com/espressif/esp-nn/blob/2c222c5e02225177b44ebf21169bc66df3c8b573/src/common/esp_nn_dot_s8_esp32s3.S
[^logistic]: Espressif, logistic lookup preparation and evaluation: https://github.com/espressif/esp-nn/blob/2c222c5e02225177b44ebf21169bc66df3c8b573/src/logistic/esp_nn_logistic_ansi.c
[^dlgru]: Espressif, ESP-DL GRU module, inspected September 12, 2026: https://github.com/espressif/esp-dl/blob/master/esp-dl/dl/module/include/dl_module_gru.hpp
[^mgu]: Zhou et al., “Minimal Gated Unit for Recurrent Neural Networks,” 2016: https://arxiv.org/abs/1603.09420
[^gtcrn]: Rong et al., GTCRN author implementation: https://github.com/Xiaobin-Rong/gtcrn
[^munet]: “µNet: Ultra-Low-Memory and Low-Complexity Speech Enhancement for Embedded Digital Signal Processors,” primary manuscript: https://arxiv.org/html/2608.21155v1
