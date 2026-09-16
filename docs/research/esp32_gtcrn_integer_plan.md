# GTCRN: a full INT8 ESP32-S3 port plan

GTCRN is a credible **memory-feasible candidate**, with a substantially harder integer implementation than the current convolutional models. A carefully packed version should fit comfortably inside the 99,000-byte model limit, including its fixed signal-processing constants. Neither its published arithmetic count nor the float adapter proves real-time S3 execution or acceptable INT8 quality. Proceed with the faithful float comparison, then a bounded integer-cell experiment before committing to the complete firmware port.

This audit uses official MIT source revision `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73`. The repository now preserves the source and file hashes in [vendor provenance](../../esp32_denoiser/vendor/gtcrn/PROVENANCE.json). The [local adapter](../../esp32_denoiser/gtcrn_model.py) keeps the upstream network and raw spectral features, but uses our causal zero-padding/overlap-add convention instead of the upstream example's centered reflect padding. It loads **no pretrained weights**. Quantization, integer export, and a GTCRN C runtime are not implemented at this audit point. [Pinned upstream model](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/gtcrn.py), [license](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/LICENSE).

## What must actually run

At 16 kHz, FFT 512/hop 256 gives 257 complex bins and 62.5 frames/s. The network consumes raw magnitude, real, and imaginary spectra. The first 65 bins pass through unchanged; a fixed ERB mapping reduces the remaining 192 bins to 64 bands, giving 129 frequency positions. A three-bin frequency neighborhood produces nine input channels. There is no upstream per-frame RMS normalization, square-root feature compression, or identity-offset mask. Introducing any of these would be a separately trained architecture change.

| Part | Exact structure relevant to the port |
|---|---|
| Encoder stem | Frequency Conv `9→16`, kernel 5/stride 2, `129→65`; grouped Conv `16→16`, kernel 5/stride 2/groups 2, `65→33` |
| Six grouped temporal blocks | Three encoder and three decoder blocks. Process eight of sixteen channels through frequency-neighborhood expansion `8→24`, pointwise `24→16`, depthwise temporal/frequency 3×3, pointwise `16→8`, attention, then shuffle with the eight bypass channels |
| Temporal dilation | Encoder `1,2,5`; decoder `5,2,1`; all temporal history is past-only |
| Six temporal attention modules | Mean-square energy over 33 bins, GRU `8→16`, Linear `16→8`, sigmoid, channelwise multiplication |
| Two dual-path recurrent blocks | Each has grouped bidirectional GRUs along the 33 frequency positions, grouped unidirectional GRUs across time, two `16→16` projections, two LayerNorms, and residual additions |
| Four LayerNorms | Each normalizes the entire current `33×16=528`-element frame, with learned gamma and beta at every element and `eps=1e-8` |
| Decoder expansion | Grouped frequency transposed Conv `16→16`, `33→65`; transposed Conv `16→2`, `65→129`; final tanh; fixed inverse ERB; complex mask multiplication |

The frequency bidirectionality does not access future audio. Its forward and reverse frequency states restart every frame. Only the temporal GRUs, attention GRUs, and convolution caches persist across audio frames. All 22 BatchNorms can be folded into their preceding convolutions for evaluation; the four LayerNorms cannot. The 15 PReLUs have one learned scalar each. [Official network](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/gtcrn.py), [official streaming graph](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/stream/gtcrn_stream.py).

## Parameters and packed-byte estimate

Direct enumeration gives **23,669 learned parameters**, **24,576 frozen ERB parameters**, and **48,245 total parameters**. The published 23.7K/updated 48.2K discrepancy is therefore resolved: the latter counts two dense copies of a fixed transform. The authors also update the compute figure from 39.6 to 33.0 MMAC/s. [Upstream accounting note](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/README.md).

Folding BatchNorm removes 580 learned affine values from the deployment graph, leaving 23,089 learned values. This is an algebraic evaluation conversion, not pruning. The following is a **packing design subtotal**, not an exported artifact:

| Item after BN folding | Count | Proposed bytes |
|---|---:|---:|
| Conv/Linear/GRU matrix weights | 17,488 | 17,488 at INT8 |
| Matrix biases, including separate GRU input/recurrent biases | 1,362 | 5,448 at INT32 |
| LayerNorm gamma | 2,112 | 2,112 at INT8 |
| LayerNorm beta | 2,112 | 8,448 at INT32 |
| PReLU slopes | 15 | 15 at INT8, scales additional |
| Per-matrix-output weight exponents | 1,362 | 1,362 at INT8 |
| Exact sparse ERB, shared by both directions | 382 coefficients plus indices | 2,040 |
| FFT analysis/synthesis window, shared | 512 | 2,048 at FP32 |
| **Subtotal** | | **38, 961** |

LayerNorm/PReLU scales, graph descriptors, activation scales, nonlinear tables, alignment, readable SIMD guards, and a format header still need to be added. Budgeting **45–55 KB** for a complete first design, including the padded-GRU option below, is reasonable. It is an estimate requiring an actual serializer and parser; the 99,000-byte acceptance limit applies to the complete serialized object. Executable firmware and vendor FFT tables are separate flash quantities.

The fixed `64×192` ERB matrix has 382 strictly nonzero FP32 coefficients; its inverse is exactly the transpose. A CSR representation requires `65×2 + 382×1 + 382×4 = 2,040` bytes. Reuse the same entries for inverse scatter-accumulation; do not store a second matrix. Sixty-two coefficients are tiny nonzero values from the upstream epsilon. Keeping all 382 preserves the stored coefficients; different summation order still needs numerical parity checks. Dropping the tiny terms is a separate approximation. This is **constant representation compression**, not learned-weight pruning. A sparse DSP implementation performs 1,910 coefficient products/frame across three input and two output channels, instead of 61,440 products/frame in the dense mapping. [Pinned ERB implementation](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/gtcrn.py).

## Proposed integer contract

FFT, windowing, fixed ERB, and final complex multiplication can remain explicitly identified floating-point DSP outside the neural boundary, as in the current firmware. Every learned neural weight, materialized neural activation, and persisted neural state must be eight-bit. INT32 biases/dot accumulators and wider reduction arithmetic are allowed arithmetic intermediates; they must not conceal a floating-point GRU or a persistent INT16 state.

Use a new graph format and runtime dispatch. The existing convolution-only formats must reject GTCRN until all operators are supported. Start with per-output-channel weight quantization, explicit per-tensor activation scales, and exact rounding/saturation at every convolution, projection, residual addition, attention product, and recurrent update. A single shared hidden exponent is especially risky here: energy, gate logits, normalized features, recurrent states, and residual branches have different useful ranges.

| Operator | Required implementation and test |
|---|---|
| Folded Conv/Linear | INT8×INT8→INT32; folded biases on accumulator grids; checked accumulator bounds; explicit output requantization |
| PReLU | Quantized learned slope and integer negative branch; no assumption that the learned slope stays positive or below 1 |
| Frequency neighborhoods/shuffle | Indexing and fixed permutations, with no floating-point work and no full-utterance tensor allocation |
| Residual sums | Align branch scales, add in a wider accumulator, saturate exactly once at the declared output boundary |
| Transposed convolutions | Preserve group/channel layout, reversed kernels where converted, frequency stride/cropping, and original boundary behavior |
| Attention energy | Sum 33 squared INT8 values in INT32, apply the squared input scale and division by 33, then quantize onto a separately calibrated energy grid |
| LayerNorm | Integer mean/variance and reciprocal-square-root path, learned integer affine transform, explicit output grid; preserve normalization over 528 elements and epsilon handling |
| GRU | Custom quantization-aware cell matching PyTorch gate order, reset placement, biases, state update, and per-step rounding |
| Sigmoid/tanh | Deterministic lookup tables or integer approximation, with QAT using the identical input/output grids and clipping |

The upstream stream conversion already replaces temporal transposed convolutions with reversed-kernel convolutions. Its pointwise and grouped frequency transposed convolutions still require correct layouts; treating all transposed weights as ordinary output-major Conv weights silently changes the model. Test nonzero biases, asymmetric kernels, groups, edges, and dilation separately. [Streaming convolution implementation](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/stream/modules/convolution.py), [conversion rules](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/stream/modules/convert.py).

### GRU and eight-bit state are the first quality gate

The PyTorch candidate uses reset **after** the recurrent candidate projection:

```text
r = sigmoid(W_ir*x + b_ir + W_hr*h + b_hr)
z = sigmoid(W_iz*x + b_iz + W_hz*h + b_hz)
n = tanh(W_in*x + b_in + r * (W_hn*h + b_hn))
h_next = (1-z)*n + z*h
```

Moving `r` inside the matrix multiplication or combining the two candidate biases changes the cell. Input and recurrent dot products may have different scales; align them explicitly before gate evaluation. [PyTorch GRU definition](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html).

Use an explicitly quantized hidden-state grid at every step, not only at sequence outputs. A signed `scale=1/128` state is a simple first experiment, but its accumulated error and saturation need long-stream testing. For probability gates, compare the simple power-of-two eight-bit grid against an endpoint-inclusive `q/255` probability grid. The latter can be stored as unsigned eight-bit or signed storage with a zero-point offset; it needs exact integer divide/requantization support beyond the current power-of-two-only kernels.

A signed zero-point-zero sigmoid grid at 1/128 tops out at 127/128. Its linear retention coefficient alone has an approximately 128-frame e-folding time, about 2.0 seconds at this hop. Unsigned 1/256 tops out at 255/256, approximately 4.1 seconds. These are quantized-gate consequences, not bounds on the full nonlinear network's memory. An endpoint-inclusive gate can represent 1, but small updates may still round away in an INT8 state. Measure both state drift and stuck-state behavior. Do not silently switch to INT16 state when claiming the strict target.

There are 2,208 GRU hidden-element updates per frame, including frequency recurrence. A direct cell implementation needs 4,416 sigmoid and 2,208 tanh evaluations, plus 48 attention sigmoids and 258 mask tanhs: **6,930 nonlinear evaluations/frame**, about 433,125/s. Lookup and requantization costs matter even though the learned matrices are small.

### LayerNorm and raw input need separate calibration

Each LayerNorm reduces 528 current-frame values and applies 528 learned gamma/beta pairs. Its statistics remain input-dependent in evaluation, so BatchNorm folding cannot remove it. Integer variance can be zero even when float variance was merely small; define epsilon and zero-variance behavior before QAT. For INT8 inputs, sum-of-squares fits INT32, but the exact expression `N*sum(q*q)-sum(q)^2` can exceed signed 32-bit range; use a checked wider reduction or an equivalently validated scaled algorithm. An integer reciprocal-square-root implementation can use range normalization, a table, and refinement. Its approximation and rounding must be reflected during QAT. [PyTorch LayerNorm definition](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html).

The raw magnitude/real/imaginary input varies with acoustic gain and FFT-bin concentration. Calibrate on the actual training mixture and gain distribution, then evaluate quiet speech, loud tones, silence, clipping, and gain changes. A static feature-channel scaling that is compensated in the first layer can preserve the float function; dynamic RMS normalization cannot. Likewise, attention energy must use its own scale: squaring activations and forcing the result onto a generic hidden grid can destroy low-energy distinctions.

## Persistent state and ML-memory budget

The official streaming cache shapes give exact batch-one element counts:

| State | Shape/count | FP32 bytes today | Strict INT8 forecast |
|---|---|---:|---:|
| Convolution histories | `2×16×16×33` | 67,584 | 16,896 |
| Inter-frame GRUs | `2×33×16` | 4,224 | 1,056 |
| Attention GRUs | `6×16` | 384 | 96 |
| **Persisted neural state** | **18,048 elements** | **72,192** | **18,048** |

Frequency-GRU temporary states reset per frame and belong in scratch memory, not persistent history. Using INT16 only for the temporal/attention recurrent states would give 19,200 persisted bytes, just 1,152 bytes more than the strict design. This is a useful diagnostic for locating quantization loss, but fails the strict eight-bit-state requirement. [Official cache definitions](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/stream/gtcrn_stream.py).

The 18,048-byte figure excludes activations, skip tensors, normalization scratch, FFT scratch, I/O, task stacks, and any SRAM model copy. A bounded implementation can retain 3,152 bytes of eight-bit encoder skip outputs and reuse frame workspaces. As an initial allocation target, reserve 48 KiB for **all** neural history/scratch, 16 KiB for the audio frontend, 8 KiB stack, and 3 KiB for FFT allocation/I/O/descriptors: 75 KiB before an optional model copy. A 55,000-byte model copy would bring that proposed allocation to 131,800 bytes, below 204,800. These are design reservations, not measured C layouts or a whole-application SRAM proof. The existing [firmware memory audit](esp32_memory_audit.md) explains the additional RTOS, instruction RAM, I2S, radio, and heap measurements required.

## Matrix arithmetic, SIMD padding, and the S3 deadline

The upstream 33.0 MMAC/s figure uses its published complexity accounting and is not an MCU measurement. A separate enumeration of the converted **streaming** graph gives these nominal matrix products, counting full kernel footprints, including boundary positions, and excluding biases, normalization, gates, elementwise products, and DSP:

| Streamed part | Matrix products/frame |
|---|---:|
| All convolutions and frequency transposed convolutions | 229,328 |
| Six attention GRUs plus their projections | 7,680 |
| Two dual-path blocks, GRUs and projections | 122,496 |
| **Neural matrix subtotal** | **359,504 = 22.469M/s** |

The dense fixed ERB adds 61,440 products/frame; sparse exact ERB reduces that to 1,910. Our narrower matrix subtotal does not replace or claim to reproduce the upstream profiler total. Biases, elementwise operations, padded lanes, address work, nonlinear evaluation, and complete audio DSP remain outside it.

The [current S3 dot wrapper](../../firmware/esp32_denoiser/denoiser.c) takes a scalar path for lengths below 16. This is an implementation gap that can be addressed, rather than a reason to reject grouped GRUs. The 960 GRU matrix rows comprise 576 rows of length 8, 96 of length 4, and 288 of length 16.

Two bounded SIMD options are worth measuring:

1. **Store each GRU row padded/aligned to 16.** GRU weights grow from 9,600 to 15,360 bytes, an extra 5,760 bytes. Pad the input to 16 with zeroes and provide the kernel's readable input guards. This has simple alignment and parser invariants and remains comfortably within the projected payload budget.
2. **Keep compact rows; pad only input.** Reading subsequent row bytes is mathematically harmless where the padded input is zero, provided every read stays inside validated storage. The existing unaligned wrapper requires `full+32` readable weight bytes and aligned guarded input. A mere 16-byte final weight guard is insufficient for every current path. Use a sufficient tail guard or copy the final rows into aligned scratch; test every row offset and tail. This avoids the 5,760-byte row padding but adds address/guard handling.

Padding all GRU dot lengths to 16 increases executed lane products from 95,616 to 211,968/frame. The complete neural matrix lane count becomes **29.741M/s**, an additional 7.272M/s. This is arithmetic overhead, not a predicted slowdown: a vector instruction can still beat several scalar products. Splitting or concatenating input/recurrent projections must preserve their separate scales and reset-after candidate semantics. Known-answer tests must include lengths 4/8, negative extremes, unaligned offsets, final rows, guard sizes, and exact agreement with scalar INT32 dots. Actual S3 timing decides between layouts.

A 240 MHz core has **3,840,000 cycles per 16 ms hop**. Using the published 33M count only as a planning denominator permits about 7.27 cycles per reported MAC inclusive of everything. Reserving 4 ms for DSP/I/O/scheduling leaves 2.88 million cycles, about 5.45 cycles per reported MAC. Neither ratio establishes feasibility. Frequency recurrence is serial across 33 positions, many dot products are short, and roughly 433K nonlinear evaluations/s add work. Measure matrix kernels, complete GRU cells, LayerNorm, full neural frames, and full audio hops separately; test flash-mapped versus internal-SRAM weights. Host timings and x86 RTF cannot establish S3 speed.

## Go/no-go conditions

The following are proposed project decision gates, not established results or guarantees:

1. **Float value:** train the faithful adapter under the same declared data/selection protocol as the strongest custom control. Advance if it improves development quality or generalization enough to justify the port. Published PESQ/SI-SNR results are supporting motivation, not our result or an automatic comparison with our custom holdout. Keep official test data sealed.
2. **Cell fidelity:** implement a scalar integer reference for one grouped GRU, attention module, and LayerNorm before the full graph. Match float cell semantics with quantization disabled, then match exact integer rounding in QAT and reference execution. Exercise silence, constants, impulses, clipping, alternating bins, scale extremes, and long streams with resets.
3. **Strict quantization:** all learned weights, materialized neural activations, and persisted neural states must be eight-bit; no float GRU/LN fallback. A proposed quality gate is at most 0.3 dB equal-utterance SI-SDRi loss from the matched float checkpoint after QAT, with PESQ/STOI and listening checks that rule out a misleading SI-only gain. The integer candidate must still improve the relevant deployed control; a small quantization loss alone is insufficient.
4. **Artifact and memory:** a validated complete binary must be at most 99,000 bytes; measured/validated ML allocations must stay below 200 KiB without PSRAM. Count constants, tables, alignment, guards, scratch, stack, and any weight copy explicitly. Preserve source/model hashes and format version.
5. **Runtime equality:** QAT/reference/C must share the same per-step integer state evolution. Test arbitrary chunk boundaries, reset, past-only causality, STFT start/flush, DC/Nyquist, and complete PCM16 frontend behavior. Compare float and actual integer waveforms on the identical development manifest.
6. **Board deadline:** on the intended S3 configuration, require no missed 16 ms hops during sustained representative tests, and target p99 complete-hop processing at or below 12 ms to leave integration margin. Record maxima, misses, heap and stack high-water marks, clock, compiler, weight placement, and I/O load. A compiled firmware image alone does not satisfy this gate.

If strict INT8 recurrence or LayerNorm loses the float advantage, retain GTCRN as a float reference or training teacher and continue with a simpler deployable student. If short-dot timing is the blocker, first test the padded/aligned layout; do not assume pruning weights will solve an operator-overhead problem. This is a faithful implementation and systems/quantization investigation based on established architecture, without a novelty claim.

## Local recurrent primitive completed

[experimental_gru.py](../../esp32_denoiser/experimental_gru.py) now provides an isolated actual-integer NumPy cell and a differentiable PyTorch fake-quantization cell. It does not wrap or modify GTCRN. The prototype uses per-row INT8 weights, INT32 biases/dots, checked Q12 accumulator alignment, Q0.7 INT8 state/candidates, signed eight-bit probability storage with `(code+128)/255`, and sigmoid/tanh lookup tables. Exporting a parameter snapshot checks worst-case dot and aligned gate-accumulator bounds. No C implementation or full-model serializer exists for this primitive.

The 22 [primitive tests](../../tests/test_experimental_gru.py) pass after independent review. They verify float control against PyTorch in both frequency directions; exact integer/fake-quantized sequence and chunk-state agreement for `8→4`, `8→8`, `8→16`, and `32→16`; the placement of the reset gate around the recurrent candidate bias; endpoint probability behavior; a 2,048-frame deterministic stream; finite nonzero QAT gradients under CPU BF16 autocast; exact snapshot parity after a parameter update; and rejection of unsupported or overflowing inputs/parameters. The additional review tests cover independent scalar lookup-table values, preserved configuration when cloning a QAT cell, rejection of a modified QAT table during integer snapshot creation, wide reset products, and overflow in the sum of individually valid aligned affine paths.

Running `.venv-esp32/bin/python -m esp32_denoiser.experimental_gru --frames 10000 --seed 816` exercises 10,000 steps per randomly initialized cell, with an extended silent middle section. All three retain INT8 state with zero saturated state values and zero clipped logits for this probe. Mean absolute hidden-state error against each unquantized PyTorch control is 0.012507 (`8→4`), 0.010266 (`8→8`), and 0.010605 (`8→16`); maximum errors are 0.044388, 0.037881, and 0.041117 respectively. These are **untrained recurrent-state diagnostics**, not speech-quality metrics or a guarantee on trained GTCRN state distributions.

The QAT cell uses float64 to reproduce integer intermediates precisely and a Python sequence loop. It is intended for numerical contract tests and a future QAT integration reference, not immediate high-throughput full-GTCRN training. A production training implementation should cache/freeze appropriate quantization metadata and batch affine work while preserving every recurrent quantization boundary. Its C port, LayerNorm, attention integration, full-model QAT, actual PCM16 evaluation, and S3 timing remain outstanding.

### Independent primitive audit, 13 September 2026

The cell preserves PyTorch's gate ordering and reset-after recurrent candidate bias. Its separately computed input/recurrent affine paths are aligned before their sums, and its candidate uses the reset gate after the recurrent projection. The unquantized control is checked against the independent PyTorch GRU implementation, including nonzero initial state and reverse frequency direction. [PyTorch's explicit equations and reset-placement note](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html).

The integer snapshot bounds each row by `abs(bias) + 128*sum(abs(weight))`, covering every partial dot sum as well as the final result. It checks the aligned input-plus-recurrent bound against INT32. Export-time alignment uses INT64; permitted shifts cannot overflow that width before validation. The reset product can reach `255*(2^31-1)`, approximately 5.48×10^11, so its INT64 intermediate is necessary. Tests now observe that product's result before logit clipping for positive and negative candidate accumulators near the INT32 limit. Hidden state returned by every integer step is actually `numpy.int8`; affine and gate-product wider values are temporary arithmetic intermediates.

The gate storage can represent probability endpoints, but the configured **logit range** determines whether its lookup table reaches them:

| Logit exponent | Reachable probability codes, divided by 255 | Reaches both 0 and 1? |
|---|---:|---|
| −6 | 30…224 | No |
| −5 | 5…250 | No |
| −4, −3, −2 | 0…255 | Yes |

At the maximum reachable update probability, grids −6 and −5 have linear retention-coefficient e-folding times of approximately 7.72 and 50.50 steps respectively: 0.123 and 0.808 seconds for temporal cells with a 16 ms step. These are not bounds on the complete nonlinear, quantized GRU's memory; state rounding and feedback can create fixed points. The default −4 grid reaches both endpoints. `storage_stats()` now exposes the reachable probability-code range explicitly. Higher precision with insufficient logit range should not be mistaken for improved recurrent fidelity.

The exact NumPy-versus-QAT tests compare different arithmetic implementations, so they are useful checks of rounding, broadcasting, recurrence, and chunk state. They share exponent selection and the original LUT generator and therefore were not wholly independent correctness proofs. The added exhaustive scalar-`math` LUT checks cover all 256 logits on all five allowed grids without calling that generator. The existing hand-calculated reset/bias case supplies a separate semantic check. None of these tests establishes C parity, trained GTCRN accuracy, acoustic quality, or S3 speed.

Two bounded defects were corrected: constructing a QAT cell from another QAT cell now inherits its nondefault grids, and an integer snapshot of the same configured QAT cell rejects modified lookup tables instead of silently rebuilding different ones. Full-model checkpoint integration must still save and restore the configuration explicitly: a bare PyTorch `state_dict` does not encode this module's dataclass exponents. QAT can also move parameters outside the exportable accumulator range; validate snapshots during future training rather than assuming float64 QAT arithmetic enforces INT32 bounds on every optimizer step.

### Independent LayerNorm primitive audit, 13 September 2026

The isolated [LayerNorm prototype](../../esp32_denoiser/experimental_layer_norm.py) now passes **18 tests** in [its focused suite](../../tests/test_experimental_layer_norm.py). It follows the vendored GTCRN graph: each of four operators normalizes the last `(33,16)` dimensions of `(batch,time,frequency,channel)` data, independently for every current frame. It uses population variance, with division by 528 rather than 527, and per-element gamma and beta. These match [PyTorch's LayerNorm contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html).

For input codes `q`, input scale `s`, and `N=528`, the implementation keeps `A=sum(q)`, `B=sum(q*q)`, and `D=N*B-A*A` exact. It does not round a mean back onto the input grid. The normalized quantity is `(N*q-A)/sqrt(D + epsilon*N*N/s²)`. The default squared-denominator grid uses 24 fractional bits; epsilon is encoded once during parameter preparation, then integer inference reads only that code. At input exponent −4 and requested epsilon 1e−8, the code is 11,973,682 and the effective epsilon is approximately 1.0000000312e−8. At the deliberately coarse exponent +8, the minimum nonzero code is 1, giving effective epsilon approximately 1.401174386e−8. The latter difference is exposed in the report, rather than claiming exact preservation of an arbitrarily small epsilon. Constant quantized frames return quantized beta, including correct saturation.

The exact maximum `D` is `floor(N²/4)*255² = 4,531,982,400`, attained by evenly split −128 and +127 inputs. It exceeds even unsigned INT32; the reduction must retain INT64. Sums and sums of squares themselves fit INT32. Constructor checks bound the encoded variance, integer-square-root search, affine products, beta addition, absolute value, and rounding addition within signed INT64. The square-root ceiling is deliberately below the full INT64 range so the independent binary-search implementation can square every trial root safely. New tests exercise odd `N=527`, actual `N=528`, and maximum supported `N=1024` at the code rails. A future C implementation must use checked multiplication or unsigned magnitude handling for negative shifted quantities: Python's defined negative left shift must not be translated into undefined signed C left shifts.

The equality claims have different scopes. Python integer inference and the Torch integer helper implement their square roots independently, but share parameter preparation and the algebraic contract. Fake quantization calls the Torch integer helper directly for its forward values, so fake-versus-Torch-integer equality is **by construction**. Its gradients use a float LayerNorm surrogate and straight-through output rounding/saturation. The added independent oracle instead calls ordinary PyTorch float64 LayerNorm with the requested epsilon, decoded quantized input, and decoded gamma/beta. A finite adversarial corpus with nonintegral means, single-code perturbations, signed affine values, rare outliers, multiple frames, and three input/output grids has exactly matching rounded output codes. This does not prove all possible rounding thresholds: integer floor-square-root and encoded epsilon are approximations to float normalization.

These diagnostics also do not include input/affine quantization error against an original trained float model. For example, an alternating ±0.001 synthetic frame, unit gamma, zero beta, and epsilon 1e−8 has float mean absolute output about 0.995, but the prototype's uncalibrated default input grid of 1/16 turns every input into zero and returns zero. A normalized 528-element frame can have a unit-gamma outlier approaching `sqrt(527)≈22.956`; the default output grid tops out at 7.9375 and clips such an outlier. An output exponent of −2 covers this unit-gamma range with less precision, but learned gamma/beta can change the range. Full-model work needs separately calibrated input/output grids for all four LayerNorms, actual activation-collapse/saturation counts, original-float-versus-QAT comparisons, and audio evaluation. The numerical helper supplies no trained quality result. Two small snapshot defects were fixed during review: BF16 affine tensors now convert safely before NumPy preparation, and subnormal finite gamma values no longer underflow automatic grid selection into `log2(0)`.

All materialized neural input/output codes and gamma values are INT8; beta is INT32 bias storage. Four `(33,16)` gamma/beta arrays occupy **10,560 bytes** in total, excluding descriptors, exponents, encoded epsilon, alignment, and any runtime model copy. One separate input/output pair occupies **1,056 bytes**, and LayerNorm adds **zero persistent state**. A two-pass native implementation can compute the statistics once and reuse scalar arithmetic while generating outputs, although residual scheduling can require both buffers to remain live. Torch's vector INT64 temporary tensors and Python allocations are not native SRAM measurements. Bare QAT `state_dict` data also omits the `options` dictionary; future checkpoint/export integration must explicitly preserve grids, epsilon encoding, and fractional precision.

Do not port the per-output integer division literally into the S3 hot path. Four LayerNorms at 62.5 frames/s would perform **132,000 wide divisions/s**, whereas their shared denominators change only **250 times/s**. Prepare a reciprocal/multiplier and shift once per operator per frame, then use bounded multiply/shift and, where required, quotient correction to retain the declared rounding rule. Its product widths and exactness need independent tests. If an approximate reciprocal changes output codes, its contract and QAT forward must change together; the current parity result cannot be carried over automatically. In particular, splitting beta addition after rounding is not automatically equivalent at signed half-way ties. This is an implementation direction, not a measured speed claim. The full native operator, GPU-capable QAT implementation, serializer, complete GTCRN integration, and physical S3 timings remain outstanding.

### Isolated native primitives, 13 September 2026

The subsequent [portable C GRU/LayerNorm port](../../firmware/experimental_int8/README.md) completes the isolated native-operator step above. Its host binding executes the C code and matches the audited NumPy references exactly across 21 new tests; 61 combined primitive tests pass. LayerNorm now uses one reciprocal division per frame with an exact quotient correction at each output. The isolated ESP32-S3 object compiles to 3,975 bytes of text with zero data/BSS, excluding linked helpers and final firmware layout. The new primitives remain separate from all current firmware dispatch/model formats. Full GTCRN integration, trained integer audio-quality validation, the model serializer and physical-board timing remain outstanding. Source hashes and compiler evidence are recorded in `output/esp32/primitive_native_build/report.json`.
