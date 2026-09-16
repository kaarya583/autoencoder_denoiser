# GTCRN integer operator boundaries

The isolated [NumPy operator module](../../esp32_denoiser/experimental_gtcrn_ops.py)
now supplies Conv2d, grouped ConvTranspose2d, Linear, eval BatchNorm folding,
PReLU, explicit activation requantization, residual addition, attention energy
and product, subband extraction, channel shuffle, and causal convolution
history. Its [27 tests](../../tests/test_experimental_gtcrn_ops.py) pass; the
combined operator, native primitive, GRU and LayerNorm suites pass 88 tests.
These are operator tests, not a complete GTCRN export or audio-quality result.

Every neural input/output/history array is INT8. Affine bias and dot products
are INT32. Per-output weight exponents cover the quantized output row,
including the correct group mapping for transposed convolutions. The checked
bound is `abs(bias)+128*sum(abs(weight)) <= INT32_MAX`, which also bounds each
partial accumulation. Requantization uses bounded INT64 and signed
nearest-rounding with ties away from zero. Activation exponents are explicit
per-tensor integers in −16…8; weight exponents are −20…4. Parameter preparation
rejects unsupported magnitudes or overflow rather than wrapping.

`folded_parameters` requires matching eval-mode affine/BatchNorm modules and
running statistics. It folds gamma, beta, running mean, running variance and
epsilon into float64 preparation weights/bias, then quantization follows.
Negative BN gamma is supported. Original parameters remain unchanged.
Canonical transposed weights have shape `[output,input/group,kT,kF]`; the
integer operation scatters onto the exact stride, padding, dilation and
`output_padding` positions. Grouped, asymmetric, dilated and nonzero-bias cases are
checked against independent ordinary PyTorch operators.

`IntegerStreamConv` handles the vendored `StreamConv2d` and
`StreamConvTranspose2d` wrappers. The latter contains an ordinary Conv2d whose
time **and frequency** kernels were already reversed by the upstream
converter. The adapter does not reverse them again. It reproduces the
wrapper's zero-insertion, frequency padding/cropping, and fixed-length causal
cache. The actual supported cache shapes are checked against the vendored
wrapper with nontrivial INT8 inputs and asymmetric weights.

## Graph wiring and shapes

The full graph wrapper must associate grids with edges, rather than assuming
one shared hidden exponent. Suggested names are `<operator-path>.input` and
`.output`, where an affine output is the result **after its folded BN**.
For depthwise streaming operators, use the wrapper's `.depth_conv` path rather
than the internal `.Conv2d`/`.ConvTranspose2d` child. These names are suggestions
for the graph's calibration schema, not a serialized format.

| Region | Operators and boundary shapes |
|---|---|
| Input | Checkpoint-specific float spectral frontend and sparse ERB produce `[B,3,T,129]`; INT8 SFE produces `[B,9,T,129]`. Preserve the checkpoint's raw-versus-RMS feature policy. |
| Encoder first two blocks | `encoder.en_convs.{0,1}.conv + .bn`, then `.act` PReLU. Frequency widths 129→65→33; output channels 16. |
| Six GTConv blocks | `encoder.en_convs.{2,3,4}` and `decoder.de_convs.{0,1,2}`. Split 16 channels into 8/8. SFE maps the first half 8→24. `point_conv1 + point_bn1`: 24→16, then `point_act`; depthwise `depth_conv + depth_bn`, then `depth_act`; `point_conv2 + point_bn2`: 16→8; then TRA and shuffle with the untouched half. |
| Attention | `.tra.energy`: mean of 33 squared frequency values, `[B,8,T]`; transpose to `[B,T,8]` for GRU 8→16; `.tra.att_fc`: 16→8; sigmoid logits; reshape probability gate to `[B,8,T,1]`; multiply the original attention features. |
| Two dual-path blocks | `dpgrnn{1,2}` operate on `[B,T,33,16]`. Grouped bidirectional GRUs run across frequency and reset each frame; `intra_fc` 16→16, then 528-element LayerNorm and residual. Grouped temporal GRUs use `[B*33,T,16]` and persisted state; `inter_fc` 16→16, reshape back, LayerNorm and residual. |
| Decoder skips | Before decoder blocks 0…4, add encoder skip indices 4,3,2,1,0 respectively, with explicit branch-grid alignment. |
| Decoder last two blocks | `decoder.de_convs.{3,4}.conv + .bn` are true grouped/non-grouped frequency transposes. Widths 33→65→129; channels 16→16→2. Block 3 uses scalar PReLU; block 4 uses tanh. |
| Output | Final bounded INT8 mask codes feed the declared dequantization and fixed inverse-ERB/complex-multiplication DSP. This module does not implement that frontend/backend or change output gain. |

An inventory test instantiates all 32 learned affine snapshots: 22 convolution
operators with BN folded, plus six attention and four dual-path Linear
operators. It also covers all 15 learned scalar PReLUs. Depthwise temporal
histories are 2,4,10 frames in the encoder and 10,4,2 in the decoder. At 16
channels and 33 bins they require 16,896 INT8 bytes. That excludes GRU state,
skip buffers, other workspaces and DSP; it is not a complete RAM bound.

## Exact elementwise semantics and calibration obligations

- PReLU stores its learned scalar/channel slopes as INT8 with an explicit
  slope exponent. Negative slopes and slopes above one are supported.
- `residual_add` first aligns both INT8 branches onto an exact finer integer
  grid, adds them in a wider intermediate and rounds once at output. It does
  not clip each branch before cancellation or round two half-code terms
  separately.
- `attention_energy` computes the frequency sum of squares in INT32, applies
  the squared input scale, divides by the frequency count and rounds once.
  Its output needs a separately calibrated nonnegative energy grid.
- `attention_product` interprets signed probability code q as `(q+128)/255`,
  including both endpoints. It combines gate multiplication, scale alignment
  and division into one output rounding. This probability representation is
  not a zero-point-zero `ActivationGrid`.
- SFE and shuffle only move INT8 values. Both shuffle halves must share a
  grid. Use `requantize_activation` explicitly or choose the attention-product
  output grid to match the untouched branch. Cache grids must remain fixed
  throughout an utterance.

`ActivationObserver` records all observed minima/maxima, value count, tensor
shapes, selected covering grid, declared training-manifest SHA256 and
checkpoint SHA256. It rejects a non-training split and nonfinite inputs, but
does not itself read manifests or establish membership; the full calibration
driver must supply audited training activations and exclude validation/test
examples. The observer itself reads no dataset files and retains no waveforms.
The chosen grid covers observed extrema, with
no implicit percentile clipping. This can sacrifice resolution when rare
outliers dominate; saturation/collapse counts and matched audio evaluation
must guide any alternative calibration policy.

The GRU primitive's allowed input exponents are −12…0, narrower than the
generic activation range. Its gate-logit range is −6…−2; state remains 1/128.
The full wrapper must respect these contracts and record any clipping rather
than silently passing incompatible grids. Each of the four LayerNorms needs
its own calibrated input/output grids. Normalization output and gate inputs
must not automatically inherit a convolution grid.

The operator metadata reports actual parameter-array bytes, weight layout,
strides, padding, groups, exponents, BN-fold status and accumulator bounds.
It excludes binary headers/alignment/guards and native scratch. The current
operators are a NumPy reference with independently tested layouts and
rounding; they do not add GPU QAT, a full graph, a C implementation, a complete
deployable blob or an ESP32-S3 timing claim.
