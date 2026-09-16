# Portable integer GTCRN operator building blocks

`firmware/experimental_int8/operators.c` and `operators.h` implement the
audited NumPy contracts from `experimental_gtcrn_ops.py`. The host binding
is `experimental_ops_native.py`. Full graph dispatch and packed-model loading
are separate integration work; this operator check does not establish trained
GTCRN integer quality or board timing.

The C code provides Linear, grouped Conv2d and ConvTranspose2d, the converted
causal convolution wrapper, PReLU, explicit grid conversion, single-rounding
residual addition, attention mean-square energy, endpoint-inclusive attention
products, lookup tables, subband expansion and channel shuffle. Convolution
tensors are contiguous `[C,T,F]`, without a batch axis. Linear accepts
`[rows,features]`. Kernels are canonical `[output,input/group,kT,kF]` and
receive no implicit reversal. Eval BatchNorm folding occurs in audited host
parameter preparation; no normalization float math occurs inside these C
operators.

Learned weights, activation tensors and convolution history are INT8. Biases
and dots are INT32. Initializers reject any row whose absolute bias plus
`128*sum(abs(weights))` exceeds INT32_MAX, so every partial accumulation fits.
Per-output weight exponents range from -20 to 4; activation exponents range
from -16 to 8. Requantization shifts are bounded by [-44,28], placing even
the largest valid INT32 left shift below 2^59. Negative values use unsigned
magnitudes and round ties away from zero; there is no signed-shift undefined
behavior or intermediate branch saturation.

Residual branches align to their finest grid before one final rounding.
Attention energy sums at most 1,024 squared INT8 codes in INT32, with a scaled
numerator at most 2^56 and denominator at most 2^50. Products interpret the
gate as `(signed_code+128)/255`, preserving both zero and one. They round the
combined rational expression once. Each energy/product call prepares one
wide reciprocal and then uses exact high-product/remainder correction for
each output. Affine requantization uses shifts rather than wide divisions.
S3 disassembly places wide divisions only in the affine shape query and
once per attention operation; none occur per affine output or attention
frequency product. The implementation is a scalar baseline, with no SIMD
dot-product integration.

All buffers remain caller-owned; the C functions allocate no heap and carry
no implicit neural state. The streaming wrapper stores the actual input-grid
codes, including dilated intermediate frames, and can update history in
place. Its explicit scratch is exactly `[C,h+1,F']` bytes, where
`h=(kT-1)*dT` and `F'=F*frequency_stride+left_padding+right_padding` after the
validated frequency transform. For a 16-channel, h=10, F=33 block this is
5,808 bytes ordinarily, or 6,160 bytes with a 35-bin transformed frequency
axis. The full graph must account for its live tensors, all histories and
handles separately.

Verification on 13 September 2026:

- 32 native tests pass; the combined NumPy-operator and native-primitive suite
  passes 80 tests. Raw INT32 affine accumulators also match an independent
  float64 PyTorch operator oracle, including groups, transpose output padding,
  asymmetric channels and signed BatchNorm scales.
- Tests cover all 256 signed input codes, nondefault and extreme grids,
  near-INT32 bounds, signed/wide PReLU products, exact attention endpoints,
  streaming upsampling/cropping, unchanged kernel orientation and in-place
  recurrent convolution history.
- A standalone AddressSanitizer/UndefinedBehaviorSanitizer harness passes
  all 625 activation-grid pairs, including 1,024-bin energy and wide products.
  Apple Clang, GCC15 and Xtensa GCC14.2.0 accept strict C99 warnings.
- ESP32-S3 object compilation with `-O2 -mlongcalls` produces 6,774 bytes of
  text and zero static data/BSS. The opaque affine handle is 96 bytes on S3
  (128 on this 64-bit host). Compiler static stack estimates are 256 bytes
  for affine execution, 160 for stream execution, 80 for energy and 96 for
  attention products, per function. These exclude nested calls and library
  helpers and are not measured stack high-water marks.

Commands, source/object hashes, disassembly, stack reports and sanitizer
harness are retained in `output/esp32/operator_native_build/`. Object size
excludes linked helper implementations and final linker placement. No
physical ESP32-S3 execution or latency claim is made by this result.
