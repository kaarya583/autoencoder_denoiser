# Isolated INT8 GRU and LayerNorm primitives

These portable C99 primitives implement the arithmetic contracts in
`esp32_denoiser/experimental_gru.py` and `experimental_layer_norm.py`. They are
**not connected to the current firmware dispatch or any model format**, and
do not provide a complete GTCRN runtime. The Python host binding is
`esp32_denoiser/experimental_native.py`.

GRU input, matrices, gate/candidate codes, output, and caller-owned recurrent
state are INT8. Biases/dot accumulators use INT32; shifts and reset products
use bounded INT64 intermediates. The gate order and reset-after candidate
bias follow PyTorch. The supplied sigmoid table stores probabilities as
`(signed_code+128)/255`; candidate/state codes use scale 1/128. The initializer
checks the dimensions, lengths, exponents, and worst-case raw/aligned
accumulator sums. It applies the supplied tables exactly; canonical LUT
generation remains part of audited host parameter preparation.

LayerNorm accepts INT8 frames/gamma and INT32 beta. Its exact variance is
`N*sum(q*q)-sum(q)^2`, with no rounded integer mean. Epsilon must already be
encoded on the declared squared-denominator grid. It uses a restoring integer
square root and rounds the combined affine result once, including beta.
All arithmetic bounds are validated at initialization.

The LayerNorm denominator is shared by a whole frame. The C implementation
computes `floor((2^64-1)/denominator)` once, uses a portable high-half 64×64
product from 32-bit limbs for each output, then corrects the quotient by at
most one. This reproduces the Python round-away result exactly while avoiding
per-output division. It uses no `__int128`, float math, heap allocation,
normalized floating activations, or persistent wider state.

Callers allocate naturally aligned handles, parameter buffers and outputs.
Parameters must remain unchanged and alive until the handle is no longer
used. A GRU step needs `hidden_size` INT8 scratch bytes, disjoint from its
other buffers; the staged update permits output to alias old state safely.
LayerNorm uses two passes and scalar scratch, and permits in-place output.
Optional diagnostic buffers have separate disjointness requirements in the
header. C cannot establish actual allocation sizes from raw pointers: callers
must honor the validated dimensions and documented buffer contracts.

Verification on 13 September 2026:

- 21 native tests pass, 61 combined native/GRU/LayerNorm tests pass.
- Exact NumPy/C checks cover all five logit grids, per-row weight scales,
  signed rounding, near-INT32 candidate accumulators with wider reset
  products, constant/rail LayerNorm inputs, nondefault variance precision,
  beta at INT32 extremes, in-place operation and 2,048-frame recurrent state.
- Independent Python arithmetic checks cover reciprocal correction and
  integer-square-root boundaries. A second review additionally checked 40,044
  signed division cases and 10,006 square roots with exact agreement.
- A standalone native harness passes AddressSanitizer and
  UndefinedBehaviorSanitizer. Clang and GCC15 accept strict C99 warnings.
- The isolated object cross-compiles for ESP32-S3 with
  `xtensa-esp-elf-gcc 14.2.0 (esp-14.2.0_20241119)`, using `-O2 -mlongcalls`:
  **3,975 bytes of object text, zero data/BSS**. This excludes linked library
  helpers, final placement/alignment and the rest of firmware. No board ran it.

The S3 compiler gives handle sizes of 52 bytes for GRU and 32 bytes for
LayerNorm. An 8→16 GRU has 2,144 bytes of supplied arrays/tables, 16 bytes of
persistent state per stream and 16 explicit scratch bytes. One 33×16 LayerNorm
has 2,640 affine-array bytes and a separate INT8 input/output pair totals 1,056
bytes; it adds zero persistent state. Four such affine sets total 10,560
bytes. Compiler stack reports are 144 bytes for the GRU step and 80 for the
LayerNorm frame **per function**, not whole call-chain bounds or measured
stack high-water marks. Model descriptors, scheduling, I/O, other operators,
library helpers and optional SRAM parameter copies still require accounting.

Build evidence and source hashes are under
`output/esp32/primitive_native_build/`. These results establish isolated
native integer arithmetic fidelity. They do not establish trained GTCRN
quality, a complete serialized model, full GPU QAT, whole-model memory,
ESP32-S3 latency or faster-than-real-time operation.
