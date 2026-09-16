# Exact shared GTCRN ERB storage

The pinned upstream revision `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73` stores a fixed 64×192 float32 matrix and its transpose as two dense parameter tensors: 98,304 bytes total. `SparseGTCRNERB.from_torch(core.erb)` verifies that the second matrix is the exact transpose, then stores one canonical CSR table:

| Component | Bytes |
|---|---:|
| 65 little-endian uint16 row offsets | 130 |
| 382 uint8 column indices | 382 |
| 382 little-endian float32 coefficients | 1,528 |
| Total | **2,040** |

All 382 nonzero coefficients are retained bit-for-bit, including the 62 coefficients smaller than 1e-10 introduced by upstream epsilon terms. No thresholding, learned pruning, renormalization or filter redesign is performed. The first 65 bins are copied unchanged. Forward maps the remaining 192 bins to 64; synthesis traverses the same rows to apply the transpose. Synthesis is not a pseudoinverse and does not promise perfect reconstruction.

The pinned table SHA256 is `112fe941ff95c9a8a28ab6a9fdd571bc6e354b89dfa17994f25bd1e14574a207`. Portable C code in `firmware/experimental_dsp/gtcrn_erb.c` uses the table directly, without heap allocation, duplicate transpose storage or persistent audio state. The 2,040-byte count excludes the handle, executable code and caller-owned input/output arrays. Three-channel forward plus two-channel inverse require 1,910 scalar multiply-add terms per frame, versus 61,440 dense terms; these are operation counts, not MCU timings.

21 focused tests cover the pinned bytes, all basis vectors, signed random/asymmetric/edge/silent inputs, both channel layouts, empty rows, unaligned payloads, stale-handle invalidation, malformed sizes/offsets/indices/coefficients, and float overflow. Ordered NumPy and host C agree exactly when the host wrapper disables FMA contraction. Dense PyTorch can use a different reduction order; tests bound the difference using the sum of absolute products and float32 precision. One initial random probe had maximum absolute difference 2.384185791015625e-7. Full-graph quantization and board-specific FFT/compiler behavior still require separate evaluation.

```sh
.venv-esp32/bin/python -m pytest tests/test_gtcrn_erb.py -q
```
