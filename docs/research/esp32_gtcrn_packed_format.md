# GTCRN packed integer model and native loading contract

`esp32_denoiser.gtcrn_integer_export` implements a real, loadable model binary for the pinned GTCRN topology. It stores every integer neural parameter and lookup table, all activation grids, and the window and sparse ERB constants used by the numerical waveform reference. Loading requires no floating model or checkpoint. This establishes serialization and numerical reproducibility; trained audio quality and ESP32 throughput require separate measurements.

The lower-level Python export accepts a prepared numerical snapshot. The real-data CLI has one route: `from_checkpoint_training` reconstructs the checkpoint's audited paired/synthetic training recipe, including its saved clean-identity probability, draws32 deterministic training crops by default with seed483, and checks separation from development audio. Development clips do not fit quantization grids.

```sh
python -m esp32_denoiser.gtcrn_integer_export \
  --checkpoint /content/esp32_runs/gtcrn_probe_inputs/student.pt \
  --manifest /content/extra_audio/development/mixtures.jsonl \
  --output /content/esp32_runs/gtcrn_probe_inputs/gtcrn_int8.bin \
  --calibration-crops 32 --calibration-seed 483 --threads 2
```

The output consists of the binary, a `.calibration.json` audit, and a `.json` size/hash summary. The audit records source implementation hashes, checkpoint/state hashes, calibration source manifests, selected source/crop hashes, numerical versions and observed grids. Its canonical JSON SHA256 is embedded in the binary. `load_gtcrn_integer(path, calibration=audit)` checks this association. An integrity digest detects changed bytes; it does not authenticate a dataset-membership or provenance claim.

## Version1 layout

All multibyte fields are little-endian. The fixed topology ID is1; a changed topology or arithmetic contract needs a new format/topology version. `SPECS` and `GRID_NAMES` are canonical, in execution order, and generate the native `layout.h` identifiers.

| Region | Offset | Bytes | Contents |
|---|---:|---:|---|
| Header |0|192|Magic `GTI8PK01`, version, sizes, topology, DSP normalization flag, RMS floor, base GRU exponents, four digests and reserved zeros|
| Grids |192|182|Signed INT8 power-of-two exponents, range−16..8|
| Alignment |374|10|Zeros|
| Operator descriptors |384|6,240|78 records ×80 bytes|
| Aligned array region |6,624|Variable|Integer arrays/LUTs and external DSP constants; maximum complete model99,000 bytes|

The header's Python struct is `<8sHHIIIIHHd4b32s32s32s32s20s`. Its four32-byte digests are complete-file integrity, source checkpoint, source state/config, and canonical calibration JSON. Integrity hashes the complete binary after replacing bytes44..75 with zeros. A zero checkpoint digest denotes a numerical snapshot without a checkpoint; the audited CLI supplies a real checkpoint digest. State and calibration digests cannot be zero.

The record struct is `<HBBbbBb8H10I16s`: ordinal, kind, flags, input exponent, output exponent, subtype, signed auxiliary exponent, eight dimensions, ten absolute array offsets, and16 bytes of type-specific metadata. Unused fields are zero. The parser checks every descriptor against its compiled topology, including direction and group arrangement. It rejects unaligned/out-of-range references, incompatible aliases, overlap, noncanonical first-reference order, extra data and nonzero padding.

The records comprise32 affine operators (including six streaming depthwise convolutions),15 PReLUs,18 GRU directions, four LayerNorms, seven activation LUTs, one ERB table and one window. New arrays start on16-byte boundaries. Byte-identical arrays with the same scalar type share storage; shapes may differ only when the extent/type is identical. No lossy compression is applied by serialization.

## Numerical interpretation

- Affine arrays contain output-major INT8 weights, INT32 biases and one INT8 exponent per output. Dimensions encode input/output channels, groups, time/frequency kernel, frequency stride/padding and temporal dilation. Temporal stride is1, temporal padding0, frequency dilation1 and output padding0. Converted streaming transpose wrappers already contain reversed ordinary Conv2d kernels; the loader never reverses them again.
- PReLUs store one INT8 slope and its power-of-two exponent. Their source channel axis is1.
- GRU direction records store input/recurrent matrices, separate biases and row exponents, plus sigmoid/tanh table references. Reset-after candidate-bias semantics are unchanged. Hidden state and GRU outputs use signed Q7. Intra-GRU direction0 then1 denotes forward/reverse frequency traversal; only the inter-GRU and temporal attention states persist between frames.
- Four LayerNorms retain33×16 INT8 gamma and INT32 beta arrays. The record auxiliary field stores gamma's exponent; its final16 bytes contain the exact uint64 epsilon code and the float64 metadata value1e−8. Integer execution reads the encoded epsilon, with24 fractional variance bits. The parser checks the epsilon code and the primitive's worst-case INT64 bounds.
- The six attention sigmoid outputs have explicit probability encoding `(signed_code+128)/255`. They do not have power-of-two grid entries. The seven standalone activation records reference actual256-byte tables; final tanh output is signed Q7. The parser verifies every stored table against its configured function/grid.
- Sparse ERB storage is2,040 bytes, including all382 original nonzeros and one shared transpose traversal. The window is512 float32 values,2,048 bytes. These are external DSP constants; neural learned arrays remain integer.

Both parsers recheck affine and GRU INT32 bounds and LayerNorm's wider arithmetic limits before accepting an artifact. The C loader uses caller-owned,8-byte-aligned handles and immutable model memory, performs no heap allocation, and installs the valid-model marker only after complete initialization. Generated LUT SHA256 constants and epsilon-code tables avoid floating nonlinear calculations in C initialization. DSP metadata validation alone reads floating values.

## Size and state evidence

The archived trained RMS-normalized artifact at `output/esp32/gtcrn_probes_02/gtcrn_normalized_probe_inputs/model_int8.bin` contains46,768 bytes; the raw-input artifact in the adjacent `gtcrn_probe_inputs` directory contains46,832 bytes. Their associated calibration and selection audits distinguish the frozen source checkpoints. These are actual payload sizes, not a final architecture selection or a quality claim. An initialized model with nondefault normalization parameters used46,816 bytes in the roundtrip tests; identical initialized parameters can create more deduplication than trained parameters. Even storing all current arrays separately gives49,969 array bytes; adding6,624 metadata bytes and a conservative maximum alignment allowance stays below60,688 bytes for this fixed schema.

Persistent neural history is18,048 INT8 bytes:16,896 convolution-history bytes,96 temporal-attention bytes and1,056 inter-GRU bytes. The numerical waveform wrapper additionally uses3,072 bytes of float32 DSP history. The separate native graph reports its own handle and workspace requirements; those are RAM allocations and do not disappear because model arrays reside in flash. The current scalar native graph's workspace is17,536 bytes; MCU stack usage and target-compiled handle sizes require their own checks.

Raw and frame-RMS modes have exact loaded/unloaded NumPy waveform parity, including all14 named history blocks after repeated frames. Tests deliberately recompute integrity hashes after corrupting dimensions, exponents, biases, LUTs, normalization epsilon, array extents and reserved fields, so rejection does not rely on the checksum alone. Direct C parser tests bypass the Python loader and check the same malformed payloads.

## Remaining deployment work

The native graph integration uses the packed arrays to execute the complete learned frame graph and compares masks and every persisted history against NumPy. `firmware/experimental_gtcrn/audio.c` now adds the full C framing/FFT/ERB/masking/OLA and PCM16 path. It reuses the existing portable/ESP-DSP FFT implementation and exact sparse ERB table, while every learned operation remains integer. Host audio state requires12,576 bytes and the neural model handle8,368 bytes: combined with18,048 neural history bytes and17,536 workspace bytes, the explicit RAM subtotal is56,528 bytes. This excludes caller audio I/O, compiler stack, vendor FFT tables and Python objects. Target-compiled sizes are reported separately.

Twenty focused audio tests cover constant-mask waveform reconstruction, DC/Nyquist conventions, flush lengths, PCM16 rounding/clipping, reset and prefix causality, invalid/overlapping buffers, and the sealed-test CLI guard. Both actual trained binaries were also compared on a three-second seeded synthetic tone/noise waveform. Maximum absolute differences against the same C neural graph with NumPy DSP were3.73×10⁻⁸ for RMS mode and2.79×10⁻⁸ for raw mode. This checks numerical DSP agreement; it is not speech-quality evidence.

```sh
python -m esp32_denoiser.gtcrn_embedded \
  --integer-model MODEL.bin --calibration MODEL.calibration.json \
  --manifest DEVELOPMENT.jsonl --output RESULT.json \
  --io-format pcm16 --compare-reference --perceptual --threads 2
```

The comparison uses identical manifest clips for full C audio and the C learned graph with NumPy DSP, and reports per-utterance metrics plus input-clipping/output-rail counts. The default full C path uses PCM16; `--io-format float32` evaluates unclipped C output. Both reset independently for each utterance and compensate the same256-sample initial overlap hop. Official-test manifests are rejected during development.

Target compilation must still measure handles, scratch, peak stack and firmware flash separately from the model payload. Actual ESP32-S3 frame deadlines, I/O buffering and sustained throughput remain the real-time go/no-go test. Quantization-aware recovery and final model selection must use training/development data while the official test remains sealed.
