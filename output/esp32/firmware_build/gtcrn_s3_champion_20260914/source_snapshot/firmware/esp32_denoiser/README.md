# ESP32 integer speech denoiser

This component contains an allocation-free INT8 neural runtime and a streaming
16 kHz audio frontend. Neural weights, activations, residual tensors, and temporal
FIFO state are INT8; biases and dot-product accumulators are INT32. FFT, feature
normalization, complex gain application, and overlap-add use float32 DSP.

The default exported model occupies **94,480 bytes**, including all biases,
exponents, headers, alignment, window coefficients, sparse ERB coefficients, and
mask scaling. This is **125.47× smaller** than the existing model's 11,854,856-byte
FP32 parameter payload. The runtime uses **8,280 bytes** of neural workspace,
plus its model handle, audio frontend state, ESP-DSP tables, and task stacks.
The benchmark reserves a 16,384-byte neural workspace and logs required and
allocated workspace separately. The model blob size excludes executable code
and vendor FFT tables.
These are format and allocation measurements, not trained audio-quality results.

## Verification and current limits

The host-compiled C neural runtime is tested against an independent integer
Python implementation on nonzero outputs and persistent temporal history.
Quantization-aware PyTorch output is compared on the same grids. The portable
audio frontend has a host FFT fallback for functional tests. A separate test
exercises the ESP-platform branch with a bit-reversed DFT stub, checking its
ESP-DSP call order, scaling, alignment, and complete waveform output.

The complete application and bootloader **compile and link for ESP32-S3** with
ESP-IDF v5.4.2, Xtensa GCC 14.2.0, and ESP-DSP 1.8.2. The initial prototype used
an untrained identity model: its application image is 322,432 bytes and its
static writable RAM is 48,268 bytes, excluding runtime heap and stacks. Build
evidence is in `output/esp32/firmware_build/`; those prototype binaries are not
the final trained denoiser. The lockfile pins the resolved ESP-DSP dependency.

ESP builds link ESP-DSP's optimized ESP32-S3 FFT. Both neural graphs retain a
portable scalar reference and support an optional guarded adapter around pinned
ESP-NN signed-dot assembly. Trained global-baseline scalar and SIMD firmware
have compiled; separate frequency-model firmware has compiled with an explicitly
untrained identity prototype. No ESP32 board has been flashed or measured. Compilation and host tests cannot establish
ESP32 real-time performance, audio quality, electrical power, or I2S reliability.

## Export a trained QAT checkpoint

From the repository root, after training has produced a QAT checkpoint:

```python
from pathlib import Path
import torch
from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig
from esp32_denoiser.quantization import configure_qat
from esp32_denoiser.export import export_model

checkpoint = torch.load("runs/qat/best.pt", map_location="cpu", weights_only=False)
model = SpectralTCN(SpectralTCNConfig(**checkpoint["model_config"]))
configure_qat(model)
model.load_state_dict(checkpoint["model"])
model.eval()
export_model(model, Path("firmware/esp32_benchmark/main/model.bin"))
```

Calibrate the shared hidden exponent using representative **training** audio
before fine-tuning QAT. The input exponent is -7 and output exponent is -7.
The selected hidden grid is stored in checkpoint buffers. Do not recalibrate
from the final test set or replace checkpoint scales during export.

## Build and measure on ESP32-S3

With ESP-IDF v5.4.2 installed and activated (the verified version):

```sh
cd firmware/esp32_benchmark
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/your-serial-port flash monitor
```

The component manager resolves ESP-DSP; retain its generated dependency lockfile
with any published benchmark result. The app uses 32 warm-up hops followed by
1,024 measured hops at 16 ms cadence. It reports full processing mean, p99,
maximum, RTF, deadline misses, and free internal heap. Timing includes PCM
conversion, forward FFT, features, integer inference, inverse FFT, and overlap-add.
It excludes I2S transfer and radio workloads. A synthetic input only exercises
timing; use held-out speech/noise pairs separately for audio-quality evaluation.

## Optional internal SRAM model placement

The default reads the model directly from mapped flash. For a controlled board
comparison, run `idf.py menuconfig`, open **ESP32 denoiser benchmark**, and enable
**Copy the model from flash to internal SRAM**
(`CONFIG_EDN_MODEL_IN_INTERNAL_SRAM`). Rebuild with the same exported model.
This allocates a 16-byte-aligned buffer with `MALLOC_CAP_INTERNAL |
MALLOC_CAP_8BIT`, copies the model before initialization, and keeps it through
all measured frames. Allocation failure stops with an explicit diagnostic;
it never silently switches back to flash. The copy is released on exit.

Both start and result JSON identify `model_storage` and
`runtime_model_copy_bytes`. Startup also records free internal byte-addressable
heap and its largest block; the result includes its minimum free heap.
Compare full-hop mean, p99, maximum, and missed deadlines for both placements
on the same board and workload. SRAM placement removes model reads from mapped
flash, but its latency benefit has **not been measured**. Code and some DSP
constants still use flash, so this setting does not make the complete pipeline
safe with flash cache disabled.

With the current 94,480-byte model, the SRAM copy plus the benchmark's reserved
neural workspace, audio state, model handle, and the ESP-DSP 512-point
bit-reversal heap allocation totals approximately **125,472 bytes
(122.53 KiB)** before stacks, allocator overhead, and other system memory.
This fits the 200 KiB ML-RAM target on paper. Confirm actual allocation and
memory watermarks on the final board, especially when adding I2S or radio work.
Both placement configurations have compiled successfully with the same identity
test model: flash placement produces a 322,816-byte application, and SRAM
placement produces a 323,104-byte application. Their separate build evidence is
in `output/esp32/firmware_build/placement_prototypes/`.

## Streaming API

Initialize `edn_model` with `edn_init`, allocate its reported `state_bytes`, then
call `edn_audio_init`. Declare `edn_audio_state` statically or allocate it with
16-byte alignment; the ESP32-S3 FFT requires that alignment. Feed exactly 256
PCM16 samples to `edn_audio_process_pcm16` per hop. The first emitted hop is
initial overlap padding; discard it, and feed one zero hop at end to flush the
last output. Keep the state between normal hops.

The neural-only `edn_process_frame` accepts 387 INT8 features and returns 514
INT8 deltas. At output exponent -7, the first 257 values are real-mask deltas,
and the rest imaginary-mask deltas. DSP gains are
`real = 1 + mask_scale * delta_real`, `imag = mask_scale * delta_imag`.
The mask scale (currently 2.0) is stored in the DSP trailer.

## Binary format and arithmetic

`EDNSI8-v2` is a little-endian format with a 32-byte header, 24-byte records for
each layer, raw INT8 weights, INT32 biases, per-output-channel INT8 exponents,
and a 3,988-byte `DSP1` trailer. The header records total blob and workspace
sizes. `export.py` is the format specification and serializer. `edn_init` checks
dimensions, regions, supported operations, workspace size, and worst-case INT32
accumulator bounds before inference.

Every neural tensor represents `integer * 2**exponent`. Requantization rounds
nearest with ties away from zero, then saturates to [-128, 127]. Residual
branches and their FIFOs share a fixed hidden exponent; their addition is INT32
followed by signed hardtanh clipping to [-6, 6] and the representable INT8 grid.
The input projection uses the same signed clipping; depthwise outputs use
ReLU6. Version 1 used ReLU6 residual states and is deliberately rejected by
this runtime. No learned parameters are decompressed to floating point inside
the neural runtime.

Vendor API references: [ESP-DSP FFT header](https://github.com/espressif/esp-dsp/blob/master/modules/fft/include/dsps_fft2r.h)
and [ESP-DSP configuration](https://github.com/espressif/esp-dsp/blob/master/Kconfig).

### Optional ESP32-S3 SIMD dot products

`CONFIG_EDN_S3_SIMD_DOT` enables the dense-layer adapter while preserving the
portable runtime's bias, saturation, per-channel power-of-two scales, and exact
rounding. It is disabled by default pending verification on the intended board.
The benchmark runs `edn_backend_self_test()` before measuring and stops if an
assembly result differs from the scalar known-answer calculation. Cases include
negative sums, extreme INT8 values, all weight alignments, inputs through 1,056 elements,
and scalar tails. The startup log identifies the selected backend.

The global TCN adapter uses a 16-byte-aligned 544-byte input buffer on the stack;
the frequency graph gathers dense inputs into a 1,088-byte stack buffer. It checks
readable model bounds around unaligned weight loads and uses scalar fallback for
unsupported lengths or insufficient guards. Matrix weights and model bytes do
not change. Host tests verify the adapter's calling contract and exact full-model
outputs using independent C stubs; they do not execute Xtensa instructions.

`esp_nn_dot_s8_esp32s3.S` is copied without modification from Espressif ESP-NN
commit `2c222c5e02225177b44ebf21169bc66df3c8b573`, component version 1.3.2,
[`src/common/esp_nn_dot_s8_esp32s3.S`](https://github.com/espressif/esp-nn/blob/2c222c5e02225177b44ebf21169bc66df3c8b573/src/common/esp_nn_dot_s8_esp32s3.S).
The original Apache-2.0 license and notices are retained in the assembly and
`ESP_NN_LICENSE.txt`. Only these raw kernels are vendored; the ESP-NN convolution
and requantization implementations are not linked into the baseline runtime.

Compare scalar/SIMD and flash/internal-SRAM settings on the same physical S3,
using the same trained model and clock/cache configuration. Successful compilation
is not a real-time result. The synthetic benchmark reports complete processing
of a hop but excludes I2S, radio load, and audio-quality evaluation.


## Frequency U-Net deployment

`frequency.c` implements the fixed `EDNFQ8-v1` graph with channel-major public
features `[3,257]` and deltas `[2,257]`. The global bottleneck flatten operation
preserves channel-major then frequency order. Every neural operation, residual,
and temporal FIFO follows the same integer scale/rounding contract as the TCN.
Pointwise and global projections use the shared raw-dot adapter when enabled;
depthwise convolutions remain portable C. No model-specific Python package is
needed in firmware.

The default prototype exports to **94,300 bytes**, including 31 layer records,
INT8 matrix weights and exponents, INT32 biases, alignment/read guards, and the
`FDS1` window/mask-scale DSP trailer. `frequency_export.py` defines the format.
The C parser validates the complete fixed topology, monotonic array ranges,
integer accumulator bounds, DSP data, and all workspace arithmetic. Its
**34,844-byte neural workspace** contains **18,816 bytes of temporal history**,
position counters, encoder skips, and reused activation storage. The S3 compiler
confirms a **10,528-byte audio state** and **40-byte model handle**. The benchmark
reserves 49,152 neural bytes and separately logs the required allocation.

Use `ednf_model_handle_bytes()` and `ednf_workspace_bytes()` for opaque bindings.
For audio, allocate a 16-byte-aligned `ednf_audio_state` or query
`ednf_audio_state_bytes()`, call `ednf_audio_init`, then supply 256-sample hops to
`ednf_audio_process` or `ednf_audio_process_pcm16`. Initial overlap and final-flush
semantics match the global TCN. Both frontends share `audio_dsp.h` for FFT setup,
normalization, quantization, complex masks, inverse FFT, and overlap-add.

Select `CONFIG_EDN_FREQUENCY_MODEL=y` in the benchmark only with an EDNFQ8 model
in `main/model.bin`. A mismatched binary stops initialization. The frequency
prototype compiled to a 327,776-byte application with 77,956 bytes of static
writable RAM; this build contains an **untrained identity model**, not a final
trained enhancement model. Its immutable build/source evidence is under
`output/esp32/firmware_build/frequency_prototype/`. Trained global-baseline build
evidence is separately under `trained_baseline/` and
`trained_baseline_shared_kernels/`.

Frequency graph tests verify exact integer stream parity, including SIMD adapter
stubs, signed extrema, nondefault channel counts/dilations, and reset. Full audio
tests also exercise ESP-DSP's bit-reversal contract, nonzero complex masks,
nondefault window/scaling, silence, flush, and PCM16 conversion. Board timing,
assembly execution, peak stack, and actual heap watermarks remain unmeasured.
