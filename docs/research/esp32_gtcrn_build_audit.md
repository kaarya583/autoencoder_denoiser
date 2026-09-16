# Complete GTCRN ESP32-S3 build and memory audit

The complete C audio path and integer GTCRN graph **compile and link for ESP32-S3** with the actual trained normalized development candidate. The packed model is **46,768 bytes**, the application binary is **298,576 bytes**, and the explicit audio/neural runtime arena is **53,904 bytes**. No physical board was flashed: these are compiler, linker and allocation-contract results, with host numerical checks. They do not establish real-time inference or observed heap/stack peaks.

The immutable evidence is [build_report.json](../../output/esp32/firmware_build/gtcrn_s3_graph/build_report.json), with the ELF, map, binary, exact compile commands, source snapshot, generated known answers and stack-usage files alongside it. The benchmark source and operating instructions are in [firmware/gtcrn_benchmark](../../firmware/gtcrn_benchmark/README.md).

## Exact candidate and toolchain

The binary is `output/esp32/gtcrn_probes_02/gtcrn_normalized_probe_inputs/model_int8.bin`, SHA256 `f7de9911b135b0fdb282857766195aba0a5f23b3479fe20bfdadcb93aca429f7`. Its adjacent calibration sidecar validates against the packed calibration digest, and its saved source checkpoint hashes to `69a3b763a82c9a0cbb9292254551b690b6278eb417f40a6d5548a0fc586f7bbc`. This is a trained development probe, not the final selected model. Its fixed frame-RMS normalization remains floating-point DSP outside the integer neural graph.

The build uses ESP-IDF 5.4.2, ESP-DSP 1.8.2, Xtensa GCC 14.2.0 (`esp-14.2.0_20241119`), `-O2 -std=gnu17 -mlongcalls`, CPU 240 MHz, 2 MiB DIO flash at 80 MHz, and no PSRAM. The default graph uses portable scalar integer kernels. The linked FFT symbol is `dsps_fft2r_fc32_aes3_`; bit reversal uses `dsps_bit_rev_fc32_ansi`. This is not a claim of a SIMD-optimized neural graph.

## Flash and initialization

The ELF contains the exact packed bytes at `0x3c028f00`, aligned to 16 bytes by an explicit assembly directive. No weight or DSP constant was omitted from the packed file to obtain the 46,768-byte count. The file includes descriptors, edge grids, INT8 learned arrays, INT32 biases, deduplicated nonlinear tables, the sparse ERB matrix and synthesis window. Executable code and vendor FFT resources are additional.

The complete application `.bin` occupies 298,576 bytes of its 1,048,576-byte factory partition, leaving 750,000 bytes. The factory configuration has no OTA partition; the spare space is not an OTA guarantee. The linker reports 18,236 bytes of static writable DRAM plus 52 bytes of RTC sections for the entire application. These include benchmark arrays and SDK globals, and are separate from heap requests. Executable IRAM and other SDK reservations must also be considered when assessing whole-device memory.

Thirty-four custom source/configuration files have matching SHA256 snapshots before and after the final build. The image links all required GTCRN and DSP symbols. Twelve complete integer output/state arrays agree exactly between NumPy and host C for the binary-bound startup vectors. The firmware contains compact FNV-1a hashes of those references and aborts its timing run if they differ on the device. The on-device startup check remains **unexecuted**; the hashes are a smoke check, not a cryptographic correctness proof.

## Internal RAM accounting

Target-compiled `sizeof` probes and the same compiled C allocation queries determine these counts. This table counts the complete audio state, so its embedded feature/mask arrays must not be added again.

| Runtime allocation or reservation | Bytes |
|---|---:|
| Model handle and zero-copy operator handles | 5,784 |
| Persistent INT8 neural histories | 18,048 |
| Reused neural workspace | 17,536 |
| Audio DSP state and sparse-ERB handle | 12,528 |
| Additional alignment in the single arena | 8 |
| **Internal runtime arena** | **53,904** |
| PCM16 input/output buffers | 1,024 |
| ESP-DSP FFT bit-reversal heap request | 960 |
| Configured main task stack | 8,192 |
| **Audio path with these explicit reservations** | **64,080** |
| Benchmark-only latency sample array | 4,096 |
| **Benchmark subtotal with model in flash** | **68,176** |
| Optional complete model copy into SRAM | 46,768 |
| **Projected benchmark subtotal with model in SRAM** | **114,944** |

Only the default flash placement was built. The SRAM-copy option is implemented, but its subtotal above is an allocation projection rather than a second measured or linked performance result. Both explicit subtotals are below 200 KiB; that limit has not been validated against an integrated microphone/radio application.

The 960-byte FFT request follows ESP-DSP 1.8.2's allocation of `2 × 240 × sizeof(uint16_t)` for the 512-point reversal table. Its S3 FFT initializer uses the ROM-provided twiddle-table pointer for this size, rather than requesting another float table from the application heap. This does not make ROM/system-reserved RAM available to the application. The packed arrays remain mapped from flash and the strict model loader uses caller-owned storage without allocating a second learned network.

The 8,192-byte main stack already covers local float PCM conversion buffers: the compiler reports a 2,080-byte frame for `edng_audio_process_pcm16`, not an additional persistent allocation. Individual `.su` entries are not a whole-call-chain peak or an interrupt-inclusive bound. ESP-IDF's Xtensa `StackType_t` is `uint8_t`; this benchmark consequently reports the stack high-water mark in **bytes**. Actual on-device high-water and free/largest internal-heap measurements remain necessary.

Do not add the whole-application static DRAM count to the table without removing overlap: PCM and latency arrays are already in `.bss`. Allocator headers, other RTOS tasks, interrupt stacks, SDK/ROM reservations, I2S/DMA, radio and other application features are outside the explicit ML subtotal.

## Remaining hardware gate

The benchmark warms up 32 frames and measures 1,024 subsequent 16 ms hops, including PCM conversion, both FFTs, sparse ERB, the full integer graph and overlap-add. It records mean/p99/max processing time, deadline misses and processing RTF. Task delays and synthetic input generation are excluded from the timed interval; I2S is absent. None of these latency fields have actual board values yet. A board run with passing known answers, sufficient memory headroom and deadline margin is required before calling this implementation real-time on ESP32-S3.

## Current QAT champion build — 14 September 2026

The exact 9.20221dB SI-SDRi development champion now also compiles and links for ESP32-S3. This build embeds the 46,768-byte QAT binary SHA-256 `dd9c1c83a44bea891fad7d47591578b57a4b3d0d54f2942a17e5402d4ca93e3f`, rather than the earlier probe described above. The model, calibration, checkpoint, and saved evaluation hashes were checked together.

Evidence is in [the champion build report](../../output/esp32/firmware_build/gtcrn_s3_champion_20260914/build_report.json). All12 host known-answer mask and complete-state comparisons passed. The existing ESP-IDF5.4.2, ESP-DSP1.8.2, and Xtensa GCC14.2.0 toolchain produced a 298,576-byte application. Target allocation sizes remain a 53,904-byte aligned runtime arena and64,080bytes including the explicit PCM, main-stack, and FFT reservations listed above. These exclude integrated microphone/radio requirements and observed whole-device peak memory.

The model is embedded byte-for-byte at16-byte-aligned address `0x3c028f00`. The application binary SHA-256 is `54253c39a05edeb1395f35edba3aafcf4d0a1f69616046225db9bad36523e727`; ELF SHA-256 is `e7d78d6cccf503f2cc6d7bf4584eae5775830e0b9ad34a8bd3fef512d77152d6`. The report SHA-256 is `350bbc3a981117b10cea89b96a3f164f0cf7b6fa7755a450388b28816a67e557`. The build used an isolated staging copy;104 original source/current-main files remained unchanged. No physical board was flashed, and this result still makes no ESP32 latency or real-time claim.
