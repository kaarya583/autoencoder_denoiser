# ESP32-S3 firmware memory and partition audit

The current frequency implementation fits the 200 KiB **ML-memory** target by allocation accounting, including an optional internal-SRAM model copy and the entire configured main-task stack. This is not a measured runtime high-water mark. No physical S3 has been run, so actual minimum free heap, largest free allocation, task stack headroom, and I2S/radio memory remain unmeasured.

The figures below come from the preserved ESP-IDF 5.4.2 / ESP-DSP 1.8.2 builds, their linker reports and symbols, the configured task stack, and validated C workspace calculations. The compiler was Xtensa GCC 14.2.0 (`esp-14.2.0_20241119`) with `-O2 -std=gnu17 -mlongcalls`. Both builds used 240 MHz, 2 MiB DIO flash at 80 MHz, and no PSRAM.

## Model storage, ML allocations, and stack

| Item, bytes | Trained global TCN build | Frequency identity prototype |
|---|---:|---:|
| Complete model blob, included in application flash | 94,480 | 94,300 |
| Required neural workspace | 8,280 | 34,844 |
| Neural workspace actually reserved in benchmark BSS | 16,384 | 49,152 |
| Unused reservation above required workspace | 8,104 | 14,308 |
| Audio state, actual 32-bit S3 layout | 13,616 | 10,528 |
| Model handle, actual 32-bit S3 layout | 32 | 40 |
| External benchmark PCM input/output arrays | 1,024 | 1,024 |
| ESP-DSP bit-reversal table heap request | 960 | 960 |
| Main-task stack configured reservation | 8,192 | 8,192 |
| Accounted ML subtotal with flash-mapped model | **40,208** | **69,896** |
| Accounted ML subtotal with full internal-SRAM model copy | **134,688** | **164,196** |

The subtotal counts the **reserved** neural workspace, not just the bytes the graph needs. It includes the entire main-task stack once. The INT8 dot input buffers (544 bytes for the global TCN, 1,088 bytes for the frequency graph), PCM-to-float conversion buffers, and temporary local variables use that stack; adding them again would double-count stack reservation. The separate startup known-answer self-test also uses stack, before audio timing. Actual maximum stack use remains unmeasured.

The frequency SRAM-copy subtotal is 160.35 KiB, leaving 40,604 bytes under the 204,800-byte target for allocator overhead and additional ML integration needs. Reducing the benchmark's 49,152-byte workspace reservation to the validated 34,844-byte requirement would lower this subtotal to 149,888 bytes, but the current firmware still reserves the larger amount.

These subtotals exclude application code in instruction RAM, unrelated RTOS task stacks and control blocks, general runtime heap use, benchmark timing-history storage, and future I2S/DMA or radio buffers. They therefore describe ML allocations, not total chip SRAM use. The model-copy allocation itself has not been executed on a board. It is an optional aligned `MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT` allocation; failure stops the benchmark rather than falling back to flash.

ESP-DSP 1.8.2 requests `2 * 240 * sizeof(uint16_t) = 960` bytes for the 512-point bit-reversal table. On S3, its FFT initialization uses the ROM-supplied table pointer for FFT sizes up to 1,024 rather than requesting a new floating-point twiddle array. Platform-reserved ROM/FFT storage is not a newly allocated model buffer. This source behavior does not replace checking the actual runtime heap.

## What the linker proves

| Build | Application binary | Static writable RAM (`.data + .bss`) | D/IRAM-resident code | Separate IRAM section |
|---|---:|---:|---:|---:|
| Trained global TCN, shared SIMD kernels | 324,352 B | 48,268 B | 38,455 B | 16,383 B |
| Frequency identity prototype, SIMD | 327,776 B | 77,956 B | 38,455 B | 16,383 B |

The static writable RAM values already include neural/audio storage, model handles, 1,024 bytes of external PCM buffers, the 4,096-byte latency-history array, and other linked globals. Do not add the ML static arrays again to these totals. With only the known 960-byte FFT allocation and the 8,192-byte main stack added, the frequency benchmark subtotal is 87,108 bytes of writable/allocated memory, or 181,408 bytes with a 94,300-byte model copy. **Those are incomplete whole-application subtotals**, because other runtime allocations and task stacks are not included.

The frequency linker report assigns 116,411 bytes to the combined D/IRAM region: 77,956 data/BSS plus 38,455 code. It reports 225,349 bytes remaining in that linker region. This is linker placement headroom, not measured free heap. Other startup allocations reduce runtime availability; the separate 16,383-byte IRAM section must not be confused with neural workspace. No PSRAM allocation is used to meet the stated ML budget.

Build evidence:

- [Trained global baseline report](../../output/esp32/firmware_build/trained_baseline_shared_kernels/build_report.json) and [linker sizes](../../output/esp32/firmware_build/trained_baseline_shared_kernels/linker_size.json).
- [Frequency identity prototype report](../../output/esp32/firmware_build/frequency_prototype/build_report.json) and [linker sizes](../../output/esp32/firmware_build/frequency_prototype/linker_size.json).

The trained global artifact uses `development_backup_01/deploy_baseline/denoiser_int8.bin`. The frequency artifact is explicitly an **untrained identity model for build verification**, not a final trained quality result. Both directories retain source snapshots. Their firmware was compiled before the later host-GCC sign-comparison portability fix; that fix separately passes GCC 15 with `-Wall -Wextra -Werror`. A final selected model needs a fresh build and provenance record.

## Partition capacity

The actual decoded partition table contains:

| Partition | Offset | Reserved size |
|---|---:|---:|
| NVS | `0x9000` | 24,576 B |
| PHY initialization | `0xf000` | 4,096 B |
| Factory application | `0x10000` | 1,048,576 B |

The frequency prototype leaves **720,800 bytes** free in the existing 1 MiB application slot. The trained global baseline leaves **724,224 bytes**. These binary sizes already contain the embedded model, executable code, and linked read-only data; do not add model bytes again. The model-only compression ratio does not apply to complete firmware size.

The configured flash capacity is 2,097,152 bytes. The current application slot ends at `0x110000`, leaving 983,040 bytes outside the listed partitions. That space is not automatically available to the application image without changing the partition table. The current layout has one factory application and **no OTA slots**. OTA capability would require a deliberate partition and update design; it is not part of the present build proof.

Decode the preserved binary table with the pinned SDK utility:

```sh
python "$IDF_PATH/components/partition_table/gen_esp32part.py" \
  output/esp32/firmware_build/frequency_prototype/partition-table.bin
```

A final board result should report model hash, architecture, scalar/SIMD setting, flash/SRAM placement, actual allocation success, required versus reserved workspace, minimum free internal heap, largest free block, main-stack high-water mark, and complete-hop deadline statistics. Current benchmark output already records the placement, workspace, heap, and deadline fields; stack high-water measurement still needs to be collected explicitly during board validation.
