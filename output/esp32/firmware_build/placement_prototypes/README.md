# Optional model placement: identity build verification

Both configurations compile and link with ESP-IDF v5.4.2, ESP-DSP 1.8.2, and Xtensa GCC 14.2.0. Both contain the same **untrained identity model**, not a trained denoiser. The original identity build evidence in the parent directory is preserved unchanged.

| Placement | Application image | Static writable RAM | Additional model allocation requested at startup |
|---|---:|---:|---:|
| Default mapped flash | 322,816 bytes | 48,268 bytes | 0 |
| Optional internal SRAM | 323,104 bytes | 48,268 bytes | 94,480 bytes |

Runtime heap includes the SRAM copy and allocator overhead; the static linker RAM figure does not. Neither runtime allocation success nor latency has been measured on a board. Each subdirectory contains its full SDK configuration, build log, linker size, and clearly marked prototype binaries. `build_report.json` records source/model/binary hashes.

The option is `CONFIG_EDN_MODEL_IN_INTERNAL_SRAM` in `firmware/esp32_benchmark/main/Kconfig.projbuild`. It defaults off. When enabled, allocation is 16-byte aligned, internal, and byte-addressable. Failure stops explicitly; there is no silent fallback. A successful copy is retained through every frame and released at benchmark exit, including model initialization or frame errors.

For the future board comparison use identical trained model bytes, board, frequency, input and runtime workload. Record full-hop mean/p99/max, missed deadlines, the logged placement/model-copy bytes, startup free heap/largest block and minimum internal byte-addressable heap. No faster-than-real-time or flash-versus-SRAM speedup claim follows from these compile checks.
