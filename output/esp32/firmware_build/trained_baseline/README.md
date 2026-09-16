# Trained development baseline firmware

Both variants contain the trained 94,480-byte EDNSI8-v2 development baseline, not the earlier identity prototype. This is not the final frequency-model frontier result. See build_report.json for the exact model/source hashes, toolchain, sizes, and configurations. Earlier identity evidence under identity_prototype/ and placement_prototypes/ has not been overwritten.

scalar_flash uses the portable integer kernel. simd_flash compiles and links the pinned ESP-NN S3 signed-dot assembly and guarded adapter. No physical board was attached: the actual Xtensa known-answer self-test and all board latency/RTF/heap measurements remain unexecuted. On a board the benchmark runs signed/extreme/alignment/tail tests against a scalar dot before timing and stops on any mismatch.

Both apps use 240 MHz, DIO flash at 80 MHz, no PSRAM, and an 8 KiB main-task stack. Linker writable RAM includes benchmark arrays and runtime infrastructure; requested runtime model-copy bytes are zero for these flash-mapped variants. SIMD's 544-byte dense scratch is stack storage, not additional static BSS. Reported storage is not a measured stack high-water mark.

Build commands are preserved in rebuild_trained_local.sh in the parent directory. Flash application.bin at 0x10000, bootloader.bin at 0x0, and partition-table.bin at 0x8000 only on a matching ESP32-S3 with the stated flash configuration. No flashing was performed.
