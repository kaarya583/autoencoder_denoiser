# ESP32-S3 toolchain verification: identity prototype only

The official ESP32-S3 application and bootloader compiled and linked with ESP-IDF v5.4.2, ESP-DSP 1.8.2, and Xtensa GCC 14.2.0. This folder records a build using an **untrained identity-initialized neural model**. These binaries are **not the trained denoiser** and are not audio-quality or board-performance results.

The application image is 322,432 bytes, including its 94,480-byte model. The ELF reports 48,268 bytes of static writable RAM; this excludes runtime heap and stacks. Of that, the benchmark reserves 16,384 bytes for neural workspace, while this model requires 8,280. Its audio-state struct occupies 13,616 bytes on the actual 32-bit target. Full linker figures are in `identity_linker_size.json` and `identity_size.log`.

`build_report.json` records versions, source/model hashes, flags, and paths. `sdkconfig.verified` records the full build configuration. Source and toolchain are retained in isolated `/private/tmp/esp32-denoiser-*` directories. No global shell profile or installation was changed.

After replacing `firmware/esp32_benchmark/main/model.bin` with the trained EDNSI8-v2 export, run `bash output/esp32/firmware_build/rebuild_local.sh` from any directory. This rebuilds the current model using the retained SDK, but deliberately does not relabel or overwrite the identity evidence in this folder. Capture new model hashes and binary evidence separately for the final trained build. Reinstall the pinned SDK first if temporary directories have been removed.

The declared ESP-DSP dependency is pinned by `firmware/esp32_benchmark/dependencies.lock`. The linked FFT is the ESP32-S3 `dsps_fft2r_fc32_aes3_` implementation; neural kernels are portable integer C. No ESP32 hardware was connected, flashed, or timed.
