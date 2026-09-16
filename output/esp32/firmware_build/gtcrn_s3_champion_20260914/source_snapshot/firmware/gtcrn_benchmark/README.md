# ESP32-S3 GTCRN benchmark

This standalone ESP-IDF project links the complete packed GTCRN integer graph and its floating-point audio DSP. It measures the PCM16 interface, including FFT, sparse ERB, the integer neural graph, inverse FFT and overlap-add. It has no microphone/I2S driver. The default neural kernels are portable scalar C; ESP-DSP supplies its ESP32-S3 FFT implementation.

The checked build uses ESP-IDF 5.4.2, ESP-DSP 1.8.2, Xtensa GCC 14.2.0, 240 MHz and no PSRAM. A successful cross-compile does not establish a real-time result. No physical board has run this project yet.

From a repository Python environment with the ESP32 training dependencies installed:

```sh
cp /path/to/validated/model_int8.bin firmware/gtcrn_benchmark/main/model.bin
python firmware/gtcrn_benchmark/generate_known_answer.py
```

The generator validates the packed binary, compares 12 complete frames and recurrent states between the NumPy and host-C integer implementations, and writes a small header bound to that exact binary SHA256. Regenerate it whenever the model changes. The binary and generated header are intentionally ignored by Git.

In an activated ESP-IDF environment:

```sh
cd firmware/gtcrn_benchmark
idf.py set-target esp32s3
idf.py build
idf.py size
idf.py -p /dev/your-board-port flash monitor
```

The packed blob is explicitly aligned to 16 bytes in a generated assembly file. By default it remains mapped from flash. The optional `GTCRN_MODEL_IN_INTERNAL_SRAM` menuconfig setting copies the complete blob into internal SRAM for a separate memory-placement measurement; it does not change the model or integer arithmetic.

Before timing, firmware checks 12 integer mask/state hashes against the binary-bound references. A mismatch aborts the run. These compact FNV-1a hashes are startup smoke checks, not cryptographic correctness proofs; the generator compares complete arrays on the host. The firmware then initializes fresh audio/neural state, warms up for 32 frames, and times 1,024 further 256-sample frames at 16 kHz. It reports mean, p99 and maximum processing time, 16 ms deadline misses, processing RTF, internal-heap diagnostics and the main task's stack high-water mark in bytes. Input generation and task delays are outside the measured interval. The stimulus is synthetic and provides no audio-quality evidence.

Real-time promotion requires an actual board log with a passing startup check, the exact model hash, complete processing latency below the hop deadline with margin, and acceptable heap/stack headroom. An I2S/radio application additionally needs its own integrated measurement. Retain separate flash/SRAM results; neither placement is assumed faster before measurement.

The trained development candidate used in the first build is recorded in `output/esp32/firmware_build/gtcrn_s3_graph/build_report.json`. It is not a final selected model. See `docs/research/esp32_gtcrn_build_audit.md` for exact linker and memory accounting.
