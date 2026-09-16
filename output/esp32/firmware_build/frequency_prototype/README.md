# Frequency-model firmware build prototype

This artifact is an UNTRAINED IDENTITY prototype used only to verify that the EDNFQ8-v1 graph, full-frequency DSP frontend, and optional S3 SIMD kernels compile and link as an ESP32-S3 application. It is not the trained baseline or a final enhancement model. No board was attached, no firmware was flashed, and no target latency or target assembly self-test was measured.

The 94,300-byte model includes graph metadata, INT8 weights/scales, INT32 biases, readable weight guards, and DSP constants. The physical 32-bit linker symbols confirm a 40-byte model handle, 10,528-byte audio state, and 49,152-byte reserved neural workspace. Required neural workspace is 34,844 bytes by the validated parser formula and host parity tests. The application size and full linker memory accounting are in build_report.json and linker_size.json.

CONFIG_EDN_FREQUENCY_MODEL=y and CONFIG_EDN_S3_SIMD_DOT=y select this mode; the model is flash mapped. The benchmark performs the known-answer signed-dot test before measuring complete PCM conversion/FFT/features/neural/iFFT/OLA hops, excluding I2S and radio work. Replace main/model.bin with a trained EDNFQ8-v1 model before producing a final firmware artifact. Source snapshots and hashes preserve exactly what was compiled here.
