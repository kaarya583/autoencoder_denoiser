# Broader training survives an actual INT8/C deployment probe

Measured 13 September 2026 on development data. This is an intermediate candidate, not the final selected model or an ESP32 timing result.

The external-development winner from epoch 10 of the broader TCN continuation was frozen before calibration. Its SHA256 is `12e4595db174a6f34b58778e112e5f986938689f0feadc39f8f368808e32e746`. Calibration used 32 training-only batches drawn from the same 50/50 paired/synthetic distribution. No QAT optimizer steps were taken. The complete exported EDNSI8-v2 payload is **94,480 bytes**, SHA256 `50ebbb78f5834f8ec20051f2d6bfb5aebda7d994bdd7b54c2179c6b0d885525e`.

| Representation | External development SI-SDRi, 500 conditions |
|---|---:|
| Frozen float student | 6.396585 dB |
| Calibrated fake-quantized graph | 6.105654 dB |
| Full C DSP, INT8 neural core, PCM16 I/O | **6.109402 dB** |
| Earlier narrow-training control, full C PCM16 | 0.810405 dB |

The float-to-deployed loss is **0.287183 dB**. Paired bootstrap over 100 base crops, retaining each crop's five SNR conditions together, gives a 95% interval for deployed-minus-float of −0.373497 to −0.204123 dB. The broader candidate exceeds the earlier deployed control by **5.298996 dB**, interval +4.704311 to +5.923497 dB, improving 479 of 500 conditions. These are development-set comparisons conditional on the observed speakers and recordings. Crop clustering does not remove dependence between different crops sharing speakers or noise recordings. [Float comparison](../../output/esp32/broad_ptq_vs_float.json), [control comparison](../../output/esp32/broad_ptq_vs_narrow_full_c.json).

On the separate 770-utterance primary holdout, full C PCM16 SI-SDRi is **7.359141 dB**, below the earlier narrow model's 7.776252 dB. The large external gain therefore comes with a measurable primary-domain tradeoff. The chosen float source was selected on external development, not substituted with the primary run's potentially unchanged narrow initialization.

## Remaining clean-speech defect

On 100 separate clean development clips, the deployed model produces mean projection gain **1.321916**, normalized waveform L1 **0.216094**, and enhanced SI-SDR **24.767055 dB**. Its PCM output touches a rail on 282 of 4,800,000 returned samples; rail contact is not by itself proof of clipping. The external noisy suite has 177 output rail samples and the primary holdout has one; all three have zero input clipping. SI-SDR's gain invariance does not excuse the approximately 32% amplification. This candidate is not accepted as a final preservation-quality result.

On a fixed 16-clip diagnostic, float32 full C audio and Torch DSP with the same integer neural core differ by approximately 0.00000019 dB in aggregate SI-SDRi. PCM16 scores 0.038616 dB lower than unclipped C on that diagnostic. This isolates arithmetic/DSP effects for those clips; it is not an all-waveform bit-exact claim or an ESP-DSP board check.

The next matched trials strengthen the level-sensitive loss and use response distillation from a fresh broader teacher. One output gain fitted using 128 clean **training** crops is 0.33397647. On the independent clean development cohort, that fixed correction gives the teacher mean gain 0.992950, normalized waveform L1 0.028749, and zero samples outside PCM range. The corrected response KD coefficient is calibrated on eight actual training batches to a 0.1 gradient-norm ratio. These teacher measurements do not establish that KD has improved a student; the matched control and KD runs must decide that. [Distillation protocol](esp32_broad_distillation.md).

## What the size and runtime claims mean

The binary contains INT8 neural weights, activations and recurrent convolution state, INT32 biases, and model-specific DSP constants. FFT, spectral features, complex masking and overlap-add remain explicitly float32 DSP. The payload is **125.475× smaller** than the historical Res-U-Net's 11,854,856 raw FP32 parameter bytes. This ratio combines architectural downsizing with quantization; INT8 alone does not give 125× compression. The unchanged small architecture has 84,738 parameters and approximately 5.212 million convolution MACs per second. Its integer neural workspace is 8,280 bytes, excluding frontend, stacks, I/O and application overhead.

No physical ESP32-S3 has been benchmarked. Complete-hop latency, real-time factor, RAM high-water marks and I2S stability remain unmeasured. No official final test or newly reserved external final suite was used. The gain-corrected perceptual results are recorded below.

The binary, frozen inputs, calibration IDs, manifests and per-utterance reports were recovered from a hash-verified 312-file backup, archive SHA256 `e2f90f97ab5680895517f788df5e31026de5eae6c80e3288ad748def0e896618`. [Recovered probe reports](../../output/esp32/development_backup_04/broad_ptq_probe/export.json).


## Training-only DSP gain correction and perceptual evaluation

A separate fit uses 64 paired and 64 LibriSpeech clean **training** crops, with actual C float output and PCM16-rounded input. It gives float64 gain 0.7415905524418878, serialized as float32 **0.7415905594825745**. This scalar is applied after overlap-add and before PCM16 conversion; it adds 4 bytes to the DSP section and changes no neural weight or state. The resulting payload is **94,484 bytes**, SHA256 `7938f862413bb42e7aa221c0f949d0432a0e879f0e6077d98e2d35d9e8f3b062`.

| Fixed development cohort | SI-SDRi | PESQ WB, noisy → enhanced | STOI, noisy → enhanced |
|---|---:|---:|---:|
| External,500conditions |6.108522dB|1.469623 →1.676253|0.813419 →0.833727|
| Primary,770utterances |7.359140dB|1.570020 →1.836581|0.812517 →0.818261|

Without this scalar the identical external run scores 6.109402dB, PESQ 1.676059, STOI 0.833725. The tiny change in SI-SDRi is due to PCM conversion/saturation, not a gain-sensitive SI-SDR formula. This is an output-level correction, not a denoising-quality breakthrough.

On the independent 100 clean development clips, mean projection gain improves from 1.321916 to**0.980664**, normalized waveform L1 from 0.216094 to**0.056535**, and output rail contacts from 282 to**2** of 4.8 million samples. Nevertheless, clean PESQ is**4.201987** versus 4.643888 input, STOI**0.977718** versus 1.0, and enhanced SI-SDR**24.877540dB**. A worst clean clip has gain 0.686771, reference RMS 0.055813, and enhanced SI-SDR 5.039063 dB: this is normal-level speech, not a near-silence artifact. The correction does not solve suppression or waveform-shape errors.

The source float student already damages clean speech: enhanced SI-SDR 26.353467 dB, mean gain 1.329471, PESQ 4.248586, STOI 0.981424. Stronger level-sensitive loss and increased clean exposure are therefore training ablations, not attempts to hide an exclusively quantization-induced problem. The new matched clean-exposure arm raises identity examples from 3% to 15% in both paired and synthetic sources; it uses the same frozen initial student and fresh optimizer as the no-KD control. Its outcome remains pending.

The optional scalar occupies existing tail padding in the S3 audio-state struct: actual Xtensa compiler checks preserve 13,616 bytes of audio state and 32 bytes of model handle. Host struct sizes alone would not prove that. No physical board timing has been measured.

All gain-fit details, float/PCM per-utterance reports and binaries are in the verified 427-member [backup05](../../output/esp32/development_backup_05/broad_ptq_gain_probe/export.json). The 114,757,858-byte archive has SHA256 `15f1f5c62f5c85485f36c19a6806a2a8da5fc07afdc79e0a62e2628c0369cdfa`; Drive holds two hash-verified-size parts because the connector caps individual files at 100 MiB. [Restore manifest](../../output/esp32/backup_05_parts/RESTORE_esp32_backup_05.json).
