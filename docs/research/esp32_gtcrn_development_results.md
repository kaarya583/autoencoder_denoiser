# GTCRN integer development results — 13 September 2026

The complete neural graph now has an independently checked NumPy integer reference, a packed model format, and a scalar C implementation. Learned weights, materialized neural activations, and persistent neural histories are INT8. Biases and dot accumulators are INT32; bounded wider integer intermediates are used for normalization, recurrent gate arithmetic and requantization. FFT, spectral feature construction, inverse ERB, complex mask application and overlap-add are external floating point DSP. This is not a claim that all audio processing uses eight-bit arithmetic.

These are development experiments. Official VoiceBank test data and the separately reserved external final test remain sealed. No physical ESP32 timing or microphone test has been performed.

## Frozen controls

The raw checkpoint was selected at epoch 25 on the external development cohort; its selection score was 8.6870 dB SI-SDRi. The independently trained RMS-normalized checkpoint was selected at epoch 15, at 8.5446 dB. Both have 23,669 learned parameters. Later training checkpoints have higher float scores, but those scores do not describe these frozen packed artifacts.

| Frozen model | Complete packed model | Source checkpoint SHA256 | Packed SHA256 |
|---|---:|---|---|
| Raw spectral GTCRN | 46,832 bytes | `a9ee8b173d0618b3c43c2c1d529c2e36a3e3d09d4acbcabdcdb030fc718e6a0f` | `3bfbc5c7ada8f87927e5d0e3bba0346d661c3bf4d020f1a39ca7e7f61fcabca4` |
| Frame-RMS GTCRN | 46,768 bytes | `69a3b763a82c9a0cbb9292254551b690b6278eb417f40a6d5548a0fc586f7bbc` | `f7de9911b135b0fdb282857766195aba0a5f23b3479fe20bfdadcb93aca429f7` |

Packed sizes include graph metadata, scales, stored nonlinear tables, all quantized parameter arrays, the exact sparse ERB transform and analysis window, alignment and integrity fields. Firmware code and runtime memory are separate. The RMS payload is about 253.5 times smaller than the historical Res-U-Net's 11,854,856 raw FP32 parameter bytes. That comparison combines a different architecture and quantization; INT8 alone gives roughly four times smaller weights, not 253.5 times. There is no matched final quality comparison against that historical model.

The packed artifacts and training-calibration audit files are preserved under `output/esp32/gtcrn_probes_02/`. Each audit reconstructs the checkpoint's actual training mixture recipe, uses 32 deterministic training crops with seed 483, checks source separation, and binds the checkpoint and calibration to the binary. No development waveforms determine the grids.

## Numerical quality gates

On all 500 external development mixtures, replacing only the recurrent modules with the audited integer recurrent implementation changed the raw model's SI-SDRi from 8.6875196 to 8.6653548 dB: a 0.0221649 dB loss. Surrounding neural operators and DSP remained float in that experiment. This is not a full-INT8 quality result.

The whole-graph numerical probe selected the same four development clips with seed 482, indices 153, 266, 304 and 451:

| Frozen model | Float SI-SDRi | Full INT8 neural SI-SDRi | Integer minus float |
|---|---:|---:|---:|
| Raw spectral | 10.42308 dB | 6.94499 dB | −3.47809 dB |
| Frame-RMS | 11.39607 dB | 10.56208 dB | −0.83400 dB |

These four clips diagnose quantization sensitivity; they are not a final score or a substitute for the full cohort. The checkpoints have different training histories and weights, so this does not isolate normalization as the sole cause of the difference. It motivates a training-only range-calibration investigation. The raw model's input ERB codes were zero for approximately 86.4% of observed values on these clips, versus 49.0% for the normalized model; neither input edge touched a rail. Lack of clipping therefore does not establish adequate quantization resolution.

The full scalar C graph matches every output code and all 14 persistent histories against the NumPy integer graph in dedicated tests. Waveform parity tests currently share the floating NumPy DSP. Native FFT/ERB/overlap-add/PCM16 integration is in progress. Both actual packed models are undergoing full 500-mixture C neural evaluation, including PESQ and STOI.

## Memory and remaining work

An ESP32-S3 cross-compiled size probe measures 5,784 bytes for the initialized model handle, 18,048 bytes for persistent INT8 neural state, 17,536 bytes for reusable neural workspace, and 645 bytes for input features/output mask: 42,013 explicit bytes before alignment, DSP, stack, RTOS, I/O buffers or any weight copy. No heap is used by the C neural graph. These sizes support further integration; they do not prove the complete application fits its runtime memory budget.

The coherent source bundle `7d52ed6af7bac6a2537730d2da36b7b7e8b86502edd03dffe4fa0127eed446a9` contains 178 source files and passed 620 local tests before verified installation in Colab. The tests establish numerical and software invariants, not device speed.

Next decisions use training-only error localization and power-of-two range selection, full-cohort C inference, full C DSP parity, and the ESP-IDF linked memory map. QAT recovery and a sustained physical-board deadline test remain required before claiming a final real-time model.

## Full C PCM16 results and compute allocation update

Training-only MSE range calibration improved the frozen raw model from 4.66212 to **7.44083 dB SI-SDRi**, and the normalized model from 7.59172 to **8.23048 dB**, on all 500 external development mixtures using the complete C PCM16 pipeline. The normalized MSE artifact is **47,024 bytes**, SHA256 `d2536ea3e0190b03b86406df1608c75c19e0ed842586666eae5361d3fb6b9a57`. Its PESQ is 1.86341 (noisy 1.46962), and STOI is 0.85107 (noisy 0.81342). Its matched float checkpoint scores 8.54447 dB with floating waveform I/O; the 0.31399 dB difference includes quantization and PCM16 effects. Full reports are in `output/esp32/gtcrn_probes_04/`. These remain development results.

Following the user's request to favor Colab compute over agent usage, further architecture exploration was stopped. Two large continuations are queued after the current raw/RMS training runs: 240 epochs, 43,208 training mixtures per epoch, batch 48, and a 12-hour per-run limit. They automatically calibrate on training data and evaluate the packed C model on external mixtures, clean speech and the primary development cohort. A longer QAT run is separately gated on an eight-step GPU training pilot and the normalized parent's completion. It selects checkpoints using actual C PCM16 inference. The full QAT numerical CUDA acceptance suite passed 23 tests. No sustained board timing has been measured.

## Update: 13 September 2026, 22:45 UTC snapshot

The first full QAT round (250 optimizer updates) produced a 46,768-byte packed model with **9.2022137622 dB SI-SDR improvement** over the same 500 external development clips, using the actual C PCM16 pipeline. All 500 clips were valid. This improves the earlier 8.2304770745 dB packed result by 0.9717366878 dB. Packed SHA-256: `dd9c1c83a44bea891fad7d47591578b57a4b3d0d54f2942a17e5402d4ca93e3f`. The checkpoint, binary, calibration, and full per-clip evaluation were recovered from a verified Drive snapshot; their hashes also match the latest snapshot inventory. Local artifacts are under `output/esp32/qat_development_2200/gtcrn_long_qat_batch16/candidates/epoch0001-ulylgdqg`.

The larger raw float run reached 10.0189958113 dB on the external development set at epoch 60. As of 22:45 UTC, the raw and normalized large runs were at epochs 62 and 55; their budget remains 240 epochs each or their configured time/early-stop limits. QAT was in round 2, last logged at step 100. The float and INT8 numbers are from different checkpoints and must not be interpreted as a measured quantization loss.

Drive backups continued every 15 minutes, latest at 22:45 UTC. Colab showed 82.71 compute units remaining and one active session; its browser connection was stalled during inspection, so these findings were obtained from the immutable backup artifacts. Final tests remain sealed and physical ESP32 latency remains unmeasured.

## Verified overnight outcome — 14 September 2026

The final available Drive snapshot is from 04:00 UTC (13 September 21:00 Pacific). Member hashes were verified, and each result below matches that snapshot's inventory. The saved raw and normalized large runs reached epochs 158 and 151 of 240. Neither has a completed training summary or deployment export in the final inventory. At 18:45 UTC, Colab was connected to a different, idle L4 machine with no `/content/esp32_runs` directory. The precise termination time and cause are unknown.

QAT completed two rounds and reached its six-hour cap during round 3; that partial round was discarded. Round 1 remained the selected winner at 9.20221 dB SI-SDRi and 46,768 packed bytes. All three planned full C PCM16 development evaluations finished. This is the same selected artifact reported earlier; the new information is completion of the perceptual evaluations.

| Development cohort | Count | SI-SDRi (dB) | PESQ WB noisy → enhanced | STOI noisy → enhanced |
|---|---:|---:|---:|---:|
| External mixtures | 500 | 9.20221 | 1.46962 → 1.98632 | 0.81342 → 0.86463 |
| Primary VoiceBank | 770 | 7.48364 | 1.57002 → 1.92417 | 0.81252 → 0.82495 |
| Clean preservation | 100 | Not an enhancement comparison | 4.64389 → 4.51453 | 1.00000 → 0.99471 |

Every cohort had zero invalid evaluations. Clean output SI-SDR was 45.87784 dB; improvement relative to identical clean input is not a useful denoising score. External host C throughput was 0.16962 RTF, which is not an ESP32 timing measurement. Official tests remain unused.

The best float external result improved to 10.08138 dB at raw epoch 150. The normalized branch reached 9.63342 dB at epoch 150. These float checkpoints differ from the QAT parent, so their gaps are not controlled quantization-loss estimates. Best and last checkpoints are preserved in the Drive backup chain. Training is currently stopped; the larger runs still require resumption or deliberate finalization before export and evaluation.
