# Quality, level control, and evaluation audit

Audit date: 12 September 2026. This review covers the current training objective, data mixing, waveform alignment, checkpoint selection, and matched development comparisons. It does not establish final-test quality or physical ESP32 throughput.

## Finding

No material SI-SDR implementation, crop alignment, padding-mask, or synthetic-SNR mixing defect was found. The principal observed limitations are generalization outside the VoiceBank training mixtures and insufficient control of output amplitude. The current evidence supports testing broader supervised training and level-sensitive losses before spending the remaining architecture budget on more parameters.

The verified local archive contains a full-C zero-depthwise-bias result of **7.77625 dB SI-SDR improvement** on 770 held-out VoiceBank training-origin utterances. This split holds out speakers p226/p287; it does not hold out the entire mixture/noise domain. Its noisy baseline is 6.01674 dB. See the [archived full-C evaluation](../../output/esp32/development_backup_02/zero_bias_embedded_validation.json) and [experiment protocol](esp32_experiments.md).

The following external-development results are now present in verified local **backup 03**: [500-mixture report](../../output/esp32/development_backup_03/zero_bias_external_mixtures.json) and [100-clean-clip report](../../output/esp32/development_backup_03/zero_bias_external_clean.json). Both evaluate complete C PCM16 inference with the same zero-bias integer model as the VoiceBank report, SHA-256 `3d58fa7c301a746ba226e6d25ff7b4ac4190bd29e7e97893e08d694128914907`. These are development results, not final-test evidence:

| External development condition | Reported result |
|---|---:|
| 500 LibriSpeech/MUSAN mixture conditions | Noisy SI-SDR 3.5130 → enhanced 4.32344 dB; improvement **0.810405456 dB** |
| Same mixtures, PESQ | 1.46962 → 1.48656 |
| Same mixtures, STOI | 0.813419 → 0.803853 |
| SI-SDR regressions | 144 / 500 conditions |
| Improvement at requested active SNR −5 / 0 / 5 / 10 / 20 dB | 0.814 / 1.286 / 1.281 / 1.006 / **−0.336 dB** |
| Separate 100 clean-input clips | Enhanced SI-SDR 29.4275 dB; PESQ 4.5201; STOI 0.996757 |
| Same clean inputs, mean projection gain | **1.5655215865** |

The high clean-input SI-SDR coexists with a large amplitude error. SI-SDR is intentionally invariant to a common output gain, so it cannot distinguish faithful level preservation from clean speech amplified by approximately 1.57×. Perceptual measures may also be insensitive to some overall level changes. Projection gain, normalized waveform error, and PCM clipping must remain explicit criteria.

## What the implementation checks establish

The [SI-SDR implementation](../../esp32_denoiser/metrics.py) removes the mean using only valid samples, evaluates each utterance separately in float64, subtracts the paired noisy score for improvement, and gives utterances equal weight. Silent references are undefined; zero estimates on valid speech receive the capped worst score. Evaluation rejects nonfinite model outputs. Every archived 770-utterance comparison has a complete finite cohort and the same noisy baseline.

The [training loss](../../esp32_denoiser/train.py) combines negative SI-SDR with waveform mean absolute error divided by the clean reference RMS. Padded samples do not enter either term. Pure clean-input amplification has essentially no SI-SDR penalty, explaining why the original small waveform coefficient can leave a level error without invalidating the metric implementation.

The [spectral loss](../../esp32_denoiser/losses.py) normalizes the estimate and reference by the **same reference RMS** and combines compressed magnitude and complex errors on 256-, 512-, and 1024-point analysis grids. Consequently it penalizes a change in output gain. It does not independently normalize each signal and accidentally discard that information. Its training-only coefficient calibration measures relative parameter-gradient norms, not a guaranteed optimal quality tradeoff.

The [model frontend](../../esp32_denoiser/model.py) uses a causal 512-sample analysis frame with a 256-sample hop, explicit initial overlap padding, and a final flush. Feature normalization uses the current frame only. The full-C frontend tests compare the host FFT and an independently implemented ESP-DSP contract stub with the Torch frontend, including DC/Nyquist, signed complex masks, altered window constants, partial final hops, reset, and PCM16 clipping. These checks found no one-hop shift or future-frame leakage. They are correctness tests, not ESP32 board measurements.

The [dynamic mixer](../../esp32_denoiser/extra_data.py) computes noise RMS on the same active speech mask used for speech RMS, scales the noise for the requested SNR, and applies a shared pair gain/peak reduction. Clean and noisy waveforms are not normalized independently. Original speech speakers and noise recordings are held out before cropping. Requested **active-segment SNR** is different from whole-utterance, zero-mean noisy SI-SDR; a 3.513 dB measured noisy baseline does not itself contradict the configured SNR sweep.

The [development generator](../../esp32_denoiser/development.py) reuses a base-crop seed across its five SNR conditions, preserving the underlying clean speech and noise segment. The 500 mixture records therefore contain 100 base crops, not 500 independent acoustic examples. The paired comparison utility supports resampling complete base crops and explicitly states that even these clusters can share speakers or original recordings.

## Quantified comparisons and interpretation

The new [comparison utility](../../esp32_denoiser/comparison.py) requires identical ID sets, rejects duplicate IDs and inconsistent stored improvements, verifies noisy baselines, and checks manifest hashes and scored sample counts when available. It accepts both training `per_utterance` and deployment `utterances` schemas. Each metric reports its own common finite cohort; missing PESQ/STOI values cannot silently change the SI-SDR denominator. Training exports that omit invalid IDs are rejected because a complete pairing cannot be reconstructed.

Paired percentile intervals below use 10,000 deterministic resamples, seed 2026, and equal utterance weighting. They are conditional on the two evaluated VoiceBank speakers and recordings, not independent-speaker population confidence intervals.

| Candidate minus reference | Mean paired change | 95% utterance bootstrap interval | Winning utterances |
|---|---:|---:|---:|
| Width-256 float teacher minus width-64 zero-bias float, SI-SDRi | +0.78142 dB | [0.71759, 0.84312] | 714 / 770 |
| Depth-10 INT8 minus zero-bias INT8, SI-SDRi | −0.02095 dB | [−0.05329, 0.01147] | 350 / 770 |
| Same INT8 pair, PESQ | +0.030823 | [0.025432, 0.036354] | 500 / 770 |
| Same INT8 pair, STOI | +0.000729 | [−0.000415, 0.001890] | 400 / 770 |

The archived deeper model has a consistent small PESQ improvement on this cohort. Its SI-SDR and STOI differences do not resolve a clear winner at this bootstrap precision. The teacher's in-domain advantage is much larger, but does not establish that its response is a good target on unseen LibriSpeech/MUSAN conditions. Evaluate that teacher on the fixed external development set and strengthen its broader supervision before relying on distillation there.

The teacher's later [complete CPU perceptual evaluation](../../output/esp32/development_backup_03/teacher_validation_perceptual.json) has 770 valid utterances and no metric errors: **8.7205483819 dB SI-SDRi**, **PESQ 2.1088951447**, **STOI 0.8434212923**. Its training-loop selected SI-SDRi is **8.7204909427 dB**; the approximately 0.000057 dB evaluation difference is numerical, not a new checkpoint improvement. Compared with the zero-bias full-C integer report, this float reference is higher by **0.94430 dB**, **0.21956 PESQ**, and **0.01597 STOI**. This additional comparison crosses architecture, precision and training budget; it does not measure quantization-only loss or demonstrate a distillation gain.

Evidence: [teacher paired comparison](../../output/esp32/paired_comparisons/teacher_minus_zero_bias.json), [integer architecture paired comparison](../../output/esp32/paired_comparisons/depth10_minus_zero_bias_int8.json).

## Minimal level-control experiments

The implemented [level-loss configuration](../../configs/esp32_zero_bias_level_float.json) matches the existing zero-bias fine-tuning control: same starting checkpoint, training data, seed, optimizer restart, learning rate, and budget. Its only experimental change is **waveform_loss_weight 0.1 → 1.0**. Model dimensions, integer arithmetic, inference MACs, payload size, and DSP are unchanged. The default remains 0.1, preserving existing runs. Regression tests confirm identical default behavior and a tenfold stronger restoring gradient for pure amplification without an artificial SI-SDR reward.

Compare this control with the already planned spectral continuation and with a matched broader-data continuation. Report mixture SI-SDRi, PESQ/STOI, clean projection gain, normalized waveform L1, and input/output clipping on the same IDs. Inspect the last checkpoint as well as the primary SI-SDR winner: SI-SDR-only selection may discard a useful level correction. The external-development selector preserves a separate SI-SDR winner, but does not itself optimize amplitude fidelity.

For any combined level-plus-spectral experiment, calibrate the auxiliary gradient ratio against the matching primary loss:

```python
from functools import partial
primary = partial(speech_loss, waveform_loss_weight=config.waveform_loss_weight)
calibration = calibrate_spectral_weight(model, training_batches, primary)
```

A **fixed output-gain control is proposed only**, not implemented or measured. Fit one scalar from a frozen, disclosed set of **training-only clean-input examples**, with the model and all neural weights frozen. One simple estimator minimizes equal-utterance reference-RMS-normalized squared error:

\[
g = \frac{\sum_i \operatorname{mean}(\hat{s}_i s_i)/r_i^2}
         {\sum_i \operatorname{mean}(\hat{s}_i^2)/r_i^2},
\qquad r_i=\max(\mathrm{RMS}(s_i),10^{-4}).
\]

Use valid samples, exclude undefined/silent references from this calibration, require a finite positive scalar, and freeze it before any development evaluation. Do **not** set it to the reciprocal of the external clean-set mean gain: that would calibrate on the evaluation set. Apply the fixed scalar before PCM clipping. A successful control should improve level/L1/clipping while preserving pre-clipping SI-SDR; any post-clipping SI change must be identified as an interface effect. Global scaling alone cannot remove the external noise-domain error and does not constitute a learned enhancement gain. Additional gain metadata and multiply cost would need explicit accounting if this control were ever adopted in firmware.

## Secondary observations

- The archived 3-second crop setup pads many naturally short VoiceBank clips: approximately 63.9% of training utterances are shorter than three seconds, corresponding to approximately 14.0% padding when sampled once each at that crop size. Loss masks handle this correctly. Training BatchNorm statistics can nevertheless include padded frames; that may affect a BN ablation without invalidating evaluation. Longer crops of the same short clips would increase padding rather than provide more useful context.
- Common waveform gain augmentation is mostly redundant for this particular network above the RMS floor: its normalized spectral input, SI-SDR, and reference-RMS-normalized L1 are approximately invariant to shared gain. Noise-level augmentation changes the task and remains useful. Adding diverse noise recordings and real speech context is the more direct generalization intervention.
- Target-informed mask diagnostics in the archive substantially exceed the learned model's score, including on the current INT8 output grid. They are not achievable learned-model scores or formal SI-SDR upper bounds. They give no immediate reason to widen mask ranges and sacrifice quantization precision.
- The checkpoint companion now serializes writers with an OS lock and rejects changed model architectures in an existing selector directory. It copies atomic training checkpoints to an immutable candidate snapshot and preserves an external-development winner separately. The official test remains outside both selection paths.
