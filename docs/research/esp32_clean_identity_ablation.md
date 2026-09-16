# Clean-speech exposure ablation

The new [broad-clean configuration](../../configs/esp32_zero_bias_broad_clean_float.json)
changes clean-identity exposure from the historical 3% to 15%. Both the paired
VoiceBank loader and the dynamically mixed LibriSpeech/MUSAN loader use this
same probability, so the expected fraction remains 15% under the 50/50 source
mixture. An identity example uses the clean target as its noisy input after
the existing augmentation policy. Validation keeps its original noisy audio.

This is a matched alternative to
`esp32_zero_bias_broad_distillation_control_float.json`: the same frozen
`/content/esp32_runs/broad_kd_inputs/student.pt`, a fresh optimizer, 40 epochs,
patience 40, learning rate 1e-4, waveform-loss weight 0.1, no spectral loss, and
no distillation. Only `clean_identity_probability` and the output directory
`/content/esp32_runs/float_zero_bias_broad_clean` differ. Increasing exposure
also reduces the fraction of noisy training examples at a fixed epoch length;
the experiment must measure preservation and denoising together.

The motivation is a concrete remaining preservation failure: the calibrated
TCN has mean clean projection gain around 0.9807, but a normal-level clean
development clip reportedly has SI-SDR 5.04 dB and projection gain 0.6868.
These are observations supplied by the current evaluation run, not results
of this untrained ablation. Stronger waveform-level supervision is a separate
queued arm; this experiment isolates identity exposure in its configuration.

`TrainConfig.clean_identity_probability` defaults to 0.03 and accepts finite
numeric values in [0,1]. Checkpoints record it in both training configuration
and provenance. Same-phase optimizer resumes reject a changed value before
overwriting the run's configuration; changing the recipe requires
`resume_optimizer=false`. Checkpoints that omit the field retain the old 0.03
meaning. The strict teacher audit and checkpoint calibration reader require
agreement between the new configuration/provenance records and report the
actual value. The audited `extra_data.py` and `mixtures.py` files are unchanged.
If this arm advances to QAT training, its continuation configuration must
explicitly retain 0.15; the global default remains 0.03. Checkpoint-derived
calibration probes read the recorded value automatically.

Seventeen focused tests cover both dataset endpoints using actual emitted
waveforms, unchanged validation input, legacy/default resume equivalence,
rejected recipe changes, valid optimizer restarts, actual fresh broader
teacher provenance and the matched configuration. Combined clean-exposure,
training, distillation, teacher-gain, hybrid-QAT and checkpoint-calibration
tests pass 97 cases. Calibration tests also replay legacy crops exactly and
verify both source types at probabilities 0, 0.15 and 1.

Promotion still requires matched primary 770, external 500 and clean 100
development evaluation, including full native PCM16 output after any
quantization/export. Inspect clean-clip failures and gain distributions as
well as means; retain the existing denoising metrics and external checkpoint
selection protocol. The final test sources remain sealed. No improvement or
training completion is claimed by adding this configuration.
