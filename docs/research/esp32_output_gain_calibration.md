# Training-only deployed output-gain calibration

`esp32_denoiser.output_gain` fits one positive scalar for an existing EDNSI8 binary. It leaves the neural graph, neural weights, state and binary unchanged. The source binary must already fit the 99,000-byte budget and have serialized `output_gain == 1.0`; an existing correction cannot be stacked accidentally.

The default cohort contains exactly 64 distinct paired-training clean recordings and 64 LibriSpeech-training recordings, with one deterministic random crop of at most three seconds per recording. Crops with reference RMS ≤1e-5 are recorded and skipped. Insufficient eligible recordings fail instead of silently shrinking the cohort. There is no noise, gain augmentation, level normalization, development response fitting, or download.

Before any calibration inference, the supplied primary development, external mixture, external clean, and complete held-out LibriSpeech source manifests form explicit exclusions by speaker, ID, group, canonical path and file-content hash. Whole-file lengths/rates/channels and declared content hashes are checked. Selected training content is also hashed and excluded against those held-out files. The binary, all six manifests and audited audio must remain unchanged throughout the fit. These checks depend on complete, correct operator-supplied source manifests; they do not detect equivalent audio in a different codec or prove an earlier model's training lineage.

For each included crop, the original clean waveform is the reference. Input is rounded to signed PCM16 using nearest-even rounding and clipping, decoded at 1/32768, and passed through the actual full-C float-output frontend. The response therefore includes the C integer neural core and C DSP, before output saturation. The scalar minimizes the mean of each crop's squared error divided by its original reference mean-square energy:

```text
gain = sum_i mean(output_i * clean_i) / mean(clean_i^2)
       -------------------------------------------------
       sum_i mean(output_i^2) / mean(clean_i^2)
```

The report retains the float64 fit, normalized sums, before/after normalized MSE, raw/corrected projection gain, input clipping counts, source binary SHA, manifest SHAs, and each selected recording's path/hash/offset/length. The serializer separately rounds the fitted value to the deployed float32 constant. The least-squares fit does not directly optimize output clipping, PESQ or SI-SDR and does not guarantee development-quality improvement.

An example invocation after the source binary has been frozen:

```sh
python -m esp32_denoiser.output_gain \
  --integer-model /content/esp32_runs/gain_inputs/denoiser_int8.bin \
  --paired-train-manifest /content/voicebank/manifests/train.jsonl \
  --speech-train-manifest /content/extra_audio/manifests/speech_train.jsonl \
  --primary-development-manifest /content/voicebank/manifests/val.jsonl \
  --speech-development-manifest /content/extra_audio/manifests/speech_val.jsonl \
  --external-mixtures-manifest /content/extra_audio/development/mixtures.jsonl \
  --external-clean-manifest /content/extra_audio/development/clean.jsonl \
  --examples-per-source 64 --crop-seconds 3 --seed 2026 \
  --output /content/esp32_runs/gain_inputs/training_gain.json
```

The command writes a new JSON report only and refuses to overwrite an existing report. Apply its scalar separately with `with_output_gain`, retain both original and corrected model hashes, and compare their actual full-C PCM16 outputs on the fixed development suites. Verify payload size and the exact float32 stored gain. The official final set remains sealed until the model-selection protocol permits evaluation.

18 focused fixture tests pass. They use actual C inference to recover the expected correction for a known 1.5× neural gain, independently recompute the normalized fit from recorded crops, check deterministic sampling/RNG preservation, reject speaker/ID/group/path/content overlap and stale source files, prevent correction stacking, reject a silent or nonpositive fit, and reject binary/manifest/audio changes during calibration.
