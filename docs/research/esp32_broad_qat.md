# Promoting the broader-data student through QAT

The next deployment candidate must start from the **immutable float checkpoint selected on external development**, retain the broader training distribution during calibration and QAT, and finish with evaluation of its exact C PCM16 deployment path. The primary `best.pt` is selected on the 770-utterance VoiceBank holdout; after broader continuation it can still contain the original narrow-data initialization. It must not silently replace the externally selected checkpoint.

This is a promotion procedure, not a completed quality claim. The [QAT configuration](../../configs/esp32_zero_bias_broad_qat.json) targets the existing signed SpectralTCN with the zero-depthwise-bias initialization recipe. It does not enable the experimental GRU or GTCRN integer paths.

## Freeze the float source before starting QAT

Read the chosen float run's `external_development/best.json` once. Its `selected_checkpoint` identifies an immutable `selected_<sha256>.pt`; `checkpoint_sha256` binds the exact bytes, and `manifest_sha256` binds the development cohort. The `source_checkpoint` field is informational and may refer to a mutable `last.pt`; never use it as the promoted source. [Selector implementation](../../esp32_denoiser/development_checkpoints.py).

Verify the selected file's SHA256, float phase, model kind/configuration, and recorded source provenance. Copy those exact bytes to `/content/esp32_runs/broad_qat_inputs/student.pt`, refusing to overwrite an existing different artifact. Preserve a frozen copy of the selection JSON, its own SHA256, the selected checkpoint SHA256, the originating run, the selection metric/cohort, and any continuation lineage. Check that the destination hash equals the selected source hash. The live selector may continue improving its pointer; this QAT run must retain its frozen input.

Choose the source using the declared external-development rule together with the established primary and clean-preservation checks. If several float continuations or distillation variants are being compared, record that choice before QAT. Do not assume a later epoch, the best primary score, or the best-looking listening sample identifies the same checkpoint.

## Calibrate on the hybrid training loader

The current training sequence is correct: create paired training data, construct `DynamicMixtureDataset` from the training-only LibriSpeech/MUSAN manifests, wrap both in `HybridTrainingDataset`, then build the training loader. Float-to-QAT calibration consumes at most 32 batches from that loader before configuring quantization and creating the fresh optimizer. The hybrid probability is 0.5; with 32-sample batches, calibration sees up to 1,024 sampled crops, not every training recording. The observed proportions vary by the seeded sampling process. [Training implementation](../../esp32_denoiser/train.py), [hybrid sampler](../../esp32_denoiser/mixtures.py).

Calibration observes bounded hidden activations and the pointwise residual branches before addition. The chosen exponent is a shared hidden grid; it is stored in the calibration record and quantization buffers. The checkpoint's legacy `calibration.source="training manifest only"` label refers to the loader source and does not describe the mixture by itself: also retain `provenance.added_training_sources`, including both source-manifest hashes, recording counts, probability, and epoch length. [Calibration implementation](../../esp32_denoiser/quantization.py).

The configured QAT run uses:

| Setting | Value |
|---|---|
| Frozen source | `/content/esp32_runs/broad_qat_inputs/student.pt` |
| Phase / optimizer | QAT / fresh optimizer, `resume_optimizer=false` |
| Training mixture | 50% paired data, 50% synthetic draws in expectation |
| Synthetic sources | Existing `speech_train.jsonl` and `noise_train.jsonl` |
| Samples per epoch | 10,802; 338 batches at batch size 32, including the final partial batch |
| Crop / AMP | Three seconds / disabled |
| Schedule | Up to 40 epochs, patience 40, learning rate 1e-4, minimum 2e-5 |
| Loss | Existing supervised SI-SDR plus 0.1 level-sensitive waveform term; no added spectral-loss weight |
| Teacher | None; response-distillation weight 0 |

The no-teacher setting is deliberate for this first QAT promotion. A teacher-backed QAT ablation would need its own fixed configuration, reviewed lineage, and matched control. Neither paired validation nor external development supplies calibration or optimization batches.

The [new regression](../../tests/test_esp32_broad_qat.py) runs real float training followed by real QAT. During calibration it observes every actual waveform entering the floating model, checks exact equality with samples emitted by the paired/synthetic datasets, confirms that both sources contribute among 64 crops / 32 batches, and rejects accidental validation access. It then loads and resumes QAT, verifying that every saved activation exponent survives and calibration is not repeated. This checks the integration, rather than just the hybrid sampler in isolation.

## Preserve the QAT winner and export its saved grids

Run the existing external-development selector against the QAT run, with the **same frozen development manifest** used for the float selection. It evaluates the checkpoint's active fake-quantized graph, including its stored grids. Retain a separate QAT `external_development/best.json` and immutable selected checkpoint; the selector does not change the primary winner.

Once the QAT candidate is selected, freeze its checkpoint and sidecar exactly as for the float source. Load that exact checkpoint through `evaluate.load_checkpoint`, which constructs the quantized modules before strictly loading their saved buffers. Export the loaded model with `export.export_model(..., max_bytes=99000)`. Do not initialize a fresh model, run a new calibration, or reconfigure default exponents after loading the chosen weights. [Checkpoint loader](../../esp32_denoiser/evaluate.py), [integer exporter](../../esp32_denoiser/export.py).

Record the chosen float and QAT checkpoint hashes, selection-sidecar hashes, complete binary SHA256/byte count, input/hidden/output exponents, source hashes, and manifest hashes together. The existing architecture typically packs to 94,480 bytes, but acceptance uses the **actual selected binary's size**, including its constants/metadata. A checkpoint pickle size or INT8 weight-array count is insufficient. INT8 conversion of an unchanged FP32 weight array is approximately 4×; the project's roughly 125× comparison additionally includes architectural downsizing and its explicitly defined baseline.

## Evaluate complete, paired development cohorts

Evaluate all three representations on identical unaugmented manifests, retaining every per-utterance record:

| Cohort | Required records | Purpose |
|---|---:|---|
| Primary training-origin holdout | 770 | Compare with established narrow-data controls |
| Frozen external development mixtures | 500 | Unseen held-out speech/noise groups and the fixed SNR sweep |
| Frozen clean-preservation suite | 100 | Detect level changes, speech damage, and clipping |

The three representations are the frozen float source, the selected QAT checkpoint, and the actual exported binary through **full C DSP plus INT8 neural inference plus PCM16 I/O**. Use `python -m esp32_denoiser.evaluate --checkpoint ...` for each checkpoint and `python -m esp32_denoiser.embedded --integer-model ... --io-format pcm16` for deployment evaluation. In these command forms, the checkpoint/model argument is the frozen artifact and the required `--manifest`/`--output` arguments identify the concrete cohort/report. `evaluate --integer-model` exercises C neural inference with Torch DSP; it is useful for isolation but does not substitute for the full C frontend. `embedded --compare-reference` additionally isolates Torch-versus-C DSP and unclipped-versus-PCM16 behavior. [Full frontend evaluator](../../esp32_denoiser/embedded.py).

Use equal-utterance SI-SDRi, PESQ/STOI when available, and the existing level/preservation diagnostics. For clean input, SI-SDRi against the capped near-perfect noisy baseline is not the main preservation statistic; inspect enhanced SI-SDR, projection gain, normalized waveform error, and clipping. Retain PCM16 I/O statistics and count invalid/missing perceptual scores explicitly. Host processing RTF remains a host result, not ESP32 latency.

Compare float→QAT and QAT→C **by exact utterance IDs and matched baselines**, using [comparison.py](../../esp32_denoiser/comparison.py). For the external SNR sweep, pass its manifest and `cluster_key="base_crop"` so repeated SNR conditions of one crop stay together during bootstrap resampling. Report point differences and intervals on these development cohorts, without implying independent unseen-speaker replication.

Before promotion, require exact NumPy/C integer state/output parity on representative features, reset/chunk/edge tests, complete PCM16 waveform checks, the actual payload at or below 99,000 bytes, and preservation of the broader-development advantage. Investigate material float→QAT loss, C-only loss, gain collapse, or clipping rather than selecting a checkpoint by an aggregate SI-SDRi score alone. Freeze the final model, grids, frontend, and selection rule before opening the official test. Actual S3 full-hop timing and RAM measurements remain separate deployment gates.
