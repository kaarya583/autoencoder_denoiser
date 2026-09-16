# Operating the sealed external-evaluation preparer

[final_evaluation.py](../../esp32_denoiser/final_evaluation.py) implements the [frozen protocol](esp32_final_evaluation_protocol.md). It has no downloader, model loader, inference function, metric calculation, or training integration. The actual final archive and mixtures have **not** been opened or prepared by the fixture tests accompanying this implementation.

The operator must first finish selection and create a model-freeze JSON. Every declared file path is resolved relative to that JSON, and every file must be nonempty and match its lowercase SHA256. The preparer validates the entire declaration before reading the final speech archive. At least one artifact has role `model`, at least one has role `baseline`, and the paths must be distinct. Additional artifacts can bind source files, calibration records, DSP/gain metadata, and implementation bundles.

The freeze object has these fields:

| Field | Required value or structure |
|---|---|
| `version` | `1` |
| `selection_frozen` | `true` |
| `test_used_for_selection` | `false` |
| `plan_sha256` | SHA256 of the existing `sealed_external_test_plan.json` bytes |
| `artifacts` | Nonempty list of objects with `role`, `path`, and `sha256`; include model and baseline roles |
| `inference_specification` | Nonempty object recording each chosen architecture, checkpoint/binary, quantization grids, DSP, output gain, implementation, and comparison policy |
| `source_manifests` | Object with exactly the eight roles listed below; each value contains `path` and `sha256` |

Required source-manifest roles are `paired_train`, `paired_development`, `speech_train`, `speech_val`, `noise_train`, `noise_val`, `development_mixtures`, and `development_clean`. Paired manifests must use training-origin records; existing external suites must use development records. The noise holdout and both development-manifest hashes must also equal those already recorded by the sealed reservation plan. Additional training corpora would require explicitly extending and reviewing the preparer's exclusion schema before use; omitting such a corpus is not an acceptable freeze.

Freeze the selected float model, its integer counterpart, and comparison baselines together. Include the exact implementation and output-gain/calibration metadata as additional hashed artifacts rather than depending on mutable run directories. This declaration is an operator commitment backed by byte hashes; the preparer does not infer whether arbitrary bytes are a valid model or authenticate the author's historical training claims.

## Audit before rendering

After the freeze exists, supply an already downloaded official LibriSpeech `test-clean.tar.gz`. The expected publisher MD5 is fixed in the module at `32fa31d27d2e1cad72775fee3f4849a9`; there is no runtime checksum-override flag. The command below only audits file hashes, exclusions, and reserved noise metadata:

```sh
python -m esp32_denoiser.final_evaluation \
  --plan output/esp32/sealed_external_test_plan.json \
  --freeze /content/esp32_runs/final_freeze/model_freeze.json \
  --speech-archive /content/final_audio/archives/test-clean.tar.gz
```

These paths identify the intended future operational layout; this command has not been run on real final audio. Audit is the CLI default. A bad or empty model freeze is rejected before archive access. Every reserved MUSAN entry must match its exact held-out record, current file hash, sample count, sample rate and mono channel count. Reserved IDs/groups/content cannot overlap training sources or either development suite's original sources.

The audit reads paired source audio to obtain exclusion file hashes where original paired manifests have no hashes. This is a one-time potentially multi-GB read. Extra-source original hashes are read from the frozen manifests. Hash comparisons mean **source file bytes**, not a decoded-audio or cross-codec fingerprint; speaker/ID/group exclusion provides the complementary corpus-level check.

## Explicit preparation

Only after final selection and the successful audit, add `--prepare` and a new output directory:

```sh
python -m esp32_denoiser.final_evaluation \
  --plan output/esp32/sealed_external_test_plan.json \
  --freeze /content/esp32_runs/final_freeze/model_freeze.json \
  --speech-archive /content/final_audio/archives/test-clean.tar.gz \
  --prepare --output-dir /content/final_audio/sealed_external
```

Preparation extracts only canonical `LibriSpeech/test-clean/<speaker>/<chapter>/<utterance>.flac` recordings. It rejects path traversal, absolute paths, duplicate members, symbolic/hard links, special files, unexpected FLAC identities, oversized members, truncation, non-mono audio, and sample rates other than 16 kHz. Every extracted final speech speaker, original ID/group, and file hash is checked against the declared training/development exclusions.

The public recipe is fixed to seed 20260913, 200 base crops, five SNRs −5/0/5/10/20 dB, and 100 clean examples. Only recordings at least three seconds long are eligible, so every output contains 48,000 valid samples. Each mixture base uses the same speech/noise samples at all five SNRs; separate seeded namespaces choose clean examples, with duplicate exact speech crops prohibited across suites. A seeded permutation cycles through all 29 reserved noise recordings, making coverage explicit; short noise recordings are tiled. Activity detection and crop loading reuse the existing pure helpers. Noise scaling uses active reference frames within 20 dB of the maximum, and a shared peak attenuation at 0.99 preserves the pair's SNR. Clean and noisy references are deterministic IEEE FLOAT WAVs; the embedded evaluator performs its usual PCM16 conversion later.

All work is staged in a temporary sibling directory. Before publishing, the preparer rechecks the model freeze and every bound artifact/manifest, the sealed plan, the archive, and reserved noise bytes. A failure leaves the requested final output unpublished. Existing output directories are rejected rather than overwritten.

Published files include:

- `mixtures.jsonl`: 1,000 matched mixture conditions.
- `clean.jsonl`: 100 clean-input examples.
- `audio/`: deterministic reference and noisy WAVs.
- `speech_sources.jsonl` and `sources/speech/`: original final speech inventory and extracted FLACs.
- `model_freeze.json` and `sealed_plan.json`: exact declaration snapshots.
- `provenance.json`: manifest hashes, archive hashes, freeze hash, source counts, mixer rules, helper-source hashes, and an explicit declaration that no inference occurred.

All rendered records have `source_split="test"` and `split="test"`. Existing training-split and development-selector guards reject them. Preparation does not automatically score them; final inference remains a separate explicit operation after verifying the published hashes. The original reservation plan's status describes when it was created; the new preparation provenance describes the rendered corpus.

The 17 fixture tests cover freeze-before-archive ordering, artifact/plan/archive/noise corruption, development-used noise rejection, unsafe archive members, final speaker/ID/group/hash overlap, exact matched SNRs, deterministic FLOAT WAV bytes, RNG preservation, clean/mixed crop separation, training/selection rejection, full 1,000+100 orchestration, and failure to publish after a mid-preparation artifact change. They use a synthetic archive with a test-only patched checksum. A tiny renderer test writes real fixture WAVs; the full-count orchestration test substitutes a lightweight writer to avoid creating 422 MB of fixture audio. None supplies a real final-test score.
