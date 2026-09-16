# Matched distillation from a fresh broad teacher

Prepared 13 September 2026. This is a planned controlled experiment, not a claim that distillation has improved the deployed model. The broader teacher must pass the source-lineage checks before calibration or training. The live clean-input audit found substantial teacher amplification: **do not launch raw broad response KD**. Fit the training-only response correction below first.

## Fixed inputs and control

Freeze the current broad student's **external-development selected checkpoint** and the fresh broad spectral teacher's **primary-validation `best.pt`** into:

- `/content/esp32_runs/broad_kd_inputs/student.pt`
- `/content/esp32_runs/broad_kd_inputs/teacher.pt`

Read the student's `external_development/best.json` once, resolve its immutable `selected_checkpoint`, and verify `checkpoint_sha256` against the file before and after copying. Preserve the pointer JSON alongside the frozen student: it identifies the external development manifest, selection score, and exact checkpoint epoch. The broad run's primary `best.pt` can still contain its original narrow-domain initialization and must not be substituted for the external winner. Record source/destination paths and SHA-256 for both models. For the teacher, preserve the corresponding primary selection manifest and `best_validation.json`; these are separate writes, so verify their scores against the checkpoint or snapshot after its source run finishes. Keep both snapshots fixed throughout the matched runs and preserve them in backups. The teacher must be fresh supervised training with `resume=None`, no pretrained/upstream teacher, and an explicit exclusion of final-test selection.

The checked-in [no-KD control](../../configs/esp32_zero_bias_broad_distillation_control_float.json) resumes the frozen student with a restarted optimizer. It uses the existing paired data plus 50% dynamically generated LibriSpeech/MUSAN examples, 10,802 samples per epoch, 3-second crops, batch 32, learning rate 0.0001 with a 0.00002 floor, a two-hour/40-epoch ceiling, and patience 40. This avoids ending the broader-data screen early just because its primary score remains below the narrow initialization. Its objective remains SI-SDR plus normalized waveform L1 at coefficient 0.1. This isolates KD from broader data, initialization, level-loss, and spectral-loss changes.

Both runs use seed 2026 by default, identical augmentation and crop policy, and identical primary and external development sets. The teacher is only called on training batches. All neural widths, output masks, feature/DSP constants, and eventual INT8 export contracts remain unchanged.

## Source-lineage admission

`FrozenTeacher` now accepts two explicit tracks: the existing fresh paired-only teacher, and a fresh teacher trained by the repository's known `DynamicMixtureDataset`/`HybridTrainingDataset` recipe. Broad-teacher checks require:

- Both configured synthetic training manifest paths, recognized source/asset metadata, and the `train` partition.
- Exact embedded training-manifest SHA-256 and source counts; matching synthetic sampling probability and samples per epoch; known configuration fields.
- Adjacent `speech_val.jsonl`/`noise_val.jsonl`, with matching preparation-version, sample-rate, per-partition SHA-256, counts, and groups.
- Disjoint train/validation source IDs, original speaker/recording groups, resolved paths, and declared source-content hashes, plus no synthetic-training overlap with the paired teacher/student selection sets.
- Checkpoint `source_sha256` values for `extra_data.py` and `mixtures.py` matching the reviewed current mixer implementation. The saved `train.py` hash is required and recorded; unrelated subsequent loss/guard edits do not have to leave the whole file unchanged.

The resulting teacher provenance includes the configured source paths, resolved partition paths and hashes, preparation-sidecar hashes, fixed recipe values, and source-code verification scope. Extra held-out speech speakers also join the forbidden KD-calibration cohort.

These checks do not rehash every source audio file or reconstruct stochastic training mixtures. Source-content hashes are preparation-time records. The original checkpoint does not embed extra validation partition hashes; their current consistency is established against adjacent preparation provenance and recorded during this audit. The original trainer's source hash identifies historical code for immutable-bundle review; configuration records alone cannot prove arbitrary third-party code executed that recipe.

The teacher uses its original primary-validation winner; the student uses its explicitly documented external-development winner. An externally selected teacher or a teacher with other selection history needs a separate explicit selection-provenance review. Renaming an arbitrary checkpoint does not establish that history; preserve each snapshot's source and selection evidence.

## Calibrate before creating the KD configuration

First fit one response gain from the teacher's already-audited **clean training** manifests:

```sh
python -m esp32_denoiser.teacher_gain \
  --teacher /content/esp32_runs/broad_kd_inputs/teacher.pt \
  --validation-manifest /content/voicebank/manifests/val.jsonl \
  --output /content/esp32_runs/broad_kd_inputs/teacher_gain.json \
  --device cuda --examples-per-source 64 --batch-size 8 --crop-seconds 3 --seed 2026
```

The validation manifest is used only to reject selection leakage during lineage verification; its audio is never supplied to this fit. The calibrator selects distinct paired-clean and LibriSpeech training recordings, hashes the actual selected audio files, and records exact crop offsets, lengths, seed, source-manifest hashes, and the frozen teacher SHA. It minimizes equal-example reference-RMS-normalized squared error with one positive scalar. No noise, augmentation, or clipping enters this fit. It changes neither teacher weights nor student inference.

Load the corrected teacher with `FrozenTeacher(teacher_path, validation_path, "cuda", gain_calibration=gain_path)`. Loading verifies the fit, model/source identity, selected training records, and audio hashes. Compare its corrected clean-input gain/L1/clipping on the frozen development set; do not tune the fitted scalar there. A global scalar can correct overall level but cannot fix a noise-dependent or frequency-dependent teacher error.

Only then instantiate the frozen student with `load_checkpoint` and the same paired/dynamic/hybrid training datasets used by the control. Use eight training-only batches and `calibrate_distillation_weight` with `target_ratio=0.1` **against the corrected teacher**. The resulting coefficient record includes the correction file SHA and output gain. The primary-loss callback must match the control, for example:

```python
from functools import partial
primary = partial(speech_loss, waveform_loss_weight=control["waveform_loss_weight"])
calibration = calibrate_distillation_weight(
    student, teacher, training_batches, primary,
    target_ratio=0.1, max_batches=8,
)
```

Retain batch gradient norms, gradient cosine diagnostics, teacher SHA-256, student snapshot SHA-256, dataset hashes, seed, and the resulting coefficient. Calibration temporarily uses evaluation mode and frozen full-precision teacher responses. Its coefficient balances local training gradient scales; it is not a measured optimum. Do not reuse the earlier paired-only teacher's coefficient.

Generate the KD twin only after that calibration has produced a finite positive coefficient. For example, with `broad_distillation_calibration.json` saved alongside the frozen inputs:

```python
import hashlib
import json
import math
from pathlib import Path

control_path = Path("configs/esp32_zero_bias_broad_distillation_control_float.json")
inputs = Path("/content/esp32_runs/broad_kd_inputs")
teacher_path = inputs / "teacher.pt"
gain_path = inputs / "teacher_gain.json"
calibration = json.loads((inputs / "broad_distillation_calibration.json").read_text())
assert calibration["teacher_checkpoint_sha256"] == hashlib.sha256(teacher_path.read_bytes()).hexdigest()
assert calibration["teacher_gain_calibration_sha256"] == hashlib.sha256(gain_path.read_bytes()).hexdigest()
assert calibration["teacher_output_gain"] == json.loads(gain_path.read_text())["output_gain"]
weight = calibration["distillation_weight"]
assert math.isfinite(weight) and weight > 0
settings = json.loads(control_path.read_text())
settings.update(
    output_dir="/content/esp32_runs/float_zero_bias_broad_distillation",
    teacher_checkpoint=str(teacher_path),
    teacher_gain_calibration=str(gain_path),
    distillation_weight=weight,
)
Path("configs/esp32_zero_bias_broad_distillation_float.json").write_text(
    json.dumps(settings, indent=2) + "\n"
)
```

The generated configuration changes exactly the output directory, teacher path, teacher response-correction file, and calibrated KD coefficient. Gating remains disabled for this matched screen. A subsequent gate or level-loss change is a separate experiment. Optimizer resume verifies both the teacher checkpoint hash and correction-file hash; modifying either requires a new matched run.

## Selection and acceptance

Run the same external-development companion for both trials so broader gains are not lost solely because the primary VoiceBank score decreases. Keep the official test sealed. Report primary and external SI-SDRi/PESQ/STOI, clean-input projection gain and waveform error, and PCM clipping. Check teacher level fidelity before interpreting KD as a quality improvement: response distillation can also transfer an unwanted gain bias.

Use [paired comparisons](../../esp32_denoiser/comparison.py) with the same IDs and noisy baselines. For external mixture comparisons, pass the frozen development manifest and `cluster_key="base_crop"`; its 100 crops each have five correlated SNR conditions. The resulting interval is conditional on the observed crops/speakers/recordings and does not establish population performance for new speakers or noise recordings.

Only candidates that improve the relevant quality criteria should proceed to matched QAT, packed export, and complete C PCM16 evaluation. Float or fake-quantized gains do not establish deployed quality, and a successful firmware compile does not establish physical-board timing.
