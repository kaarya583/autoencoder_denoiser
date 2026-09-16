# Training a stronger sub-99 KB speech denoiser

## Recommendation

Use the remaining compute to test three sources of improvement separately: richer training mixtures, a better optimization objective, and knowledge distillation from a teacher with an auditable training history. Keep the deployed model-data limit at 99,000 bytes, keep the causal streaming contract, and rank candidates by complete C PCM16 validation SI-SDR improvement. An 8 dB result is an intermediate measurement, not a stopping criterion or a theoretical ceiling.

The most defensible immediate teacher is a newly trained, wider version of the spectral network using the permitted training speakers. DeepFilterNet3 and Demucs DNS64 are legitimate pretrained comparison models, but their public provenance does not establish every exclusion needed for the current speaker-and-noise holdout claim. MP-SENet's released best VoiceBank checkpoint has a more direct selection-provenance problem. These distinctions should affect which experiments can support an untouched-test claim, rather than preventing useful exploratory comparisons.

Start broad-data training with LibriSpeech `train-clean-100`, the non-speech portion of MUSAN, and optionally the RIR portion of OpenSLR28. This is practical on a 235 GB disk and avoids the roughly terabyte-scale unpacked DNS5 collection. Add larger speech or noise collections only when a smaller controlled experiment demonstrates a useful gain. The following plan proposes experiments; it does not assign a final quality score to the ongoing runs.

## Current experimental contract

The repository's present protocol uses 10,802 VoiceBank-DEMAND training utterances from 26 speakers and all 770 utterances from p226 and p287 for validation. Official test audio remains sealed. The student uses 16 kHz audio, a 512-sample analysis window, a 256-sample hop, a causal neural core, and integer neural deployment with external float32 DSP. The current data and metric implementations retain paired alignment, use full validation utterances, compute zero-mean SI-SDR, and count each valid utterance equally. These are project facts from [data.py](../../esp32_denoiser/data.py), [metrics.py](../../esp32_denoiser/metrics.py), and [embedded.py](../../esp32_denoiser/embedded.py).

The objective is

\[
\max_\theta\;\frac{1}{|V|}\sum_{u\in V}
\left[\operatorname{SI\!\!\text{-}\!SDR}(f^{\mathrm{C,PCM16}}_\theta(x_u),s_u)
-\operatorname{SI\!\!\text{-}\!SDR}(x_u,s_u)\right],
\]

subject to the byte budget, supported integer operations, causal state, and eventual device timing and memory limits. The complete PCM16 result is decisive because input conversion and output clipping can change quality. Float validation screens experiments efficiently; it does not replace final integer validation. SI-SDR remains the primary selection metric, while PESQ, STOI, clipping statistics, clean-speech preservation, and an external development suite reveal regressions that one average may hide.

## Pretrained teachers and provenance

| Candidate | Verified primary-source facts | Role in this project |
|---|---|---|
| **DeepFilterNet3** | The official implementation supports 48 kHz enhancement, a `DeepFilterNet3` checkpoint, state reset, and explicit delay compensation. Its paper reports training on DNS4 with PTDB and VCTK oversampled 10× and describes its VoiceBank/DEMAND test as unseen. Code is MIT/Apache-2.0 dual licensed.[^1][^2][^3] | Best permissive-code pretrained candidate to investigate. Do not assert that p226/p287 or every held-out noise recording was excluded without an actual checkpoint training manifest. Use an isolated environment and explicitly choose DFN3; some README examples still name DFN2. |
| **Demucs DNS64** | Official loader supplies a 16 kHz `dns64` checkpoint. DNS64 is distinguished from `master64`, which additionally uses Valentini/VoiceBank. The repository is CC-BY-NC 4.0.[^4][^5] | A convenient research teacher or reference at the student's sample rate. Preserve the noncommercial provenance; do not describe this as a permissive commercial teacher. DNS-only is not proof of unseen DEMAND noises. |
| **MP-SENet `g_best_vb`** | The official repository provides a best VoiceBank checkpoint. Its training code defaults validation to `VoiceBank+DEMAND/test.txt` and chooses the best checkpoint using validation PESQ. Code carries an MIT license.[^6][^7][^8] | Exclude the distributed best checkpoint from the strict untouched-test lineage unless independent evidence establishes a different selection set. A fresh MP-SENet trained with the project's train/validation manifests is a legitimate alternative, with higher integration cost. |
| **Fresh wider spectral teacher** | A project-controlled experiment can preserve the exact speech/noise exclusions, feature framing, and training objective. | Recommended strict-track teacher. Start around 0.5–2 million parameters; no MCU-size constraint applies to a training-only teacher. Require a demonstrated validation advantage before investing in KD. |
| **Earlier repository teacher selected using official test** | The recorded project history identifies test-based selection. | Retain only as a historical comparison. Distillation does not remove information transmitted by a checkpoint selected on test results. |

**There are two distinct contamination checks.** First, a teacher must not have optimized weights, hyperparameters, or checkpoint choice using the official test. Second, a claim of generalization to unseen speakers and noise recordings requires those sources to be absent from the teacher's training as well as the student's. A third-party statement that a test set is unseen is useful evidence, but it does not automatically establish exclusions for this project's custom p226/p287 validation split.

DNS 2020 uses LibriVox speech and includes additional noise from Freesound **and DEMAND**. Consequently, even the DNS-only Demucs checkpoint cannot be assumed to preserve VoiceBank's unseen-noise protocol. Later DNS collections include VCTK and mixed source licenses. For the strict track, exclude all added VCTK and DEMAND material until recording-level provenance can enforce the required exclusions.[^9][^10] This is a protocol decision, not a claim that every third-party checkpoint is contaminated.

For reproducibility, save the teacher's repository commit, checkpoint SHA-256, configuration, license files, training-source declaration, and adapter implementation. The exact published download locations are the [Demucs DNS64 file](https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th), [DFN3 archive](https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip), and [MP-SENet checkpoint directory](https://github.com/yxlu-0102/MP-SENet/tree/main/best_ckpt).[^1][^4][^6] Remote checkpoint byte sizes were not verified; measure them before download rather than treating parameter-count estimates as actual archive sizes.

## Data expansion within the runtime limits

| Data | Published download scale and license | Proposed use |
|---|---|---|
| **LibriSpeech train-clean-100** | 100 hours of 16 kHz read English; 6.3 GB compressed; CC BY 4.0. Official archive MD5: `2a93770f6d5c6c964bc36631d331a522`.[^11][^12] | First added speech source. Preserve FLAC on disk and decode crops; split by speaker before mixing. Avoid `train-other-500` as an initial clean target because its harder recordings need additional quality review. |
| **LibriSpeech dev-clean / dev-other** | Approximately 337 MB / 314 MB compressed; same corpus license.[^11] | Fixed external development mixtures, separate from the official VoiceBank test. These are development data once used for decisions. With external teachers, disclose possible LibriVox source overlap instead of claiming teacher-unseen speech. |
| **MUSAN** | Whole archive approximately 11 GB; corpus contains music, speech, and noise; CC BY 4.0.[^13] | Start with non-speech noise. Add selected music only as a declared condition. Hold out whole original recordings before cropping. Do not silently train suppression of competing speech: selecting a target speaker is a different task from generic denoising. |
| **OpenSLR28** | Approximately 1.3 GB; 16 kHz, 16-bit RIR/noise collection; Apache 2.0. Some point-source noise originates from MUSAN.[^14] | Use RIRs first. Split by room/source collection. If its noise is also used, group it with its MUSAN ancestors to prevent a cropped recording crossing splits. |
| **LibriSpeech train-clean-360** | 360 hours, approximately 23 GB compressed; CC BY 4.0.[^11] | Second expansion only after the 100-hour experiment earns more budget. Keep compressed audio and generate mixtures on demand. |
| **DNS5** | About 550 GB archived / 1 TB unpacked; README lists 58 GB noise, 5.9 GB RIR, and 827 GB clean speech. Component licenses differ.[^10] | Do not download the full collection into this runtime. A selected, attributable noise subset may be useful later; exclude DEMAND and retain per-file source/license/checksum records. |

A 16 kHz mono float32 stream consumes **0.2304 GB per hour**. Thus 100 hours of decoded clean speech alone consumes 23.04 GB; materializing clean, noisy, and teacher outputs for the same 100 hours consumes 69.12 GB before archives and checkpoints. These are arithmetic storage estimates, not download sizes. The efficient layout retains compressed source audio, mixes on demand, and caches only teacher outputs for a bounded deterministic mixture bank. A 25-hour teacher-output bank costs about 5.76 GB as float32, excluding its source audio.

Keep at least 40–50 GB free for extraction, artifacts, and interrupted downloads. Avoid decoding entire corpora into the 53 GB RAM: use crop reads or a bounded cache. Store mixture specifications containing source IDs, offsets, SNR, gain, RIR, random seed, and teacher checkpoint hash. A cache entry is valid only for that exact specification. Additional noise scaling or filtering after teacher prediction invalidates a cached target unless the same transformation is explicitly valid and applied to all corresponding signals.

## Mixing and augmentation priorities

The following probabilities and ranges are proposed initial settings, not published optima. Compare each change against the same model, initialization, optimizer-step budget, and validation set.

1. **Preserve the paired-data anchor.** Begin with 50% existing VoiceBank training pairs and 50% fresh synthetic mixtures. Test 75:25 and 25:75 only after measuring whether broader data improves external conditions or dilutes VoiceBank performance. Do not change all augmentation settings simultaneously.
2. **Sample meaningful noise levels.** Start dynamic mixtures at uniformly sampled -5 to +20 dB active-speech SNR; include approximately 10% very clean/+20 to +35 dB examples and 3–5% clean identity examples. Compute noise scaling over active speech or a documented segmental activity rule, so long silence and isolated impulses do not dominate RMS. Record realized SNR after common peak management. DNS explicitly motivates segmental active-region SNR for transient noises.[^9]
3. **Broaden recordings before broadening labels.** Hold out entire noise recordings and rooms, then crop them. Mix 10–20% of synthetic examples from two non-speech noises with independently sampled levels. An unseen crop of a training recording is not an unseen noise recording.
4. **Maintain exact paired alignment.** Apply the same sample offset, gain, sample-rate conversion, and declared channel filter to clean and noisy audio. For paired noise remixing, the residual is `noisy - clean`; audit its residual speech before using it as a general noise bank. Never resample clean and noisy with different filters or independently normalize their peaks.
5. **Add mild room variation separately.** Begin with 10–20% reverberant synthetic examples. For a denoising target, use `target = speech * room_speech` and `mixture = target + noise * room_noise`; preserve the speech-room response in the target. If removing late reverberation is desired, define a separate early-response target and preserve the direct-path timing. Demucs reported that indiscriminate dereverberation could introduce artifacts, which argues for a controlled target choice.[^15]
6. **Exercise the actual input path.** Use realistic level variation, a small proportion of PCM16-rounded inputs, mild channel-response variation, and occasionally clipped inputs with an explicitly defined target. Apply physical microphone coloration before constructing the desired target. Leave validation untouched. Evaluate low-level speech, clean speech, transients and output rails separately.
7. **Train useful temporal history.** Compare the current three-second crops with a short preceding context segment, excluding the context from the loss. This reduces the fraction of scored frames produced from artificial zero state. Use only preceding audio, and match this history for the teacher and student; extra context is a training expense, not added deployment lookahead.

The standard VoiceBank protocol holds out both test speakers and five DEMAND noise types. Adding all DEMAND recordings to a new noise bank would undermine that unseen-condition claim even if no noisy test mixture were downloaded.[^16] A strict additional-source allowlist is simpler than trying to infer unrecorded ancestry after mixing.

## Losses and distillation experiments

### Keep SI-SDR primary while testing spectral supervision

Retain the existing negative zero-mean SI-SDR term and its level-preserving waveform penalty as the reference. Compare it with a small compressed complex-spectrum or multiresolution STFT term computed from the **reconstructed waveform**, with padding masked and each utterance reduced separately. Proposed analysis windows are 256, 512, and 1024 samples; these are loss computations and do not alter the student's causal inference graph.

Use a small weight grid, such as three settings whose observed training gradients make the auxiliary term approximately 5%, 10%, or 25% of the SI-SDR gradient norm. Keep the coefficient fixed within each trial after that training-only calibration. This avoids assuming that numerical weights transfer between differently normalized losses. A compressed spectral exponent around 0.3–0.5 is a reasonable experimental range. Do not add a PESQ discriminator first: it increases complexity and selects a different optimization emphasis when the stated priority is SI-SDR. Demucs provides primary evidence that spectral supervision and simple augmentations can help, but its reported PESQ gains are not forecasts of this student's SI-SDR gain.[^15]

### Distillation is an experiment, not a guaranteed improvement

Tiny speech-enhancement research gives a useful caution. In a 2023 study, a roughly 60k-parameter baseline achieved 6.34 dB SDR improvement, ordinary output KD 6.35 dB, and two-stage feature KD followed by supervised training 6.77 dB. A 30k-parameter student improved from 4.42 to 5.52 dB. The paper used DNS data, PSA supervision and SDR—not this project's SI-SDR protocol—and trained for two million steps. Its evidence supports trying two-stage KD, not promising an extra decibel here.[^17]

A 2024 latent-mixup study also found that ordinary teacher-output matching could fail to improve students. Its approximately 93k-parameter CMGAN derivative increased PESQ from 3.10 to 3.18 with latent-mixup KD. This is evidence about training, not an ESP32 runtime result or proof that a 93k-parameter model fits a 99 KB complete export.[^16]

Run the following controlled progression:

| Trial | Training objective | What it isolates |
|---|---|---|
| S0 | Existing supervised objective | Reproducible reference for the selected architecture |
| S1 | Supervised objective plus one spectral auxiliary loss | Whether clean-target supervision is already sufficient to obtain the gain |
| K1 | Supervised objective plus a low-weight, stop-gradient teacher-output loss | Cheap test of response KD; reject it if it loses validation SI-SDR |
| K2 | First 20–25% of steps use teacher feature similarity; remaining steps use supervised objective only | A two-stage initialization strategy without permanently replacing clean ground truth |
| K3, conditional | Same-clean/different-noise latent mixing with a compatible teacher | A more expensive representation experiment after K2 demonstrates value |

A fresh wider teacher sharing the student's frame grid makes K2 practical: compare normalized temporal similarity matrices or train-only projection adapters at one or two corresponding layers. Avoid every-layer Gram matrices or large feature caches until profiling demonstrates affordability. Remove all adapters from export and verify that the deployed graph and binary size remain unchanged. A noncausal teacher is allowed during training, but its outputs must not become student inputs; the student's causal graph and future-perturbation tests remain mandatory. Cross-network noncausal-to-causal distillation has been studied, but it does not eliminate the student's information constraint.[^18]

Teacher predictions are imperfect targets. Keep clean supervision as the decisive objective and finish promising KD students with supervised fine-tuning. If the teacher is worse on a known training example, do not force imitation of its waveform; a training-only quality gate or lower KD weight is an explicit ablation, not a test-set filter. Do not choose the teacher or its postfilter by official-test PESQ or SI-SDR.

### Teacher waveform adapter requirements

For DFN3, resample the noisy input from 16 to 48 kHz, call the explicit model with delay compensation and postfilter disabled initially, then resample the output back with a declared deterministic filter. Check sample count, impulse timing, identity behavior where applicable, and waveform alignment against clean training references. The official Python implementation resets hidden state and trims its compensated output to the original input length.[^1] For Demucs, explicitly choose whether the teacher uses offline or streaming inference; its normalization differs between those modes.[^15]

Cache output only after every noisy-input augmentation for that mixture has been fixed. For crop-based generation, provide preceding context and any teacher-required future context, then extract the correctly aligned center target. The student receives only its permitted input history. Use float32 teacher targets so no accidental PCM clipping becomes supervision, and keep source IDs and teacher hashes with every cached item.

## Proposed compute allocation

Allocate the remaining authorized units rather than assuming all 100 are still available. If all 100 remain, the table below totals 100 units. The approximately 65 L4-hours interpretation is a session planning assumption—0.65 hours per unit—not a guaranteed Colab price or runtime entitlement. Recalculate from the active runtime's displayed consumption and measured throughput before each phase. Scale the table to remaining units while retaining an evaluation/recovery reserve.

| Phase | Units | Approximate hours at the planning rate | Decision rule |
|---|---:|---:|---|
| Objective/augmentation screens | 12 | 7.8 | Compare 4–6 short, matched-step variants; continue only the strongest two |
| Broader-data experiments | 22 | 14.3 | Anchor-plus-synthetic mixture, then one speech-scale increase if useful |
| Teacher and adapters | 17 | 11.1 | Train one auditable teacher; benchmark pretrained references only within their declared provenance track |
| Distillation | 18 | 11.7 | Compare K1/K2 against equally trained S1; attempt K3 only if justified |
| Repeat finalists and QAT | 18 | 11.7 | At least two seeds for close contenders; QAT the best deployable candidates |
| Full integer validation and final evaluation | 8 | 5.2 | Complete C PCM16, paired metric denominators, optional perceptual scores and artifacts |
| Recovery and packaging reserve | 5 | 3.3 | Interruptions, export fixes, reproducibility bundle; no idle spending requirement |

Use optimizer steps and presented audio hours to compare runs. An epoch over 100 hours of new speech is not comparable with an epoch over the current 26-speaker corpus. A practical first screen is 5,000–10,000 optimizer steps per setting, followed by 20,000–50,000 steps for survivors, adjusted to measured cost. Validate the same fixed full set periodically; do not select using a convenient changing subset.

Before using much of the teacher allocation, measure whether the current mask parameterization has a low ceiling: construct a validation-only oracle complex mask under the actual mask bounds, reconstruct through the same analysis/synthesis path, and score it. This is an analytical diagnostic, not an achievable model score. If the bounded oracle is poor, extra training cannot solve the restriction. If the oracle is strong but a much wider teacher remains weak, investigate features, loss and optimization before spending on KD. If the teacher clearly improves while the student stalls, capacity-aware distillation becomes more compelling.

Do not stop at a chosen absolute dB score. Promote changes that improve the measured deployed frontier, and stop an unproductive branch when equal-budget validation and repeat seeds show no gain. A difference of a few hundredths of a decibel from one seed is weak evidence. Report per-speaker results and paired utterance deltas; a confidence interval over 770 utterances does not imply broad population coverage when only two validation speakers are represented.

## Evaluation and claims after broader training

Keep the existing 770-utterance validation set for continuity and add a fixed external development suite with separately held-out speakers, original noise recordings, rooms and SNR strata. A useful proposed suite has roughly 1,000 mixtures spanning stationary noise, transients, music, mild reverb and low-level speech. Report its categories independently of VoiceBank; do not average it into the original benchmark and retain the same label. Also retain a small fixed clean-speech suite to detect unnecessary distortion.

Teacher and student training must share the allowed-source manifest. Label pretrained-teacher experiments separately when source exclusions cannot be verified. A previous test-selected teacher cannot enter the strict lineage through feature targets, waveform pseudo-labels or initialization. Once all architecture, augmentation, teacher, checkpoint and quantization choices are finished, freeze their hashes and score the official test once for the selected comparison.

Final reports should state the added training datasets and hours, teacher provenance, model bytes including metadata/DSP constants, exact PCM16 processing path, SI-SDR definition and denominators, and host-versus-board timing distinction. If PESQ/STOI reject different utterances across models, compute comparative means on common valid IDs. A model trained with broader speech is a valid stronger system; describe it as trained with additional data rather than comparing it as an identical-data reproduction.

## Sources

Primary sources were checked for this report in September 2026. Repository pages are mutable; pin commits and record artifact hashes when implementing a run.

[^1]: Hendrik Schröter and contributors. [DeepFilterNet `df/enhance.py`](https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/enhance.py). Official model names, defaults, model download path, state reset and delay compensation.
[^2]: Hendrik Schröter, Tobias Rosenkranz, Alberto N. Escalante-B. and Andreas Maier. [DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement](https://arxiv.org/html/2305.08227v1). Interspeech 2023, especially §4 training provenance and evaluation.
[^3]: DeepFilterNet contributors. [Official repository licensing section](https://github.com/Rikorose/DeepFilterNet#license). MIT/Apache-2.0 code licensing; archive-specific notices should also be retained.
[^4]: Facebook Research. [Denoiser pretrained model loader](https://github.com/facebookresearch/denoiser/blob/main/denoiser/pretrained.py). DNS48, DNS64, master64 and Valentini checkpoint URLs and model constructors.
[^5]: Facebook Research. [Denoiser README](https://github.com/facebookresearch/denoiser#readme) and [LICENSE](https://github.com/facebookresearch/denoiser/blob/main/LICENSE). Distinction between DNS-only and DNS-plus-Valentini models; CC-BY-NC 4.0 repository terms.
[^6]: Ye-Xin Lu, Yang Ai and Zhen-Hua Ling. [MP-SENet official README](https://github.com/yxlu-0102/MP-SENet#readme). 2023 architecture, pretrained VoiceBank checkpoint and inference instructions; expanded journal paper published 2025.
[^7]: MP-SENet authors. [Official `train.py`](https://github.com/yxlu-0102/MP-SENet/blob/main/train.py). `--input_validation_file` defaults to `VoiceBank+DEMAND/test.txt`; `g_best` selected by validation PESQ.
[^8]: Ye-Xin Lu. [MP-SENet LICENSE](https://github.com/yxlu-0102/MP-SENet/blob/main/LICENSE). MIT, 2023.
[^9]: Chandan K. A. Reddy and colleagues. [The Interspeech 2020 Deep Noise Suppression Challenge: Datasets, Subjective Testing Framework, and Challenge Results](https://arxiv.org/pdf/2005.13981). 2020, §§2.1–2.3: speech sources, DEMAND noise inclusion and active-region mixing.
[^10]: Microsoft. [DNS Challenge official README](https://github.com/microsoft/DNS-Challenge/blob/master/README.md). DNS5 data sizes, component sources, licensing and SHA1 manifest; README identifies the ICASSP 2023 challenge.
[^11]: Vassil Panayotov and Daniel Povey. [OpenSLR12: LibriSpeech](https://www.openslr.org/12/). 2015 corpus; official subset sizes, 16 kHz format and CC BY 4.0 declaration.
[^12]: OpenSLR. [LibriSpeech archive MD5 checksums](https://www.openslr.org/resources/12/md5sum.txt).
[^13]: David Snyder, Guoguo Chen and Daniel Povey. [OpenSLR17: MUSAN](https://www.openslr.org/17/). 2015 corpus; download size, contents and CC BY 4.0 declaration.
[^14]: OpenSLR. [OpenSLR28: Room Impulse Response and Noise Database](https://www.openslr.org/28/). ICASSP 2017 augmentation dataset; format, Apache 2.0 declaration and MUSAN ancestry.
[^15]: Alexandre Défossez, Gabriel Synnaeve and Yossi Adi. [Real Time Speech Enhancement in the Waveform Domain](https://arxiv.org/pdf/2006.12847). Interspeech 2020, §§2.3, 3.1–3.3: losses, augmentation, streaming normalization and reverberation findings.
[^16]: Behnam Gholami, Mostafa El-Khamy and KeeBong Song. [Knowledge Distillation for Tiny Speech Enhancement with Latent Feature Augmentation](https://www.isca-archive.org/interspeech_2024/gholami24_interspeech.pdf). Interspeech 2024, pp. 652–656; same-clean/different-noise latent mixing, VoiceBank protocol and measured student results.
[^17]: Rayan Daod Nathoo, Mikolaj Kegler and Marko Stamenovic. [Two-Step Knowledge Distillation for Tiny Speech Enhancement](https://arxiv.org/html/2309.08144v1). September 2023, §§2.4–3.3; training budget and Tables 1, 2 and 4. Reported SDR is not treated as project SI-SDR.
[^18]: Hyun Joon Park, Wooseok Shin, Jin Sob Kim and Sung Won Han. [Leveraging Non-Causal Knowledge via Cross-Network Knowledge Distillation for Real-Time Speech Enhancement](https://pure.korea.ac.kr/en/publications/leveraging-non-causal-knowledge-via-cross-network-knowledge-disti/). IEEE Signal Processing Letters 31, 1129–1133, 2024; DOI 10.1109/LSP.2024.3388956.
