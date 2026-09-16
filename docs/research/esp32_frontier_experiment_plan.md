# ESP32 frontier experiment contract

The user authorized up to **100 Colab compute units** for this project. Eight decibels is not a performance ceiling or a stopping rule. We select the best measured quality under the deployment constraints, while terminating trials that do not justify their compute. Additional unit purchases have not been authorized.

## Deployment constraints

| Quantity | Working constraint | Evidence status |
|---|---|---|
| Primary target | ESP32-S3, 240 MHz, no required PSRAM | Target assumption; no physical board attached |
| Model payload | At most 99,000 bytes, including integer parameters, quantization metadata and DSP constants | Enforced on actual exported binaries |
| Compression reference | 11,854,856 raw FP32 parameter bytes in the earlier 2,963,714-parameter teacher | Payload comparison; executable firmware is additional storage |
| Neural arithmetic | INT8 weights, activations and persistent neural state; INT32 biases/accumulators | Baseline C implementation verified; each new graph requires its own verification |
| Audio DSP | Float32 FFT, features and reconstruction outside the integer neural graph | Explicit deployment precision, not an all-integer audio pipeline claim |
| Working memory | Aim below 200 KiB for model copy, neural workspace, DSP and working buffers | Must account for the actual linked application and runtime allocation |
| Neural compute | Explore approximately 5–30 million MACs per second | Screening range, not a measured ESP32 throughput guarantee |
| Audio | Mono, 16 kHz, FFT 512, hop 256 | Identical framing across controls; no neural future frames |
| Real time | Every 16 ms audio hop processed before its deadline, with headroom | Requires a physical board; host RTF and successful compilation are insufficient |

The 99,000-byte limit represents a 119.75× reduction relative to the historical raw FP32 parameter payload. INT8 alone contributes at most 4× for the same parameters; the rest must come from architecture and parameter reduction. The complete firmware application, bootloader and partition table are reported separately.

## Experiments and promotion

1. **Measured controls.** Retain the signed spectral TCN, its deeper variant, and the zero-depthwise-bias initialization trial. Compare float, QAT, actual C neural inference and complete C PCM16 audio. These establish the reference frontier and deployment pipeline.
2. **Frequency sharing.** Train a full-bin, convolution-only frequency U-Net with a global temporal branch. Test both the approximately 27.6 MMAC/s configuration and a smaller configuration. The default has 83,170 learned parameters and a measured 94,300-byte packed model. Its C neural workspace is 34,844 bytes; an ESP32-S3 firmware build passes, while board runtime remains unmeasured. A matched encoder-BatchNorm trial folds the normalization away for deployment. A separate shared-GRU variant is currently float-only until integer recurrence is implemented and verified.
3. **Objective ablation.** Compare direct SI-SDR supervision against a small multiresolution compressed-spectrum term. Calibrate a starting coefficient from training-only gradient norms, then hold it fixed. Match initialization, total optimizer steps, augmentation and validation when attributing a gain to the loss.
4. **Broader mixtures.** Retain a paired VoiceBank training anchor while adding LibriSpeech speech and MUSAN non-speech noise. Split speakers, original noise recordings and optional simulated rooms before generating crops. Report external development conditions separately.
5. **Auditable teacher and distillation.** Train a wider teacher on the permitted sources. Require a measured teacher advantage before promoting KD. Preserve the teacher's lineage and checkpoint hash; earlier test-selected checkpoints do not enter the strict experiment lineage.
6. **Quantized selection.** Only complete C PCM16 validation can establish the winning deployable model. A promising float result is provisional until its graph, binary, numerical agreement and memory use are validated. Follow with repeat seeds and matched clean-speech/perceptual checks for close candidates.
7. **Frozen final evaluation.** Freeze architecture, sources, training choices, quantization and checkpoint hashes before preparing the official test set. The [final-evaluation protocol](esp32_final_evaluation_protocol.md) also reserves an independent source combination for generalization testing. Compare the frozen systems once; no test-based architecture or checkpoint selection is permitted.

Target-informed spectral mask diagnostics inspect whether the current gain bounds are a serious restriction. They use clean development targets and cannot be deployed. Their scores are neither attainable-model forecasts nor rigorous SI-SDR upper bounds after overlap-add.

### Development selection and generalization

The primary ranking remains equal-utterance SI-SDR improvement over all 770 VoiceBank development clips. The additional LibriSpeech/MUSAN suite is a separate generalization screen: 100 held-out source crops repeated at five fixed SNRs (-5, 0, 5, 10, 20 dB), plus 100 separate clean-speech clips. Report the five SNR cohorts separately and the overall mean; repeated SNR versions of one crop are correlated, so confidence intervals must resample whole crop groups. Do not blend this suite into the primary score or present it as an unseen final test after using it for development decisions.

Retain a Pareto comparison if broader training improves unseen-source performance while reducing VoiceBank quality. A stronger VoiceBank result alone does not establish a stronger general-purpose denoiser. Report clean-speech enhanced SI-SDR, projection gain, and amplitude-preserving error rather than interpreting its difference from the implementation's capped 80 dB identity baseline. PESQ, STOI, clipping and listening checks remain necessary companion evidence. No aggregate weighting or clean/perceptual acceptance threshold has been empirically established yet.

## Budget and recovery

The active L4 session initially displayed approximately 1.54 units per hour. That is an observed rate, not a guaranteed price or session duration. Track the remaining balance in Colab and preserve an evaluation/recovery reserve. Favor successive promotion over training every configuration to the same long schedule. An authorized ceiling does not require spending units on failed branches.

Each run writes its configuration, validation records, best and last checkpoints, training history and provenance. New checkpoints also contain source-file and manifest hashes. Completed artifacts are downloaded locally and copied into the user's private **ESP32 Frontier Experiments** Drive folder. The initial development backup includes the complete baseline C result and learned checkpoints. Do not depend on ephemeral Colab storage as the only copy.

## Research basis and novelty standard

Frequency-shared convolution, depthwise convolution, U-Nets, temporal convolutions, QAT, distillation and deep filtering already have prior art. The current proposal is a project design hypothesis. A novelty claim requires a specific distinction, controlled ablations and a tangible quality/compute/memory benefit; an architecture name is not evidence.

The detailed architecture and systems reviews accompany the [training research](esp32_frontier_training.md). The [UL-UNAS paper](https://arxiv.org/html/2503.00340v2) motivates efficient frequency processing but includes recurrent and attention components whose CPU complexity does not establish full-INT8 ESP32 feasibility. Its published quality metrics must not be compared directly with this project's custom speaker-held-out validation SI-SDR improvement.
