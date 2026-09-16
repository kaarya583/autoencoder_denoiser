# Final evaluation protocol

Prepared 13 September 2026. The current search uses development data repeatedly;
its winning scores are not final-test estimates. The original project's older
test-selected checkpoint remains historical evidence only.

## Freeze before scoring

Before opening final audio to any model, record the chosen architecture,
checkpoint hashes, quantization calibration, packed binary hash, DSP constants,
any output-gain calibration, inference implementation, and comparison models.
The official VoiceBank test remains sealed throughout the current search.
Evaluate its full utterances once after this freeze, using the complete C PCM16
path for each deployable candidate and the same explicit alignment/metric rules.
Do not tune against the resulting scores.

## Separate unseen-source evaluation

The existing 500 LibriSpeech/MUSAN conditions are development data. To obtain an
independent generalization check, the [frozen source inventory](../../output/esp32/sealed_external_test_plan.json)
reserves **29 MUSAN noise recordings** that were absent from both frozen
development suites. Their original IDs, groups and content hashes are disjoint
from the noise used for development. They already belong to the held-out source
partition, so they are also excluded from broader training. Reservation used
only source membership; no audio quality or model output influenced it.

Combine these recordings with the official LibriSpeech **test-clean** partition
after final model selection. OpenSLR publishes its archive and checksum;
the planned archive MD5 is `32fa31d27d2e1cad72775fee3f4849a9`.
Verify its speakers, IDs and content against all current training/development
speech before rendering. [Official catalog](https://www.openslr.org/12/),
[archive checksums](https://openslr.trmal.net/resources/12/md5sum.txt).

The fixed plan is 200 base crops of three seconds each, with the same crop at
active SNRs −5, 0, 5, 10 and 20 dB, plus 100 separate clean-input examples.
The seed, source hashes and mixer settings must be committed to the final
manifest before inference. Reuse the established active-SNR definition and
shared peak scaling; preserve unquantized reference WAVs and let the embedded
interface perform its normal PCM16 conversion. Neither final speech audio nor
these mixtures has yet been downloaded, rendered or evaluated by this work.

## Report the limits of the evidence

Report equal-utterance SI-SDR improvement and the noisy baseline, PESQ/STOI on
their common valid cohorts, clean-input gain and normalized waveform error,
and output clipping. Keep VoiceBank and the new mixture corpus separate.
The 1,000 mixture conditions are 200 correlated base crops with only 29 original
noise recordings. Report results by SNR and original noise group; uncertainty
must acknowledge shared crops, speakers and recordings. More conditions do not
create more independent noise sources.

Freeze both float and integer comparison checkpoints together so quantization
loss is measured on identical signals. A final-test result cannot establish
on-board speed: physical ESP32-S3 hop timing, RAM peaks, I2S operation and
acoustic delay require their own measurements.
