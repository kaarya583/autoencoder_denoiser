# ESP32 training continuation — 14 September 2026

The replacement L4 runtime was observed with 74.2 compute units remaining, at approximately 1.54 units/hour. The complete backup chain through `esp32_20260914T040022Z_33aac2b4.zip` restored 594 files from 38 archives. Every member and the final inventory passed SHA-256 verification before atomic installation. Source release `eceb96444d0988d487f7a087190b5baf5437d54f949190dbf0ef67ec1e6a64d3` remains unchanged.

The continuation preserves the two large runs' saved optimizer/scheduler/scaler state and recorded augmentation recipe. They resume from raw epoch158 and normalized epoch151, targeting240 with at most five additional training hours per branch. Existing patience60 remains effective, so completion may be earlier. The checkpoint selectors retain their previous best candidates and explicitly verify that the final checkpoint was scored.

Each selected float winner receives 128 training-crop MSE calibration, a packed INT8 export, and full C PCM16 development evaluation on external500, clean100, and primary770, including PESQ/STOI. Both exports must complete before the next QAT stage.

The next six-hour QAT run starts from the better of those two new INT8 exports: batch16, up to80 rounds, 250 steps/4,000 examples per round, LR3e-5 with5e-6 floor, and the same waveform-loss weight0.1. It preserves optimizer checkpoints and immutable evaluated candidates. The final champion is chosen among the historical9.20221dB artifact, the stronger new plain INT8 export, and the new QAT candidate, with identical input hashes and strict improvement required to replace an incumbent.

The notebook remains attached to the actual supervisor and streams saved training progress. GPU jobs run independently of browser rendering. Immutable Drive backups continue every15minutes, and completion requires final evaluations plus a verified backup. No official test data is read. Physical ESP32 timing remains unmeasured.

Control scripts are in `output/esp32/continue_colab_20260914.py`, `resume_large_20260914.py`, `post_large_qat_20260914.py`, `rehydrate_data_20260914.py`, and `restore_snapshot_chain.py`. The resume coordinator passed10 focused tests; restoration passed5 synthetic corruption/path/transition tests. The QAT selector passed its focused cohort/champion checks, including retention of a superior plain INT8 export.

Status at19:15UTC: supervisor16204 running; the first new backup `esp32_20260914T191520Z_13ca9778.zip` succeeded. Audio download/audit precedes GPU training. See `output/esp32/run_status.json` for subsequent verified progress.

## Verified training restart, 19:34 UTC

Both runs advanced: raw epoch159 and normalized epoch152, on training PIDs20244 and20246. The L4 showed99% utilization and3972MiB allocated. The23 GPU acceptance tests passed in21.79seconds. The completed data audit matched the original extra-data source archives and regenerated all1,200 development WAV hashes.

VoiceBank has all23,144 expected split/utterance/role identities, and its five source parquet archives match the previous runtime byte-for-byte. Full regenerated FLOAT WAV container hashes differ because their PEAK chunks include creation timestamps. Three representative files were verified against their previous full-file hashes after replacing only that timestamp in memory; no on-disk audio was modified. Container hashes must not be described as universally identical across these recoveries.

The normalized checkpoint entered continuation with50 stale epochs against patience60, so it may stop well before240 if its primary validation score does not improve. Export and complete development evaluation still follow. The raw checkpoint entered with11 stale epochs.

The historical9.20221dB/46,768-byte QAT champion was also successfully cross-compiled for ESP32-S3 using the existing toolchain. The resulting application is298,576bytes, with a53,904-byte runtime arena and12 passing host known-answer frame/state comparisons. See [the build report](../../output/esp32/firmware_build/gtcrn_s3_champion_20260914/build_report.json). No physical board timing was performed.
