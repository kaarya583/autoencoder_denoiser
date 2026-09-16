# Colab recovery — 13 September 2026

The former Colab runtime was unavailable at recovery. A fresh NVIDIA L4 runtime was allocated with 89.33 compute units remaining. No checkpoints were found in its `/content` directory.

Verified recovery inputs:

- Full backup05: SHA-256 `15f1f5c62f5c85485f36c19a6806a2a8da5fc07afdc79e0a62e2628c0369cdfa`, 427 files.
- GTCRN probes04: SHA-256 `a556328412867fc43dee78d4654cd683651a9816a187ac833b06ba9622dd75f5`, 31 files.
- Source release: SHA-256 `eceb96444d0988d487f7a087190b5baf5437d54f949190dbf0ef67ec1e6a64d3`, 191 files.

All archive members were checked against their embedded SHA-256 manifests before extraction. The restored source passed 23 QAT/GRU tests on the new GPU in 25.34 seconds.

The raw GTCRN training checkpoint is epoch43 (`76cde5b5fc5756f428dcae241f4c9a1db2b44425c8861041c7ede75ac7b65380`); normalized GTCRN is epoch25 (`d4ea91b91e473184dc56389d5a00696993a3c38397c5a8050bf723a9d02d5914`). These preserve optimizer and scheduler state. Training progress after the backed-up checkpoints was not recoverable from the expired runtime. No later training result should be inferred from the prior queues.

The 47,024-byte normalized INT8 artifact and its 8.23048 dB development SI-SDR improvement report survived in probes04. This remains a development result, with no physical ESP32 timing measurement.

Drive is mounted at `/content/drive/MyDrive/Colab Notebooks/ESP32 Frontier Experiments`. Recovery keeps the notebook attached to the actual training process and adds periodic immutable checkpoint backups in this private folder. A Colab runtime lifetime is still finite; the backups protect progress when it ends.

Data restoration uses the same pinned VoiceBank mirror and original LibriSpeech/MUSAN archives. VoiceBank cache filenames incorporate download modification times, so recovery must match utterance metadata, alias the regenerated audio to the original paths, and preserve the backed-up manifests. Development mixtures must pass their original hash checks before training resumes. Official test sets remain sealed.

See `output/esp32/run_status.json` for the last verified recovery stage. The recovery scripts are in `output/esp32/recover_colab_training.py`, `recover_colab_qat.py`, and `colab_durable_backup.py`.

## Verified recovery outcome, 18:50 UTC

Training resumed successfully and advanced to raw epoch48 and normalized epoch30. The eight-step QAT pilot completed; its long-run wrapper waits for the normalized parent. Both 240-epoch float continuations are queued behind the resumed parent runs. These are running/queued experiments, not completed results.

The data audit verified 23,144 VoiceBank WAV aliases, restored the exact original manifest hashes, verified the original extra-data archives, and reproduced all 1,200 development WAV hashes.

The raw checkpoint selector initially rejected two newly explicit default configuration fields. `recover_selector_metadata.py` verified the immutable epoch35 checkpoint, preserved the original record, and added only `normalize_input=false` and `rms_floor=0.0001`; weights, metrics, and checkpoint identity were unchanged. The restarted selector successfully evaluated epoch48. `recover_colab_completion.py` now supervises the repaired selector, larger queue, surviving QAT wrapper, and Drive backup process. The obsolete monitor was terminated individually; active training jobs were preserved.

The first full Drive snapshot contains497 files and passed remote readback verification: `recovery_snapshots/esp32_20260913T184442Z_ebec1f8a.zip`, SHA-256 `53a45455569f6d1bf98da3d1bf27ce2ffd8a2e027564deb03f99af0ab7572734`. Later snapshots contain changed files and link to their preceding archive. For another recovery, start from the full archive and apply deltas in sequence using their manifests; do not restore only the newest delta.

## Notebook connection repair, 14 September 01:22 UTC

The primary notebook now reports Connected on the original L4 machine (`2acfc3687f2c`). Its terminal remained responsive even while the frontend showed Connecting / Resuming execution. Sending SIGINT only to notebook kernel PID8290 ended the foreground status observer and cleared the stalled primary interface. No runtime restart or process-group signal was used.

The notebook now schedules `colab_async_status.watch_training()` with `asyncio.create_task` and returns immediately. The observer reads actual supervisor identity and training progress; it owns no GPU processes. A foreground asynchronous observer still stalled the diagnostic reload, so it was replaced with this background task. The diagnostic tab was closed; the primary notebook remained Connected. Automatic reconnection from an independently reloaded tab was not established.

All five tracked processes retained their original PIDs and creation times: supervisor16131, large queue16148, QAT wrapper13184, backup14617, and QAT trainer32490. Training advanced to raw epoch109 and normalized epoch102; QAT reached round3, step175. GPU utilization was99%, with9383MiB used. The latest backup state recorded `esp32_20260914T011512Z_cfef43ce.zip`, SHA-256 `fd0a10166d344d28af8fb698afefda47d125a9c9ad2958ecaccf205ff86b3728`. Training remains in progress.
