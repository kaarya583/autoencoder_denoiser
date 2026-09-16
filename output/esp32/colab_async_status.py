"""Await actual training completion without blocking Colab's event loop.

Notebook entry point: await watch_training(). This observer owns no GPU jobs.
"""
from pathlib import Path
import asyncio
import json
import time

import psutil

ROOT = Path('/content/esp32_runs')


def training_status():
    epochs = {}
    for name in ('float_gtcrn_broad_large', 'float_gtcrn_broad_normalized_large'):
        path = ROOT / name / 'history.jsonl'
        if path.exists():
            # Ignore a writer's unfinished last line.
            completed = path.read_text().rpartition('\n')[0].splitlines()
            if completed:
                epochs[name] = json.loads(completed[-1])['epoch']
    return dict(event='training_status', epochs=epochs,
                all_complete=(ROOT / 'recovery_all_complete.json').is_file())


async def watch_training():
    record = json.loads((ROOT / 'recovery_completion_launches.json').read_text())
    process = psutil.Process(record['supervisor_pid'])
    if (process.create_time() != record['supervisor_create_time'] or
            str(ROOT / 'recover_colab_completion.py') not in process.cmdline()):
        raise RuntimeError('Training supervisor identity changed; inspect before attaching')
    print('Connected to active training. Drive checkpoints continue every 15 minutes.', flush=True)
    last_report = 0
    while True:
        try:
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        if time.monotonic() - last_report >= 600:
            print(json.dumps(await asyncio.to_thread(training_status)), flush=True)
            last_report = time.monotonic()
        await asyncio.sleep(10)
    if not (ROOT / 'recovery_all_complete.json').is_file():
        raise RuntimeError((ROOT / 'recovery_completion.log').read_text()[-3000:])
    print('Training, evaluation, and final Drive backup completed.', flush=True)
