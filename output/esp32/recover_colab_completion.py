"""Supervise surviving GPU jobs after the audited legacy selector repair."""
from pathlib import Path
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import time

import psutil

R = Path('/content/esp32_runs')
P = Path('/content/esp32_project')
E = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def matching(script):
    expected = str(R / script)
    matches = []
    for process in psutil.process_iter():
        try:
            if expected in process.cmdline()[1:]:
                matches.append(process)
        except psutil.NoSuchProcess:
            continue
    return matches


def existing(script):
    matches = matching(script)
    if len(matches) != 1:
        raise RuntimeError(f'Expected one live {script}: {[p.pid for p in matches]}')
    return matches[0]


def alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def start(command, log):
    with (R / log).open('x') as handle:
        return subprocess.Popen(command, cwd=P, env=E, stdout=handle, stderr=subprocess.STDOUT)


def main():
    with (R / 'recovery_completion.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (R / 'recovery_completion_launches.json').exists():
            raise RuntimeError('Repair launch record exists; inspect before restarting')
        old_supervisor = existing('colab_recovery_supervisor.py')
        qat = existing('recover_colab_qat.py')
        backup = existing('colab_durable_backup.py')
        if matching('large_colab_round.py'):
            raise RuntimeError('An existing large queue must be inspected before relaunch')
        # This migration verifies and preserves the selected checkpoint and metrics.
        subprocess.run([sys.executable, str(R / 'recover_selector_metadata.py')], cwd=P, env=E, check=True)
        selector = start([sys.executable, '-u', '-m', 'esp32_denoiser.development_checkpoints',
            '--run', str(R / 'float_gtcrn_broad'), '--manifest', '/content/extra_audio/development/mixtures.jsonl',
            '--device', 'cuda', '--every-epochs', '5', '--max-hours', '5'], 'recovery_raw_selector_repaired.log')
        large = start([sys.executable, '-u', str(R / 'large_colab_round.py')], 'recovery_large_queue_repaired.log')
        record = dict(created_unix=time.time(), supervisor_pid=os.getpid(),
            supervisor_create_time=psutil.Process().create_time(), selector_pid=selector.pid,
            large_queue_pid=large.pid, qat_pid=qat.pid, backup_pid=backup.pid,
            replaced_supervisor_pid=old_supervisor.pid)
        (R / 'recovery_completion_launches.json').write_text(json.dumps(record, indent=2))
        # Terminate only the obsolete monitor. All actual jobs are preserved.
        old_supervisor.terminate()
        old_supervisor.wait(timeout=15)
        print(json.dumps(dict(event='repaired_supervision_active', **record)), flush=True)
        deadline = time.monotonic() + 20 * 3600
        while True:
            for name, process in [('raw selector', selector), ('large queue', large)]:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f'{name} failed with code {process.returncode}; inspect its repair log')
            if not alive(backup):
                raise RuntimeError('Drive backup process stopped; actual training may still be running')
            qat_done = not alive(qat)
            if qat_done and not (R / 'long_qat_complete.json').is_file():
                raise RuntimeError('QAT stopped without completion; inspect recovery_qat.log and pilot log')
            if selector.poll() == 0 and large.poll() == 0 and qat_done:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Recovery completion exceeded its twenty-hour limit')
            time.sleep(15)
        for marker in ['large_round_complete.json', 'long_qat_complete.json']:
            if not (R / marker).is_file():
                raise RuntimeError(f'Missing completion record: {marker}')
        backup.terminate()
        backup.wait(timeout=30)
        spec = importlib.util.spec_from_file_location('durable_backup', R / 'colab_durable_backup.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = json.loads(module.STATE.read_text()) if module.STATE.exists() else {}
        module.snapshot(state)
        (R / 'recovery_all_complete.json').write_text(json.dumps(dict(completed_unix=time.time(), official_test_used=False)))
        print('Recovered training, development evaluation, and final Drive backup completed.', flush=True)


if __name__ == '__main__':
    main()
