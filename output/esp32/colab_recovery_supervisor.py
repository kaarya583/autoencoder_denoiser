"""Keep the notebook attached to real recovery/training jobs and Drive backups."""
from pathlib import Path
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import time

ROOT = Path('/content/esp32_runs')
PROJECT = Path('/content/esp32_project')
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def start(script, log):
    with (ROOT / log).open('a') as handle:
        return subprocess.Popen([sys.executable, '-u', str(ROOT / script)], cwd=PROJECT,
                                env=ENV, stdout=handle, stderr=subprocess.STDOUT)


def main():
    with (ROOT / 'recovery_supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        training = start('recover_colab_training.py', 'recovery_training.log')
        qat = start('recover_colab_qat.py', 'recovery_qat.log')
        backup = None
        last_report = 0
        print('Recovery attached: verifying data, then resuming saved training.', flush=True)
        while training.poll() is None or qat.poll() is None:
            ready = (ROOT / 'recovery_data_ready.json').exists()
            if ready and backup is None:
                backup = start('colab_durable_backup.py', 'durable_backup.log')
                print('Data verified; training and 15-minute Drive backups are active.', flush=True)
            if training.poll() not in (None, 0) and not ready:
                qat.terminate()
                qat.wait()
                raise RuntimeError('Data recovery failed; inspect recovery_training.log. No training was launched.')
            if backup is not None and backup.poll() is not None:
                raise RuntimeError('Drive backup process exited; inspect durable_backup.log. Training may remain active.')
            if time.monotonic() - last_report >= 600:
                print(json.dumps(dict(event='recovery_jobs', data_ready=ready,
                      training_returncode=training.poll(), qat_returncode=qat.poll(),
                      drive_backup_running=backup is not None)), flush=True)
                last_report = time.monotonic()
            time.sleep(10)
        if backup is not None:
            backup.terminate()
            backup.wait()
            spec = importlib.util.spec_from_file_location('durable_backup', ROOT / 'colab_durable_backup.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            previous = json.loads(module.STATE.read_text()) if module.STATE.exists() else {}
            module.snapshot(previous)
        if training.returncode or qat.returncode:
            raise RuntimeError(f'Inspect recovery logs: training={training.returncode}, QAT={qat.returncode}')
        print('Training and final Drive backup complete. Official tests remain sealed.', flush=True)


if __name__ == '__main__':
    main()
