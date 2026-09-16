"""Run actual recovery, training, export, and QAT jobs with durable Drive backups."""
from pathlib import Path
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path('/content/esp32_runs')
PROJECT = Path('/content/esp32_project')
DRIVE = Path('/content/drive/MyDrive/Colab Notebooks/ESP32 Frontier Experiments')
CONTROL = ROOT / 'continuation_20260914'
ENV = {**os.environ, 'PYTHONPATH': str(PROJECT), 'OMP_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
TARGET = 'esp32_20260914T040022Z_33aac2b4.zip'


def report_progress(stop):
    previous = {}
    while not stop.wait(30):
        for name in ('float_gtcrn_broad_large', 'float_gtcrn_broad_normalized_large'):
            path = ROOT / name / 'history.jsonl'
            try:
                lines = path.read_text().rpartition('\n')[0].splitlines()
                epoch = json.loads(lines[-1])['epoch']
            except (OSError, IndexError, ValueError):
                continue
            if previous.get(name) != epoch:
                previous[name] = epoch
                print(json.dumps(dict(event='saved_training_progress', run=name, epoch=epoch)), flush=True)
        path = ROOT / 'post_large_qat_20260914/training.log'
        if path.exists():
            lines = path.read_text().rpartition('\n')[0].splitlines()
            progress = next((line for line in reversed(lines) if 'qat_progress' in line), None)
            if progress and previous.get('qat') != progress:
                previous['qat'] = progress
                print(progress, flush=True)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def run(command, label):
    print(json.dumps(dict(event='stage_started', stage=label)), flush=True)
    with (CONTROL / (label + '.log')).open('a') as log:
        with subprocess.Popen(command, cwd=PROJECT, env=ENV, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = child.wait()
    if code:
        raise RuntimeError(f'{label} failed with exit status {code}; inspect {log.name}')
    print(json.dumps(dict(event='stage_completed', stage=label)), flush=True)


def main():
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL / 'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (CONTROL / 'complete.json').exists():
            print('This continuation already completed; no jobs relaunched.', flush=True)
            return
        state_path = ROOT / 'durable_backup_state.json'
        if not state_path.exists():
            archive = DRIVE / 'recovery_snapshots' / TARGET
            with zipfile.ZipFile(archive) as bundle:
                manifest = json.loads(bundle.read('DURABLE_BACKUP_MANIFEST.json'))
            write_json(state_path, dict(archive=TARGET, base_archive=manifest['base_archive'],
                inventory=manifest['inventory_sha256'],
                archive_sha256='bf48918f323753595e0adfe29b493bae163e9f30408045b022990ba39d73b1c5',
                created_unix=manifest['created_unix']))
        with (CONTROL / 'backup.log').open('a') as log:
            backup = subprocess.Popen([sys.executable, '-u', str(ROOT / 'colab_durable_backup.py')],
                cwd=PROJECT, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        import psutil
        write_json(CONTROL / 'processes.json', dict(supervisor_pid=os.getpid(),
            supervisor_create_time=psutil.Process().create_time(), backup_pid=backup.pid,
            backup_create_time=psutil.Process(backup.pid).create_time(), started_unix=time.time()))
        outcome = dict(status='running', official_test_used=False)
        stop_reporting = threading.Event()
        reporter = threading.Thread(target=report_progress, args=(stop_reporting,), daemon=True)
        reporter.start()
        try:
            deadline = time.monotonic() + 180
            while json.loads(state_path.read_text())['archive'] == TARGET:
                if backup.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Initial recovery backup did not complete; inspect backup.log')
                time.sleep(2)
            ready = ROOT / 'recovery_20260914/data_ready.json'
            if not ready.exists():
                run([sys.executable, '-u', str(ROOT / 'rehydrate_data_20260914.py')], 'data_audit')
            run([sys.executable, '-m', 'pytest', '-q', 'tests/test_gtcrn_qat.py',
                 'tests/test_gtcrn_qat_gru.py'], 'gpu_acceptance')
            run([sys.executable, '-u', str(ROOT / 'resume_large_20260914.py'),
                 '--source-bundle', str(DRIVE / 'source_eceb96444d0988d4.zip'),
                 '--training-hours', '5', '--evaluation-hours', '2', '--execute'], 'large_continuation')
            run([sys.executable, '-u', str(ROOT / 'post_large_qat_20260914.py')], 'post_large_qat')
            outcome['status'] = 'training_and_evaluation_complete'
        except BaseException as error:
            outcome.update(status='failed', error=repr(error))
            raise
        finally:
            stop_reporting.set()
            reporter.join(timeout=5)
            write_json(CONTROL / 'outcome.json', {**outcome, 'updated_unix': time.time()})
            if backup.poll() is None:
                backup.terminate()
                backup.wait(timeout=120)
            spec = importlib.util.spec_from_file_location('durable_backup_final', ROOT / 'colab_durable_backup.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with (ROOT / 'durable_backup.lock').open('a+') as backup_lock:
                fcntl.flock(backup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                final_backup = module.snapshot(json.loads(state_path.read_text()))
            if outcome['status'] == 'training_and_evaluation_complete':
                write_json(CONTROL / 'complete.json', dict(completed_unix=time.time(),
                    final_backup=final_backup['archive'], final_backup_sha256=final_backup['archive_sha256'],
                    official_test_used=False, physical_esp32_timing_measured=False))
                print('Continuation, quantization, development evaluation, and final Drive backup completed.', flush=True)


if __name__ == '__main__':
    main()
