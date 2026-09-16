"""Promote the completed broad continuation's external winner into matched QAT."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

root = Path('/content/esp32_runs')
project = Path('/content/esp32_project')
run = root / 'float_zero_bias_broad_continue'
deadline = time.monotonic() + 5 * 3600
while not (run / 'summary.json').exists():
    if time.monotonic() >= deadline:
        raise TimeoutError('The float predecessor did not finish')
    time.sleep(15)
completed = json.loads((run / 'history.jsonl').read_text().splitlines()[-1])['epoch']
deadline = time.monotonic() + 600
while True:
    history = run / 'external_development/history.jsonl'
    rows = history.read_text().splitlines() if history.exists() else []
    if rows and json.loads(rows[-1])['model']['checkpoint_epoch'] == completed:
        break
    if time.monotonic() >= deadline:
        raise TimeoutError('The final float checkpoint has not been scored on external development')
    time.sleep(15)

marker = run / 'external_development/best.json'
marker_bytes = marker.read_bytes()
selection = json.loads(marker_bytes)
blob = Path(selection['selected_checkpoint']).read_bytes()
assert hashlib.sha256(blob).hexdigest() == selection['checkpoint_sha256']
assert selection['model']['precision'] == 'float32 PyTorch'
inputs = root / 'broad_qat_inputs'
inputs.mkdir(exist_ok=False)
(inputs / 'student.pt').write_bytes(blob)
(inputs / 'selection.json').write_text(json.dumps({
    'selection': selection, 'marker_path': str(marker),
    'marker_sha256': hashlib.sha256(marker_bytes).hexdigest(),
    'completed_float_epoch': completed}, indent=2) + '\n')
config = project / 'configs/esp32_zero_bias_broad_qat.json'
settings = json.loads(config.read_text())
assert settings['resume'] == str(inputs / 'student.pt')
assert settings['phase'] == 'qat' and settings['resume_optimizer'] is False
output = Path(settings['output_dir'])
assert not output.exists(), 'Refusing to overwrite a QAT run'
print(json.dumps({'event': 'frozen_float_source', 'epoch': selection['model']['checkpoint_epoch'],
                  'external_si_sdri': selection['validation']['si_sdri'],
                  'checkpoint_sha256': selection['checkpoint_sha256'],
                  'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest()}), flush=True)
with (root / 'qat_zero_bias_broad.log').open('w') as training_log, \
        (root / 'qat_zero_bias_broad_external_watch.log').open('w') as validation_log:
    training = subprocess.Popen(['python3', '-u', '-m', 'esp32_denoiser.train', '--config', str(config)],
                                cwd=project, stdout=training_log, stderr=subprocess.STDOUT)
    validation = subprocess.Popen(['python3', '-u', '-m', 'esp32_denoiser.development_checkpoints',
                                  '--run', str(output), '--manifest',
                                  '/content/extra_audio/development/mixtures.jsonl',
                                  '--device', 'cuda', '--every-epochs', '5', '--max-hours', '3'],
                                 cwd=project, stdout=validation_log, stderr=subprocess.STDOUT)
    print('QAT_PID', training.pid, 'QAT_EXTERNAL_PID', validation.pid, flush=True)
    code = training.wait()
    if code:
        validation.terminate()
        validation.wait()
        raise RuntimeError(f'QAT failed, code={code}')
    assert validation.wait(timeout=600) == 0
print('BROAD_QAT_ROUND_COMPLETE', flush=True)
