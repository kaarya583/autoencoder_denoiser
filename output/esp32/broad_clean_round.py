"""Run the matched clean-exposure ablation after the existing KD treatment."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

root = Path('/content/esp32_runs')
project = Path('/content/esp32_project')
control_path = project / 'configs/esp32_zero_bias_broad_distillation_control_float.json'
config_path = project / 'configs/esp32_zero_bias_broad_clean_float.json'
control = json.loads(control_path.read_text())
settings = json.loads(config_path.read_text())
differences = {k for k in control.keys() | settings.keys() if control.get(k) != settings.get(k)}
assert differences <= {'output_dir', 'clean_identity_probability'}
assert settings['clean_identity_probability'] == .15
assert control.get('clean_identity_probability', .03) == .03
assert settings['resume_optimizer'] is False and settings['distillation_weight'] == 0
source = Path(settings['resume'])
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_sha == '12e4595db174a6f34b58778e112e5f986938689f0feadc39f8f368808e32e746'
output = Path(settings['output_dir'])
assert not output.exists(), 'Refusing to overwrite an existing trial'
treatment = json.loads((root / 'broad_distillation_calibrated_config.json').read_text())
predecessor = Path(treatment['output_dir'])
deadline = time.monotonic() + 3 * 3600
while not (predecessor / 'summary.json').exists():
    if time.monotonic() >= deadline:
        raise TimeoutError('The matched KD predecessor did not finish')
    time.sleep(15)
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
assert not output.exists()
print(json.dumps({'event': 'clean_exposure_start', 'source_sha256': source_sha,
                  'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  'clean_identity_probability': .15, 'time': time.time()}), flush=True)
with (root / (output.name + '.log')).open('w') as training_log, \
        (root / (output.name + '_external_watch.log')).open('w') as validation_log:
    training = subprocess.Popen(['python3', '-u', '-m', 'esp32_denoiser.train',
                                 '--config', str(config_path)], cwd=project,
                                stdout=training_log, stderr=subprocess.STDOUT)
    validation = subprocess.Popen(['python3', '-u', '-m', 'esp32_denoiser.development_checkpoints',
                                   '--run', str(output), '--manifest',
                                   '/content/extra_audio/development/mixtures.jsonl',
                                   '--device', 'cuda', '--every-epochs', '5', '--max-hours', '3'],
                                  cwd=project, stdout=validation_log, stderr=subprocess.STDOUT)
    print('CLEAN_TRAIN_PID', training.pid, 'EXTERNAL_PID', validation.pid, flush=True)
    code = training.wait()
    if code:
        validation.terminate()
        validation.wait()
        raise RuntimeError(f'Clean-exposure training failed: {code}')
    assert validation.wait(timeout=600) == 0
print('BROAD_CLEAN_ROUND_COMPLETE', flush=True)
