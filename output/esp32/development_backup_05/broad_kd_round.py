"""Finite Colab queue for the frozen, gain-corrected broad KD comparison."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/content/esp32_runs')
PROJECT = Path('/content/esp32_project')
INPUTS = ROOT / 'broad_kd_inputs'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    control = PROJECT / 'configs/esp32_zero_bias_broad_distillation_control_float.json'
    treatment = ROOT / 'broad_distillation_calibrated_config.json'
    a, b = [json.loads(path.read_text()) for path in (control, treatment)]
    allowed = {'output_dir', 'teacher_checkpoint', 'teacher_gain_calibration', 'distillation_weight'}
    assert {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)} <= allowed
    assert a['resume'] == b['resume'] == str(INPUTS / 'student.pt')
    assert a['resume_optimizer'] is False and not a.get('teacher_checkpoint')
    calibration = json.loads((INPUTS / 'broad_distillation_calibration.json').read_text())
    assert calibration['teacher_checkpoint_sha256'] == digest(INPUTS / 'teacher.pt')
    assert calibration['teacher_gain_calibration_sha256'] == digest(INPUTS / 'teacher_gain.json')
    assert b['distillation_weight'] == calibration['distillation_weight'] > 0
    snapshots = {name: digest(INPUTS / name) for name in ('student.pt', 'teacher.pt', 'teacher_gain.json')}
    for settings in (a, b):
        assert not Path(settings['output_dir']).exists(), 'Refusing to overwrite a trial'

    # Preserve the useful spectral-loss arm already running in the replaced queue.
    process_path = Path('/proc/61608/cmdline')
    while process_path.exists() and b'esp32_zero_bias_spectral_float.json' in process_path.read_bytes():
        time.sleep(15)
    assert (ROOT / 'float_zero_bias_spectral/summary.json').exists(), 'Spectral predecessor did not complete'

    for config, settings in ((control, a), (treatment, b)):
        assert snapshots == {name: digest(INPUTS / name) for name in snapshots}, 'Frozen inputs changed'
        output = Path(settings['output_dir'])
        assert not output.exists(), 'Refusing to overwrite a trial'
        print(json.dumps({'event': 'start', 'config': str(config), 'config_sha256': digest(config),
                          'frozen_inputs': snapshots, 'time': time.time()}), flush=True)
        with (ROOT / (output.name + '.log')).open('w') as training_log, \
                (ROOT / (output.name + '_external_watch.log')).open('w') as validation_log:
            training = subprocess.Popen(['python3', '-u', '-m', 'esp32_denoiser.train',
                                         '--config', str(config)], cwd=PROJECT,
                                        stdout=training_log, stderr=subprocess.STDOUT)
            validation = subprocess.Popen(['python3', '-u', '-m', 'esp32_denoiser.development_checkpoints',
                                           '--run', str(output), '--manifest',
                                           '/content/extra_audio/development/mixtures.jsonl',
                                           '--device', 'cuda', '--every-epochs', '5', '--max-hours', '3'],
                                          cwd=PROJECT, stdout=validation_log, stderr=subprocess.STDOUT)
            code = training.wait()
            if code:
                validation.terminate()
                validation.wait()
                raise RuntimeError(f'Training failed: {config}, code={code}')
            validation_code = validation.wait(timeout=600)
            assert validation_code == 0, 'External development selection failed'
        print(json.dumps({'event': 'complete', 'output': str(output), 'time': time.time()}), flush=True)


if __name__ == '__main__':
    main()
