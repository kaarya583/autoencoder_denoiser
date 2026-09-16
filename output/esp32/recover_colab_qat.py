"""Recover the authorized QAT pilot and long run after data restoration."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path('/content/esp32_runs')
PROJECT = Path('/content/esp32_project')
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def command(args, log):
    with log.open('w') as handle:
        subprocess.run(args, cwd=PROJECT, env=ENV, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    deadline = time.monotonic() + 3 * 3600
    while not (ROOT / 'recovery_data_ready.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Recovery data preparation did not finish within three hours')
        time.sleep(30)
    pilot = ROOT / 'gtcrn_qat_gpu_pilot'
    if pilot.exists():
        raise FileExistsError('Inspect existing QAT pilot before attempting a duplicate run')
    inputs = ROOT / 'gtcrn_normalized_probe_inputs'
    config = dict(
        source_checkpoint=str(inputs / 'student.pt'),
        integer_model=str(inputs / 'model_mse_int8.bin'),
        calibration=str(inputs / 'model_mse_int8.calibration.json'),
        development_manifest='/content/extra_audio/development/mixtures.jsonl',
        output_dir=str(pilot), epochs=1, batch_size=2, workers=0,
        max_steps_per_epoch=8, max_validation_utterances=4,
        max_hours=.5, device='cuda',
    )
    for field in ('source_checkpoint', 'integer_model', 'calibration'):
        if not Path(config[field]).is_file():
            raise FileNotFoundError(config[field])
    config_path = ROOT / 'gtcrn_qat_gpu_pilot_config.json'
    config_path.write_text(json.dumps(config, indent=2))
    command([sys.executable, '-u', '-m', 'esp32_denoiser.gtcrn_qat_train',
             '--config', str(config_path)], ROOT / 'gtcrn_qat_gpu_pilot.log')
    summary = json.loads((pilot / 'summary.json').read_text())
    if summary['status'] != 'complete' or summary['completed_epoch'] != 1:
        raise RuntimeError(f'QAT pilot did not complete: {summary}')
    command([sys.executable, '-u', str(ROOT / 'long_qat_round.py')], ROOT / 'long_qat_round.log')


if __name__ == '__main__':
    main()
