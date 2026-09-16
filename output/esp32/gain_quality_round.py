"""Fit a training-only deployment gain, then score immutable development cohorts."""
import sys
sys.path.insert(0, '/content/esp32_project')

import hashlib
import json
from pathlib import Path
import subprocess
import torch

from esp32_denoiser.output_gain import calibrate_integer_output_gain
from esp32_denoiser.export import with_output_gain

torch.set_num_threads(2)
root = Path('/content/esp32_runs')
output = root / 'broad_ptq_gain_probe'
output.mkdir(exist_ok=False)
binary = root / 'broad_ptq_probe/denoiser_int8.bin'
expected = '50ebbb78f5834f8ec20051f2d6bfb5aebda7d994bdd7b54c2179c6b0d885525e'
assert hashlib.sha256(binary.read_bytes()).hexdigest() == expected
calibration = calibrate_integer_output_gain(
    binary,
    paired_train_manifest='/content/voicebank/manifests/train.jsonl',
    speech_train_manifest='/content/extra_audio/manifests/speech_train.jsonl',
    primary_development_manifest='/content/voicebank/manifests/val.jsonl',
    speech_development_manifest='/content/extra_audio/manifests/speech_val.jsonl',
    external_mixtures_manifest='/content/extra_audio/development/mixtures.jsonl',
    external_clean_manifest='/content/extra_audio/development/clean.jsonl',
    examples_per_source=64, crop_seconds=3, seed=2026)
(output / 'calibration.json').write_text(json.dumps(calibration, indent=2, allow_nan=False) + '\n')
assert calibration['source_model']['sha256'] == expected
deployed = output / 'denoiser_int8.bin'
export = with_output_gain(binary, deployed, calibration['output_gain'], max_bytes=99000)
export['calibration_sha256'] = hashlib.sha256((output / 'calibration.json').read_bytes()).hexdigest()
export['source_bundle_sha256'] = 'c4acc2155adc62cdfbb42fe904a5502361d1c1260dfabb60860c6aeb82815fef'
(output / 'export.json').write_text(json.dumps(export, indent=2) + '\n')
print('GAIN_FIT', calibration['output_gain'], calibration['diagnostics'], flush=True)
print('GAIN_EXPORT', export, flush=True)

cohorts = [('clean', '/content/extra_audio/development/clean.jsonl'),
           ('external', '/content/extra_audio/development/mixtures.jsonl'),
           ('primary', '/content/voicebank/manifests/val.jsonl')]
for name, manifest in cohorts:
    subprocess.run([sys.executable, '-u', '-m', 'esp32_denoiser.embedded',
                    '--integer-model', str(deployed), '--manifest', manifest,
                    '--output', str(output / (name + '_full_c_pcm16.json')),
                    '--io-format', 'pcm16', '--perceptual', '--threads', '2'],
                   cwd='/content/esp32_project', check=True)
    result = json.loads((output / (name + '_full_c_pcm16.json')).read_text())
    print('CORRECTED', name, result['summary']['si_sdri'],
          result['preservation']['enhanced']['projection_gain'],
          result['perceptual']['summary'], flush=True)

# Preserve a direct perceptual comparison against exactly the same original binary.
subprocess.run([sys.executable, '-u', '-m', 'esp32_denoiser.embedded',
                '--integer-model', str(binary), '--manifest', cohorts[1][1],
                '--output', str(output / 'uncorrected_external_full_c_pcm16.json'),
                '--io-format', 'pcm16', '--perceptual', '--threads', '2'],
               cwd='/content/esp32_project', check=True)
subprocess.run([sys.executable, '-u', '-m', 'esp32_denoiser.evaluate',
                '--checkpoint', str(root / 'broad_kd_inputs/student.pt'),
                '--manifest', cohorts[0][1], '--device', 'cpu', '--threads', '2',
                '--output', str(output / 'source_float_clean.json'), '--perceptual'],
               cwd='/content/esp32_project', check=True)
print('GAIN_QUALITY_ROUND_COMPLETE', flush=True)
