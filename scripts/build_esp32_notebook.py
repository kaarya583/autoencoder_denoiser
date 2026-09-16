"""Create a portable Colab notebook from the current reviewed source bundle."""
import base64
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(ROOT/'scripts/package_esp32_colab.py')], check=True)
payload = base64.b64encode((ROOT/'output/esp32/esp32_training_source.zip').read_bytes()).decode()
cells = []
def markdown(text):
    cells.append({'cell_type':'markdown', 'metadata':{}, 'source':text.splitlines(keepends=True)})
def code(text):
    cells.append({'cell_type':'code', 'metadata':{}, 'source':text.splitlines(keepends=True), 'execution_count':None, 'outputs':[]})

markdown('''# ESP32-S3 speech enhancement: SI-SDR training and integer evaluation

This notebook embeds the source bundle, including tests. Choose an L4 GPU runtime. The default reproduces the strongest completed integer control: the signed spectral TCN with zero depthwise-bias initialization. Frequency U-Net, folded encoder BatchNorm, recurrent, broader-data, spectral-loss, and distillation configurations are also included. The official test split is reserved until all architecture and checkpoint selection is complete.

Default architecture: 84,738 parameters, 5.212 MMAC/s neural arithmetic, 256-sample hops at 16 kHz. INT8 exported model data must be ≤99,000 bytes. Hardware speed requires the separate ESP32-S3 benchmark; Colab/host timing is not board timing.

The completed control scored 7.7763 dB SI-SDR improvement on the 770-clip speaker-held-out development split through complete C PCM16 inference, with a 94,480-byte model. This is a development result; architecture search remains open and eight decibels is not a cap. No board runtime or architecture-novelty claim is established.

Training uses early stopping and a wall-time limit that can overrun by one validation pass; preparation and QAT calibration are outside the loop timer. Monitor the current Colab compute-unit rate before running. No additional credits are purchased. Download checkpoints before the runtime is deleted.
''')
code("""import subprocess, sys
# Keep Colab's preinstalled GPU PyTorch; install only the small missing tools.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'soundfile', 'scipy', 'pyarrow', 'huggingface_hub', 'pytest', 'einops==0.8.1', 'threadpoolctl==3.6.0'], check=True)
# Ancillary final reporting only; these metrics never select checkpoints.
PERCEPTUAL = False  # Set True to report PESQ WB and ordinary STOI on final test.
if PERCEPTUAL:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'pesq==0.0.4', 'pystoi==0.4.1'], check=True)
import torch
assert torch.cuda.is_available(), 'Select a GPU runtime before training.'
print(torch.__version__, torch.cuda.get_device_name())
""")
code("""import base64, io, zipfile, pathlib, os, json, hashlib
source_bundle_b64 = '"""+payload+"""'
project = pathlib.Path('/content/esp32_project'); project.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(source_bundle_b64))) as bundle:
    for info in bundle.infolist():
        assert not pathlib.Path(info.filename).is_absolute() and '..' not in pathlib.Path(info.filename).parts
    bundle.extractall(project)
os.chdir(project)
if str(project) not in sys.path: sys.path.insert(0, str(project))
manifest = json.loads((project/'SOURCE_MANIFEST.json').read_text())
assert all(hashlib.sha256((project/p).read_bytes()).hexdigest() == digest for p,digest in manifest.items())
result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], capture_output=True, text=True)
print(result.stdout, result.stderr)
result.check_returncode()
""")
markdown('''## Data and holdout

The original Edinburgh download server rejected the Colab runtime. The fallback is the public `JacobLinCool/VoiceBank-DEMAND-16k` mirror pinned to commit `4497db342d7312978c45690591fda86117831940`. Its resampling method and byte identity to the original archives are unverified and recorded in provenance. Audio is decoded into float32 WAV without another resampling pass.

Training excludes speakers p226 and p287. The noisy and clean signals always use identical crop offsets and shared gain. Training may vary noise amplitude and include clean identity examples. Validation uses full, unmodified utterances with equal utterance weighting.
''')
code("""os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
from esp32_denoiser.data import prepare_voicebank_parquet
manifests = prepare_voicebank_parquet('/content/voicebank', download=True, include_test=False)
print(pathlib.Path('/content/voicebank/manifests/provenance.json').read_text())
""")
code("""def run_logged(command):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout: print(line, end='', flush=True)
        if process.wait(): raise RuntimeError(f'Command failed: {command}')
run_logged([sys.executable, '-u', '-m', 'esp32_denoiser.train', '--config', 'configs/esp32_pilot.json'])
""")
markdown('''## Full training

The pilot checks that learning works. The zero-bias full run starts from reproducible initialization and selects checkpoints only on all 770 validation clips from p226 and p287. It writes `float_zero_bias`; QAT starts from that run's best checkpoint, calibrates a shared fixed hidden-state scale on training audio, and optimizes the deployed INT8 grids. Resume a phase with its corresponding `last.pt`; float-to-QAT starts a new optimization phase. Use fresh output directories when reproducing a trial so earlier evidence is preserved.

Complete the architecture, training-source, loss, and distillation comparisons before proceeding to the final-test cells. The default sequence reproduces a six-block control; it does not establish a globally optimal model. The included frequency and frequency-BN configurations use the same framing and a 94,300-byte integer model. GRU candidates remain float-only and cannot be selected for deployment until their integer graph is implemented and verified.
''')
code("""# Change both paths together for an integer-supported alternative, such as
# esp32_frequency_bn_float.json and esp32_frequency_bn_qat.json.
# GRU configurations are float-only and cannot enter the deployment stage yet.
FLOAT_CONFIG = pathlib.Path('configs/esp32_zero_bias_float.json')
QAT_CONFIG = pathlib.Path('configs/esp32_zero_bias_qat.json')
float_config = json.loads(FLOAT_CONFIG.read_text())
qat_config = json.loads(QAT_CONFIG.read_text())
FLOAT_CHECKPOINT = pathlib.Path(float_config['output_dir']) / 'best.pt'
QAT_CHECKPOINT = pathlib.Path(qat_config['output_dir']) / 'best.pt'
float_checkpoint, qat_checkpoint = FLOAT_CHECKPOINT, QAT_CHECKPOINT
assert float_config['phase'] == 'float' and qat_config['phase'] == 'qat'
assert pathlib.Path(qat_config['resume']) == FLOAT_CHECKPOINT
assert not float_config.get('max_val_batches') and not qat_config.get('max_val_batches')
run_logged([sys.executable, '-u', '-m', 'esp32_denoiser.train', '--config', str(FLOAT_CONFIG)])
run_logged([sys.executable, '-u', '-m', 'esp32_denoiser.train', '--config', str(QAT_CONFIG)])
""")
code("""from esp32_denoiser.evaluate import load_checkpoint
from esp32_denoiser.models import checkpoint_kind
model, checkpoint_provenance = load_checkpoint(qat_checkpoint)
assert checkpoint_provenance['phase'] == 'qat' and model.config.activation_mode == 'signed'
integer_model = pathlib.Path('/content/esp32_runs/deploy/denoiser_int8.bin')
kind = checkpoint_kind(checkpoint_provenance)
if kind == 'spectral_tcn':
    from esp32_denoiser.export import export_model
elif kind == 'frequency_unet':
    from esp32_denoiser.frequency_export import export_frequency_model as export_model
else:
    raise ValueError(f'No verified integer exporter for {kind}')
report = export_model(model, integer_model)
assert report['format'] in ('EDNSI8-v2', 'EDNFQ8-v1') and integer_model.stat().st_size <= 99000
print(json.dumps(report, indent=2))
""")
markdown('''## Integer validation, final test, and downloads

Compare float, fake-quantized, and complete C PCM16 results on validation before freezing selection. `esp32_denoiser.embedded` runs the full C analysis/features/neural/synthesis path with PCM16 input conversion and output clipping. Its reference comparison also reports the unclipped C float32 path and the C neural core with Python DSP. Inspect clipping counts and all validation denominators.

The host uses a portable FFT; ESP32-S3 uses ESP-DSP. Host timing therefore cannot establish board speed or acoustic latency. The separate ESP-IDF benchmark measures complete-hop timing on a board.

The initial acceptance target is at most 0.3 dB SI-SDRi loss from the selected float model to the complete C PCM16 model. Passing validation freezes model hashes before the official test is downloaded. Test results remain final reporting data; do not select a later architecture or checkpoint from them. Optional PESQ WB and STOI reuse the same test enhancements, preserve shared signal gain, and report invalid denominators and package versions.
''')
code("""# Set this only after the full architecture/training search is finished.
# It prevents Run All from consuming the official test during an early trial.
FINAL_SELECTION_READY = False
# Match the selected float and QAT models on full development utterances.
for label, argument, source in [
    ('float', '--checkpoint', str(float_checkpoint)),
    ('qat', '--checkpoint', str(qat_checkpoint)),
]:
    run_logged([sys.executable, '-u', '-m', 'esp32_denoiser.evaluate', argument, source,
                '--manifest', str(manifests['val']),
                '--output', f'/content/esp32_runs/{label}_validation.json'])
run_logged([sys.executable, '-u', '-m', 'esp32_denoiser.embedded',
            '--integer-model', str(integer_model), '--manifest', str(manifests['val']),
            '--output', '/content/esp32_runs/embedded_validation.json',
            '--io-format', 'pcm16', '--compare-reference'])
validation_reports = {label: json.loads(pathlib.Path(f'/content/esp32_runs/{label}_validation.json').read_text())
                      for label in ('float', 'qat', 'embedded')}
validation = {label: value['summary'] for label, value in validation_reports.items()}
print(json.dumps(validation, indent=2))
expected_ids = {json.loads(line)['id'] for line in pathlib.Path(manifests['val']).read_text().splitlines() if line.strip()}
assert len(expected_ids) == 770, 'Expected the complete two-speaker VoiceBank validation split.'
for value in validation_reports.values():
    assert not value['limited_evaluation']
    assert value['summary']['valid_utterances'] == len(expected_ids)
    assert len(value['utterances']) == len(expected_ids)
    assert {row['id'] for row in value['utterances']} == expected_ids
quantization_drop = validation['float']['si_sdri'] - validation['embedded']['si_sdri']
assert quantization_drop <= 0.3, 'Quantization loss exceeds the target; refine QAT before unlocking test.'
assert validation['embedded']['si_sdri'] > 0, 'PCM16 enhancement did not improve SI-SDR.'
assert integer_model.stat().st_size <= 99000
print(json.dumps(validation_reports['embedded']['model']['io_statistics'], indent=2))
print(json.dumps(validation_reports['embedded']['comparison'], indent=2))

def file_sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
frozen_selection = {
    'selection_split': 'training speakers p226 and p287; full validation only',
    'validation_manifest_sha256': file_sha256(manifests['val']),
    'float_checkpoint': str(float_checkpoint), 'float_sha256': file_sha256(float_checkpoint),
    'qat_checkpoint': str(qat_checkpoint), 'qat_sha256': file_sha256(qat_checkpoint),
    'integer_model': str(integer_model), 'integer_sha256': file_sha256(integer_model),
    'float_to_pcm16_si_sdri_drop_db': quantization_drop,
    'model_bytes': integer_model.stat().st_size,
}
for label, hash_key in [('float', 'float_sha256'), ('qat', 'qat_sha256'), ('embedded', 'integer_sha256')]:
    assert validation_reports[label]['model']['model_sha256'] == frozen_selection[hash_key]
assert FINAL_SELECTION_READY, 'Development search is still open; do not freeze or access test yet.'
selection_path = pathlib.Path('/content/esp32_runs/frozen_selection.json')
if selection_path.exists():
    assert json.loads(selection_path.read_text()) == frozen_selection, 'Selection is already frozen; do not replace it after test access.'
else:
    selection_path.write_text(json.dumps(frozen_selection, indent=2) + '\\n')
""")
code("""# Final test only after checkpoint selection and integer validation are complete.
# Once scored, keep this result fixed; further tuning needs a new holdout.
assert FINAL_SELECTION_READY, 'Development search is still open; keep official test sealed.'
frozen_selection = json.loads(pathlib.Path('/content/esp32_runs/frozen_selection.json').read_text())
for path_key, hash_key in [('float_checkpoint', 'float_sha256'), ('qat_checkpoint', 'qat_sha256'),
                          ('integer_model', 'integer_sha256')]:
    assert file_sha256(frozen_selection[path_key]) == frozen_selection[hash_key], 'Selected model changed.'
assert file_sha256(manifests['val']) == frozen_selection['validation_manifest_sha256']
manifests = prepare_voicebank_parquet('/content/voicebank', download=True, include_test=True)
perceptual_args = ['--perceptual'] if PERCEPTUAL else []
for label, module, argument, source, digest in [
    ('float', 'esp32_denoiser.evaluate', '--checkpoint', frozen_selection['float_checkpoint'], frozen_selection['float_sha256']),
    ('embedded', 'esp32_denoiser.embedded', '--integer-model', frozen_selection['integer_model'], frozen_selection['integer_sha256']),
]:
    output = pathlib.Path(f'/content/esp32_runs/{label}_test.json')
    if output.exists():
        saved = json.loads(output.read_text())
        assert saved['model']['model_sha256'] == digest
        assert saved['manifest_sha256'] == file_sha256(manifests['test'])
        assert not saved['limited_evaluation']
        assert ('perceptual' in saved) == PERCEPTUAL, 'Keep reporting settings fixed when reusing test results.'
        print(f'Reusing unchanged final report: {output}')
        continue
    extra = ['--io-format', 'pcm16'] if label == 'embedded' else []
    run_logged([sys.executable, '-u', '-m', module, argument, source,
                '--manifest', str(manifests['test']),
                '--output', str(output), '--audio-dir', f'/content/esp32_runs/{label}_examples',
                *extra, *perceptual_args])
final_report = json.loads(pathlib.Path('/content/esp32_runs/embedded_test.json').read_text())
print(json.dumps({key: final_report[key] for key in ('summary', 'model', 'perceptual') if key in final_report}, indent=2))
""")
code("""# Save all evidence, parameters, and reproducible source before disconnecting.
import shutil
source_copy = pathlib.Path('/content/esp32_runs/source')
# Copy exactly the audited source inventory; omit Python caches, SDK downloads,
# compiled libraries and any other files generated inside the working tree.
for relative in [*manifest, 'SOURCE_MANIFEST.json']:
    destination = source_copy / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(project / relative, destination)
manifest_copy = pathlib.Path('/content/esp32_runs/manifests')
manifest_copy.mkdir(parents=True, exist_ok=True)
for source in manifests.values():
    shutil.copy2(source, manifest_copy / pathlib.Path(source).name)
shutil.copy('/content/voicebank/manifests/provenance.json', '/content/esp32_runs/dataset_provenance.json')
archive = shutil.make_archive('/content/esp32_run_artifacts', 'zip', '/content/esp32_runs')
from google.colab import files
files.download(archive)
""")
notebook={'nbformat':4,'nbformat_minor':4,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'},'accelerator':'GPU'},'cells':cells}
path=ROOT/'notebooks/ESP32_S3_Training.ipynb'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(notebook,indent=1)+'\n')
print(path)
