"""Restore audited training data, then resume only the two saved GTCRN runs.

Run from /content/esp32_project after source, backup05, and probes04 restoration.
No official test data is downloaded. A failed audit starts no training jobs.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path('/content/esp32_runs')
PROJECT = Path('/content/esp32_project')
VOICE = Path('/content/voicebank')
EXTRA = Path('/content/extra_audio')
RECOVERY = ROOT / 'recovery_20260913'
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'MKL_NUM_THREADS': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1'}
SOURCE_MANIFEST_SHA = 'b32851d42af2442ac051e49711d2639244dd8a8506585cc24cea9e7c32e4a9a2'
LARGE_SCRIPT_SHA = 'ad3abfb423ca52201a94154533ba507bb96831a21e65c5d1914d8793f8fcd8c8'
RUNS = {
    'float_gtcrn_broad': (43, '76cde5b5fc5756f428dcae241f4c9a1db2b44425c8861041c7ede75ac7b65380',
                         '0b50904a1ee67c264b2686b88ae34683e302173efabb62ef483d140c479361d4'),
    'float_gtcrn_broad_normalized': (25, 'd4ea91b91e473184dc56389d5a00696993a3c38397c5a8050bf723a9d02d5914',
                                    'c3d04ebd86dff2bc860b0faea8fd96c31f1e09881776b44dc304ddf412e75604'),
}
CHILDREN = []
MANIFESTS = {
    'voicebank_manifests/train.jsonl': 'c0b9ea9aa7b13d30aadd30f76274f6b2f932d802ae85c55426c2c3e13a4a247c',
    'voicebank_manifests/val.jsonl': 'c1e0ccf95766f1542da3e8eebc1cf33107ca6ecf3ccf59c7a2d8078c171f5668',
    'extra_manifests/speech_train.jsonl': '24acf8d2789c019fb966b39754bd71376e016b4d1169cd611031f94271d7ac77',
    'extra_manifests/noise_train.jsonl': 'e953b3952bb66377ad6bc67b93cb1b2148511cca952b41b0fb038a93a79a8b03',
    'extra_manifests/speech_val.jsonl': 'd89456f5f272cdd1678f3450b172f4c6354611bda7193f5ac9deac354561729d',
    'extra_manifests/noise_val.jsonl': 'fa9349454796e08f76c3e9e02b2edc1519bb69f3819a1b84b30435b98137f711',
    'external_development_data/mixtures.jsonl': 'b82497d1febcfba26e1449ac57596e8f6983071e82d48482a1d6de00a5a84742',
    'external_development_data/clean.jsonl': '5095c806d052665f14ca8c2ca8789caaf6020c425e80f8daae752fcd4db6c808',
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def event(name, **fields):
    print(json.dumps(dict(event=name, unix=time.time(), **fields)), flush=True)


def preserve(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(sha(source) == sha(destination), f'Immutable recovery input differs: {destination}')
    else:
        with destination.open('xb') as handle:
            handle.write(source.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_path(root, path):
    resolved = path.resolve()
    require(resolved.is_relative_to(root.resolve()), f'Path escapes data root: {path}')
    return resolved


def audit_source():
    manifest_path = PROJECT / 'SOURCE_MANIFEST.json'
    require(sha(manifest_path) == SOURCE_MANIFEST_SHA, 'Unexpected source release manifest')
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest.items():
        path = safe_path(PROJECT, PROJECT / relative)
        require(sha(path) == expected, f'Source mismatch: {relative}')
    require(sha(ROOT / 'large_colab_round.py') == LARGE_SCRIPT_SHA, 'Large-run queue script changed')
    return manifest


def freeze_inputs():
    for relative, expected in MANIFESTS.items():
        source = ROOT / relative
        require(sha(source) == expected, f'Backup manifest mismatch: {relative}')
        preserve(source, RECOVERY / 'inputs' / relative)
    for folder in ('voicebank_manifests', 'extra_manifests', 'external_development_data'):
        preserve(ROOT / folder / 'provenance.json', RECOVERY / 'inputs' / folder / 'provenance.json')
    for name, (_, checkpoint_sha, config_sha) in RUNS.items():
        run = ROOT / name
        require(not (run / 'summary.json').exists(), f'Unexpected completed predecessor: {name}')
        require(not (ROOT / (name + '_large')).exists(), f'Large-run directory already exists: {name}')
        require(sha(run / 'last.pt') == checkpoint_sha, f'Restored checkpoint mismatch: {name}')
        require(sha(run / 'config.json') == config_sha, f'Restored configuration mismatch: {name}')
        for filename in ('last.pt', 'best.pt', 'config.json', 'provenance.json', 'history.jsonl'):
            preserve(run / filename, RECOVERY / 'inputs' / name / filename)


def wait_for_data(marker):
    deadline = time.monotonic() + 3 * 3600
    while not marker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Data preparation did not finish: {marker}')
        time.sleep(15)


def prepare_voice():
    import soundfile as sf

    wait_for_data(ROOT / 'recovery_voicebank_download.done')
    mapping_path = RECOVERY / 'voicebank_path_mapping.jsonl'
    temporary = mapping_path.with_suffix('.tmp')
    count = 0
    with temporary.open('w') as mapping:
        for split in ('train', 'val'):
            generated = VOICE / 'manifests' / (split + '.jsonl')
            original = RECOVERY / 'inputs/voicebank_manifests' / generated.name
            old_rows, new_rows = rows(original), rows(generated)
            old = {row['id']: row for row in old_rows}
            new = {row['id']: row for row in new_rows}
            require(len(old) == len(old_rows) == len(new_rows) == len(new), 'Duplicate VoiceBank IDs')
            require(old.keys() == new.keys(), f'VoiceBank {split} IDs changed')
            for identifier, before in old.items():
                after = new[identifier]
                require({k: v for k, v in before.items() if k not in ('clean', 'noisy')} ==
                        {k: v for k, v in after.items() if k not in ('clean', 'noisy')},
                        f'VoiceBank metadata changed: {identifier}')
                for role in ('clean', 'noisy'):
                    source = safe_path(VOICE, generated.parent / after[role])
                    destination = safe_path(VOICE, generated.parent / before[role])
                    info = sf.info(source)
                    require(info.samplerate == before['sample_rate'] and info.channels == 1 and
                            info.frames == before['samples'], f'Invalid decoded VoiceBank WAV: {identifier}')
                    digest = sha(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        require(sha(destination) == digest, f'Existing VoiceBank alias differs: {destination}')
                    else:
                        try:
                            os.link(source, destination)
                        except OSError:
                            shutil.copyfile(source, destination)
                        require(sha(destination) == digest, 'VoiceBank alias copy verification failed')
                    mapping.write(json.dumps(dict(id=identifier, split=split, role=role,
                        generated_path=str(source), restored_path=str(destination), wav_sha256=digest)) + '\n')
                    count += 1
            shutil.copyfile(generated, RECOVERY / ('voicebank_regenerated_' + split + '.jsonl'))
            shutil.copyfile(original, generated)
            require(sha(generated) == sha(original), 'Restored VoiceBank manifest differs')
    temporary.replace(mapping_path)
    archives = {str(path.relative_to(VOICE)): sha(path) for path in
                sorted((VOICE / 'archives/hf16k/data').glob('train-*.parquet'))}
    require(len(archives) == 5, 'Expected exactly five VoiceBank training shards')
    evidence = dict(files=count, mapping_sha256=sha(mapping_path), parquet_sha256=archives,
        policy='Pinned mirror re-decoded; metadata matched by ID; original manifest paths restored via verified aliases. '
               'Backup has no original waveform hashes, so this does not claim an independent historical byte comparison.')
    write_json(RECOVERY / 'voicebank_rehydration.json', evidence)
    event('voicebank_ready', files=count)
    return evidence


def prepare_extra():
    wait_for_data(ROOT / 'recovery_extra_download.done')
    original = json.loads((RECOVERY / 'inputs/extra_manifests/provenance.json').read_text())
    generated = json.loads((EXTRA / 'manifests/provenance.json').read_text())
    for kind in ('speech', 'noise'):
        expected = original['sources'][kind]
        require(generated['sources'][kind]['sha256'] == expected['sha256'], f'{kind} source identity changed')
        archive = EXTRA / 'archives' / expected['filename']
        require(sha(archive) == expected['sha256'], f'{kind} archive bytes changed')
    result = {}
    for name in ('speech_train', 'speech_val', 'noise_train', 'noise_val'):
        path = EXTRA / 'manifests' / (name + '.jsonl')
        result[name] = sha(path)
        require(result[name] == MANIFESTS[f'extra_manifests/{name}.jsonl'], f'Extra manifest changed: {name}')
    event('extra_sources_ready')
    return result


def prepare_development():
    from esp32_denoiser.development import prepare_development as render
    outputs = render(EXTRA, num_mixtures=500, clean_examples=100, crop_seconds=3, seed=2026)
    count = 0
    for suite in ('mixtures', 'clean'):
        path = outputs[suite]
        require(sha(path) == MANIFESTS[f'external_development_data/{suite}.jsonl'],
                f'Rendered development manifest changed: {suite}')
        for row in rows(path):
            for role in ('clean', 'noisy'):
                waveform = safe_path(EXTRA, path.parent / row[role])
                require(sha(waveform) == row[role + '_sha256'], 'Rendered development WAV differs')
                count += 1
    event('development_ready', verified_wav_files=count)
    return {suite: sha(outputs[suite]) for suite in ('mixtures', 'clean')}


def launch(command, log_name):
    with (RECOVERY / log_name).open('a') as log:
        child = subprocess.Popen(command, cwd=PROJECT, env=ENV, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN.append((child, log_name))
    return dict(pid=child.pid, command=command, log=str(RECOVERY / log_name))


def main():
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    RECOVERY.mkdir(parents=True, exist_ok=True)
    with (RECOVERY / 'recovery.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (RECOVERY / 'launches.json').exists(), 'Launch record exists; inspect processes before retrying')
        source = audit_source()
        freeze_inputs()
        write_json(RECOVERY / 'provenance.json', dict(created_unix=time.time(), source_sha256=source,
            recovery_script_sha256=sha(Path(__file__)), source_bundle_sha256=
            'eceb96444d0988d487f7a087190b5baf5437d54f949190dbf0ef67ec1e6a64d3',
            official_test_accessed=False, resumed_runs=list(RUNS)))
        event('rehydration_started')
        with ThreadPoolExecutor(max_workers=2) as pool:
            voice = pool.submit(prepare_voice)
            extra = pool.submit(prepare_extra)
            voice_evidence, extra_evidence = voice.result(), extra.result()
        development = prepare_development()
        audit_source()
        import torch
        require(torch.cuda.is_available(), 'CUDA unavailable; training was not launched')
        write_json(ROOT / 'recovery_data_ready.json', dict(created_unix=time.time(),
            voicebank=voice_evidence, extra_manifests=extra_evidence, development_manifests=development,
            source_manifest_sha256=SOURCE_MANIFEST_SHA, official_test_accessed=False))
        launches = []
        # Write before the first subprocess: partial launches require inspection, not blind replay.
        write_json(RECOVERY / 'launches.json', dict(status='launching', processes=launches))
        for name, (epoch, digest, _) in RUNS.items():
            inputs = RECOVERY / 'inputs' / name
            checkpoint = inputs / 'last.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            require(saved['epoch'] == epoch and sha(checkpoint) == digest, f'Resume checkpoint changed: {name}')
            config = json.loads((inputs / 'config.json').read_text())
            history = rows(inputs / 'history.jsonl')
            require(history[-1]['epoch'] == epoch, 'Checkpoint/history epoch mismatch')
            elapsed = float(history[-1]['elapsed_seconds'])
            remaining = min(6.0, float(config['max_hours']) - elapsed / 3600)
            require(remaining > 0, f'No predecessor time budget remains: {name}')
            config.update(resume=str(checkpoint), resume_optimizer=True, max_hours=remaining)
            config_path = RECOVERY / (name + '_resume_config.json')
            write_json(config_path, config)
            launches.append(launch([sys.executable, '-u', '-m', 'esp32_denoiser.train', '--config', str(config_path)],
                                   name + '_training.log'))
            write_json(RECOVERY / 'launches.json', dict(status='launching', processes=launches))
            launches.append(launch([sys.executable, '-u', '-m', 'esp32_denoiser.development_checkpoints',
                '--run', str(ROOT / name), '--manifest', str(EXTRA / 'development/mixtures.jsonl'),
                '--device', 'cuda', '--every-epochs', '5', '--max-hours', str(remaining + .5)],
                name + '_external_watch.log'))
            write_json(RECOVERY / 'launches.json', dict(status='launching', processes=launches))
            event('float_resumed', run=name, next_epoch=epoch + 1, remaining_hours=remaining)
        launches.append(launch([sys.executable, '-u', str(ROOT / 'large_colab_round.py')], 'large_colab_round.log'))
        write_json(RECOVERY / 'launches.json', dict(status='launched', processes=launches))
        event('recovery_launched', process_count=len(launches))
        # The visible notebook cell remains attached to actual training work.
        # This is a job supervisor; it never sends synthetic activity to Colab.
        pending = list(CHILDREN)
        failed = []
        while pending:
            for child, name in list(pending):
                code = child.poll()
                if code is None:
                    continue
                pending.remove((child, name))
                event('child_completed', log=name, returncode=code)
                if code:
                    failed.append(dict(log=name, returncode=code))
                    # A failed predecessor/selector cannot satisfy the large queue.
                    for queued, queued_name in pending:
                        if queued_name == 'large_colab_round.log' and queued.poll() is None:
                            queued.terminate()
            if pending:
                time.sleep(15)
        write_json(RECOVERY / 'completion.json', dict(completed_unix=time.time(), failures=failed))
        require(not failed, f'Recovered jobs failed; inspect logs: {failed}')
        event('recovery_training_complete')


if __name__ == '__main__':
    main()
