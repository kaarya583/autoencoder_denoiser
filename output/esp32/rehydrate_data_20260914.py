"""Audit freshly downloaded training data against the restored experiment inputs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
import time

import psutil

ROOT = Path('/content/esp32_runs')
WORK = Path('/content/recovery_20260914')


def main():
    deadline = time.monotonic() + 3 * 3600
    while not all((WORK / (name + '.done')).is_file() for name in ('voicebank', 'extra')):
        if time.monotonic() > deadline:
            raise TimeoutError('Audio download exceeded three hours')
        process = psutil.Process(int((WORK / 'download_pid').read_text()))
        if process.status() == psutil.STATUS_ZOMBIE or not any(
                'prepare_voicebank_parquet' in arg for arg in process.cmdline()):
            raise RuntimeError('Audio downloader stopped: ' + (WORK / 'downloads.log').read_text()[-2000:])
        time.sleep(15)

    helper = ROOT / 'recover_colab_training.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest() != '775e0bf0344d3d2f4bbd9b64070b7f6a019ecdda19ff87e92e97193e5f4d8fb2':
        raise RuntimeError('Restored data rehydration helper changed')
    sys.path.insert(0, '/content/esp32_project')
    spec = importlib.util.spec_from_file_location('original_rehydration', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.audit_source()
    module.RECOVERY = ROOT / 'recovery_20260914' / 'data_audit'
    module.RECOVERY.mkdir(parents=True, exist_ok=False)
    for relative, expected in module.MANIFESTS.items():
        source = ROOT / relative
        module.require(module.sha(source) == expected, 'Restored manifest changed: ' + relative)
        module.preserve(source, module.RECOVERY / 'inputs' / relative)
    for folder in ('voicebank_manifests', 'extra_manifests', 'external_development_data'):
        module.preserve(ROOT / folder / 'provenance.json',
                        module.RECOVERY / 'inputs' / folder / 'provenance.json')
    # The fresh downloader's completion markers above supersede historical markers.
    module.wait_for_data = lambda marker: None
    with ThreadPoolExecutor(max_workers=2) as pool:
        voice = pool.submit(module.prepare_voice)
        extra = pool.submit(module.prepare_extra)
        evidence = dict(voicebank=voice.result(), extra_manifests=extra.result())
    evidence['development_manifests'] = module.prepare_development()
    evidence.update(created_unix=time.time(), official_test_accessed=False,
                    source_manifest_sha256=module.SOURCE_MANIFEST_SHA)
    module.audit_source()
    marker = ROOT / 'recovery_20260914' / 'data_ready.json'
    module.write_json(marker, evidence)
    print(json.dumps(dict(event='rehydration_complete', marker=str(marker))), flush=True)


if __name__ == '__main__':
    main()
