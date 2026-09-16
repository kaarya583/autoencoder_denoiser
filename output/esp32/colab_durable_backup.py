"""Drive recovery chain: one full ZIP, then changed files every 15 minutes.

Each ZIP contains a SHA-256 manifest, the preceding ZIP name, and removed paths.
Restore the full ZIP followed by its deltas in order, applying removed paths.
Archives are never deleted. Atomic checkpoint inodes are read once; log tails
may contain an unfinished line. Immutable selected checkpoints accompany their
captured best.json pointers even when the selector advances during a snapshot.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import time
import uuid
import zipfile

ROOT = Path('/content/esp32_runs')
DRIVE = Path('/content/drive/MyDrive/Colab Notebooks/ESP32 Frontier Experiments')
STATE = ROOT / 'durable_backup_state.json'
INTERVAL_SECONDS = 15 * 60
MAX_SECONDS = 24 * 3600


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(previous):
    if not DRIVE.is_dir() or not os.path.ismount('/content/drive'):
        raise RuntimeError('Google Drive is not mounted; refusing a local-only backup')
    destination = DRIVE / 'recovery_snapshots'
    destination.mkdir(exist_ok=True)
    pinned = {}
    selected = set()
    for marker in sorted(ROOT.glob('*/external_development/best.json')):
        data = marker.read_bytes()
        record = json.loads(data)
        checkpoint = Path(record['selected_checkpoint']).resolve()
        if not checkpoint.is_relative_to(ROOT.resolve()):
            raise RuntimeError('Selected checkpoint is outside the run root')
        checkpoint_data = checkpoint.read_bytes()
        if digest(checkpoint_data) != record['checkpoint_sha256']:
            raise RuntimeError('Selected checkpoint hash mismatch')
        pinned[str(marker.relative_to(ROOT))] = data
        pinned[str(checkpoint.relative_to(ROOT))] = checkpoint_data
        selected.add(checkpoint)
    inventory = {}
    files = {}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts:
            continue
        if path.name.endswith(('.tmp', '.part', '.lock')) or path.name in ('evaluating.pt', STATE.name):
            continue
        if path.name.startswith('selected_') and path.suffix == '.pt' and path.resolve() not in selected:
            continue
        files[str(path.relative_to(ROOT))] = path
    for root, prefix in [(Path('/content/voicebank/manifests'), 'voicebank_manifests'),
                         (Path('/content/extra_audio/manifests'), 'extra_manifests'),
                         (Path('/content/extra_audio/development'), 'external_development_data')]:
        if root.exists():
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.jsonl'):
                    files[prefix + '/' + str(path.relative_to(root))] = path
    # Captured pointers and selected bytes win over any later live-file version.
    all_names = sorted(set(files) | set(pinned))
    old_hashes = previous.get('inventory', {})
    timestamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    name = f'esp32_{timestamp}_{uuid.uuid4().hex[:8]}.zip'
    local = Path('/content') / name
    included = {}
    with zipfile.ZipFile(local, 'x', zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for relative in all_names:
            try:
                data = pinned[relative] if relative in pinned else files[relative].read_bytes()
            except FileNotFoundError:
                # Transient files can disappear; mandatory pinned artifacts cannot.
                if relative in pinned:
                    raise
                continue
            current = digest(data)
            inventory[relative] = current
            if old_hashes.get(relative) != current:
                bundle.writestr(relative, data)
                included[relative] = current
        removed = sorted(set(old_hashes) - set(inventory))
        manifest = dict(schema=1, created_unix=time.time(), mode='delta' if previous else 'full',
            previous_archive=previous.get('archive'), base_archive=previous.get('base_archive', name),
            included_sha256=included, inventory_sha256=inventory, removed=removed,
            note='No training audio. Apply deltas in sequence; active logs may have an incomplete trailing line.')
        bundle.writestr('DURABLE_BACKUP_MANIFEST.json', json.dumps(manifest, indent=2))
    # Copy an exclusively named temporary object, verify bytes on Drive, then publish.
    target = destination / name
    temporary = destination / (name + '.part')
    local_sha = hashlib.sha256()
    with local.open('rb') as source, temporary.open('xb') as output:
        for block in iter(lambda: source.read(8 << 20), b''):
            local_sha.update(block)
            output.write(block)
    remote_sha = hashlib.sha256()
    with temporary.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            remote_sha.update(block)
    if local_sha.digest() != remote_sha.digest():
        raise RuntimeError('Drive backup readback hash mismatch')
    if target.exists():
        raise RuntimeError('Refusing to replace an existing Drive archive')
    temporary.rename(target)
    state = dict(archive=name, base_archive=manifest['base_archive'], inventory=inventory,
                 archive_sha256=local_sha.hexdigest(), created_unix=manifest['created_unix'])
    pending = STATE.with_suffix('.tmp')
    pending.write_text(json.dumps(state, indent=2))
    pending.replace(STATE)
    local.unlink()
    print(json.dumps(dict(event='durable_backup', archive=str(target), included=len(included),
                          sha256=state['archive_sha256'], mode=manifest['mode'])), flush=True)
    return state


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'durable_backup.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(STATE.read_text()) if STATE.exists() else {}
        if previous and not (DRIVE / 'recovery_snapshots' / previous['archive']).is_file():
            raise RuntimeError('The previous Drive archive is missing; inspect the recovery chain')
        deadline = time.monotonic() + MAX_SECONDS
        while True:
            try:
                previous = snapshot(previous)
            except Exception as error:
                print(json.dumps(dict(event='durable_backup_failed', error=repr(error))), flush=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(INTERVAL_SECONDS, remaining))


if __name__ == '__main__':
    main()
