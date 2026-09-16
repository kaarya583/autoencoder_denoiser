"""One-time, audited canonicalization of the legacy raw GTCRN selector marker.

Only adds normalize_input=False and rms_floor=0.0001, defaults proven by loading
the hash-verified selected checkpoint. Weights, metrics, hashes, and paths stay
unchanged. Must run while no selector holds this run's selector.lock.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import sys
import time

PROJECT = Path('/content/esp32_project')
RUN = Path('/content/esp32_runs/float_gtcrn_broad')
CHECKPOINT_SHA = '03e98970e65d6be7076b6b13e54943fd74eb119a02301c90b45d6ad1aa3b67a7'
DEFAULTS = {'normalize_input': False, 'rms_floor': 0.0001}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def preserve(path, data):
    if path.exists():
        require(path.read_bytes() == data, f'Immutable audit artifact differs: {path}')
    else:
        with path.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())


def atomic_write(path, data):
    temporary = path.with_name(path.name + '.migration.tmp')
    with temporary.open('wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    sys.path.insert(0, str(PROJECT))
    from esp32_denoiser.evaluate import load_checkpoint
    from esp32_denoiser.gtcrn_model import GTCRNConfig
    import torch

    directory = RUN / 'external_development'
    marker = directory / 'best.json'
    audit_dir = directory / 'recovery_metadata_20260913'
    with (directory / 'selector.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_bytes = marker.read_bytes()
        original = json.loads(original_bytes)
        checkpoint = Path(original['selected_checkpoint']).resolve()
        require(checkpoint.parent == directory.resolve(), 'Selected checkpoint is outside this selector directory')
        checkpoint_bytes = checkpoint.read_bytes()
        require(digest(checkpoint_bytes) == original['checkpoint_sha256'] == CHECKPOINT_SHA,
                'The original selected checkpoint identity changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model, current = load_checkpoint(checkpoint, 'cpu')
        require(current['model_sha256'] == CHECKPOINT_SHA and saved['epoch'] == 35,
                'Unexpected loaded checkpoint identity')
        require(current['model_kind'] == original['model']['model_kind'] == 'gtcrn', 'Unexpected model kind')
        for field in ('checkpoint_epoch', 'phase', 'precision'):
            require(current[field] == original['model'][field], f'Metadata changed beyond config defaults: {field}')
        effective = current['model_config']
        require(model.config == GTCRNConfig.from_checkpoint(saved['model_config']), 'Effective checkpoint config mismatch')
        require(all(key not in saved['model_config'] for key in DEFAULTS), 'Checkpoint is not the legacy raw model')
        require(all(effective[key] == value for key, value in DEFAULTS.items()), 'Current legacy defaults changed')
        audit_path = audit_dir / 'migration.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            require(digest(original_bytes) == audit['after_sha256'], 'Previously migrated marker changed; inspect manually')
            require(original['model']['model_config'] == effective, 'Migrated config differs from checkpoint')
            print(json.dumps(dict(status='already_applied', audit=str(audit_path))), flush=True)
            return
        previous = original['model']['model_config']
        require(set(effective) - set(previous) == set(DEFAULTS), 'Unexpected added architecture fields')
        require(set(previous) - set(effective) == set(), 'Architecture fields were removed')
        require(all(previous[key] == effective[key] for key in previous), 'Existing architecture values changed')
        require(previous == saved['model_config'], 'Legacy marker differs from the stored checkpoint config')
        updated = json.loads(original_bytes)
        updated['model']['model_config'] = effective
        updated_bytes = (json.dumps(updated, indent=2) + '\n').encode()
        roundtrip = json.loads(updated_bytes)
        roundtrip['model']['model_config'] = previous
        require(roundtrip == original, 'Migration would alter values beyond the two config defaults')
        audit_dir.mkdir(exist_ok=True)
        preserve(audit_dir / 'original_best.json', original_bytes)
        preserve(audit_dir / 'selected_checkpoint.pt', checkpoint_bytes)
        preserve(audit_dir / 'canonical_best.json', updated_bytes)
        audit = dict(schema=1, created_unix=time.time(), checkpoint_sha256=CHECKPOINT_SHA,
            before_sha256=digest(original_bytes), after_sha256=digest(updated_bytes), added_defaults=DEFAULTS,
            before_model_config=previous, after_model_config=effective,
            script_sha256=digest(Path(__file__).read_bytes()), source_sha256={
                str(path.relative_to(PROJECT)): digest(path.read_bytes()) for path in
                (PROJECT / 'esp32_denoiser/evaluate.py', PROJECT / 'esp32_denoiser/gtcrn_model.py')},
            invariant='Only two omitted defaults were made explicit; model weights, metric results, '
                      'checkpoint identity, and all other marker fields are unchanged.')
        preserve(audit_dir / 'planned_migration.json', (json.dumps(audit, indent=2) + '\n').encode())
        require(marker.read_bytes() == original_bytes, 'Selector marker changed during migration')
        atomic_write(marker, updated_bytes)
        require(marker.read_bytes() == updated_bytes and checkpoint.read_bytes() == checkpoint_bytes,
                'Post-migration marker/checkpoint verification failed')
        atomic_write(audit_path, (json.dumps(audit, indent=2) + '\n').encode())
        print(json.dumps(dict(status='applied', added_defaults=DEFAULTS,
                              checkpoint_sha256=CHECKPOINT_SHA, audit=str(audit_path))), flush=True)


if __name__ == '__main__':
    main()
