"""Verify a pinned Drive snapshot chain and atomically restore its exact inventory.

No training audio is included in these backups. Restoration writes only the
run directory; backed-up data manifests retain their backup prefixes. Existing
nonempty output is rejected unless --quarantine-existing is explicitly given,
in which case the complete old directory is preserved beside the replacement.
The audit is a sibling file, so the restored directory contains exactly the
files authenticated by the final pinned inventory.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
import zipfile

DEFAULT_SNAPSHOTS = '/content/drive/MyDrive/Colab Notebooks/ESP32 Frontier Experiments/recovery_snapshots'
DEFAULT_TARGET = 'esp32_20260914T040022Z_33aac2b4.zip'
DEFAULT_SHA256 = 'bf48918f323753595e0adfe29b493bae163e9f30408045b022990ba39d73b1c5'
MANIFEST_NAME = 'DURABLE_BACKUP_MANIFEST.json'
HASH_RE = re.compile(r'[0-9a-f]{64}\Z')
MAX_CHAIN = 512
MAX_MANIFEST_BYTES = 64 << 20
MAX_UNCOMPRESSED_BYTES = 32 << 30


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def relative_name(value):
    require(isinstance(value, str) and value and '\\' not in value and '\x00' not in value
            and ':' not in value, f'Unsafe relative path: {value!r}')
    parsed = PurePosixPath(value)
    require(not parsed.is_absolute() and str(parsed) == value
            and all(part not in ('', '.', '..') for part in value.split('/')),
            f'Unsafe relative path: {value!r}')
    return value


def archive_name(value):
    relative_name(value)
    require('/' not in value and value.endswith('.zip'), f'Unsafe archive name: {value!r}')
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key!r}')
        result[key] = value
    return result


def hash_mapping(value, field):
    require(isinstance(value, dict), f'{field} must be a mapping')
    for name, digest in value.items():
        relative_name(name)
        require(name != MANIFEST_NAME, 'Manifest name is reserved')
        require(isinstance(digest, str) and HASH_RE.fullmatch(digest), f'Invalid hash for {name}')
    names = set(value)
    for name in names:
        require(not any(str(parent) in names for parent in PurePosixPath(name).parents if str(parent) != '.'),
                f'File/directory path collision: {name}')
    return value


def read_manifest(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), 'Duplicate ZIP member name')
        require(MANIFEST_NAME in names, 'Missing backup manifest')
        for info in infos:
            relative_name(info.filename)
            kind = stat.S_IFMT(info.external_attr >> 16)
            require(not info.is_dir() and kind in (0, stat.S_IFREG), f'Non-regular ZIP member: {info.filename}')
            require(not info.flag_bits & 1, 'Encrypted ZIP members are not supported')
        require(sum(info.file_size for info in infos) <= MAX_UNCOMPRESSED_BYTES, 'Archive exceeds restoration size limit')
        require(archive.getinfo(MANIFEST_NAME).file_size <= MAX_MANIFEST_BYTES, 'Manifest is too large')
        data = archive.read(MANIFEST_NAME)
        manifest = json.loads(data, object_pairs_hook=unique_object)
        require(isinstance(manifest, dict) and type(manifest.get('schema')) is int and manifest['schema'] == 1,
                'Unsupported backup schema')
        require(manifest.get('mode') in ('full', 'delta'), 'Invalid backup mode')
        created = manifest.get('created_unix')
        require(type(created) in (int, float) and math.isfinite(created) and created > 0, 'Invalid creation timestamp')
        archive_name(manifest.get('base_archive'))
        if manifest.get('previous_archive') is not None:
            archive_name(manifest['previous_archive'])
        included = hash_mapping(manifest.get('included_sha256'), 'included_sha256')
        hash_mapping(manifest.get('inventory_sha256'), 'inventory_sha256')
        removed = manifest.get('removed')
        require(isinstance(removed, list), 'removed must be a list')
        for name in removed:
            relative_name(name)
        require(len(removed) == len(set(removed)), 'Duplicate removed path')
        require(set(names) == set(included) | {MANIFEST_NAME}, 'ZIP members do not exactly match included_sha256')
        return manifest, hashlib.sha256(data).hexdigest()


def validate_transition(previous_inventory, manifest, previous_name, base_name, name):
    included = manifest['included_sha256']
    inventory = manifest['inventory_sha256']
    removed = set(manifest['removed'])
    if previous_name is None:
        require(manifest['mode'] == 'full' and manifest['previous_archive'] is None, 'Chain does not begin with a full backup')
        require(manifest['base_archive'] == name and not removed, 'Invalid full backup base/removals')
        require(included == inventory, 'Full backup inventory is incomplete')
    else:
        require(manifest['mode'] == 'delta' and manifest['previous_archive'] == previous_name,
                'Broken predecessor link')
        require(manifest['base_archive'] == base_name, 'Base archive changes within chain')
        require(removed == set(previous_inventory) - set(inventory), 'Removed paths do not match inventory transition')
        changed = {key: value for key, value in inventory.items() if previous_inventory.get(key) != value}
        require(included == changed, 'Included files do not match inventory transition')
    reconstructed = {key: value for key, value in previous_inventory.items() if key not in removed}
    reconstructed.update(included)
    require(reconstructed == inventory, 'Inventory transition cannot reproduce declared snapshot')


def verify_destination(path, quarantine):
    require(path.is_absolute() and path != Path(path.anchor), 'Destination must be an absolute non-root directory')
    for ancestor in (path, *path.parents):
        require(not ancestor.is_symlink(), f'Destination has a symlink component: {ancestor}')
    require(not path.exists() or path.is_dir(), 'Destination exists and is not a directory')
    require(quarantine or not path.exists() or not any(path.iterdir()),
            'Destination is nonempty; explicitly use --quarantine-existing to preserve and replace it')


def restore(snapshots, destination, target=DEFAULT_TARGET, target_sha256=DEFAULT_SHA256, *, quarantine=False):
    snapshots = Path(snapshots).resolve()
    destination = Path(os.path.abspath(destination))
    archive_name(target)
    require(isinstance(target_sha256, str) and HASH_RE.fullmatch(target_sha256), 'Invalid target SHA-256')
    require(snapshots.is_dir(), 'Snapshot directory is missing')
    verify_destination(destination, quarantine)
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    audit_path = destination.parent / f'{destination.name}_restore_audit_{session}.json'
    with tempfile.TemporaryDirectory(prefix=destination.name + '_restore_', dir=destination.parent) as temporary:
        work = Path(temporary)
        cache = work / 'archives'
        staged = work / 'files'
        cache.mkdir()
        staged.mkdir()
        chain = []
        seen = set()
        name = target
        while True:
            require(name not in seen and len(seen) < MAX_CHAIN, 'Snapshot chain cycles or exceeds length limit')
            seen.add(name)
            source = snapshots / name
            require(source.is_file() and not source.is_symlink(), f'Archive missing or symlinked: {name}')
            cached = cache / name
            shutil.copyfile(source, cached)
            digest = sha_file(cached)
            if name == target:
                require(digest == target_sha256, 'Pinned target archive SHA-256 mismatch')
            manifest, manifest_sha = read_manifest(cached)
            chain.append(dict(name=name, archive_sha256=digest, manifest_sha256=manifest_sha,
                              manifest=manifest, cached=cached))
            if manifest['mode'] == 'full':
                require(manifest['previous_archive'] is None, 'Full backup unexpectedly references a predecessor')
                break
            require(manifest['previous_archive'] is not None, 'Delta backup has no predecessor')
            name = manifest['previous_archive']
        chain.reverse()
        inventory = {}
        previous_name = None
        base_name = chain[0]['name']
        total_bytes = 0
        previous_time = 0
        for entry in chain:
            manifest = entry['manifest']
            validate_transition(inventory, manifest, previous_name, base_name, entry['name'])
            require(manifest['created_unix'] >= previous_time, 'Archive timestamps move backwards')
            for relative in manifest['removed']:
                path = staged / relative
                require(path.is_file() and not path.is_symlink(), f'Removed tracked file is absent: {relative}')
                path.unlink()
                parent = path.parent
                while parent != staged and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
            with zipfile.ZipFile(entry['cached']) as archive:
                for relative, expected in manifest['included_sha256'].items():
                    path = staged / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.open(relative) as source, path.open('wb') as output:
                        for block in iter(lambda: source.read(8 << 20), b''):
                            total_bytes += len(block)
                            require(total_bytes <= MAX_UNCOMPRESSED_BYTES, 'Chain exceeds restoration size limit')
                            digest.update(block)
                            output.write(block)
                    require(digest.hexdigest() == expected, f'ZIP member SHA-256 mismatch: {entry["name"]}:{relative}')
            inventory = manifest['inventory_sha256']
            previous_name = entry['name']
            previous_time = manifest['created_unix']
        require(previous_name == target, 'Chain does not terminate at pinned target')
        actual = {str(path.relative_to(staged)): sha_file(path) for path in staged.rglob('*') if path.is_file()}
        require(actual == inventory, 'Restored files do not exactly match the pinned final inventory')
        # Recheck immediately before committing; no live directory was touched during validation.
        verify_destination(destination, quarantine)
        prior = None
        audit = dict(schema=1, target_archive=target, target_archive_sha256=target_sha256,
            base_archive=base_name, destination=str(destination), files=len(inventory),
            final_inventory_sha256=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
            inventory=inventory, archive_chain=[{key: item[key] for key in
                ('name', 'archive_sha256', 'manifest_sha256')} for item in chain],
            verified_member_bytes=total_bytes, created_unix=time.time(), status='verified_not_installed',
            note='Final content hashes are anchored by the pinned target archive; predecessor archives '
                 'have individually verified ZIP member hashes. No training audio was restored.')
        with audit_path.open('x') as handle:
            json.dump(audit, handle, indent=2)
        if destination.exists():
            if any(destination.iterdir()):
                prior = destination.with_name(destination.name + '_pre_restore_' + session)
                require(not prior.exists(), 'Quarantine destination unexpectedly exists')
                destination.rename(prior)
            else:
                destination.rmdir()
        try:
            require(not destination.exists(), 'Destination appeared before installation')
            staged.rename(destination)
        except BaseException:
            if prior is not None and not destination.exists():
                prior.rename(destination)
            raise
        audit.update(status='restored', completed_unix=time.time(), quarantined_directory=str(prior) if prior else None)
        pending = audit_path.with_suffix('.tmp')
        pending.write_text(json.dumps(audit, indent=2) + '\n')
        pending.replace(audit_path)
        return dict(destination=str(destination), files=len(inventory), archives=len(chain),
                    audit=str(audit_path), quarantined_directory=audit['quarantined_directory'])


def self_test():
    """Small synthetic success, removal, corruption, path, and overwrite checks."""
    import unittest

    class RestoreTests(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory()
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name).resolve()
            self.snapshots = self.root / 'snapshots'
            self.snapshots.mkdir()
            self.dest = self.root / 'runs'

        def archive(self, name, files, inventory=None, previous=None, removed=(), override=None):
            hashes = {key: hashlib.sha256(value).hexdigest() for key, value in files.items()}
            manifest = dict(schema=1, created_unix=2 if previous else 1, mode='delta' if previous else 'full',
                previous_archive=previous, base_archive='base.zip', included_sha256=hashes,
                inventory_sha256=hashes if inventory is None else inventory, removed=list(removed))
            if override:
                manifest.update(override)
            path = self.snapshots / name
            with zipfile.ZipFile(path, 'w') as bundle:
                for key, value in files.items():
                    bundle.writestr(key, value)
                bundle.writestr(MANIFEST_NAME, json.dumps(manifest))
            return path, manifest

        def test_delta_and_removal(self):
            self.archive('base.zip', {'old.txt': b'old', 'keep/a.pt': b'a'})
            final = {'keep/a.pt': hashlib.sha256(b'b').hexdigest()}
            path, _ = self.archive('final.zip', {'keep/a.pt': b'b'}, final, 'base.zip', ('old.txt',))
            result = restore(self.snapshots, self.dest, path.name, sha_file(path))
            self.assertEqual((self.dest / 'keep/a.pt').read_bytes(), b'b')
            self.assertFalse((self.dest / 'old.txt').exists())
            self.assertEqual(result['files'], 1)

        def test_corrupt_member(self):
            path, _ = self.archive('base.zip', {'a': b'bad'}, override={
                'included_sha256': {'a': '0' * 64}, 'inventory_sha256': {'a': '0' * 64}})
            with self.assertRaisesRegex(ValueError, 'member SHA-256 mismatch'):
                restore(self.snapshots, self.dest, path.name, sha_file(path))
            self.assertFalse(self.dest.exists())

        def test_traversal(self):
            path, _ = self.archive('base.zip', {'../escape': b'bad'})
            with self.assertRaisesRegex(ValueError, 'Unsafe relative path'):
                restore(self.snapshots, self.dest, path.name, sha_file(path))
            self.assertFalse((self.root / 'escape').exists())

        def test_nonempty_guard_and_quarantine(self):
            path, _ = self.archive('base.zip', {'new': b'new'})
            self.dest.mkdir()
            (self.dest / 'existing').write_bytes(b'keep')
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                restore(self.snapshots, self.dest, path.name, sha_file(path))
            result = restore(self.snapshots, self.dest, path.name, sha_file(path), quarantine=True)
            self.assertEqual((Path(result['quarantined_directory']) / 'existing').read_bytes(), b'keep')
            self.assertEqual({path.name for path in self.dest.iterdir()}, {'new'})

        def test_pinned_hash(self):
            path, _ = self.archive('base.zip', {'a': b'a'})
            with self.assertRaisesRegex(ValueError, 'Pinned target archive'):
                restore(self.snapshots, self.dest, path.name, '0' * 64)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RestoreTests))
    require(result.wasSuccessful(), 'Synthetic restoration tests failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', default=DEFAULT_SNAPSHOTS)
    parser.add_argument('--destination', default='/content/esp32_runs')
    parser.add_argument('--target', default=DEFAULT_TARGET)
    parser.add_argument('--target-sha256', default=DEFAULT_SHA256)
    parser.add_argument('--quarantine-existing', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(json.dumps(restore(args.snapshots, args.destination, args.target, args.target_sha256,
                                 quarantine=args.quarantine_existing)), flush=True)


if __name__ == '__main__':
    main()
