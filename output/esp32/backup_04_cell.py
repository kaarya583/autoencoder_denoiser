from pathlib import Path
import json,hashlib,zipfile,time
from google.colab import files
runs=Path('/content/esp32_runs')
archive=Path('/content/esp32_progress_backup_04.zip')
selected=set()
for marker in runs.glob('*/external_development/best.json'):
    record=json.loads(marker.read_text())
    selected.add(Path(record['selected_checkpoint']).resolve())
manifest={}
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as bundle:
    for p in sorted(runs.rglob('*')):
        if not p.is_file() or p.name.endswith('.tmp') or '__pycache__' in p.parts: continue
        if p.name.startswith('selected_') and p.suffix=='.pt' and p.resolve() not in selected: continue
        if p.name=='evaluating.pt': continue
        data=p.read_bytes(); name=str(p.relative_to(runs))
        bundle.writestr(name,data); manifest[name]=hashlib.sha256(data).hexdigest()
    for root,prefix in [(Path('/content/voicebank/manifests'),'voicebank_manifests'),(Path('/content/extra_audio/manifests'),'extra_manifests'),(Path('/content/extra_audio/development'),'external_development_data')]:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl'):
                data=p.read_bytes();name=prefix+'/'+str(p.relative_to(root))
                bundle.writestr(name,data);manifest[name]=hashlib.sha256(data).hexdigest()
    bundle.writestr('BACKUP_MANIFEST.json',json.dumps({'created_unix':time.time(),'files':manifest,'note':'Atomic checkpoint files; active logs may end with an incomplete last line. No training audio included.'},indent=2))
print('BACKUP04_BYTES',archive.stat().st_size)
print('BACKUP04_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest())
print('BACKUP04_FILES',len(manifest))
files.download(str(archive))
