import pathlib,subprocess,sys,os,signal,time,json
from esp32_denoiser.extra_data import SOURCES,_check_archive
old=pathlib.Path('/proc/32917/cmdline')
if old.exists() and b'esp32_denoiser.extra_data' in old.read_bytes():
 os.kill(32917,signal.SIGTERM)
 for _ in range(50):
  if not old.exists(): break
  time.sleep(.1)
 if old.exists() and b'esp32_denoiser.extra_data' in old.read_bytes(): raise RuntimeError('Old downloader did not stop')
root=pathlib.Path('/content/extra_audio/archives'); root.mkdir(exist_ok=True,parents=True); records=[]
for kind in ['speech','noise']:
 source=SOURCES[kind]; target=root/source.filename
 if not target.exists():
  part=target.with_suffix(target.suffix+'.part')
  command=['aria2c','--continue=true','--auto-file-renaming=false','--allow-overwrite=true','--file-allocation=none','--max-connection-per-server=4','--split=4','--min-split-size=32M','--max-tries=5','--retry-wait=5','--summary-interval=30','--console-log-level=warn','--dir',str(root),'--out',part.name,source.urls[0]]
  print('Downloading',source.filename,flush=True); subprocess.run(command,check=True)
  hashes=_check_archive(part,source); part.replace(target)
 else: hashes=_check_archive(target,source)
 records.append({'source':source.urls[0],'file':target.name,'bytes':target.stat().st_size,**hashes})
 pathlib.Path('/content/esp32_runs/parallel_downloads.json').write_text(json.dumps(records,indent=2))
 print('Verified archive',target.name,flush=True)
subprocess.run([sys.executable,'-u','-m','esp32_denoiser.extra_data','--root','/content/extra_audio'],check=True)
