import pathlib,subprocess,json,time,hashlib
r=pathlib.Path('/content/esp32_runs');project=pathlib.Path('/content/esp32_project')
# Reuse the GPU slot of the original frequency trial after it terminates.
p=pathlib.Path('/proc/30868/cmdline')
while p.exists() and b'esp32_frequency_float.json' in p.read_bytes():time.sleep(15)
configs=[project/'configs/esp32_zero_bias_broad_float.json',project/'configs/esp32_zero_bias_finetune_float.json',project/'configs/esp32_zero_bias_spectral_float.json',project/'configs/esp32_zero_bias_distillation_control_float.json',r/'distillation_calibrated_config.json']
for config in configs:
 if not config.exists():raise RuntimeError('Missing calibrated configuration: '+str(config))
 d=json.loads(config.read_text());out=pathlib.Path(d['output_dir']);assert not out.exists(),'Refusing to overwrite '+str(out)
 record={'event':'start','config':str(config),'sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'output':str(out),'unix_time':time.time()};print(json.dumps(record),flush=True)
 with open(r/(out.name+'.log'),'w') as log:
  process=subprocess.Popen(['python','-u','-m','esp32_denoiser.train','--config',str(config)],cwd=project,stdout=log,stderr=subprocess.STDOUT);code=process.wait()
 print(json.dumps({'event':'finished','output':str(out),'returncode':code,'unix_time':time.time()}),flush=True)
 if code:raise RuntimeError('Training failed: '+str(config))
