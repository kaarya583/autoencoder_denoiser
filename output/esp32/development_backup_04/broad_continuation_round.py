import pathlib,json,hashlib,shutil,subprocess,sys,time,os,signal
r=pathlib.Path('/content/esp32_runs');old=r/'float_frequency_gru';pid=47063
cmd=pathlib.Path(f'/proc/{pid}/cmdline')
if cmd.exists():
 assert 'configs/esp32_frequency_gru_float.json' in cmd.read_bytes().decode().replace(chr(0),' ')
 os.kill(pid,signal.SIGTERM)
 for _ in range(50):
  if not cmd.exists():break
  time.sleep(.1)
 assert not cmd.exists(),'Process did not exit; do not launch new jobs'
h=[json.loads(s) for s in (old/'history.jsonl').read_text().splitlines()];best=max(h,key=lambda x:x['si_sdri']);summary={'status':'pruned_uncompetitive','epoch':h[-1]['epoch'],'best_si_sdri':best['si_sdri'],'reason':'Matched encoder-BN shared-GRU branch is persistently stronger; this is manual pruning, not convergence.'};(old/'summary.json').write_text(json.dumps(summary,indent=2));print('PRUNED',summary,flush=True)
inputs=r/'broad_continuation_inputs';inputs.mkdir(exist_ok=False);marker=r/'float_zero_bias_broad/external_development/best.json';record=json.loads(marker.read_text());source=pathlib.Path(record['selected_checkpoint']);blob=source.read_bytes();assert hashlib.sha256(blob).hexdigest()==record['checkpoint_sha256'];(inputs/'student.pt').write_bytes(blob);(inputs/'selection.json').write_text(json.dumps(record,indent=2));print('FROZEN_STUDENT',record['model']['checkpoint_epoch'],record['validation']['si_sdri'],record['checkpoint_sha256'],flush=True)
base=json.loads(pathlib.Path('configs/esp32_zero_bias_broad_float.json').read_text());base.update(epochs=80,patience=80,max_hours=4,learning_rate=.0002,min_learning_rate=.00005,resume=str(inputs/'student.pt'),resume_optimizer=False)
for suffix,weight in [('continue',.1),('level',1.)]:
 config=pathlib.Path('configs')/f'esp32_zero_bias_broad_{suffix}_float.json';settings={**base,'output_dir':str(r/f'float_zero_bias_broad_{suffix}'),'waveform_loss_weight':weight};assert not pathlib.Path(settings['output_dir']).exists();config.write_text(json.dumps(settings,indent=2)+'\n');print('START',suffix,hashlib.sha256(config.read_bytes()).hexdigest(),flush=True)
 log=open(r/f'float_zero_bias_broad_{suffix}.log','w');p=subprocess.Popen([sys.executable,'-m','esp32_denoiser.train','--config',str(config)],stdout=log,stderr=subprocess.STDOUT)
 wlog=open(r/f'float_zero_bias_broad_{suffix}_external_watch.log','w');watch=subprocess.Popen([sys.executable,'-m','esp32_denoiser.development_checkpoints','--run',settings['output_dir'],'--manifest','/content/extra_audio/development/mixtures.jsonl','--device','cuda','--every-epochs','5','--max-hours','5'],stdout=wlog,stderr=subprocess.STDOUT)
 code=p.wait();print('FINISHED',suffix,code,flush=True);assert code==0;assert watch.wait(timeout=180)==0
