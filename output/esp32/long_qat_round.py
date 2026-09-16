"""Launch one substantial QAT run after the GPU pilot and parent float run."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path('/content/esp32_runs');P=Path('/content/esp32_project');D='/content/extra_audio/development/mixtures.jsonl'
E={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}

def wait_file(path,hours):
    deadline=time.monotonic()+hours*3600
    while not path.exists():
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        time.sleep(60)

def run(cmd,log):
    with log.open('w') as f:return subprocess.call(cmd,cwd=P,env=E,stdout=f,stderr=subprocess.STDOUT)

wait_file(R/'gtcrn_qat_gpu_pilot/summary.json',.6)
pilot=json.loads((R/'gtcrn_qat_gpu_pilot/summary.json').read_text())
assert pilot['status']=='complete' and pilot['completed_epoch']==1
history=[json.loads(x) for x in (R/'gtcrn_qat_gpu_pilot/history.jsonl').read_text().splitlines()]
assert history[-1]['optimizer_steps']==8
parent=R/'float_gtcrn_broad_normalized'
wait_file(parent/'summary.json',8)
last_epoch=json.loads((parent/'history.jsonl').read_text().splitlines()[-1])['epoch']
deadline=time.monotonic()+900
while True:
    h=parent/'external_development/history.jsonl'
    rows=[json.loads(x) for x in h.read_text().splitlines()] if h.exists() else []
    if rows and rows[-1]['model']['checkpoint_epoch']==last_epoch:break
    if time.monotonic()>deadline:raise TimeoutError('Parent final development evaluation')
    time.sleep(30)
selection=json.loads((parent/'external_development/best.json').read_text())
source=Path(selection['selected_checkpoint']).read_bytes()
assert hashlib.sha256(source).hexdigest()==selection['checkpoint_sha256']
inputs=R/'gtcrn_long_qat_inputs';inputs.mkdir(exist_ok=False)
(inputs/'float.pt').write_bytes(source);(inputs/'selection.json').write_text(json.dumps(selection,indent=2))
command=[sys.executable,'-m','esp32_denoiser.gtcrn_mse_calibration','--checkpoint',str(inputs/'float.pt'),
    '--manifest',D,'--output',str(inputs/'model.bin'),'--calibration-crops','128','--threads','2']
assert run(command,inputs/'calibration.log')==0
# Each restart gets a separate output directory and retains failure evidence.
# Only a verified CUDA out-of-memory failure permits reducing the batch.
for batch in [16,8,4]:
    output=R/f'gtcrn_long_qat_batch{batch}'
    config=dict(source_checkpoint=str(inputs/'float.pt'),integer_model=str(inputs/'model.bin'),
        calibration=str(inputs/'model.calibration.json'),development_manifest=D,output_dir=str(output),
        epochs=80,batch_size=batch,workers=2,max_steps_per_epoch=4000//batch,
        max_hours=6,patience=20,device='cuda',learning_rate=3e-5,min_learning_rate=5e-6,
        waveform_loss_weight=.1)
    path=R/f'gtcrn_long_qat_batch{batch}_config.json';path.write_text(json.dumps(config,indent=2))
    log=R/f'gtcrn_long_qat_batch{batch}.log'
    print(json.dumps(dict(event='long_qat_started',batch_size=batch,optimizer_steps_per_round=4000//batch,
        rounds=80,max_hours=6,parent_si_sdri=selection['validation']['si_sdri'])),flush=True)
    code=run([sys.executable,'-u','-m','esp32_denoiser.gtcrn_qat_train','--config',str(path)],log)
    if code==0:
        best=json.loads((output/'best.json').read_text());candidate=Path(best['directory'])
        for name,manifest in [('external',D),('clean','/content/extra_audio/development/clean.jsonl'),
                              ('primary','/content/voicebank/manifests/val.jsonl')]:
            assert run([sys.executable,'-m','esp32_denoiser.gtcrn_embedded','--integer-model',str(candidate/'model.bin'),
                '--calibration',str(candidate/'model.calibration.json'),'--manifest',manifest,
                '--output',str(output/(name+'_final_development.json')),'--perceptual','--threads','2'],
                output/(name+'_final_development.log'))==0
        (R/'long_qat_complete.json').write_text(json.dumps(dict(output=str(output),best=best,official_test_used=False)))
        break
    if 'CUDA out of memory' not in log.read_text():raise RuntimeError('QAT failed; inspect '+str(log))
else:raise RuntimeError('QAT cannot fit the available GPU memory')
