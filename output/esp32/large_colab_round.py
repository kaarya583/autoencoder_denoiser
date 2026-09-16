"""Long GPU continuations and automatic packed C development evaluation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, json, os, subprocess, sys, time

ROOT=Path('/content/esp32_runs')
PROJECT=Path('/content/esp32_project')
DEV='/content/extra_audio/development/mixtures.jsonl'
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}

def run_command(command, log):
    with log.open('w') as handle:
        subprocess.run(command,cwd=PROJECT,env=ENV,stdout=handle,stderr=subprocess.STDOUT,check=True)

def branch(name):
    predecessor=ROOT/name
    deadline=time.monotonic()+8*3600
    while not (predecessor/'summary.json').exists():
        if time.monotonic()>deadline: raise TimeoutError(name+' predecessor still unfinished')
        time.sleep(60)
    # The existing selector must score the predecessor's final checkpoint.
    rows=[json.loads(x) for x in (predecessor/'history.jsonl').read_text().splitlines() if x.strip()]
    final_epoch=rows[-1]['epoch']
    deadline=time.monotonic()+900
    while True:
        history=predecessor/'external_development/history.jsonl'
        observed=[json.loads(x) for x in history.read_text().splitlines() if x.strip()] if history.exists() else []
        if observed and observed[-1]['model']['checkpoint_epoch']==final_epoch: break
        if time.monotonic()>deadline: raise TimeoutError(name+' final development selection not completed')
        time.sleep(30)
    marker=predecessor/'external_development/best.json'
    selection_bytes=marker.read_bytes(); selection=json.loads(selection_bytes)
    source=Path(selection['selected_checkpoint']).read_bytes()
    assert hashlib.sha256(source).hexdigest()==selection['checkpoint_sha256']
    assert selection['model']['precision']=='float32 PyTorch'
    destination=ROOT/(name+'_large')
    destination.mkdir(exist_ok=False)
    frozen=destination/'parent.pt'; frozen.write_bytes(source)
    (destination/'parent_selection.json').write_bytes(selection_bytes)
    config=json.loads((predecessor/'config.json').read_text())
    config.update(output_dir=str(destination),resume=str(frozen),resume_optimizer=False,
        epochs=240,batch_size=48,epoch_samples=43208,learning_rate=2e-4,min_learning_rate=1e-5,
        max_hours=12,patience=60,workers=2,eval_batch_size=16,amp=True,
        max_steps_per_epoch=None,max_val_batches=None)
    config_path=destination/'large_config.json'
    config_path.write_text(json.dumps(config,indent=2))
    with (destination/'training.log').open('w') as train_log, (destination/'external_watch.log').open('w') as eval_log:
        training=subprocess.Popen([sys.executable,'-u','-m','esp32_denoiser.train','--config',str(config_path)],
            cwd=PROJECT,env=ENV,stdout=train_log,stderr=subprocess.STDOUT)
        validation=subprocess.Popen([sys.executable,'-u','-m','esp32_denoiser.development_checkpoints',
            '--run',str(destination),'--manifest',DEV,'--device','cuda','--every-epochs','10','--max-hours','13'],
            cwd=PROJECT,env=ENV,stdout=eval_log,stderr=subprocess.STDOUT)
        (destination/'processes.json').write_text(json.dumps(dict(training_pid=training.pid,validation_pid=validation.pid)))
        print(json.dumps(dict(event='large_training_started',run=name,epochs=240,examples_per_epoch=43208,
            batch_size=48,max_hours=12,parent_si_sdri=selection['validation']['si_sdri'],training_pid=training.pid)),flush=True)
        code=training.wait()
        if code:
            validation.terminate();validation.wait();raise RuntimeError(name+' training failed '+str(code))
        if validation.wait(timeout=1200):raise RuntimeError(name+' selector failed')
    winner=json.loads((destination/'external_development/best.json').read_text())
    data=Path(winner['selected_checkpoint']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==winner['checkpoint_sha256']
    deploy=destination/'deployment';deploy.mkdir(exist_ok=False)
    (deploy/'float.pt').write_bytes(data)
    (deploy/'selection.json').write_text(json.dumps(winner,indent=2))
    run_command([sys.executable,'-m','esp32_denoiser.gtcrn_mse_calibration','--checkpoint',str(deploy/'float.pt'),
        '--manifest',DEV,'--output',str(deploy/'model.bin'),'--calibration-crops','128','--calibration-seed','483',
        '--threads','2'],deploy/'calibration.log')
    for cohort,manifest in [('external',DEV),('clean','/content/extra_audio/development/clean.jsonl'),
                             ('primary','/content/voicebank/manifests/val.jsonl')]:
        run_command([sys.executable,'-m','esp32_denoiser.gtcrn_embedded','--integer-model',str(deploy/'model.bin'),
            '--calibration',str(deploy/'model.calibration.json'),'--manifest',manifest,
            '--output',str(deploy/(cohort+'.json')),'--io-format','pcm16','--perceptual','--threads','2'],
            deploy/(cohort+'.log'))
    print(json.dumps(dict(event='large_round_complete',run=name,model_bytes=(deploy/'model.bin').stat().st_size)),flush=True)
    return name

if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(branch,['float_gtcrn_broad','float_gtcrn_broad_normalized']))
    (ROOT/'large_round_complete.json').write_text(json.dumps(dict(completed=results,official_test_used=False)))
