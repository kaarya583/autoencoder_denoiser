import json,pathlib,torch,hashlib,shutil
from threadpoolctl import threadpool_limits
from esp32_denoiser.evaluate import load_checkpoint,evaluate_manifest,_json_finite
r=pathlib.Path('/content/esp32_runs');out=r/'broader_quality_probes';out.mkdir(exist_ok=False)
torch.set_num_threads(2)
selected=json.loads((r/'float_zero_bias_broad/external_development/best.json').read_text())
teacher=out/'broad_teacher_snapshot.pt';shutil.copyfile(r/'float_spectral_teacher_broad/best.pt',teacher)
jobs=[('zero_bias_float_external',r/'float_zero_bias/best.pt','mixtures'),('paired_teacher_float_external',r/'float_spectral_teacher/best.pt','mixtures'),('broad_student_float_clean',pathlib.Path(selected['selected_checkpoint']),'clean'),('broad_teacher_float_clean',teacher,'clean')]
with threadpool_limits(limits=1,user_api='blas'):
 for name,source,suite in jobs:
  model,metadata=load_checkpoint(source,'cpu');metadata.update(source=str(source),model_sha256=hashlib.sha256(source.read_bytes()).hexdigest());result=evaluate_manifest(model,pathlib.Path('/content/extra_audio/development')/(suite+'.jsonl'),device='cpu',perceptual=False);result['model']=metadata;(out/(name+'.json')).write_text(json.dumps(_json_finite(result),indent=2,allow_nan=False));print('PROBE',name,metadata['checkpoint_epoch'],result['summary']['si_sdri'],flush=True);del model
