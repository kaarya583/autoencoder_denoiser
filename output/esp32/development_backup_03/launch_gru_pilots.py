import torch,json,pathlib,subprocess,hashlib
from torch.utils.data import DataLoader
from esp32_denoiser.data import PairedAudioDataset,pad_collate
from esp32_denoiser.models import build_model
from esp32_denoiser.train import speech_loss,seed_everything
r=pathlib.Path('/content/esp32_project'); runs=pathlib.Path('/content/esp32_runs')
for n,h in json.loads((r/'SOURCE_MANIFEST.json').read_text()).items():
 assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
torch.set_num_threads(2)
loader=DataLoader(PairedAudioDataset('/content/voicebank/manifests/train.jsonl',crop_seconds=3),batch_size=8,shuffle=True,collate_fn=pad_collate)
results=[]
for bn in [False,True]:
 seed_everything(2026);m=build_model('frequency_gru',{'encoder_batch_norm':bn}).cuda().train();opt=torch.optim.AdamW(m.parameters(),lr=.001);scores=[]
 for step,b in enumerate(loader):
  if step==8:break
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   y=m(b['noisy'].cuda());loss=speech_loss(y,b['clean'].cuda(),b['length'].cuda())
  assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();scores.append(float(loss.detach()))
 results.append({'batch_norm':bn,'losses':scores,'finite_gradients':True});del m,opt;torch.cuda.empty_cache()
(runs/'gru_cuda_pilot.json').write_text(json.dumps(results,indent=2))
for tag in ['frequency_gru','frequency_gru_bn']:
 assert not (runs/('float_'+tag)).exists(),tag
 log=open(runs/(tag+'_float.log'),'w');p=subprocess.Popen(['python','-m','esp32_denoiser.train','--config','configs/esp32_'+tag+'_float.json'],cwd=r,stdout=log,stderr=subprocess.STDOUT);print('LAUNCHED',tag,p.pid,flush=True)
