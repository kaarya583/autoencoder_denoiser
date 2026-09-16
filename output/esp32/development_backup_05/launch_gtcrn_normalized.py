import sys
sys.path.insert(0, '/content/esp32_project')
import torch,json,time,subprocess,sys,pathlib
from torch.utils.data import DataLoader
from esp32_denoiser.models import build_model
from esp32_denoiser.data import PairedAudioDataset,pad_collate
from esp32_denoiser.train import seed_everything,speech_loss
torch.set_num_threads(2);seed_everything(2026)
m=build_model('gtcrn',{'normalize_input':True}).cuda().train();o=torch.optim.AdamW(m.parameters(),lr=.001)
l=DataLoader(PairedAudioDataset('/content/voicebank/manifests/train.jsonl',crop_seconds=3,gain_db=(-6,6),noise_scale_db=(-6,6),clean_identity_prob=.03),batch_size=8,shuffle=True,collate_fn=pad_collate,num_workers=0)
losses=[];t=time.monotonic()
for i,b in enumerate(l):
 if i==8:break
 o.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  y=m(b['noisy'].cuda());v=speech_loss(y,b['clean'].cuda(),b['length'].cuda())
 assert torch.isfinite(v);v.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);torch.nn.utils.clip_grad_norm_(m.parameters(),5);o.step();losses.append(v.item())
r=pathlib.Path('/content/esp32_runs');(r/'gtcrn_normalized_cuda_pilot.json').write_text(json.dumps({'steps':8,'losses':losses,'finite_gradients':True,'elapsed_seconds':time.monotonic()-t,'stats':m.model_stats()},indent=2));print('PILOT_PASS',losses,flush=True)
for n in ['esp32_gtcrn_broad_normalized_float']:
 c=pathlib.Path('configs')/(n+'.json');d=json.loads(c.read_text());assert not pathlib.Path(d['output_dir']).exists();log=open(r/(n+'.log'),'w');p=subprocess.Popen([sys.executable,'-m','esp32_denoiser.train','--config',str(c)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(n,p.pid,flush=True)
log=open(r/'float_gtcrn_broad_normalized_external_watch.log','w');p=subprocess.Popen([sys.executable,'-m','esp32_denoiser.development_checkpoints','--run',str(r/'float_gtcrn_broad_normalized'),'--manifest','/content/extra_audio/development/mixtures.jsonl','--device','cuda','--every-epochs','5','--max-hours','7'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print('GTCRN_WATCH',p.pid,flush=True)
