import pathlib,json,torch
from torch.utils.data import DataLoader
from esp32_denoiser.data import PairedAudioDataset,pad_collate
from esp32_denoiser.evaluate import load_checkpoint
from esp32_denoiser.distillation import FrozenTeacher,calibrate_distillation_weight
from esp32_denoiser.train import speech_loss,seed_everything
torch.set_num_threads(2);seed_everything(2026);r=pathlib.Path('/content/esp32_runs')
m,_=load_checkpoint(r/'float_zero_bias/best.pt',device='cuda');teacher=FrozenTeacher(r/'float_spectral_teacher/best.pt','/content/voicebank/manifests/val.jsonl','cuda')
data=PairedAudioDataset('/content/voicebank/manifests/train.jsonl',crop_seconds=3,gain_db=(-6,6),noise_scale_db=(-6,6),clean_identity_prob=.03)
loader=DataLoader(data,batch_size=32,shuffle=True,collate_fn=pad_collate)
d=calibrate_distillation_weight(m,teacher,loader,speech_loss,target_ratio=.1,max_batches=8);(r/'distillation_calibration.json').write_text(json.dumps(d,indent=2))
c=json.loads(pathlib.Path('configs/esp32_zero_bias_distillation_float.json').read_text());c['distillation_weight']=d['distillation_weight'];(r/'distillation_calibrated_config.json').write_text(json.dumps(c,indent=2));print('KD_WEIGHT',d['distillation_weight'],'BATCHES',d['accepted_batches'],flush=True)
