import pathlib,json,hashlib,torch,subprocess,sys
from torch.utils.data import DataLoader
from esp32_denoiser.evaluate import load_checkpoint
from esp32_denoiser.train import seed_everything,validate
from esp32_denoiser.data import PairedAudioDataset,pad_collate
from esp32_denoiser.extra_data import DynamicMixtureDataset
from esp32_denoiser.mixtures import HybridTrainingDataset
from esp32_denoiser.models import configure_model_qat,calibrate_model_hidden_exponent
from esp32_denoiser.export import export_model
r=pathlib.Path('/content/esp32_runs');out=r/'broad_ptq_probe';out.mkdir(exist_ok=False);source=r/'broad_kd_inputs/student.pt';torch.set_num_threads(2);seed_everything(2026);model,metadata=load_checkpoint(source,'cuda');paired=PairedAudioDataset('/content/voicebank/manifests/train.jsonl',crop_seconds=3,gain_db=(-6,6),noise_scale_db=(-6,6),clean_identity_prob=.03);synthetic=DynamicMixtureDataset('/content/extra_audio/manifests/speech_train.jsonl','/content/extra_audio/manifests/noise_train.jsonl',crop_seconds=3,snr_db=(-5,20),gain_db=(-6,6),clean_identity_prob=.03);hybrid=HybridTrainingDataset(paired,synthetic,synthetic_probability=.5,epoch_samples=10802);loader=DataLoader(hybrid,batch_size=32,shuffle=True,collate_fn=pad_collate,num_workers=0,generator=torch.Generator().manual_seed(2002026));ids=[]
def waves():
 for batch in loader:
  ids.extend(batch['id']);yield batch['noisy'].cuda()
exponent=calibrate_model_hidden_exponent(model,'spectral_tcn',waves(),max_batches=32);configure_model_qat(model,'spectral_tcn',hidden_exponent=exponent);model.cuda().eval();validation=validate(model,DataLoader(PairedAudioDataset('/content/extra_audio/development/mixtures.jsonl',crop_seconds=None,random_crop=False),batch_size=8,collate_fn=pad_collate),torch.device('cuda'));(out/'fake_quant_external_validation.json').write_text(json.dumps(validation,indent=2));model.cpu();binary=out/'denoiser_int8.bin';report=export_model(model,binary,max_bytes=99000);report.update(quantization_method='post-training diagnostic; zero QAT optimizer steps',float_source=str(source),float_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),hidden_exponent=exponent,calibration_source='50/50 training-only paired and synthetic mixture',calibration_ids=ids,seed=2026,sampler_seed=2002026,synthetic_probability=.5,source_bundle_sha256='a050696f9b6e2ae9446b6ada668761c2ff9a0c372e1b30a53d958d072f03b293');(out/'export.json').write_text(json.dumps(report,indent=2));print('PTQ_EXPORTED',binary.stat().st_size,exponent,validation['si_sdri'],flush=True);del model;torch.cuda.empty_cache()
for name,manifest in [('external','/content/extra_audio/development/mixtures.jsonl'),('clean','/content/extra_audio/development/clean.jsonl'),('primary','/content/voicebank/manifests/val.jsonl')]:
 subprocess.run([sys.executable,'-m','esp32_denoiser.embedded','--integer-model',str(binary),'--manifest',manifest,'--output',str(out/(name+'_full_c_pcm16.json')),'--io-format','pcm16'],check=True)
subprocess.run([sys.executable,'-m','esp32_denoiser.embedded','--integer-model',str(binary),'--manifest','/content/extra_audio/development/mixtures.jsonl','--output',str(out/'frontend_parity_16.json'),'--io-format','pcm16','--compare-reference','--max-utterances','16'],check=True);print('PTQ_PROBE_COMPLETE',flush=True)
