import pathlib, subprocess, sys, json
from esp32_denoiser.evaluate import load_checkpoint
from esp32_denoiser.export import export_model
r=pathlib.Path('/content/esp32_runs'); d=r/'deploy_depth10'; d.mkdir(exist_ok=True)
m,_=load_checkpoint(r/'qat_depth10/best.pt'); print(json.dumps(export_model(m,d/'denoiser_int8.bin')),flush=True)
subprocess.run([sys.executable,'-u','-m','esp32_denoiser.embedded','--integer-model',str(d/'denoiser_int8.bin'),'--manifest','/content/voicebank/manifests/val.jsonl','--output',str(r/'depth10_embedded_validation.json'),'--perceptual'],check=True)
subprocess.run([sys.executable,'-u','-m','esp32_denoiser.evaluate','--checkpoint',str(r/'float_depth10/best.pt'),'--manifest','/content/voicebank/manifests/val.jsonl','--output',str(r/'depth10_float_validation_perceptual.json'),'--perceptual'],check=True)
