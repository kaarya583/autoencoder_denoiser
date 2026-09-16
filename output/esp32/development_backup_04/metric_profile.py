import time,json,numpy as np,torch
from threadpoolctl import threadpool_info,threadpool_limits
from esp32_denoiser.data import PairedAudioDataset
from esp32_denoiser.development import preservation_metrics
from pesq import pesq
from pystoi import stoi
b=PairedAudioDataset('/content/extra_audio/development/mixtures.jsonl',crop_seconds=None,random_crop=False)[0];c=b['clean'].numpy();n=b['noisy'].numpy();print('POOLS',json.dumps(threadpool_info()),flush=True)
for limit in [None,1,2]:
 with threadpool_limits(limits=limit):
  for name,f in [('preservation',lambda:preservation_metrics(n,c,n)),('PESQ',lambda:pesq(16000,c,n,'wb')),('STOI',lambda:stoi(c,n,16000,extended=False))]:
   t=time.perf_counter();values=[f() for _ in range(5)];print(limit,name,(time.perf_counter()-t)/5,flush=True)
