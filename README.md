# Autoencoder Denoiser

Project to identify and denoise shifting noise distributions in audio data using autoencoders via robust and adaptive models.

## Notebooks

- **`Adaptive_Autoencoders_Project.ipynb`** — latent-channel autoencoders, RF routing on pure latent noise, adaptive switching demo.
- **`MoE_Denoiser_Baseline.ipynb`** — **start here for MoE routing on pure noise only (no speech)**: waveform figures, RF vs neural router, confusion matrices and training curves. No LibriSpeech download required.

## Python package: `moe_baseline/`

Runnable training and fair routing comparison (aligned with adaptive notebook cell 12):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full LibriSpeech pipeline: latent RF + waveform RF + speech RF + MoE training
python run_moe_baseline.py

# RF routers only (needs LibriSpeech for speech router)
python run_moe_baseline.py --routers-only
```

| Goal | What to run |
|------|-------------|
| Pure-noise routing + visuals | **`MoE_Denoiser_Baseline.ipynb`** (cells 1–9) |
| **MoE demo** (instant; companion to adaptive notebook) | **`MoE_Demo.ipynb`** — loads precomputed WAVs |
| Regenerate MoE WAVs (optional) | `python scripts/run_moe_presentation_demo.py` |
| Full denoising + all router types | `python run_moe_baseline.py` |

**Presentation outputs:** `outputs/moe_presentation_demo/` (waveforms, routing timeline, confusion matrices, segment NR). For best denoising: `python scripts/run_moe_presentation_demo.py --quality-train --retrain`.

**Waveform RF** (notebook): log-FFT features on 512-sample **pure noise figures** — no speech.

**Latent fair router** (script): pure latent noise — same task as adaptive `noise_clf`.

**Deployment router** (script): spectral features on noisy LibriSpeech frames.

**Entry point:** [`run_moe_baseline.py`](run_moe_baseline.py)  
**Modules:** [`moe_baseline/`](moe_baseline/) (`config`, `librispeech`, `noise`, `features`, `model`, `routers`, `train`)
