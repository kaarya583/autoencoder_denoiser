"""Target-informed mask diagnostics, never a learned model or an SI-SDR bound."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import PairedAudioDataset
from .metrics import si_sdr, summarize_utterances
from .quantization import round_away


@torch.inference_mode()
def target_masks(noisy: torch.Tensor, clean: torch.Tensor) -> dict[str, torch.Tensor]:
    """Use clean targets to diagnose restrictions of the current synthesis head.

    Pointwise spectral-error-optimal gains need not be waveform SI-SDR-optimal
    after overlap-add. These diagnostics are not upper bounds or deployable
    scores, and cannot predict a small network's attainable performance.
    """
    if noisy.shape != clean.shape or noisy.ndim != 2 or noisy.shape[-1] < 1:
        raise ValueError("Expected aligned, nonempty [batch,samples] audio")
    n, hop, size = noisy.shape[-1], 256, 512
    window = torch.hann_window(size, device=noisy.device).sqrt()
    padded = [F.pad(x, (hop, (-n) % hop + hop)) for x in (noisy, clean)]
    spectra = [torch.fft.rfft(x.unfold(-1, size, hop) * window) for x in padded]
    x, s = spectra
    ratio = s * x.conj() / x.abs().square().clamp_min(1e-12)
    real, imag = ratio.real.clamp(-1, 3), ratio.imag.clamp(-2, 2)
    dr = round_away((real - 1) / 2 * 128).clamp(-128, 127) / 128
    di = round_away(imag / 2 * 128).clamp(-128, 127) / 128
    gains = {
        "positive_attenuation": ratio.real.clamp(0, 1),
        "bounded_complex_float": torch.complex(real, imag),
        "bounded_complex_int8_grid": torch.complex(1 + 2 * dr, 2 * di),
    }
    weights = window.square().view(1, -1, 1).expand(1, -1, x.shape[1])
    total = padded[0].shape[-1]
    denominator = F.fold(weights, (1, total), kernel_size=(1, size), stride=(1, hop))[0, 0, 0].clamp_min(1e-8)
    result = {}
    for name, gain in gains.items():
        synthesis = torch.fft.irfft(x * gain, n=size) * window
        audio = F.fold(synthesis.transpose(1, 2), (1, total), kernel_size=(1, size), stride=(1, hop))[:, 0, 0]
        result[name] = (audio / denominator)[:, hop:hop + n]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    dataset = PairedAudioDataset(args.manifest, crop_seconds=None, random_crop=False)
    if any(r.get("source_split") != "train" for r in dataset.records):
        raise ValueError("Diagnostics are restricted to development audio; do not inspect official test targets")
    rows = {}
    for i, item in enumerate(dataset):
        noisy, clean = item["noisy"][None], item["clean"][None]
        baseline = float(si_sdr(noisy, clean)[0])
        for name, audio in target_masks(noisy, clean).items():
            score = float(si_sdr(audio, clean)[0])
            rows.setdefault(name, []).append({"id": item["id"], "si_sdr_noisy": baseline,
                                              "si_sdr_enhanced": score, "si_sdri": score - baseline})
        if (i + 1) % 100 == 0:
            print(json.dumps({"completed": i + 1, "total": len(dataset)}), flush=True)
    result = {"meaning": "target-informed spectral mask diagnostics; not learned quality or SI-SDR upper bounds",
              "official_test_used": False,
              "results": {name: {"summary": summarize_utterances(values), "per_utterance": values}
                          for name, values in rows.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({name: value["summary"] for name, value in result["results"].items()}), flush=True)


if __name__ == "__main__":
    main()
