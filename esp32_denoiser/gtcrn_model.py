"""Fresh GTCRN reference with the project's causal waveform framing.

The MIT network is vendored at a pinned upstream revision. No pretrained
weights are loaded. The default raw spectral features/operators stay
upstream-identical. An explicit frame-RMS input ablation is optional; the
waveform boundary convention differs from upstream's reflect-padded
``torch.stft`` example. Integer inference is deliberately unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .vendor.gtcrn.network import GTCRN


UPSTREAM_REVISION = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"


@dataclass(frozen=True)
class GTCRNConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    upstream_revision: str = UPSTREAM_REVISION
    framing: str = "causal_zero_overlap_v1"
    normalize_input: bool = False
    rms_floor: float = 1e-4

    def __post_init__(self):
        if (self.sample_rate, self.n_fft, self.hop_length) != (16000, 512, 256):
            raise ValueError("The faithful GTCRN reference requires 16 kHz, FFT512 and hop256")
        if self.upstream_revision != UPSTREAM_REVISION or self.framing != "causal_zero_overlap_v1":
            raise ValueError("Unsupported GTCRN source revision or waveform framing")
        if not isinstance(self.normalize_input, bool):
            raise ValueError("normalize_input must be a boolean")
        if (isinstance(self.rms_floor, bool) or not isinstance(self.rms_floor, (int, float))
                or not math.isfinite(self.rms_floor) or self.rms_floor <= 0
                or not torch.finfo(torch.float32).tiny <= self.rms_floor * self.rms_floor <= torch.finfo(torch.float32).max):
            raise ValueError("rms_floor must be positive with a representable normal float32 square")

    @classmethod
    def from_checkpoint(cls, values: dict) -> "GTCRNConfig":
        return cls(**values)


def _audio_shape(audio: Tensor):
    if audio.ndim != 2 or min(audio.shape) < 1 or not audio.is_floating_point():
        raise ValueError("Audio must have floating shape [batch, positive sample count]")


def _spectrum(frames: Tensor, window: Tensor) -> Tensor:
    # Keep FFT arithmetic float32 even when the network trains with autocast.
    spectrum = torch.fft.rfft(frames.float() * window, n=512)
    return torch.view_as_real(spectrum).permute(0, 2, 1, 3)


def _network_spectrum(frames: Tensor, window: Tensor, config: GTCRNConfig) -> tuple[Tensor, Tensor | None]:
    """Optionally normalize each causal analysis frame, without temporal state.

    RMS includes all 512 unwindowed samples (and framing zeros). Dividing
    the windowed FFT by ``512 * max(RMS, floor)`` bounds every complex bin
    magnitude by ``sqrt(mean(window**2))`` via Cauchy-Schwarz. The returned
    scale restores the original spectral amplitude after the network.
    Clamp variance before sqrt so silence also has finite input gradients.
    """
    spectrum = _spectrum(frames, window)
    if not config.normalize_input:
        return spectrum, None
    variance = frames.float().square().mean(dim=-1)
    scale = config.n_fft * variance.clamp_min(config.rms_floor ** 2).sqrt()
    scale = scale[:, None, :, None]
    return spectrum / scale, scale


def _synthesis(spectrum: Tensor, window: Tensor) -> Tensor:
    complex_spectrum = torch.view_as_complex(spectrum.float().contiguous()).transpose(1, 2)
    return torch.fft.irfft(complex_spectrum, n=512) * window


class GTCRNDenoiser(nn.Module):
    """Trainable waveform adapter; ``forward([B,N])`` returns aligned [B,N].

Upstream expects raw magnitude/real/imaginary STFT features, which remain
the default. ``normalize_input=True`` explicitly selects a frame-RMS
ablation and restores amplitude before overlap-add. It changes the input
recipe, adds no learned parameters and does not alter the vendored graph.
Training uses the parallel upstream graph. ``make_streaming`` creates a
separate frozen snapshot for inference, without registering duplicate
parameters in this trainable model or changing the caller's RNG state.
"""

    def __init__(self, config: GTCRNConfig | None = None):
        super().__init__()
        self.config = config or GTCRNConfig()
        self.core = GTCRN()
        self.register_buffer("window", torch.hann_window(512).sqrt())

    def forward(self, noisy: Tensor) -> Tensor:
        _audio_shape(noisy)
        n, hop = noisy.shape[-1], self.config.hop_length
        padded = F.pad(noisy.float(), (hop, (-n) % hop + hop))
        frames = padded.unfold(-1, self.config.n_fft, hop)
        spectrum, scale = _network_spectrum(frames, self.window, self.config)
        enhanced = self.core(spectrum)
        if scale is not None:
            enhanced = enhanced * scale
        synthesis = _synthesis(enhanced, self.window)
        total = padded.shape[-1]
        output = F.fold(synthesis.transpose(1, 2), (1, total),
                        kernel_size=(1, self.config.n_fft), stride=(1, hop))[:, 0, 0]
        weights = self.window.square().view(1, -1, 1).expand(1, -1, frames.shape[1])
        denominator = F.fold(weights, (1, total), kernel_size=(1, self.config.n_fft),
                             stride=(1, hop))[0, 0, 0]
        return (output / denominator.clamp_min(1e-8))[:, hop:hop + n]

    def make_streaming(self) -> "GTCRNStreamer":
        if self.training:
            raise ValueError("Call eval() before creating a frozen GTCRN streaming snapshot")
        return GTCRNStreamer(self)

    def model_stats(self) -> dict:
        # Learned architecture count must not change when a teacher is frozen.
        frozen = sum(p.numel() for p in self.core.erb.parameters())
        learned = sum(p.numel() for p in self.core.parameters()) - frozen
        erb = self.core.erb.erb_fc.weight
        return {
            "precision": "float-only; INT8 QAT/export/runtime are not implemented",
            "learned_parameters": learned,
            "fixed_erb_parameters": frozen,
            "total_parameters_including_fixed": learned + frozen,
            "learned_parameter_bytes_float32": learned * 4,
            "fixed_erb_bytes_float32": frozen * 4,
            "state_dict_bytes": sum(t.numel() * t.element_size() for t in self.state_dict().values()),
            "fixed_erb_shape": list(erb.shape),
            "fixed_erb_nonzero_values_one_direction": int(torch.count_nonzero(erb)),
            "fixed_erb_transpose_duplicate": True,
            "macs_per_second": 33_000_000,
            "macs_status": "published upstream accounting; not measured MCU cycles or an INT8 runtime",
            "context_frames": None,
            "temporal_context": "causal recurrent history; intra-frame bidirectional frequency GRUs",
            "neural_state_elements": 18_048,
            "neural_state_bytes_float32": 72_192,
            "audio_overlap_state_bytes_float32": 3 * 256 * 4,
            "state_scope": "batch1 persistent state only; activations, weight snapshot and FFT scratch excluded",
            "packed_bytes_status": "unavailable; no integer exporter",
            "upstream_revision": self.config.upstream_revision,
            "framing": self.config.framing,
            "input_normalization": "frame_rms" if self.config.normalize_input else "raw",
            "input_rms_floor": self.config.rms_floor if self.config.normalize_input else None,
            "input_normalization_scope": "float32 external DSP; per analysis frame, no additional persistent state",
            "constructor_initialization": "fresh; the adapter does not load pretrained weights",
        }


@dataclass
class GTCRNStreamState:
    analysis: Tensor
    synthesis: Tensor
    synthesis_weight: Tensor
    convolution: Tensor
    attention: Tensor
    recurrent: Tensor


class GTCRNStreamer:
    """Frozen float inference snapshot of the official streaming graph.

The snapshot owns one converted parameter copy. It is not a deployable
integer graph. Changes to the training model require a new snapshot.
State caches are updated in place: a consumed state must not be reused to
branch a stream without cloning its tensors. Process fixed 256-sample hops,
discard the initial output hop, and flush one zero hop at the end.
"""

    def __init__(self, source: GTCRNDenoiser):
        from .vendor.gtcrn.convert import convert_to_stream
        from .vendor.gtcrn.streaming import StreamGTCRN

        self.config = source.config
        self.device = source.window.device
        with torch.random.fork_rng(devices=[]):
            self.network = StreamGTCRN().to(self.device).eval()
        convert_to_stream(self.network, source.core)
        self.network.requires_grad_(False)
        self.window = source.window.detach().float().clone()

    def init_stream_state(self, batch_size: int = 1) -> GTCRNStreamState:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        def zeros(*shape):
            return torch.zeros(shape, device=self.device, dtype=torch.float32)

        return GTCRNStreamState(
            analysis=zeros(batch_size, 256), synthesis=zeros(batch_size, 256),
            synthesis_weight=zeros(batch_size, 256),
            convolution=zeros(2, batch_size, 16, 16, 33),
            attention=zeros(2, 3, 1, batch_size, 16),
            recurrent=zeros(2, 1, batch_size * 33, 16),
        )

    @torch.no_grad()
    def stream_step(self, audio: Tensor, state: GTCRNStreamState) -> tuple[Tensor, GTCRNStreamState]:
        _audio_shape(audio)
        if audio.shape != state.analysis.shape or audio.shape[-1] != 256 or audio.device != self.device:
            raise ValueError("Streaming audio must match the state batch/device and contain one 256-sample hop")
        frames = torch.cat((state.analysis, audio.float()), dim=-1).unsqueeze(1)
        with torch.autocast(device_type=self.device.type, enabled=False):
            spectrum, scale = _network_spectrum(frames, self.window, self.config)
            enhanced, convolution, attention, recurrent = self.network(
                spectrum, state.convolution, state.attention, state.recurrent)
            if scale is not None:
                enhanced = enhanced * scale
            synthesis = _synthesis(enhanced, self.window)[:, 0]
        weight = self.window.square()
        output = (state.synthesis + synthesis[:, :256]) / (state.synthesis_weight + weight[:256]).clamp_min(1e-8)
        next_state = GTCRNStreamState(audio.float().clone(), synthesis[:, 256:].clone(),
                                     weight[256:].expand_as(audio).clone(), convolution, attention, recurrent)
        return output, next_state

    @torch.no_grad()
    def denoise(self, audio: Tensor) -> Tensor:
        """Convenience full-utterance streaming check, with reset and flush."""
        _audio_shape(audio)
        n = audio.shape[-1]
        padded = F.pad(audio, (0, (-n) % 256 + 256))
        state = self.init_stream_state(audio.shape[0])
        outputs = []
        for start in range(0, padded.shape[-1], 256):
            output, state = self.stream_step(padded[:, start:start + 256], state)
            outputs.append(output)
        return torch.cat(outputs[1:], dim=-1)[:, :n]
