"""Causal, small spectral speech enhancer with a convolution-only neural core.

The feature extractor and overlap-add are DSP operations. The neural core has
explicit quantization boundaries and requires only pointwise/depthwise
convolution, clipping, addition, and bounded history when deployed.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SpectralTCNConfig:
    width: int = 64
    activation_mode: str = "signed"
    feature_layout: str = "erb_complex"
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    mask_scale: float = 2.0

    def __post_init__(self):
        # The feature layout and embedded exporter deliberately have one
        # sample-rate/framing contract rather than silently changing semantics.
        if (self.sample_rate, self.n_fft, self.hop_length) != (16000, 512, 256):
            raise ValueError("SpectralTCN requires 16 kHz audio, FFT 512, hop 256")
        if self.activation_mode not in {"signed", "relu"}:
            raise ValueError("activation_mode must be signed or relu")
        if self.feature_layout not in {"erb_complex", "fullmag_lowphase"}:
            raise ValueError("feature_layout must be erb_complex or fullmag_lowphase")
        if self.width < 1 or not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError("width and every dilation must be positive")
        if not math.isfinite(self.mask_scale) or self.mask_scale <= 0:
            raise ValueError("mask_scale must be finite and positive")

    @classmethod
    def from_checkpoint(cls, values: dict) -> "SpectralTCNConfig":
        """Interpret historical defaults only when reading saved checkpoints."""
        values = dict(values)
        # Missing activation metadata belongs to the historical ReLU model;
        # new constructor calls intentionally default to signed residuals.
        values.setdefault("activation_mode", "relu")
        values.setdefault("feature_layout", "erb_complex")
        if "dilations" in values:
            values["dilations"] = tuple(values["dilations"])
        return cls(**values)


@dataclass
class StreamState:
    analysis: Tensor
    synthesis: Tensor
    synthesis_weight: Tensor
    temporal: tuple[Tensor, ...]


class CausalTCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int, activation_mode: str = "signed"):
        super().__init__()
        self.dilation = dilation
        self.history_length = 2 * dilation
        self.depthwise = nn.Conv1d(width, width, 3, dilation=dilation,
                                   groups=width, bias=True)
        self.depth_activation = nn.ReLU6()
        self.pointwise = nn.Conv1d(width, width, 1, bias=True)
        self.residual_quant = nn.Identity()
        # Signed residual state preserves an identity gradient through silence
        # and negative features; post-add ReLU can kill the entire small stack.
        self.output_activation = nn.Hardtanh(-6.0, 6.0) if activation_mode == "signed" else nn.ReLU6()
        # Keep an initially deep residual stack numerically well behaved.
        with torch.no_grad():
            self.pointwise.weight.mul_(0.1)
            self.pointwise.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.depth_activation(
            self.depthwise(F.pad(x, (self.history_length, 0))))
        return self.output_activation(self.residual_quant(x + self.pointwise(hidden)))

    def stream_step(self, x: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        joined = torch.cat((history, x), dim=-1)
        hidden = self.depth_activation(self.depthwise(joined))
        out = self.output_activation(self.residual_quant(x + self.pointwise(hidden)))
        return out, joined[..., -self.history_length:]


class SpectralTCN(nn.Module):
    """Full-bin complex gains driven by 387 causal spectral features.

    Default: 84,738 learned parameters and 5.212 MMAC/s of convolution at
    62.5 frames/s. Those counts do not establish device latency or model bytes.
    The default merges high-frequency complex features into sparse ERB bands.
    The experimental fullmag_lowphase layout uses all 257 magnitudes and the
    real/imaginary components of the first 65 bins, at the same neural cost.
    Output gains cover all 257 original bins in both layouts.
    """

    feature_size = 387
    output_size = 514
    low_bins = 65
    high_bands = 64

    def __init__(self, config: SpectralTCNConfig | None = None):
        super().__init__()
        self.config = config or SpectralTCNConfig()
        width = self.config.width
        self.input_quant = nn.Identity()
        self.input_proj = nn.Conv1d(self.feature_size, width, 1, bias=True)
        self.input_activation = nn.Hardtanh(-6.0, 6.0) if self.config.activation_mode == "signed" else nn.ReLU6()
        self.blocks = nn.ModuleList(
            CausalTCNBlock(width, d, self.config.activation_mode) for d in self.config.dilations)
        self.head = nn.Conv1d(width, self.output_size, 1, bias=True)
        self.head_activation = nn.Hardtanh(-1.0, 1.0)
        self.head_quant = nn.Identity()
        # Delta-complex-mask output starts as an exact identity, including QAT.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        window = torch.hann_window(self.config.n_fft, periodic=True).sqrt()
        self.register_buffer("window", window)
        if self.config.feature_layout == "erb_complex":
            self._register_filterbank()

    def _register_filterbank(self):
        # At most two contributions per high-frequency bin, with no dense
        # 192-by-64 matrices. Average each band's spectrum to bound features.
        frequencies = torch.arange(self.low_bins, 257, dtype=torch.float32)
        frequencies *= self.config.sample_rate / self.config.n_fft
        erb = 21.4 * torch.log10(1 + 0.00437 * frequencies)
        position = (erb - erb[0]) / (erb[-1] - erb[0]) * (self.high_bands - 1)
        lower = position.floor().long().clamp(0, self.high_bands - 1)
        upper = (lower + 1).clamp(max=self.high_bands - 1)
        upper_weight = position - lower
        lower_weight = 1 - upper_weight
        denominator = torch.zeros(self.high_bands)
        denominator.index_add_(0, lower, lower_weight)
        denominator.index_add_(0, upper, upper_weight)
        self.register_buffer("erb_lower", lower)
        self.register_buffer("erb_upper", upper)
        self.register_buffer("erb_lower_weight", lower_weight / denominator[lower])
        self.register_buffer("erb_upper_weight", upper_weight / denominator[upper])

    def _merge_bands(self, x: Tensor) -> Tensor:
        """Map [...,257] values to [...,129] with sparse triangular bands."""
        high = x[..., self.low_bins:]
        merged = x.new_zeros((*x.shape[:-1], self.high_bands))
        merged.index_add_(-1, self.erb_lower, high * self.erb_lower_weight)
        merged.index_add_(-1, self.erb_upper, high * self.erb_upper_weight)
        return torch.cat((x[..., :self.low_bins], merged), dim=-1)

    def frame_features(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        """Return original complex FFT and bounded neural features.

        Args:
            frames: [batch, frames, 512] unwindowed waveform frames.
        Returns:
            spectrum [batch, frames, 257], features [batch, 387, frames].
        RMS uses the current analysis frame only. It cannot leak later frames
        and avoids dependence on the duration or peak level of an utterance.
        """
        spectrum = torch.fft.rfft(frames * self.window, n=self.config.n_fft)
        rms = frames.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        normalized = spectrum / (self.config.n_fft * rms)
        if self.config.feature_layout == "fullmag_lowphase":
            root = normalized.abs().clamp_min(1e-8).sqrt()
            low_root = root[..., :self.low_bins]
            real = normalized.real[..., :self.low_bins] / low_root
            imag = normalized.imag[..., :self.low_bins] / low_root
        else:
            magnitude = self._merge_bands(normalized.abs())
            root = magnitude.clamp_min(1e-8).sqrt()
            real = self._merge_bands(normalized.real) / root
            imag = self._merge_bands(normalized.imag) / root
        features = torch.cat((root, real, imag), dim=-1)
        return spectrum, features.transpose(1, 2)

    def forward_features(self, features: Tensor) -> Tensor:
        """Neural core: [B,387,T] features -> [B,514,T] complex-mask deltas."""
        hidden = self.input_activation(self.input_proj(self.input_quant(features)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head_quant(self.head_activation(self.head(hidden)))

    def apply_mask(self, spectrum: Tensor, deltas: Tensor) -> Tensor:
        # Neural autocast may return bfloat16, which torch.complex cannot use.
        # Keep the DSP path at the spectrum's real precision.
        deltas = deltas.to(spectrum.real.dtype).transpose(1, 2)
        real, imag = deltas.split(257, dim=-1)
        # A scale of two permits negative real gains (phase flips), unlike
        # positive-only attenuation, while retaining the identity at zero.
        scale = self.config.mask_scale
        return spectrum * torch.complex(1.0 + scale * real, scale * imag)

    def forward(self, noisy: Tensor) -> Tensor:
        """Denoise [B,N] samples, preserving exact length and alignment.

        A left overlap pad initializes streaming history. One extra zero hop
        flushes the final overlap, exactly as in ``stream_step``. Overlap-add
        is written explicitly because torch.istft(center=False) rejects Hann
        endpoints, and utterance-wide normalization would violate streaming.
        """
        if noisy.ndim != 2 or noisy.shape[-1] == 0:
            raise ValueError("noisy must have shape [batch, positive sample count]")
        n = noisy.shape[-1]
        hop = self.config.hop_length
        tail = (-n) % hop
        padded = F.pad(noisy, (hop, tail + hop))
        frames = padded.unfold(-1, self.config.n_fft, hop)
        spectrum, features = self.frame_features(frames)
        enhanced = self.apply_mask(spectrum, self.forward_features(features))
        synthesis = torch.fft.irfft(enhanced, n=self.config.n_fft) * self.window
        total = padded.shape[-1]
        output = F.fold(synthesis.transpose(1, 2), (1, total),
                        kernel_size=(1, self.config.n_fft), stride=(1, hop))
        weights = self.window.square().view(1, -1, 1).expand(1, -1, frames.shape[1])
        denominator = F.fold(weights, (1, total),
                             kernel_size=(1, self.config.n_fft), stride=(1, hop))
        output = output[:, 0, 0] / denominator[0, 0, 0].clamp_min(1e-8)
        return output[:, hop:hop + n]

    def init_stream_state(self, batch_size: int = 1, device=None, dtype=None) -> StreamState:
        parameter = self.input_proj.weight
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        hop = self.config.hop_length
        def zeros(*shape):
            return torch.zeros(shape, device=device, dtype=dtype)
        return StreamState(
            analysis=zeros(batch_size, hop),
            synthesis=zeros(batch_size, hop),
            synthesis_weight=zeros(batch_size, hop),
            temporal=tuple(zeros(batch_size, self.config.width, block.history_length)
                           for block in self.blocks),
        )

    def stream_step(self, audio: Tensor, state: StreamState) -> tuple[Tensor, StreamState]:
        """Consume and emit one 256-sample hop.

        Discard the first emitted hop (initial overlap), and feed one zero hop
        at end-of-stream to flush the final output. No state is reset between
        normal audio hops. Delay accounting must include this overlap.
        """
        hop = self.config.hop_length
        if audio.ndim != 2 or audio.shape[-1] != hop:
            raise ValueError("stream_step expects [batch,256] audio")
        if audio.shape != state.analysis.shape or len(state.temporal) != len(self.blocks):
            raise ValueError("stream state does not match the model or audio batch")
        frames = torch.cat((state.analysis, audio), dim=-1).unsqueeze(1)
        spectrum, features = self.frame_features(frames)
        hidden = self.input_activation(self.input_proj(self.input_quant(features)))
        histories = []
        for block, history in zip(self.blocks, state.temporal):
            hidden, next_history = block.stream_step(hidden, history)
            histories.append(next_history)
        deltas = self.head_quant(self.head_activation(self.head(hidden)))
        enhanced = self.apply_mask(spectrum, deltas)
        synthesis = torch.fft.irfft(enhanced, n=self.config.n_fft)[:, 0] * self.window
        weight = self.window.square()
        denominator = state.synthesis_weight + weight[:hop]
        output = (state.synthesis + synthesis[:, :hop]) / denominator.clamp_min(1e-8)
        next_state = StreamState(audio, synthesis[:, hop:],
                                 weight[hop:].expand(audio.shape[0], -1), tuple(histories))
        return output, next_state

    def model_stats(self) -> dict[str, int | float]:
        parameters = sum(p.numel() for p in self.parameters())
        convolution_weights = sum(m.weight.numel() for m in self.modules()
                                  if isinstance(m, nn.Conv1d))
        state_values = sum(self.config.width * block.history_length for block in self.blocks)
        return {
            "learned_parameters": parameters,
            "convolution_weights": convolution_weights,
            "macs_per_second": convolution_weights * self.config.sample_rate / self.config.hop_length,
            "context_frames": 1 + 2 * sum(self.config.dilations),
            "neural_state_bytes_int8": state_values,
        }
