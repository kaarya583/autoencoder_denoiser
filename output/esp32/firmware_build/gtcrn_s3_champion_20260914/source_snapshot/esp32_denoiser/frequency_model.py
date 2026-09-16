"""Frequency-shared U-Net with local and global causal temporal processing.

This model shares the existing waveform analysis/synthesis implementation, but
has a separate neural graph and EDNFQ8 integer export format. Model statistics
include a byte estimate; frequency_export measures the actual packed payload.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import CausalTCNBlock, SpectralTCN
from .frequency_normalization import EncoderBatchNormConv2d


@dataclass(frozen=True)
class FrequencyUNetConfig:
    encoder_channels: tuple[int, int, int] = (16, 24, 32)
    global_width: int = 32
    local_dilations: tuple[int, ...] = (1, 2, 4)
    global_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    mask_scale: float = 2.0
    activation_mode: str = "signed"
    feature_layout: str = "full_complex257"
    encoder_batch_norm: bool = False

    def __post_init__(self):
        if (self.sample_rate, self.n_fft, self.hop_length) != (16000, 512, 256):
            raise ValueError("FrequencyUNet requires 16 kHz audio, FFT512, hop256")
        if len(self.encoder_channels) != 3 or any(c < 1 for c in self.encoder_channels):
            raise ValueError("encoder_channels must contain three positive widths")
        if self.global_width < 1 or not self.local_dilations or not self.global_dilations:
            raise ValueError("Positive global width and nonempty temporal stacks are required")
        if any(d < 1 for d in (*self.local_dilations, *self.global_dilations)):
            raise ValueError("Temporal dilations must be positive")
        if self.activation_mode != "signed" or self.feature_layout != "full_complex257":
            raise ValueError("FrequencyUNet requires signed activations and full_complex257 features")
        if not math.isfinite(self.mask_scale) or self.mask_scale <= 0:
            raise ValueError("mask_scale must be finite and positive")
        if not isinstance(self.encoder_batch_norm, bool):
            raise ValueError("encoder_batch_norm must be boolean")

    @classmethod
    def from_checkpoint(cls, values: dict) -> "FrequencyUNetConfig":
        values = dict(values)
        for key in ("encoder_channels", "local_dilations", "global_dilations"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass
class FrequencyStreamState:
    analysis: Tensor
    synthesis: Tensor
    synthesis_weight: Tensor
    local: tuple[Tensor, ...]
    global_temporal: tuple[Tensor, ...]


class FrequencyDownsample(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.depthwise = nn.Conv2d(inputs, inputs, (1, 5), stride=(1, 2),
                                   padding=(0, 2), groups=inputs)
        self.depth_activation = nn.ReLU6()
        self.pointwise = nn.Conv2d(inputs, outputs, 1)
        self.output_activation = nn.Hardtanh(-6.0, 6.0)
        nn.init.zeros_(self.depthwise.bias)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_activation(self.pointwise(self.depth_activation(self.depthwise(x))))


class LocalTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.history_length = 2 * dilation
        self.depthwise = nn.Conv2d(channels, channels, (3, 3),
                                   dilation=(dilation, 1), groups=channels)
        self.depth_activation = nn.ReLU6()
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.residual_quant = nn.Identity()
        self.output_activation = nn.Hardtanh(-6.0, 6.0)
        nn.init.zeros_(self.depthwise.bias)
        nn.init.zeros_(self.pointwise.bias)
        with torch.no_grad():
            self.pointwise.weight.mul_(0.1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.depth_activation(self.depthwise(F.pad(x, (1, 1, self.history_length, 0))))
        return self.output_activation(self.residual_quant(x + self.pointwise(hidden)))

    def stream_step(self, x: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        joined = torch.cat((history, x), dim=2)
        hidden = self.depth_activation(self.depthwise(F.pad(joined, (1, 1, 0, 0))))
        output = self.output_activation(self.residual_quant(x + self.pointwise(hidden)))
        return output, joined[:, :, -self.history_length:]


class FrequencyUpsample(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.pointwise = nn.Conv2d(inputs, outputs, 1)
        self.point_activation = nn.ReLU6()
        self.depthwise = nn.Conv2d(outputs, outputs, (1, 5), padding=(0, 2), groups=outputs)
        self.skip_quant = nn.Identity()
        self.output_activation = nn.Hardtanh(-6.0, 6.0)
        nn.init.zeros_(self.pointwise.bias)
        nn.init.zeros_(self.depthwise.bias)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        # Explicit repeat/crop has an unambiguous embedded implementation.
        x = x.repeat_interleave(2, dim=-1)[..., :skip.shape[-1]]
        x = self.depthwise(self.point_activation(self.pointwise(x)))
        return self.output_activation(self.skip_quant(x + skip))


class FrequencyUNet(SpectralTCN):
    """Full-bin complex masks from a frequency-shared encoder and decoder.

    Waveform ``forward`` and ``apply_mask`` are inherited solely to share the
    tested DSP contract. The constructor creates only this model's neural graph.
    ``forward_features`` accepts [B,3,T,257] and returns [B,514,T].
    """

    feature_size = 771
    frequency_sizes = (257, 129, 65, 33)

    def __init__(self, config: FrequencyUNetConfig | None = None):
        nn.Module.__init__(self)
        self.config = config or FrequencyUNetConfig()
        c1, c2, c3 = self.config.encoder_channels
        hidden = self.config.global_width
        self.input_quant = nn.Identity()
        self.stem = nn.Conv2d(3, c1, (1, 5), stride=(1, 2), padding=(0, 2))
        self.stem_activation = nn.Hardtanh(-6.0, 6.0)
        self.down1 = FrequencyDownsample(c1, c2)
        self.down2 = FrequencyDownsample(c2, c3)
        if self.config.encoder_batch_norm:
            self.stem = EncoderBatchNormConv2d.from_float(self.stem)
            for down in (self.down1, self.down2):
                down.depthwise = EncoderBatchNormConv2d.from_float(down.depthwise)
                down.pointwise = EncoderBatchNormConv2d.from_float(down.pointwise)
        self.local_blocks = nn.ModuleList(LocalTemporalBlock(c3, d) for d in self.config.local_dilations)
        self.global_in = nn.Conv1d(c3 * 33, hidden, 1)
        self.global_in_activation = nn.Hardtanh(-6.0, 6.0)
        self.global_blocks = nn.ModuleList(CausalTCNBlock(hidden, d) for d in self.config.global_dilations)
        for block in self.global_blocks:
            nn.init.zeros_(block.depthwise.bias)
        self.global_out = nn.Conv1d(hidden, c3 * 33, 1)
        self.global_residual_quant = nn.Identity()
        self.global_out_activation = nn.Hardtanh(-6.0, 6.0)
        with torch.no_grad():
            self.global_out.weight.mul_(0.1)
            self.global_out.bias.zero_()
        self.up1 = FrequencyUpsample(c3, c2)
        self.up2 = FrequencyUpsample(c2, c1)
        self.head_dw = nn.Conv2d(c1, c1, (1, 5), padding=(0, 2), groups=c1)
        self.head_dw_activation = nn.ReLU6()
        nn.init.zeros_(self.head_dw.bias)
        self.head = nn.Conv2d(c1, 2, 1)
        self.head_activation = nn.Hardtanh(-1.0, 1.0)
        self.head_quant = nn.Identity()
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("window", torch.hann_window(self.config.n_fft, periodic=True).sqrt())

    def frame_features(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        spectrum = torch.fft.rfft(frames * self.window, n=self.config.n_fft)
        rms = frames.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        normalized = spectrum / (self.config.n_fft * rms)
        root = normalized.abs().clamp_min(1e-8).sqrt()
        features = torch.stack((root, normalized.real / root, normalized.imag / root), dim=1)
        return spectrum, features

    def _encode(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        skip1 = self.stem_activation(self.stem(self.input_quant(features)))
        skip2 = self.down1(skip1)
        return skip1, skip2, self.down2(skip2)

    @staticmethod
    def _flatten_frequency(x: Tensor) -> Tensor:
        # Channel-major then frequency, independently for every time frame.
        return x.permute(0, 1, 3, 2).reshape(x.shape[0], x.shape[1] * x.shape[3], x.shape[2])

    def _fuse_global(self, x: Tensor, hidden: Tensor) -> Tensor:
        residual = self.global_out(hidden)
        residual = residual.reshape(x.shape[0], x.shape[1], x.shape[3], x.shape[2]).permute(0, 1, 3, 2)
        return self.global_out_activation(self.global_residual_quant(x + residual))

    def _decode(self, x: Tensor, skip1: Tensor, skip2: Tensor) -> Tensor:
        x = self.up2(self.up1(x, skip2), skip1)
        x = x.repeat_interleave(2, dim=-1)[..., :257]
        x = self.head_dw_activation(self.head_dw(x))
        deltas = self.head_quant(self.head_activation(self.head(x)))
        return self._flatten_frequency(deltas)

    def forward_features(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != 3 or features.shape[-1] != 257:
            raise ValueError("FrequencyUNet features must have shape [B,3,T,257]")
        skip1, skip2, x = self._encode(features)
        for block in self.local_blocks:
            x = block(x)
        hidden = self.global_in_activation(self.global_in(self._flatten_frequency(x)))
        for block in self.global_blocks:
            hidden = block(hidden)
        return self._decode(self._fuse_global(x, hidden), skip1, skip2)

    def init_stream_state(self, batch_size: int = 1, device=None, dtype=None) -> FrequencyStreamState:
        parameter = self.stem.weight
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype

        def zeros(*shape):
            return torch.zeros(shape, device=device, dtype=dtype)

        hop = self.config.hop_length
        return FrequencyStreamState(
            analysis=zeros(batch_size, hop), synthesis=zeros(batch_size, hop),
            synthesis_weight=zeros(batch_size, hop),
            local=tuple(zeros(batch_size, self.config.encoder_channels[-1], block.history_length, 33)
                        for block in self.local_blocks),
            global_temporal=tuple(zeros(batch_size, self.config.global_width, block.history_length)
                                  for block in self.global_blocks),
        )

    def stream_step(self, audio: Tensor, state: FrequencyStreamState) -> tuple[Tensor, FrequencyStreamState]:
        hop = self.config.hop_length
        if audio.ndim != 2 or audio.shape[-1] != hop or audio.shape != state.analysis.shape:
            raise ValueError("stream_step expects matching audio/state [B,256]")
        if len(state.local) != len(self.local_blocks) or len(state.global_temporal) != len(self.global_blocks):
            raise ValueError("Temporal state does not match this model")
        frames = torch.cat((state.analysis, audio), dim=-1).unsqueeze(1)
        spectrum, features = self.frame_features(frames)
        skip1, skip2, x = self._encode(features)
        local_states = []
        for block, history in zip(self.local_blocks, state.local):
            x, next_history = block.stream_step(x, history)
            local_states.append(next_history)
        hidden = self.global_in_activation(self.global_in(self._flatten_frequency(x)))
        global_states = []
        for block, history in zip(self.global_blocks, state.global_temporal):
            hidden, next_history = block.stream_step(hidden, history)
            global_states.append(next_history)
        deltas = self._decode(self._fuse_global(x, hidden), skip1, skip2)
        enhanced = self.apply_mask(spectrum, deltas)
        synthesis = torch.fft.irfft(enhanced, n=self.config.n_fft)[:, 0] * self.window
        weight = self.window.square()
        denominator = state.synthesis_weight + weight[:hop]
        output = (state.synthesis + synthesis[:, :hop]) / denominator.clamp_min(1e-8)
        next_state = FrequencyStreamState(audio, synthesis[:, hop:],
                                          weight[hop:].expand(audio.shape[0], -1),
                                          tuple(local_states), tuple(global_states))
        return output, next_state

    def model_stats(self) -> dict:
        def weights(module):
            return sum(m.weight.numel() for m in module.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d)))

        convolutions = [m for m in self.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d))]
        weight_count = sum(m.weight.numel() for m in convolutions)
        bias_count = sum(m.bias.numel() for m in convolutions if m.bias is not None)
        macs = (weights(self.stem) * 129 + weights(self.down1) * 65 + weights(self.down2) * 33
                + weights(self.local_blocks) * 33 + weights(self.global_in)
                + weights(self.global_blocks) + weights(self.global_out)
                + weights(self.up1) * 65 + weights(self.up2) * 129
                + (weights(self.head_dw) + weights(self.head)) * 257)
        local_state = 2 * sum(self.config.local_dilations) * self.config.encoder_channels[-1] * 33
        global_state = 2 * sum(self.config.global_dilations) * self.config.global_width
        # Budget includes INT32 biases, per-output exponents, 32-byte layer
        # records, worst-case 16-byte alignment, window and graph descriptors.
        packed_estimate = weight_count + 5 * bias_count + 47 * len(convolutions) + 64 + 2068 + 256
        return {
            "learned_parameters": sum(p.numel() for p in self.parameters()),
            "deployed_parameters": weight_count + bias_count,
            "training_normalization_parameters": sum(p.numel() for m in self.modules()
                                                       if isinstance(m, nn.BatchNorm2d)
                                                       for p in m.parameters()),
            "convolution_weights": weight_count, "convolution_biases": bias_count,
            "convolution_layers": len(convolutions), "macs_per_frame": macs,
            "macs_per_second": macs * self.config.sample_rate / self.config.hop_length,
            "context_frames": 1 + 2 * sum(self.config.local_dilations) + 2 * sum(self.config.global_dilations),
            "local_state_bytes_int8": local_state, "global_state_bytes_int8": global_state,
            "neural_state_bytes_int8": local_state + global_state,
            "estimated_packed_bytes_upper": packed_estimate,
            "packed_bytes_status": "design estimate; frequency_export measures the actual EDNFQ8 payload",
        }
