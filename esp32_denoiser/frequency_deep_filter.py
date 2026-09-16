"""Float-only low-frequency causal deep-filter correction for FrequencyUNet.

The normal complex mask stays intact. A zero-initialized head adds a bounded
linear combination of the current and preceding low-frequency noisy spectra.
No future spectra, extra lookahead, custom downloads, QAT or integer export.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn

from .frequency_model import FrequencyStreamState, FrequencyUNet, FrequencyUNetConfig


@dataclass(frozen=True)
class FrequencyDeepFilterConfig:
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
    encoder_batch_norm: bool = True
    filter_order: int = 5
    filter_bins: int = 65
    filter_scale: float = 0.5

    def __post_init__(self):
        if not isinstance(self.filter_order, int) or self.filter_order < 1:
            raise ValueError("filter_order must be a positive integer")
        if not isinstance(self.filter_bins, int) or not 1 <= self.filter_bins <= 257:
            raise ValueError("filter_bins must be an integer in [1,257]")
        if not math.isfinite(self.filter_scale) or self.filter_scale <= 0:
            raise ValueError("filter_scale must be finite and positive")
        self.convolution_config()

    def convolution_config(self) -> FrequencyUNetConfig:
        values = asdict(self)
        for name in ("filter_order", "filter_bins", "filter_scale"):
            values.pop(name)
        return FrequencyUNetConfig(**values)

    @classmethod
    def from_checkpoint(cls, values: dict) -> "FrequencyDeepFilterConfig":
        values = dict(values)
        for name in ("encoder_channels", "local_dilations", "global_dilations"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


@dataclass
class FrequencyDeepFilterStreamState(FrequencyStreamState):
    spectral_history: Tensor  # [B,order-1,low_bins], oldest to newest, complex.


class FrequencyDeepFilter(FrequencyUNet):
    """Preserve the baseline and add a causal low-band complex correction.

    Waveform input/output is [B,N]. ``forward_features`` takes [B,3,T,257]
    and returns [B,514+2*order*low_bins,T]. The first514 values are ordinary
    mask deltas. Additional channels are real taps then imaginary taps, ordered
    by lag (current frame first), then frequency. Values are clipped to +/-1;
    ``filter_scale`` converts them to complex correction coefficients in DSP.
    """

    def __init__(self, config: FrequencyDeepFilterConfig | None = None):
        config = config or FrequencyDeepFilterConfig()
        super().__init__(config.convolution_config())
        self.config = config
        self.filter_head = nn.Conv2d(config.encoder_channels[0], 2 * config.filter_order, 1)
        self.filter_activation = nn.Hardtanh(-1.0, 1.0)
        nn.init.zeros_(self.filter_head.weight)
        nn.init.zeros_(self.filter_head.bias)
        self.output_size = 514 + 2 * config.filter_order * config.filter_bins

    def _decode(self, x: Tensor, skip1: Tensor, skip2: Tensor) -> Tensor:
        x = self.up2(self.up1(x, skip2), skip1)
        x = x.repeat_interleave(2, dim=-1)[..., :257]
        x = self.head_dw_activation(self.head_dw(x))
        masks = self.head_quant(self.head_activation(self.head(x)))
        coefficients = self.filter_activation(self.filter_head(x[..., :self.config.filter_bins]))
        return torch.cat((self._flatten_frequency(masks), self._flatten_frequency(coefficients)), dim=1)

    def apply_filter(self, spectrum: Tensor, predictions: Tensor,
                     history: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Apply the predicted mask/correction and return bounded past state."""
        if (spectrum.ndim != 3 or spectrum.shape[-1] != 257
                or predictions.shape != (spectrum.shape[0], self.output_size, spectrum.shape[1])):
            raise ValueError("Expected complex spectra [B,T,257] and matching deep-filter predictions")
        order, bins = self.config.filter_order, self.config.filter_bins
        low = spectrum[..., :bins]
        expected_history = (spectrum.shape[0], order - 1, bins)
        if history is None:
            history = spectrum.new_zeros(expected_history)
        elif history.shape != expected_history:
            raise ValueError("Spectral history does not match filter order, bins or batch")
        joined = torch.cat((history, low), dim=1)
        # unfold visits oldest-to-newest; reversing exposes lag0,lag1,... .
        taps = joined.unfold(1, order, 1).permute(0, 1, 3, 2).flip(2)
        values = predictions[:, 514:].to(spectrum.real.dtype)
        values = values.reshape(spectrum.shape[0], 2, order, bins, spectrum.shape[1]).permute(0, 4, 1, 2, 3)
        coefficients = torch.complex(values[:, :, 0], values[:, :, 1]) * self.config.filter_scale
        correction = (coefficients * taps).sum(dim=2)
        masked = super().apply_mask(spectrum, predictions[:, :514])
        enhanced = torch.cat((masked[..., :bins] + correction, masked[..., bins:]), dim=-1)
        # clone avoids retaining an extra current-frame backing allocation in
        # persistent state. The order1 ablation needs no spectral history.
        next_history = joined[:, -(order - 1):].clone() if order > 1 else joined[:, :0].clone()
        return enhanced, next_history

    def apply_mask(self, spectrum: Tensor, predictions: Tensor) -> Tensor:
        return self.apply_filter(spectrum, predictions)[0]

    def init_stream_state(self, batch_size: int = 1, device=None, dtype=None) -> FrequencyDeepFilterStreamState:
        base = super().init_stream_state(batch_size, device, dtype)
        complex_dtype = torch.complex128 if base.analysis.dtype == torch.float64 else torch.complex64
        history = torch.zeros(batch_size, self.config.filter_order - 1, self.config.filter_bins,
                              device=base.analysis.device, dtype=complex_dtype)
        return FrequencyDeepFilterStreamState(base.analysis, base.synthesis, base.synthesis_weight,
                                               base.local, base.global_temporal, history)

    def stream_step(self, audio: Tensor, state: FrequencyDeepFilterStreamState) -> tuple[Tensor, FrequencyDeepFilterStreamState]:
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
        predictions = self._decode(self._fuse_global(x, hidden), skip1, skip2)
        enhanced, spectral_history = self.apply_filter(spectrum, predictions, state.spectral_history)
        synthesis = torch.fft.irfft(enhanced, n=self.config.n_fft)[:, 0] * self.window
        weight = self.window.square()
        denominator = state.synthesis_weight + weight[:hop]
        output = (state.synthesis + synthesis[:, :hop]) / denominator.clamp_min(1e-8)
        next_state = FrequencyDeepFilterStreamState(
            audio, synthesis[:, hop:], weight[hop:].expand(audio.shape[0], -1),
            tuple(local_states), tuple(global_states), spectral_history,
        )
        return output, next_state

    def model_stats(self) -> dict:
        stats = super().model_stats()
        order, bins = self.config.filter_order, self.config.filter_bins
        rate = self.config.sample_rate / self.config.hop_length
        added_macs = self.filter_head.weight.numel() * bins
        stats["macs_per_frame"] += added_macs
        stats["macs_per_second"] = stats["macs_per_frame"] * rate
        stats["context_frames"] = max(stats["context_frames"], order)
        stats.pop("estimated_packed_bytes_upper", None)
        stats.update({
            "precision": "float-only; deep-filter QAT and integer export are not implemented",
            "packed_bytes_status": "unavailable; this deep-filter graph has no integer exporter",
            "neural_state_bytes_int8_status": "forecast only; float neural states are currently used",
            "deep_filter_added_parameters": sum(p.numel() for p in self.filter_head.parameters()),
            "deep_filter_added_neural_macs_per_frame": added_macs,
            "deep_filter_complex_products_per_second": order * bins * rate,
            "deep_filter_real_products_per_second": 4 * order * bins * rate,
            "deep_filter_coefficient_scale_products_per_second": 2 * order * bins * rate,
            "deep_filter_spectral_history_bytes_float32": 8 * (order - 1) * bins,
            "neural_output_values_per_frame": self.output_size,
            "additional_lookahead_frames": 0,
        })
        return stats
