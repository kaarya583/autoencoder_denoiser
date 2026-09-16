"""Float-only frequency U-Net ablation with a shared temporal GRU.

The encoder, global TCN, decoder, and audio DSP match FrequencyUNet. One GRU
shares weights across frequency locations while retaining independent states.
Integer memory figures are forecasts: this graph has no QAT or integer export.
"""

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .frequency_model import FrequencyUNet, FrequencyUNetConfig


@dataclass(frozen=True)
class FrequencyGRUConfig:
    encoder_channels: tuple[int, int, int] = (16, 24, 32)
    recurrent_hidden: int = 16
    global_width: int = 32
    global_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    mask_scale: float = 2.0
    activation_mode: str = "signed"
    feature_layout: str = "full_complex257"
    encoder_batch_norm: bool = False

    def __post_init__(self):
        if not isinstance(self.recurrent_hidden, int) or self.recurrent_hidden < 1:
            raise ValueError("recurrent_hidden must be a positive integer")
        self.convolution_config()  # Validate the shared DSP and convolution fields.

    def convolution_config(self) -> FrequencyUNetConfig:
        values = asdict(self)
        values.pop("recurrent_hidden")
        return FrequencyUNetConfig(**values)

    @classmethod
    def from_checkpoint(cls, values: dict) -> "FrequencyGRUConfig":
        values = dict(values)
        for name in ("encoder_channels", "global_dilations"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


@dataclass
class FrequencyGRUStreamState:
    analysis: Tensor
    synthesis: Tensor
    synthesis_weight: Tensor
    recurrent: Tensor  # [1, batch * frequency, hidden], one state per band.
    global_temporal: tuple[Tensor, ...]


class FrequencyGRU(FrequencyUNet):
    """Frequency-shared GRU, with the same waveform/feature contract as the TCN.

    ``forward_features`` accepts [B,3,T,257] and returns [B,514,T]. Calling the
    ordinary waveform forward starts a fresh utterance; stream_step preserves
    recurrent state until the caller explicitly creates a new state.
    """

    def __init__(self, config: FrequencyGRUConfig | None = None):
        config = config or FrequencyGRUConfig()
        # Construct common modules through the baseline constructor. This also
        # keeps their matched-seed initial weights identical for this ablation.
        super().__init__(config.convolution_config())
        del self.local_blocks
        self.config = config
        channels = config.encoder_channels[-1]
        self.local_gru = nn.GRU(channels, config.recurrent_hidden, batch_first=True)
        self.local_projection = nn.Linear(config.recurrent_hidden, channels)
        self.local_residual_quant = nn.Identity()
        self.local_activation = nn.Hardtanh(-6.0, 6.0)
        with torch.no_grad():
            self.local_projection.weight.mul_(0.1)
            self.local_projection.bias.zero_()

    def _recur(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch, channels, frames, frequencies = x.shape
        sequence = x.permute(0, 3, 2, 1).reshape(batch * frequencies, frames, channels)
        hidden, next_state = self.local_gru(sequence, state)
        branch = self.local_projection(hidden).reshape(batch, frequencies, frames, channels)
        branch = branch.permute(0, 3, 2, 1)
        return self.local_activation(self.local_residual_quant(x + branch)), next_state

    def forward_features(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != 3 or features.shape[-1] != 257:
            raise ValueError("FrequencyGRU features must have shape [B,3,T,257]")
        skip1, skip2, x = self._encode(features)
        x, _ = self._recur(x)
        hidden = self.global_in_activation(self.global_in(self._flatten_frequency(x)))
        for block in self.global_blocks:
            hidden = block(hidden)
        return self._decode(self._fuse_global(x, hidden), skip1, skip2)

    def init_stream_state(self, batch_size: int = 1, device=None, dtype=None) -> FrequencyGRUStreamState:
        parameter = self.stem.weight
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype

        def zeros(*shape):
            return torch.zeros(shape, device=device, dtype=dtype)

        hop = self.config.hop_length
        return FrequencyGRUStreamState(
            analysis=zeros(batch_size, hop), synthesis=zeros(batch_size, hop),
            synthesis_weight=zeros(batch_size, hop),
            recurrent=zeros(1, batch_size * 33, self.config.recurrent_hidden),
            global_temporal=tuple(zeros(batch_size, self.config.global_width, block.history_length)
                                  for block in self.global_blocks),
        )

    def stream_step(self, audio: Tensor, state: FrequencyGRUStreamState) -> tuple[Tensor, FrequencyGRUStreamState]:
        hop = self.config.hop_length
        if audio.ndim != 2 or audio.shape[-1] != hop or audio.shape != state.analysis.shape:
            raise ValueError("stream_step expects matching audio/state [B,256]")
        if (state.recurrent.shape != (1, audio.shape[0] * 33, self.config.recurrent_hidden)
                or len(state.global_temporal) != len(self.global_blocks)):
            raise ValueError("Recurrent state does not match this model")
        frames = torch.cat((state.analysis, audio), dim=-1).unsqueeze(1)
        spectrum, features = self.frame_features(frames)
        skip1, skip2, x = self._encode(features)
        x, recurrent = self._recur(x, state.recurrent)
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
        next_state = FrequencyGRUStreamState(
            audio, synthesis[:, hop:], weight[hop:].expand(audio.shape[0], -1),
            recurrent, tuple(global_states),
        )
        return output, next_state

    def model_stats(self) -> dict:
        def weights(module):
            return sum(m.weight.numel() for m in module.modules()
                       if isinstance(m, (nn.Conv1d, nn.Conv2d)))

        convolutions = [m for m in self.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d))]
        weight_count = sum(m.weight.numel() for m in convolutions)
        bias_count = sum(m.bias.numel() for m in convolutions if m.bias is not None)
        recurrent_weights = self.local_gru.weight_ih_l0.numel() + self.local_gru.weight_hh_l0.numel()
        recurrent_biases = self.local_gru.bias_ih_l0.numel() + self.local_gru.bias_hh_l0.numel()
        projection_weights = self.local_projection.weight.numel()
        projection_biases = self.local_projection.bias.numel()
        macs = (weights(self.stem) * 129 + weights(self.down1) * 65 + weights(self.down2) * 33
                + (recurrent_weights + projection_weights) * 33 + weights(self.global_in)
                + weights(self.global_blocks) + weights(self.global_out)
                + weights(self.up1) * 65 + weights(self.up2) * 129
                + (weights(self.head_dw) + weights(self.head)) * 257)
        recurrent_state = 33 * self.config.recurrent_hidden
        global_state = 2 * sum(self.config.global_dilations) * self.config.global_width
        return {
            "precision": "float-only; QAT and integer export are not implemented",
            "learned_parameters": sum(p.numel() for p in self.parameters()),
            "deployed_parameters": (weight_count + bias_count + recurrent_weights + recurrent_biases
                                     + projection_weights + projection_biases),
            "training_normalization_parameters": sum(p.numel() for m in self.modules()
                                                       if isinstance(m, nn.BatchNorm2d)
                                                       for p in m.parameters()),
            "convolution_weights": weight_count, "convolution_biases": bias_count,
            "convolution_layers": len(convolutions),
            "recurrent_weights": recurrent_weights, "recurrent_biases": recurrent_biases,
            "projection_weights": projection_weights, "projection_biases": projection_biases,
            "macs_per_frame": macs,
            "macs_per_second": macs * self.config.sample_rate / self.config.hop_length,
            "macs_status": "matrix products only; GRU nonlinearities and elementwise gates excluded",
            "context_frames": None,
            "temporal_context": "recurrent history; no future audio frames",
            "recurrent_state_bytes_int8": recurrent_state,
            "global_state_bytes_int8": global_state,
            "neural_state_bytes_int8": recurrent_state + global_state,
            "neural_state_bytes_int8_status": "forecast only; float states are currently used",
            "neural_state_bytes_float32": 4 * (recurrent_state + global_state),
            "packed_bytes_status": "unavailable; this recurrent graph has no integer exporter",
        }
