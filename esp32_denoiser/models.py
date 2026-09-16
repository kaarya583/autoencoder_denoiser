"""Explicit architecture dispatch, with backward-compatible checkpoint loading."""
from __future__ import annotations

from .model import SpectralTCN, SpectralTCNConfig


def build_model(kind: str, options: dict, *, checkpoint: bool = False):
    if kind == "spectral_tcn":
        config = SpectralTCNConfig.from_checkpoint(options) if checkpoint else SpectralTCNConfig(**options)
        return SpectralTCN(config)
    if kind == "frequency_unet":
        from .frequency_model import FrequencyUNet, FrequencyUNetConfig
        config = FrequencyUNetConfig.from_checkpoint(options)
        return FrequencyUNet(config)
    if kind == "frequency_gru":
        from .frequency_gru import FrequencyGRU, FrequencyGRUConfig
        return FrequencyGRU(FrequencyGRUConfig.from_checkpoint(options))
    if kind == "frequency_deep_filter":
        from .frequency_deep_filter import FrequencyDeepFilter, FrequencyDeepFilterConfig
        return FrequencyDeepFilter(FrequencyDeepFilterConfig.from_checkpoint(options))
    if kind == "gtcrn":
        from .gtcrn_model import GTCRNDenoiser, GTCRNConfig
        return GTCRNDenoiser(GTCRNConfig.from_checkpoint(options))
    raise ValueError(f"Unknown model_kind: {kind}")


def checkpoint_kind(checkpoint: dict) -> str:
    return checkpoint.get("model_kind", "spectral_tcn")


def configure_model_qat(model, kind: str, **kwargs):
    if kind in {"frequency_gru", "frequency_deep_filter", "gtcrn"}:
        raise ValueError(f"{kind} is float-only; INT8 QAT is not implemented")
    if kind == "spectral_tcn":
        from .quantization import configure_qat
        return configure_qat(model, **kwargs)
    if kind == "frequency_unet":
        from .frequency_quantization import configure_frequency_qat
        return configure_frequency_qat(model, **kwargs)
    raise ValueError(f"QAT is not implemented for {kind}")


def calibrate_model_hidden_exponent(model, kind: str, waveforms, **kwargs):
    if kind in {"frequency_gru", "frequency_deep_filter", "gtcrn"}:
        raise ValueError(f"{kind} is float-only; INT8 calibration is not implemented")
    if kind == "spectral_tcn":
        from .quantization import calibrate_hidden_exponent
        return calibrate_hidden_exponent(model, waveforms, **kwargs)
    if kind == "frequency_unet":
        from .frequency_quantization import calibrate_frequency_hidden_exponent
        return calibrate_frequency_hidden_exponent(model, waveforms, **kwargs)
    raise ValueError(f"QAT calibration is not implemented for {kind}")
