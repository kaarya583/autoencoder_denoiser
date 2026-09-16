"""Data-free, function-preserving equalization of GTCRN's interior channels.

This is conventional cross-layer equalization, restricted to twelve safe
Conv/BN/PReLU/Conv chains. It changes no graph operators or tensor shapes.
Its weight-range objective does not guarantee lower activation error or
better audio quality; recalibrate and compare the actual integer model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from .experimental_gtcrn_ops import folded_parameters
from .gtcrn_model import GTCRNConfig, GTCRNDenoiser


@dataclass(frozen=True)
class EqualizationConfig:
    passes: int = 4
    max_channel_exponent: int = 4

    def __post_init__(self):
        for name, upper in (("passes", 32), ("max_channel_exponent", 8)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer between 1 and {upper}")


BLOCKS = tuple([f"core.encoder.en_convs.{i}" for i in (2, 3, 4)]
               + [f"core.decoder.de_convs.{i}" for i in (0, 1, 2)])


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        data = value.numpy().tobytes()
        digest.update(len(header).to_bytes(8, "little")); digest.update(header)
        digest.update(len(data).to_bytes(8, "little")); digest.update(data)
    return digest.hexdigest()


def _input_ranges(layer, batch_norm):
    weight, _ = folded_parameters(layer, batch_norm)
    # Canonical folded weights are [output,input/group,time,frequency],
    # including ConvTranspose2d; group membership must remain explicit.
    grouped = weight.reshape(layer.groups, layer.out_channels // layer.groups,
                             layer.in_channels // layer.groups, *layer.kernel_size)
    return np.abs(grouped).max(axis=(1, 3, 4)).reshape(layer.in_channels)


@torch.no_grad()
def _divide_input_channels(layer, scale):
    """Compensate positive input scales in the layer's native Torch layout."""
    if type(layer) not in (nn.Conv2d, nn.ConvTranspose2d):
        raise TypeError("Equalization accepts native Conv2d/ConvTranspose2d only")
    scale = torch.as_tensor(scale, dtype=layer.weight.dtype, device=layer.weight.device)
    if scale.shape != (layer.in_channels,) or not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("Input channel scales must be positive finite values of the correct shape")
    if isinstance(layer, nn.ConvTranspose2d):
        layer.weight.div_(scale[:, None, None, None])
    else:
        grouped = layer.weight.view(layer.groups, layer.out_channels // layer.groups,
                                    layer.in_channels // layer.groups, *layer.kernel_size)
        grouped.div_(scale.view(layer.groups, 1, layer.in_channels // layer.groups, 1, 1))
    # The downstream convolution bias is independent of its input: unchanged.


def _validate_model(source):
    if type(source) is not GTCRNDenoiser or any(module.training for module in source.modules()):
        raise ValueError("Equalization requires the native GTCRNDenoiser in eval mode")
    for value in source.state_dict().values():
        if value.device.type != "cpu" or (value.is_floating_point() and
                (value.dtype != torch.float32 or not torch.isfinite(value).all())):
            raise ValueError("Equalization requires finite CPU float32 parameters/buffers")
    for path in BLOCKS:
        block = source.get_submodule(path)
        for conv_name, bn_name in (("point_conv1", "point_bn1"), ("depth_conv", "depth_bn"),
                                   ("point_conv2", "point_bn2")):
            conv, bn = getattr(block, conv_name), getattr(block, bn_name)
            if (type(conv) not in (nn.Conv2d, nn.ConvTranspose2d) or type(bn) is not nn.BatchNorm2d
                    or not bn.affine or not bn.track_running_stats):
                raise ValueError("Equalization requires the original convolution/affine-BN topology")
            folded_parameters(conv, bn)
        if (type(block.point_act) is not nn.PReLU or type(block.depth_act) is not nn.PReLU
                or block.point_conv1.out_channels != block.depth_conv.in_channels
                or block.depth_conv.out_channels != block.point_conv2.in_channels):
            raise ValueError("Equalization requires uncompromised interior PReLU chains")


@torch.inference_mode()
def verify_equalized_parity(source, transformed):
    """Synthetic waveform and streaming probes; no dataset or quality fitting."""
    generator = torch.Generator().manual_seed(48261)
    time = torch.arange(3073, dtype=torch.float32) / 16000
    probes = [torch.zeros(1, 1), torch.randn(2, 1025, generator=generator) * .04,
              (torch.sin(time * (2*torch.pi*213)) * .13 + .03)[None],
              torch.randn(1, 773, generator=generator) * 1e-6,
              torch.randn(1, 1027, generator=generator) * .7]
    maximum, maximum_stream = 0., 0.
    for probe in probes:
        expected, actual = source(probe), transformed(probe)
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise ValueError("Nonfinite waveform during equalization verification")
        maximum = max(maximum, float((expected-actual).abs().max()))
        if not torch.allclose(actual, expected, rtol=3e-5, atol=3e-6):
            raise ValueError("Equalization did not preserve float waveform behavior")
    probe = torch.randn(1, 4609, generator=generator) * .06
    expected, actual = source.make_streaming().denoise(probe), transformed.make_streaming().denoise(probe)
    maximum_stream = float((expected-actual).abs().max())
    if not torch.isfinite(actual).all() or not torch.allclose(actual, expected, rtol=3e-5, atol=3e-6):
        raise ValueError("Equalization did not preserve streaming/reset/flush behavior")
    return dict(waveform_max_abs_error=maximum, streaming_max_abs_error=maximum_stream,
                waveform_batches=len(probes), streaming_samples=probe.numel(),
                rtol=3e-5, atol=3e-6, data_source="deterministic synthetic probes only",
                scope="numerical function-preservation check, not an audio-quality evaluation")


@torch.no_grad()
def equalize_gtcrn(source: GTCRNDenoiser, config: EqualizationConfig | None = None):
    """Return a verified independent model copy and its complete transform log.

    For folded first-layer output range a and second-layer input range b,
    choose D=2**round(log2(sqrt(b/a))). Scale upstream BN gamma/beta by D
    and divide downstream convolution input weights by D. Both folded
    upstream weights AND bias therefore scale by D. PReLU commutes with D
    because D is strictly positive. BN running statistics remain unchanged.

    Ranges use maximum absolute folded matrix coefficients, not audio or
    activation statistics. Zero matrix ranges are left alone. Cumulative
    per-chain exponents stay within +/-max_channel_exponent over all passes.
    No scaling crosses a shuffle, residual, attention, recurrent or LN edge.
    """
    config = config or EqualizationConfig()
    if not isinstance(config, EqualizationConfig):
        raise TypeError("config must be EqualizationConfig")
    _validate_model(source)
    initial_sha = _state_sha256(source)
    transformed = deepcopy(source)
    cumulative, records = {}, []
    allowed = set()
    pairs = (("point_conv1", "point_bn1", "depth_conv", "depth_bn"),
             ("depth_conv", "depth_bn", "point_conv2", "point_bn2"))
    for pass_index in range(config.passes):
        changed = 0
        for path in BLOCKS:
            block = transformed.get_submodule(path)
            for left, left_bn, right, right_bn in pairs:
                upstream, normalization = getattr(block, left), getattr(block, left_bn)
                downstream, next_bn = getattr(block, right), getattr(block, right_bn)
                name = f"{path}.{left}->{right}"
                previous = cumulative.setdefault(name, np.zeros(upstream.out_channels, np.int64))
                left_weight, left_bias = folded_parameters(upstream, normalization)
                a = np.abs(left_weight).reshape(upstream.out_channels, -1).max(axis=1)
                b = _input_ranges(downstream, next_bn)
                active = (a > 0) & (b > 0)
                step = np.zeros_like(previous)
                step[active] = np.rint((np.log2(b[active])-np.log2(a[active]))*.5).astype(np.int64)
                step = np.clip(step, -config.max_channel_exponent-previous,
                               config.max_channel_exponent-previous)
                scale = np.exp2(step.astype(np.float64))
                values = torch.from_numpy(scale.astype(np.float32))
                normalization.weight.mul_(values); normalization.bias.mul_(values)
                _divide_input_channels(downstream, values)
                previous += step
                new_weight, new_bias = folded_parameters(upstream, normalization)
                # Check the folded algebra as well as the later waveform gate.
                np.testing.assert_array_equal(new_weight, left_weight*scale[:, None, None, None])
                np.testing.assert_array_equal(new_bias, left_bias*scale)
                changed += int(np.count_nonzero(step))
                records.append(dict(pass_index=pass_index, chain=name,
                    upstream_bn=path+"."+left_bn, downstream_conv=path+"."+right,
                    downstream_type=type(downstream).__name__, groups=downstream.groups,
                    step_exponents=step.tolist(), cumulative_exponents=previous.tolist(),
                    left_output_weight_ranges_before=a.tolist(), right_input_weight_ranges_before=b.tolist(),
                    left_output_weight_ranges_after=(a*scale).tolist(),
                    right_input_weight_ranges_after=(b/scale).tolist(),
                    zero_range_channels=np.flatnonzero(~active).tolist()))
                allowed.update((path+"."+left_bn+".weight", path+"."+left_bn+".bias",
                                path+"."+right+".weight"))
        if not changed:
            break
    _validate_model(transformed)
    actual_changes = [name for name, value in source.state_dict().items()
                      if not torch.equal(value, transformed.state_dict()[name])]
    if not set(actual_changes) <= allowed or _state_sha256(source) != initial_sha:
        raise ValueError("Equalization changed an unsafe tensor or mutated the source")
    parity = verify_equalized_parity(source, transformed)
    report = dict(format="gtcrn_channel_equalization_v1", config=asdict(config),
        method="bounded positive power-of-two cross-layer weight-range equalization",
        scale_rounding="nearest integer log2, ties to even", data_fitting=False,
        source_state_sha256=initial_sha, transformed_state_sha256=_state_sha256(transformed),
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        changed_tensors=actual_changes, passes_run=pass_index+1, chains=records,
        cumulative_exponents={name: value.tolist() for name, value in cumulative.items()},
        waveform_parity=parity,
        limitations="Weight ranges exclude biases from scale selection; biases are rescaled exactly. No measured INT8 quality gain. Recalibrate the transformed source; old calibration fingerprints are invalid.")
    return transformed, report


def prepare_equalized_checkpoint(source, destination, *, config=None, expected_source_sha256=None):
    """Write a new initialization artifact, preserving the audited train recipe.

    The original file is read once for deserialization, then its hash is
    rechecked before publication. No optimizer or stale grids are copied.
    Fresh optimization must explicitly use resume_optimizer=False.
    """
    source, destination = Path(source), Path(destination)
    sidecar = destination.with_suffix(".equalization.json")
    if source.resolve() in (destination.resolve(), sidecar.resolve()):
        raise ValueError("Equalized artifacts must differ from the source checkpoint")
    if destination.exists() or sidecar.exists():
        raise FileExistsError("Equalization destination or report already exists")
    content = source.read_bytes(); source_sha = hashlib.sha256(content).hexdigest()
    if expected_source_sha256 is not None and source_sha != expected_source_sha256:
        raise ValueError("Source checkpoint differs from the expected frozen SHA256")
    checkpoint = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    if (not isinstance(checkpoint, dict) or checkpoint.get("model_kind") != "gtcrn"
            or checkpoint.get("phase") != "float"):
        raise ValueError("Equalization requires a float GTCRN checkpoint")
    provenance, training = checkpoint.get("provenance"), checkpoint.get("train_config")
    if (not isinstance(provenance, dict) or provenance.get("test_used_for_selection") is not False
            or not isinstance(training, dict)):
        raise ValueError("Equalization requires recorded training provenance without test selection")
    if provenance.get("channel_equalization"):
        raise ValueError("Refusing to stack equalization on an already transformed checkpoint")
    with torch.random.fork_rng(devices=[]):
        model = GTCRNDenoiser(GTCRNConfig.from_checkpoint(checkpoint["model_config"])).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    transformed, report = equalize_gtcrn(model, config)
    lineage = dict(source_checkpoint=str(source.resolve()), source_checkpoint_sha256=source_sha,
                   source_checkpoint_epoch=checkpoint.get("epoch"), source_recorded_best_si_sdri=checkpoint.get("best_si_sdri"),
                   source_resume_checkpoint_sha256=provenance.get("resume_checkpoint_sha256"), transform=report)
    provenance = deepcopy(provenance)
    provenance["channel_equalization"] = lineage
    provenance["resume_checkpoint_sha256"] = source_sha
    provenance["calibration"] = None
    payload = dict(model=transformed.state_dict(), model_config=asdict(model.config), model_kind="gtcrn",
                   phase="float", epoch=0, initialization_only=True,
                   train_config=deepcopy(training), provenance=provenance)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
        raise ValueError("Source checkpoint changed during equalization; nothing was written")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    published = []
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name+".", delete=False) as handle:
            temporary = Path(handle.name); torch.save(payload, handle)
        result = dict(**lineage, destination=str(destination.resolve()),
                      checkpoint_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),
                      initialization_only=True, resume_optimizer=False, report_path=str(sidecar.resolve()))
        # Hard-link publication is atomic and refuses a concurrently created
        # destination instead of overwriting it with Path.replace().
        os.link(temporary, destination); published.append(destination)
        with sidecar.open("x") as handle:
            published.append(sidecar); json.dump(result, handle, indent=2); handle.write("\n")
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--max-channel-exponent", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    result = prepare_equalized_checkpoint(args.source, args.output,
        config=EqualizationConfig(args.passes, args.max_channel_exponent),
        expected_source_sha256=args.expected_source_sha256)
    print(json.dumps({name: result[name] for name in ("destination", "checkpoint_sha256", "report_path",
                                                    "initialization_only", "resume_optimizer")}, indent=2))


if __name__ == "__main__":
    main()
