"""Training-only encoder normalization that folds into ordinary convolutions."""

import torch
from torch import Tensor, nn


class EncoderBatchNormConv2d(nn.Conv2d):
    """Preserve the Conv2d API while normalizing its output during training.

    Evaluation uses fixed running statistics. Fold this module before QAT and
    optimizer construction; deployment never executes normalization separately.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_norm = nn.BatchNorm2d(self.out_channels, device=self.weight.device,
                                         dtype=self.weight.dtype)

    @classmethod
    def from_float(cls, layer: nn.Conv2d) -> "EncoderBatchNormConv2d":
        if type(layer) is not nn.Conv2d:
            raise TypeError("Encoder normalization requires an ordinary floating Conv2d")
        # Wrapping must not change initialization of later model layers in the
        # matched-seed normalization ablation. The new random weights are unused.
        with torch.random.fork_rng(devices=[layer.weight.device] if layer.weight.is_cuda else []):
            result = cls(layer.in_channels, layer.out_channels, layer.kernel_size,
                         stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                         groups=layer.groups, bias=layer.bias is not None,
                         padding_mode=layer.padding_mode, device=layer.weight.device,
                         dtype=layer.weight.dtype)
        result.weight = layer.weight
        result.bias = layer.bias
        result.train(layer.training)
        return result

    def forward(self, x: Tensor) -> Tensor:
        return self.batch_norm(super().forward(x))

    @torch.inference_mode(False)
    @torch.no_grad()
    def folded(self) -> nn.Conv2d:
        """Return the convolution equivalent to evaluation-mode BN output."""
        norm = self.batch_norm
        if not norm.track_running_stats or norm.running_mean is None or norm.running_var is None:
            raise ValueError("Folding requires fixed BatchNorm running statistics")
        with torch.random.fork_rng(devices=[self.weight.device] if self.weight.is_cuda else []):
            result = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                               stride=self.stride, padding=self.padding, dilation=self.dilation,
                               groups=self.groups, bias=True, padding_mode=self.padding_mode,
                               device=self.weight.device, dtype=self.weight.dtype)
        scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
        bias = self.bias if self.bias is not None else torch.zeros_like(norm.running_mean)
        result.weight.copy_(self.weight * scale[:, None, None, None])
        result.bias.copy_((bias - norm.running_mean) * scale + norm.bias)
        result.weight.requires_grad_(self.weight.requires_grad)
        result.bias.requires_grad_(self.bias.requires_grad if self.bias is not None else self.weight.requires_grad)
        result.train(self.training)
        return result


def fold_encoder_batch_norm(model: nn.Module) -> nn.Module:
    """Fold encoder wrappers in place, using stored evaluation statistics.

    Idempotent. The model's train/eval mode is preserved. Parameters are replaced,
    so call before constructing a QAT optimizer. Configuration is preserved so
    the training recipe remains recorded in checkpoints.
    """
    for name, layer in list(model.named_modules()):
        if isinstance(layer, EncoderBatchNormConv2d):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, layer.folded())
    return model
