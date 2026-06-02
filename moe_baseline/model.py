"""MoE denoiser module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEHardDenoiser(nn.Module):
    """y = x + r_shared(h) + sum_k pi_k * r_k(h); router logits or external route indices."""

    def __init__(
        self,
        frame_size: int,
        num_experts: int,
        hidden: int = 512,
        bottleneck: int = 128,
        use_shared_expert: bool = True,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.num_experts = num_experts
        self.use_shared_expert = use_shared_expert

        self.encoder = nn.Sequential(
            nn.Linear(frame_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
            nn.GELU(),
        )
        self.router = nn.Linear(bottleneck, num_experts)
        self.shared_expert = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.GELU(),
            nn.Linear(hidden, frame_size),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(bottleneck, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, frame_size),
                )
                for _ in range(num_experts)
            ]
        )

    def _mix_experts(self, h: torch.Tensor, route_idx: torch.Tensor) -> torch.Tensor:
        expert_out = torch.stack([e(h) for e in self.experts], dim=1)
        pi = F.one_hot(route_idx, num_classes=self.num_experts).to(h.dtype)
        return (pi.unsqueeze(-1) * expert_out).sum(dim=1)

    def forward(self, noisy: torch.Tensor):
        h = self.encoder(noisy)
        logits = self.router(h)
        pi_soft = F.softmax(logits, dim=-1)
        idx = pi_soft.argmax(dim=-1)
        pi_hard = F.one_hot(idx, num_classes=self.num_experts).to(noisy.dtype)
        pi_ste = pi_hard - pi_soft.detach() + pi_soft
        expert_out = torch.stack([e(h) for e in self.experts], dim=1)
        mix = (pi_ste.unsqueeze(-1) * expert_out).sum(dim=1)
        shared = self.shared_expert(h) if self.use_shared_expert else 0.0
        return noisy + shared + mix, logits

    def forward_with_route(self, noisy: torch.Tensor, route_idx: torch.Tensor) -> torch.Tensor:
        h = self.encoder(noisy)
        mix = self._mix_experts(h, route_idx)
        shared = self.shared_expert(h) if self.use_shared_expert else 0.0
        return noisy + shared + mix
