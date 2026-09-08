"""
Parameter-efficient adaptation (master spec section 13).

Implementation note: we use a bottleneck residual adapter (down-project ->
nonlinearity -> up-project -> residual add) applied to the frozen backbone's
shared payment state z_t, rather than literal LoRA low-rank weight-matrix
decomposition inside attention/FFN layers. This is a legitimate, commonly
used PEFT alternative and follows the same principle PRAGMA motivates
(freeze the shared backbone, train a small number of new parameters to
specialize it) -- but we do not claim it IS LoRA. See docs/novelty_boundary.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int = 16):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        # zero-init the up-projection so the adapter starts as an identity
        # function (standard adapter init trick to avoid destabilizing the
        # frozen backbone's representation at the start of adaptation)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.up(self.act(self.down(z)))

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
