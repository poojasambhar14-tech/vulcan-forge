"""
Batched frozen-backbone inference helper.

Several scripts (train_baseline_heads.py, train_world_model.py) need z_t
embeddings for a large number of pre-built windows from a FROZEN backbone.
Running that as one giant unbatched forward pass is fine at `tiny` scale
but can spike memory well past what's needed at `base` scale (attention
activations scale with batch size) and silently OOM-kill the process with
no Python traceback. This chunks the forward pass instead.
"""
from __future__ import annotations

from typing import Dict

import torch


def batched_current_state(backbone, categorical_ids: Dict[str, torch.Tensor], continuous: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    n = continuous.shape[0]
    outs = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_cat = {name: t[start:end] for name, t in categorical_ids.items()}
            batch_cont = continuous[start:end]
            outs.append(backbone.current_state(batch_cat, batch_cont))
    return torch.cat(outs, dim=0)
