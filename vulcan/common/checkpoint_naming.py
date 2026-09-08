"""
Checkpoint naming helper.

Training scripts originally hardcoded `checkpoints/mini_vulcan_pretrained.pt`
etc. regardless of config, which meant running a second config (e.g.
`base.yaml`) silently overwrote the `tiny.yaml` artifacts. To run multiple
scales side by side (see reports/base_scale_result.md), checkpoints are now
named with a model-size suffix -- except `tiny`, which keeps its original,
already-verified filenames unchanged for backward compatibility with
existing artifacts and scripts/demo_part3.py (which is intentionally not
touched by this change).
"""
from __future__ import annotations

from pathlib import Path


def ckpt_name(base_name: str, model_size: str) -> str:
    """e.g. ckpt_name("mini_vulcan_pretrained", "base") -> "mini_vulcan_pretrained_base.pt"
    ckpt_name("mini_vulcan_pretrained", "tiny") -> "mini_vulcan_pretrained.pt" (unchanged)
    """
    if model_size == "tiny":
        return f"{base_name}.pt"
    return f"{base_name}_{model_size}.pt"


def ckpt_path(base_name: str, model_size: str, ckpt_dir: str = "checkpoints") -> Path:
    return Path(ckpt_dir) / ckpt_name(base_name, model_size)
