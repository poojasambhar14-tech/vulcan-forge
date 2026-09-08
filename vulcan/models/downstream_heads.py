"""
Downstream Mini-Vulcan heads (master spec section 7 / Baseline B).

Attach lightweight per-route heads directly to the shared payment state z_t.
Training labels only exist for the route that was actually taken (bandit
feedback), so losses are masked to the taken route per example.

Baseline B policy: greedy route selection by deterministic utility computed
directly from these heads (no explicit action-conditioned dynamics function;
see vulcan/world_model for the action-conditioned alternative, Model C/D).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class DownstreamHeads(nn.Module):
    def __init__(self, d_model: int, num_routes: int):
        super().__init__()
        self.num_routes = num_routes
        self.success_head = nn.Linear(d_model, num_routes)
        self.fraud_head = nn.Linear(d_model, num_routes)
        self.abandon_head = nn.Linear(d_model, num_routes)
        self.latency_head = nn.Linear(d_model, num_routes)  # predicts log-latency

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """z: [N, d_model] (current-state representation).
        Returns raw logits/values per route, shape [N, num_routes]."""
        return {
            "success_logit": self.success_head(z),
            "fraud_logit": self.fraud_head(z),
            "abandon_logit": self.abandon_head(z),
            "log_latency": self.latency_head(z),
        }


def masked_bandit_bce_loss(
    logits: torch.Tensor, targets: torch.Tensor, taken_route: torch.Tensor
) -> torch.Tensor:
    """logits/targets: [N, num_routes] -- but we only observe the true label
    for the route in `taken_route` [N]. Compute BCE only on that column."""
    N = logits.shape[0]
    idx = torch.arange(N)
    selected_logits = logits[idx, taken_route]
    return torch.nn.functional.binary_cross_entropy_with_logits(selected_logits, targets)


def masked_latency_mse_loss(
    log_latency_pred: torch.Tensor, log_latency_target: torch.Tensor, taken_route: torch.Tensor
) -> torch.Tensor:
    N = log_latency_pred.shape[0]
    idx = torch.arange(N)
    selected = log_latency_pred[idx, taken_route]
    return torch.nn.functional.mse_loss(selected, log_latency_target)


def greedy_utility_select(
    heads_out: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    """Baseline B decision rule: pick route maximizing deterministic utility
    from direct per-route head outputs. Returns chosen route id per example
    [N]."""
    p_success = torch.sigmoid(heads_out["success_logit"])
    p_fraud = torch.sigmoid(heads_out["fraud_logit"])
    p_abandon = torch.sigmoid(heads_out["abandon_logit"])
    latency = torch.exp(heads_out["log_latency"])
    norm_latency = latency / (latency.amax(dim=-1, keepdim=True) + 1e-6)

    utility = (
        weights.get("w_success", 1.0) * p_success
        - weights.get("w_fraud", 1.0) * p_fraud
        - weights.get("w_latency", 0.3) * norm_latency
        - weights.get("w_abandon", 0.5) * p_abandon
    )
    return utility.argmax(dim=-1)
