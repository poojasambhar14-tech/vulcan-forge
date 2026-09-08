"""
Action-conditioned payment world model (master spec section 8).

    z_t = MiniVulcanBackbone.current_state(history, state)
    a_t = ActionEmbedding(candidate_route)
    WorldModel(z_t, a_t) -> predicted outcome distribution + next-state signal

    P(success, latency, fraud, abandonment, next_state | z_t, a_t)

This is the differentiating "Model C/D" alternative to the direct per-route
heads in vulcan/models/downstream_heads.py (Baseline B): instead of a
separate output slot per route, the action is explicitly embedded and
concatenated with z_t, and ONE shared dynamics function is queried once per
candidate action. Horizon is fixed at 1 (see docs/limitations.md on
compounding rollout error); horizon>1 is out of scope for this build.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ActionConditionedWorldModel(nn.Module):
    def __init__(self, d_model: int, num_routes: int, action_emb_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.action_embedding = nn.Embedding(num_routes, action_emb_dim)
        input_dim = d_model + action_emb_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.success_head = nn.Linear(hidden_dim, 1)
        self.fraud_head = nn.Linear(hidden_dim, 1)
        self.abandon_head = nn.Linear(hidden_dim, 1)
        # latency as a Gaussian over log-latency: predict (mean, log_var)
        self.latency_head = nn.Linear(hidden_dim, 2)
        # minimal next-state prediction: next value of this route's rolling
        # success signal (a genuine, observable, action-relevant next-state
        # variable -- not merely re-predicting the current input)
        self.next_state_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor, route_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        """z: [N, d_model], route_id: [N] long.
        Returns dict of prediction tensors, each [N] or [N,2] for latency."""
        a = self.action_embedding(route_id)
        h = self.trunk(torch.cat([z, a], dim=-1))
        latency_params = self.latency_head(h)
        return {
            "success_logit": self.success_head(h).squeeze(-1),
            "fraud_logit": self.fraud_head(h).squeeze(-1),
            "abandon_logit": self.abandon_head(h).squeeze(-1),
            "latency_log_mean": latency_params[:, 0],
            "latency_log_logvar": latency_params[:, 1],
            "next_state_pred": self.next_state_head(h).squeeze(-1),
        }

    def predict_all_routes(self, z: torch.Tensor, num_routes: int) -> Dict[int, Dict[str, torch.Tensor]]:
        """Convenience: query the dynamics function once per candidate
        action for a single state (or batch of states), used by the
        planner. z: [N, d_model]. Returns {route_id: {..per-example preds}}."""
        out = {}
        N = z.shape[0]
        for r in range(num_routes):
            route_ids = torch.full((N,), r, dtype=torch.long)
            out[r] = self.forward(z, route_ids)
        return out


def gaussian_nll_loss(log_mean: torch.Tensor, log_logvar: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood for a Gaussian over log-latency, with
    predicted (mean, log-variance) -- this is how the model returns
    calibrated UNCERTAINTY rather than a single point estimate."""
    var = torch.exp(log_logvar)
    return torch.mean(0.5 * log_logvar + 0.5 * (target_log - log_mean) ** 2 / var)
