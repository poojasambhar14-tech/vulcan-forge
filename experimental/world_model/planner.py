"""
Deterministic multi-objective planner (master spec section 9-10). NOT RL.

Computes utility for each candidate route from the world model's predicted
outcome distribution, applies hard constraints, and returns a full,
structured decision trace (no hidden chain-of-thought). Also implements the
confidence/uncertainty gate: if predictive uncertainty for the chosen action
is too high, fall back to a configured stable policy instead of blindly
executing the planner's raw optimum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from experimental.world_model.dynamics import ActionConditionedWorldModel


@dataclass
class PlannerWeights:
    w_success: float = 1.0
    w_fraud: float = 1.0
    w_latency: float = 0.3
    w_cost: float = 0.2
    w_abandon: float = 0.5


@dataclass
class PlannerConstraints:
    min_route_health: float = 0.15   # p_success floor to even consider a route
    max_fraud_ceiling: float = 0.5   # p_fraud ceiling
    max_retry_count: int = 3


@dataclass
class UncertaintyGateConfig:
    max_latency_logvar: float = 2.0  # above this, treat prediction as too uncertain
    fallback_route_id: int = 0


def _route_health_ok(rolling_success: float, constraints: PlannerConstraints) -> bool:
    return rolling_success >= constraints.min_route_health


def plan_decision(
    world_model: ActionConditionedWorldModel,
    z: torch.Tensor,          # [1, d_model] single current state
    num_routes: int,
    observable_route_health: Dict[int, float],  # rolling success per route, from ObservableState
    retry_count: int,
    weights: PlannerWeights,
    constraints: PlannerConstraints,
    uncertainty_cfg: UncertaintyGateConfig,
) -> Dict[str, Any]:
    """Returns a full structured decision trace: per-candidate predictions,
    utility components, uncertainty, hard-constraint filtering, the chosen
    action, and whether a confidence fallback was triggered."""
    assert z.shape[0] == 1, "plan_decision operates on one current state at a time"

    with torch.no_grad():
        preds = world_model.predict_all_routes(z, num_routes)

    candidates = []
    for r in range(num_routes):
        p = preds[r]
        p_success = float(torch.sigmoid(p["success_logit"])[0])
        p_fraud = float(torch.sigmoid(p["fraud_logit"])[0])
        p_abandon = float(torch.sigmoid(p["abandon_logit"])[0])
        latency_mean = float(torch.exp(p["latency_log_mean"])[0])
        latency_logvar = float(p["latency_log_logvar"][0])

        eligible = True
        exclusion_reasons = []
        route_health = observable_route_health.get(r, 1.0)
        if not _route_health_ok(route_health, constraints):
            eligible = False
            exclusion_reasons.append(f"route_health={route_health:.3f} < min_route_health={constraints.min_route_health}")
        if p_fraud > constraints.max_fraud_ceiling:
            eligible = False
            exclusion_reasons.append(f"p_fraud={p_fraud:.3f} > max_fraud_ceiling={constraints.max_fraud_ceiling}")
        if retry_count > constraints.max_retry_count:
            eligible = False
            exclusion_reasons.append(f"retry_count={retry_count} > max_retry_count={constraints.max_retry_count}")

        candidates.append({
            "route_id": r,
            "p_success": p_success,
            "p_fraud": p_fraud,
            "p_abandon": p_abandon,
            "expected_latency_ms": latency_mean,
            "latency_uncertainty_logvar": latency_logvar,
            "eligible": eligible,
            "exclusion_reasons": exclusion_reasons,
        })

    max_latency = max(c["expected_latency_ms"] for c in candidates) or 1.0
    for c in candidates:
        norm_latency = c["expected_latency_ms"] / max_latency
        utility = (
            weights.w_success * c["p_success"]
            - weights.w_fraud * c["p_fraud"]
            - weights.w_latency * norm_latency
            - weights.w_abandon * c["p_abandon"]
        )
        c["utility_components"] = {
            "success_term": weights.w_success * c["p_success"],
            "fraud_term": -weights.w_fraud * c["p_fraud"],
            "latency_term": -weights.w_latency * norm_latency,
            "abandon_term": -weights.w_abandon * c["p_abandon"],
        }
        c["utility"] = utility

    eligible_candidates = [c for c in candidates if c["eligible"]]
    fallback_triggered = False
    fallback_reason = None

    if not eligible_candidates:
        # nothing passes hard constraints -> fall back
        chosen = uncertainty_cfg.fallback_route_id
        fallback_triggered = True
        fallback_reason = "No candidate route satisfied hard constraints."
    else:
        best = max(eligible_candidates, key=lambda c: c["utility"])
        if best["latency_uncertainty_logvar"] > uncertainty_cfg.max_latency_logvar:
            chosen = uncertainty_cfg.fallback_route_id
            fallback_triggered = True
            fallback_reason = (
                f"Predictive uncertainty too high for best candidate "
                f"(latency_logvar={best['latency_uncertainty_logvar']:.3f} > "
                f"{uncertainty_cfg.max_latency_logvar}); falling back to stable policy."
            )
        else:
            chosen = best["route_id"]

    baseline_candidate = max(candidates, key=lambda c: c["p_success"])  # naive baseline: max success only

    return {
        "candidates": candidates,
        "chosen_action": chosen,
        "confidence_status": "UNSURE" if fallback_triggered else "CONFIDENT",
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "utility_delta_vs_naive_baseline": (
            next(c["utility"] for c in candidates if c["route_id"] == chosen)
            - baseline_candidate["utility"]
        ),
    }
