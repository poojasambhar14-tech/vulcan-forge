"""
EVALUATOR-ONLY module (see vulcan/evaluation/oracle_reference.py for the
existing isolation convention this follows). Used only in evaluation/test
code paths -- never imported by vulcan/data/generate.py or any training
script.

Computes:
  - the simulator's TRUE (oracle) pairwise route-effect sizes, i.e. how much
    P(success) genuinely differs between candidate routes for a given
    state, according to the simulator's own ground-truth structural model.
    This is the calibration reference for "is the world model's action
    sensitivity even in the right ballpark", not a target to tune toward.
  - the world model's PREDICTED pairwise route-effect sizes for the same
    states.
  - comparison statistics between the two (MAE, correlation, ranking
    accuracy).
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats as scipy_stats

from vulcan.simulator.environment import IndiaPaymentSim, ObservableState


def oracle_pairwise_effects(sim: IndiaPaymentSim, observations: List[ObservableState]) -> Dict[str, np.ndarray]:
    """For each observation, query the simulator's oracle for every route's
    true P(success), then compute:
      - mean_abs_pairwise_effect: mean over all route pairs of |p_i - p_j|
      - range_effect: max_route(p) - min_route(p)
      - per_state_probs: [N, num_routes] array of true P(success) per route
    EVALUATOR-ONLY: calls sim.oracle_outcome_distribution directly."""
    num_routes = sim.num_routes
    pairs = list(combinations(range(num_routes), 2))

    mean_abs_pairwise = np.zeros(len(observations))
    range_effect = np.zeros(len(observations))
    per_state_probs = np.zeros((len(observations), num_routes))

    for i, obs in enumerate(observations):
        dist = sim.oracle_outcome_distribution(obs)
        probs = np.array([dist[r]["p_success"] for r in range(num_routes)])
        per_state_probs[i] = probs
        diffs = [abs(probs[a] - probs[b]) for a, b in pairs]
        mean_abs_pairwise[i] = float(np.mean(diffs))
        range_effect[i] = float(probs.max() - probs.min())

    return {
        "mean_abs_pairwise_effect": mean_abs_pairwise,
        "range_effect": range_effect,
        "per_state_probs": per_state_probs,
    }


def predicted_pairwise_effects(world_model, z: torch.Tensor, num_routes: int) -> Dict[str, np.ndarray]:
    """Same quantities as oracle_pairwise_effects, but from the world
    model's predicted P(success) per route for a batch of states z."""
    pairs = list(combinations(range(num_routes), 2))
    N = z.shape[0]

    with torch.no_grad():
        preds = world_model.predict_all_routes(z, num_routes)
        probs = torch.stack([torch.sigmoid(preds[r]["success_logit"]) for r in range(num_routes)], dim=1).numpy()

    mean_abs_pairwise = np.zeros(N)
    range_effect = np.zeros(N)
    for i in range(N):
        diffs = [abs(probs[i, a] - probs[i, b]) for a, b in pairs]
        mean_abs_pairwise[i] = float(np.mean(diffs))
        range_effect[i] = float(probs[i].max() - probs[i].min())

    return {
        "mean_abs_pairwise_effect": mean_abs_pairwise,
        "range_effect": range_effect,
        "per_state_probs": probs,
    }


def compare_predicted_vs_oracle(oracle: Dict[str, np.ndarray], predicted: Dict[str, np.ndarray]) -> Dict[str, float]:
    """MAE and correlation between predicted and oracle per-state pairwise
    effect sizes, plus best-route ranking accuracy (does the model's argmax
    route match the oracle's argmax route more often than chance)."""
    oracle_effect = oracle["mean_abs_pairwise_effect"]
    pred_effect = predicted["mean_abs_pairwise_effect"]

    mae = float(np.mean(np.abs(oracle_effect - pred_effect)))

    if np.std(oracle_effect) > 1e-9 and np.std(pred_effect) > 1e-9:
        pearson_r = float(np.corrcoef(oracle_effect, pred_effect)[0, 1])
        spearman_r = float(scipy_stats.spearmanr(oracle_effect, pred_effect).correlation)
    else:
        pearson_r = float("nan")
        spearman_r = float("nan")

    oracle_best = oracle["per_state_probs"].argmax(axis=1)
    pred_best = predicted["per_state_probs"].argmax(axis=1)
    num_routes = oracle["per_state_probs"].shape[1]
    ranking_accuracy = float(np.mean(oracle_best == pred_best))
    chance_accuracy = 1.0 / num_routes

    return {
        "mae": mae,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "best_route_ranking_accuracy": ranking_accuracy,
        "chance_accuracy": chance_accuracy,
        "oracle_mean_abs_pairwise_effect": float(oracle_effect.mean()),
        "oracle_mean_range_effect": float(oracle["range_effect"].mean()),
        "predicted_mean_abs_pairwise_effect": float(pred_effect.mean()),
        "predicted_mean_range_effect": float(predicted["range_effect"].mean()),
    }
