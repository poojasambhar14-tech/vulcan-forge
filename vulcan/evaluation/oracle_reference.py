"""
EVALUATOR-ONLY module.

Nothing in `vulcan/data/` or `vulcan/models/` may import this module. It
exists to compute counterfactual/oracle information for evaluation purposes
(e.g. regret calculation, benchmark reporting) -- never for training.
"""
from __future__ import annotations

from typing import Any, Dict, List

from vulcan.simulator.environment import IndiaPaymentSim, ObservableState


def oracle_best_action_value(sim: IndiaPaymentSim, obs: ObservableState) -> Dict[str, Any]:
    """Evaluator-only: returns the true best action and its value, using the
    simulator's oracle. Used to compute regret at evaluation time."""
    dist = sim.oracle_outcome_distribution(obs)
    best_route = max(dist.keys(), key=lambda r: dist[r]["p_success"])
    return {
        "best_route": best_route,
        "distribution": dist,
    }


def compute_regret(sim: IndiaPaymentSim, obs: ObservableState, chosen_route: int, utility_fn) -> float:
    """utility_fn(route_id, dist_entry) -> float. Regret is defined as
    utility(best action) - utility(chosen action), evaluator-side only."""
    dist = sim.oracle_outcome_distribution(obs)
    utilities = {r: utility_fn(r, d) for r, d in dist.items()}
    best_utility = max(utilities.values())
    chosen_utility = utilities[chosen_route]
    return float(best_utility - chosen_utility)
