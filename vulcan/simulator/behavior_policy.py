"""
Behavior policy used ONLY to collect training data.

Produces adequate route coverage (80% "reasonable" route choice based on
cheap observable signals + 20% uniform exploration) and logs the propensity
(probability) of the action actually taken, so off-policy evaluation is
possible later. This policy must only use OBSERVABLE state — it must never
call `IndiaPaymentSim.oracle_outcome_distribution`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from vulcan.simulator.environment import ObservableState, RouteAction


@dataclass
class BehaviorPolicyConfig:
    exploration_rate: float = 0.20


class BehaviorPolicy:
    def __init__(self, cfg: BehaviorPolicyConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def _reasonable_scores(self, obs: ObservableState, actions: List[RouteAction]) -> np.ndarray:
        """Cheap heuristic using only OBSERVABLE network signals (rolling
        success rate minus timeout rate), NOT the hidden ground truth."""
        scores = []
        for a in actions:
            r = a.route_id
            success = obs.network.rolling_route_success.get(r, 0.85)
            timeout = obs.network.recent_timeout_rate.get(r, 0.02)
            scores.append(success - 0.5 * timeout)
        arr = np.array(scores, dtype=np.float64)
        # softmax over observable heuristic scores
        arr = arr - arr.max()
        exp = np.exp(arr * 6.0)
        return exp / exp.sum()

    def select_action(self, obs: ObservableState, actions: List[RouteAction]) -> Tuple[RouteAction, float]:
        """Returns (chosen_action, propensity) where propensity is the
        probability THIS policy assigned to the chosen action, needed for
        off-policy evaluation."""
        n = len(actions)
        reasonable_probs = self._reasonable_scores(obs, actions)
        uniform_probs = np.full(n, 1.0 / n)

        mixed_probs = (
            (1.0 - self.cfg.exploration_rate) * reasonable_probs
            + self.cfg.exploration_rate * uniform_probs
        )
        mixed_probs = mixed_probs / mixed_probs.sum()

        idx = self.rng.choice(n, p=mixed_probs)
        return actions[idx], float(mixed_probs[idx])
