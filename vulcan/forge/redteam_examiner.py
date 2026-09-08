"""
Red-Team Examiner (Forge pipeline stage: independent adversarial
certification).

DELIBERATELY SEPARATE from curriculum_generator.py. Its objective is NOT to
improve the challenger -- it is to find valid payment states where the
challenger performs worse than expected, especially where it regresses
relative to the champion:

    maximize  loss(challenger, x) - loss(champion, x)
    subject to  x is a valid payment scenario, x was not used for training

This is a genuine iterative local search over the input space (seed pool ->
score by champion/challenger disagreement against the evaluator-only oracle
-> keep the worst -> perturb them -> re-validate -> re-score -> repeat),
NOT a fixed table of named attack scenarios and NOT
`if shift == "issuer": generate_issuer_examples()`. The search is guided
entirely by the actual trained champion and challenger's predictions, so
its output is specific to whichever models it is run against.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from vulcan.forge.schemas import stable_hash
from vulcan.forge.scenario_validator import ScenarioValidator, ScenarioRegistry, IntendedUse
from vulcan.simulator.environment import IndiaPaymentSim, RouteAction
from vulcan.simulator.behavior_policy import BehaviorPolicy, BehaviorPolicyConfig
from vulcan.data.generate import _flatten_observation
from vulcan.evaluation.oracle_reference import oracle_best_action_value


@dataclass
class RedTeamConfig:
    seed_pool_size: int = 300
    top_k_to_refine: int = 20
    refinement_iterations: int = 3
    perturbations_per_candidate: int = 4
    amount_jitter_fractions: Tuple[float, ...] = (-0.3, -0.15, 0.15, 0.3, 0.6)
    # Minimum route-level regret increase (in oracle success-probability
    # units) required before a route flip is reported as a regression.
    # Recalibrated for the route-level scorer -- earlier values (0.4, then
    # 0.10) belonged to two different, since-replaced metrics and carry no
    # meaning here. 0.02 means the challenger's chosen route is at least 2
    # percentage points worse in true success probability than the route
    # the champion would have picked, which is materially costly at
    # payment volume while excluding numerical-noise ties.
    severity_threshold_to_report: float = 0.02
    max_failures_to_return: int = 10


def _oracle_p_success(sim: IndiaPaymentSim, obs) -> float:
    """EVALUATOR-ONLY. Used here only to score/search over candidate
    scenarios for certification purposes -- never used as a training
    signal or fed into any model input."""
    dist = sim.oracle_outcome_distribution(obs)
    # if the record has an action attached, use that route's oracle prob;
    # otherwise use the best-route oracle prob as the reference difficulty.
    return dist


def _predict_prob(predict_fn: Callable[[Dict[str, Any]], float], record: Dict[str, Any]) -> float:
    return float(predict_fn(record))


def _perturb_amount(record: Dict[str, Any], fraction: float) -> Dict[str, Any]:
    new_record = dict(record)
    new_record["amount"] = max(1.0, record["amount"] * (1.0 + fraction))
    return new_record


def run_redteam_search(
    champion_predict_fn: Callable[[Dict[str, Any]], float],
    challenger_predict_fn: Callable[[Dict[str, Any]], float],
    cfg_sim: dict,
    validator: ScenarioValidator,
    registry: ScenarioRegistry,
    seed: int,
    cfg: Optional[RedTeamConfig] = None,
    champion_route_fn: Optional[Callable[[Dict[str, Any], List[int]], int]] = None,
    challenger_route_fn: Optional[Callable[[Dict[str, Any], List[int]], int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    champion_predict_fn / challenger_predict_fn: record -> predicted
        P(success) for the record's own action_route_id (or best-known
        route if unset).

    Returns (list of discovered failure dicts, search metadata). Each
    discovered failure dict contains: scenario (record), oracle_p_success,
    champion_pred, challenger_pred, champion_loss, challenger_loss,
    regression_severity, scenario_hash.
    """
    cfg = cfg or RedTeamConfig()
    rng = np.random.default_rng(seed)

    # ---- seed pool: fresh simulator trajectory (structured simulator
    # perturbation source + rare-but-valid combinations occur naturally
    # across a long enough random trajectory) ----
    sim = IndiaPaymentSim(cfg_sim, seed=seed + 55555)
    policy = BehaviorPolicy(
        BehaviorPolicyConfig(min(0.6, cfg_sim["simulator"]["behavior_policy"]["exploration_rate"] * 3)),  # boosted exploration for red-teaming
        np.random.default_rng(seed + 55556),
    )
    seed_pool: List[Dict[str, Any]] = []
    for _ in range(cfg.seed_pool_size):
        obs = sim.current_observation()
        actions = sim.candidate_actions()
        action, propensity = policy.select_action(obs, actions)
        oracle_dist = sim.oracle_outcome_distribution(obs)  # EVALUATOR-ONLY
        outcome = sim.step(obs, action)
        rec = _flatten_observation(obs)
        rec.update(
            action_route_id=action.route_id, action_rail=action.rail, action_gateway=action.gateway,
            propensity=propensity, outcome_success=outcome.success, outcome_latency_ms=outcome.latency_ms,
            outcome_processing_cost=outcome.processing_cost, outcome_fraud_loss=outcome.fraud_loss,
            outcome_abandoned=outcome.abandoned, outcome_error_type=outcome.error_type.value,
        )
        rec["_oracle_p_success"] = oracle_dist[action.route_id]["p_success"]  # kept out of the record when persisted for training-eligibility
        # Per-route oracle success probabilities, for the route-level
        # negative-flip scorer below. EVALUATOR-ONLY, stripped (along with
        # every other _-prefixed key) before any scenario is persisted or
        # registered, so it can never reach training.
        rec["_oracle_route_p_success"] = {
            int(r): float(d["p_success"]) for r, d in oracle_dist.items()
        }
        seed_pool.append(rec)

    def score(record: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """ROUTE-LEVEL negative-flip scorer.

        Grounded in the model-update-regression literature (Yan et al.,
        "Positive-Congruent Training: Towards Regression-Free Model
        Updates", CVPR 2021), which defines the Negative Flip Rate -- the
        fraction of cases the OLD model got right that the NEW model gets
        wrong -- and shows NFR stays positive even when average error
        improves. That is exactly the failure aggregate gates are blind to.

        WHY ROUTE-LEVEL AND NOT BINARY SUCCESS: two earlier scorers were
        tried and both were measuring the wrong thing.
          1. Continuous |challenger_err| - |champion_err| vs the oracle
             probability, thresholded at 0.4. Too strict for a small
             adapter; reported zero regressions across 14 seeds.
          2. Binary-success negative flip (champion's success/fail call
             correct, challenger's wrong) at threshold 0.5. Measured
             directly and found DEGENERATE in this domain: the simulator's
             oracle p_success exceeded 0.5 for 100% of 300 sampled
             transactions (mean 0.788), and both models predicted >0.5 for
             100% of them. No sample can ever cross the boundary, so no
             binary flip can exist by construction -- and an "accuracy"
             of ~0.78 is just the base rate of a near-constant label.

        The decision a payment routing model actually makes is not
        "will this succeed" but "WHICH ROUTE should carry it". So the
        meaningful regression is a ROUTE flip: the challenger routes a
        transaction to a materially worse route than the champion did,
        measured as regret against the oracle's best available route.
        This is non-degenerate (routes genuinely differ), domain-correct,
        and directly maps onto the NFR definition with argmax-over-routes
        replacing the binary threshold.

        Returns (surrogate, detail); surrogate = challenger_regret -
        champion_regret, positive exactly when the challenger's routing
        choice is worse than the champion's, and smooth enough for the
        refinement loop to hill-climb."""
        oracle_route_p = record.get("_oracle_route_p_success")
        if not oracle_route_p:
            # No per-route oracle available for this record; fall back to
            # reporting no regression rather than guessing.
            return -1.0, {"is_negative_flip": False, "reason": "no_per_route_oracle"}

        route_ids = sorted(oracle_route_p.keys())

        def choose(predict_fn, route_fn) -> int:
            # CERTIFY THE POLICY THAT IS ACTUALLY DEPLOYED. When a
            # route_fn is supplied the red team asks the model the same
            # question production does -- which route does the full
            # multi-objective policy pick, combining success, fraud,
            # latency and abandonment. Falling back to success-only argmax
            # would certify a decision rule the system never ships,
            # leaving "are you testing what you deploy?" unanswerable.
            if route_fn is not None:
                return int(route_fn(record, route_ids))
            best_r, best_p = route_ids[0], -1.0
            for r in route_ids:
                probe = dict(record)
                probe["action_route_id"] = r
                p = _predict_prob(predict_fn, probe)
                if p > best_p:
                    best_p, best_r = p, r
            return best_r

        champ_route = choose(champion_predict_fn, champion_route_fn)
        chall_route = choose(challenger_predict_fn, challenger_route_fn)

        oracle_best_route = max(route_ids, key=lambda r: oracle_route_p[r])
        oracle_best_p = oracle_route_p[oracle_best_route]

        champ_regret = float(oracle_best_p - oracle_route_p[champ_route])
        chall_regret = float(oracle_best_p - oracle_route_p[chall_route])
        surrogate = float(chall_regret - champ_regret)

        is_negative_flip = bool(chall_route != champ_route and chall_regret > champ_regret)

        return surrogate, {
            "oracle_best_route": oracle_best_route,
            "oracle_best_p_success": oracle_best_p,
            "champion_route": champ_route,
            "challenger_route": chall_route,
            "champion_route_p_success": oracle_route_p[champ_route],
            "challenger_route_p_success": oracle_route_p[chall_route],
            "champion_regret": champ_regret,
            "challenger_regret": chall_regret,
            "is_negative_flip": is_negative_flip,
            # backward-compatible fields for existing manifest/UI readers
            "oracle_p_success": oracle_best_p,
            "champion_pred": oracle_route_p[champ_route],
            "challenger_pred": oracle_route_p[chall_route],
            "champion_loss": champ_regret,
            "challenger_loss": chall_regret,
        }

    scored_pool = []
    for rec in seed_pool:
        regression, detail = score(rec)
        scored_pool.append((regression, rec, detail))
    scored_pool.sort(key=lambda t: t[0], reverse=True)

    # ---- local refinement (embedding-neighborhood-style exploration via
    # feature-space perturbation) on the worst candidates ----
    frontier = scored_pool[: cfg.top_k_to_refine]
    for iteration in range(cfg.refinement_iterations):
        new_frontier = []
        for regression, rec, detail in frontier:
            new_frontier.append((regression, rec, detail))  # keep the incumbent
            for frac in rng.choice(cfg.amount_jitter_fractions, size=cfg.perturbations_per_candidate, replace=True):
                candidate = _perturb_amount(rec, float(frac))
                validation = validator.validate({k: v for k, v in candidate.items() if not k.startswith("_")})
                if validation != "VALID":
                    continue
                # re-derive oracle probability for the perturbed amount
                # (amount affects fraud probability + a small success effect
                # in the simulator's structural model -- EVALUATOR-ONLY)
                from vulcan.simulator.environment import ObservableState, ObservableTransaction, NetworkObservation
                # Reconstruct a minimal ObservableState-compatible query by
                # reusing the simulator's own private success-probability
                # function is not exposed publicly; instead re-derive via a
                # fresh oracle query through the existing public API using a
                # lightweight synthetic observation wrapper.
                synthetic_obs = _record_to_observable_state(candidate, sim)
                oracle_dist = sim.oracle_outcome_distribution(synthetic_obs)
                candidate["_oracle_p_success"] = oracle_dist[candidate["action_route_id"]]["p_success"]
                candidate["_oracle_route_p_success"] = {
                    int(r): float(d["p_success"]) for r, d in oracle_dist.items()
                }
                cand_regression, cand_detail = score(candidate)
                if cand_regression > regression:
                    new_frontier.append((cand_regression, candidate, cand_detail))
        new_frontier.sort(key=lambda t: t[0], reverse=True)
        frontier = new_frontier[: cfg.top_k_to_refine]

    # ---- finalize: keep candidates above the severity threshold, dedupe,
    # register as CERTIFICATION (never TRAIN) ----
    discovered = []
    seen_hashes = set()
    for regression, rec, detail in frontier:
        # Report criterion: a genuine NEGATIVE FLIP (champion decided
        # correctly, challenger decided incorrectly) with a minimum
        # combined confidence margin, so we don't report cases where both
        # models are sitting essentially on the 0.5 boundary and the
        # "flip" is numerical noise rather than a real behavioral change.
        if not detail.get("is_negative_flip", False):
            continue
        if regression < cfg.severity_threshold_to_report:
            continue
        clean_record = {k: v for k, v in rec.items() if not k.startswith("_")}
        validation = validator.validate(clean_record)
        if validation != "VALID":
            continue
        scenario_hash = stable_hash(clean_record)
        if scenario_hash in seen_hashes:
            continue
        if not registry.register(scenario_hash, IntendedUse.CERTIFICATION.value):
            continue  # would contaminate an existing TRAIN scenario -- skip
        seen_hashes.add(scenario_hash)
        discovered.append({
            "scenario": clean_record, "scenario_hash": scenario_hash,
            "regression_severity": regression, **detail,
            # Evaluator-only per-route oracle probabilities, carried through
            # so forge_loop can store a durable ROUTE INVARIANT in failure
            # memory (future challengers must be re-tested on the route
            # property that was actually discovered, not on binary success).
            # Underscore-prefixed and deliberately NOT part of `clean_record`,
            # so it can never enter training data.
            "_oracle_route_p_success": rec.get("_oracle_route_p_success"),
        })
        if len(discovered) >= cfg.max_failures_to_return:
            break

    metadata = {
        "seed_pool_size": len(seed_pool),
        "n_refined_candidates_considered": len(frontier),
        "n_failures_discovered": len(discovered),
        "severity_threshold": cfg.severity_threshold_to_report,
        "route_policy": "deployed_multi_objective_utility" if champion_route_fn is not None else "success_only_argmax",
    }
    return discovered, metadata


def _record_to_observable_state(record: Dict[str, Any], sim: IndiaPaymentSim):
    """Reconstruct a minimal ObservableState from a flattened record dict,
    for re-querying the oracle after a feature perturbation. EVALUATOR-ONLY
    helper -- never used in any training path."""
    from vulcan.simulator.environment import ObservableState, ObservableTransaction, NetworkObservation

    txn = ObservableTransaction(
        amount=record["amount"], rail=record["rail"], merchant_category=record["merchant_category"],
        merchant_segment=record["merchant_segment"], issuer=record["issuer"], device_class=record["device_class"],
        geo_bucket=record["geo_bucket"], time_bucket=record["time_bucket"],
        previous_attempts=record["previous_attempts"], retry_count=record["retry_count"],
        customer_tenure_days=record["customer_tenure_days"],
    )
    num_routes = sim.num_routes
    net = NetworkObservation(
        rolling_route_success={r: record.get(f"route_{r}_rolling_success", 0.85) for r in range(num_routes)},
        route_latency_ms={r: record.get(f"route_{r}_latency_ms", 300.0) for r in range(num_routes)},
        recent_timeout_rate={r: record.get(f"route_{r}_timeout_rate", 0.02) for r in range(num_routes)},
        route_load={r: record.get(f"route_{r}_load", 0.5) for r in range(num_routes)},
        recent_issuer_decline_rate=record.get("recent_issuer_decline_rate", 0.05),
        gateway_availability_signal={r: record.get(f"route_{r}_gw_availability", 0.95) for r in range(num_routes)},
    )
    return ObservableState(transaction=txn, network=net, step=record.get("step", 0))
