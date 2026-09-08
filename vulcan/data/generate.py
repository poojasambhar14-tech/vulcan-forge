"""
Training data generation for IndiaPaymentSim.

CRITICAL SEPARATION RULE (master spec section 4):
`generate_training_data()` may only ever call `sim.step(obs, action)`, which
returns the REALIZED outcome for the action actually taken. It must never
call `sim.oracle_outcome_distribution(...)`, which knows the true structural
probabilities for actions *not* taken.

Oracle/counterfactual information lives in a physically separate module
(`vulcan/evaluation/oracle_reference.py`) and separate output file, so it is
impossible to accidentally join it back onto the training set without an
explicit, clearly-named step.
"""
from __future__ import annotations

import inspect
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np

from vulcan.simulator.environment import IndiaPaymentSim, ObservableState
from vulcan.simulator.behavior_policy import BehaviorPolicy, BehaviorPolicyConfig


def _flatten_observation(obs: ObservableState) -> Dict[str, Any]:
    txn = obs.transaction
    net = obs.network
    flat = {
        "step": obs.step,
        "amount": txn.amount,
        "rail": txn.rail,
        "merchant_category": txn.merchant_category,
        "merchant_segment": txn.merchant_segment,
        "issuer": txn.issuer,
        "device_class": txn.device_class,
        "geo_bucket": txn.geo_bucket,
        "time_bucket": txn.time_bucket,
        "previous_attempts": txn.previous_attempts,
        "retry_count": txn.retry_count,
        "customer_tenure_days": txn.customer_tenure_days,
    }
    for r, v in net.rolling_route_success.items():
        flat[f"route_{r}_rolling_success"] = v
    for r, v in net.route_latency_ms.items():
        flat[f"route_{r}_latency_ms"] = v
    for r, v in net.recent_timeout_rate.items():
        flat[f"route_{r}_timeout_rate"] = v
    for r, v in net.route_load.items():
        flat[f"route_{r}_load"] = v
    for r, v in net.gateway_availability_signal.items():
        flat[f"route_{r}_gw_availability"] = v
    flat["recent_issuer_decline_rate"] = net.recent_issuer_decline_rate
    return flat


def _assert_no_oracle_access(sim: IndiaPaymentSim) -> None:
    """Best-effort runtime guard: inspect the call stack to make sure
    training data generation is never invoked from a context that has
    called the oracle function. This is a defense-in-depth check; the
    primary guarantee is architectural (separate module/function)."""
    stack_funcs = {frame.function for frame in inspect.stack()}
    assert "oracle_outcome_distribution" not in stack_funcs, (
        "Leakage violation: generate_training_data() must never be called "
        "from within an oracle_outcome_distribution() call stack."
    )


def generate_training_data(cfg: dict, seed: int, n_transactions: int) -> List[Dict[str, Any]]:
    """Generate (state, action_taken, observed_outcome, propensity) tuples
    ONLY. No counterfactual/oracle information is included or accessible."""
    sim = IndiaPaymentSim(cfg, seed=seed)
    _assert_no_oracle_access(sim)

    policy_cfg = BehaviorPolicyConfig(
        exploration_rate=cfg["simulator"]["behavior_policy"]["exploration_rate"]
    )
    policy_rng = np.random.default_rng(seed + 777)
    policy = BehaviorPolicy(policy_cfg, policy_rng)

    records: List[Dict[str, Any]] = []
    for _ in range(n_transactions):
        obs = sim.current_observation()
        actions = sim.candidate_actions()
        chosen_action, propensity = policy.select_action(obs, actions)

        outcome = sim.step(obs, chosen_action)

        record = _flatten_observation(obs)
        record.update(
            {
                "action_route_id": chosen_action.route_id,
                "action_rail": chosen_action.rail,
                "action_gateway": chosen_action.gateway,
                "propensity": propensity,
                "outcome_success": outcome.success,
                "outcome_latency_ms": outcome.latency_ms,
                "outcome_processing_cost": outcome.processing_cost,
                "outcome_fraud_loss": outcome.fraud_loss,
                "outcome_abandoned": outcome.abandoned,
                "outcome_error_type": outcome.error_type.value,
            }
        )
        records.append(record)

    return records


TRAINING_VISIBLE_COLUMNS_FORBIDDEN_PREFIXES = ("oracle_", "hidden_", "regime_name")


def assert_no_leaked_columns(records: List[Dict[str, Any]]) -> None:
    """Sanity check used by tests: no record may contain any oracle/hidden
    fields, regardless of how the records were produced."""
    for rec in records:
        for key in rec.keys():
            assert not key.startswith(TRAINING_VISIBLE_COLUMNS_FORBIDDEN_PREFIXES), (
                f"Leakage violation: training record contains forbidden field '{key}'"
            )
