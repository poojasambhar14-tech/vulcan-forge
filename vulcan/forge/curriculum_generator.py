"""
Curriculum Generator (Forge pipeline stage 3).

Given a diagnosed blind spot, builds a SMALL, targeted training curriculum
(bounded by `curriculum_budget`) instead of blindly sampling recent data.
Combines three sources:

  A. Hard real examples -- existing records nearest to the weak region.
  B. Controlled simulator perturbations -- new, VALID transactions generated
     by rejection-sampling the existing IndiaPaymentSim + BehaviorPolicy
     until they land in the weak region (never by hacking simulator
     internals -- the simulator stays a black box, exactly as everywhere
     else in this project).
  C. Boundary/hard-example search -- from the pooled candidates (A+B),
     ranks by an explicit information-value score favoring cases where the
     champion is confident but wrong, or genuinely uncertain.

Every candidate scenario is passed through `ScenarioValidator` before it can
enter the curriculum -- the generator's own judgment is never trusted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vulcan.forge.schemas import BlindSpot, Curriculum, CurriculumScenario, ScenarioProvenance, IntendedUse
from vulcan.forge.scenario_validator import ScenarioValidator, ScenarioRegistry
from vulcan.simulator.environment import IndiaPaymentSim
from vulcan.simulator.behavior_policy import BehaviorPolicy, BehaviorPolicyConfig
from vulcan.data.generate import _flatten_observation

SIMULATOR_VERSION = "IndiaPaymentSim-v1"  # bump manually if the simulator's structural model changes


@dataclass
class CurriculumConfig:
    curriculum_budget: int = 2000
    hard_real_fraction: float = 0.4
    perturbation_fraction: float = 0.35
    boundary_search_fraction: float = 0.25
    max_perturbation_attempts_multiplier: int = 20  # rejection-sampling budget = budget_slice * this


def _matches_blindspot(record: Dict[str, Any], blindspot: BlindSpot) -> bool:
    for dim, value in blindspot.dimensions.items():
        if dim == "amount_bucket":
            from vulcan.forge.blindspot_miner import _amount_bucket_label
            if _amount_bucket_label(float(record.get("amount", 0.0))) != value:
                return False
        else:
            if str(record.get(dim)) != value:
                return False
    return True


def _information_value(champion_predict_fn: Callable[[Dict[str, Any]], float], record: Dict[str, Any]) -> float:
    """champion_predict_fn(record) -> predicted P(success) for the taken
    action. Information value rewards cases where the champion is
    confident but wrong, or genuinely uncertain (near 0.5) -- both are
    informative training signal; a confident-and-correct case is not."""
    p = champion_predict_fn(record)
    true_success = float(bool(record.get("outcome_success", False)))
    confidently_wrong = abs(p - true_success) if abs(p - 0.5) > 0.3 else 0.0
    uncertainty = 1.0 - 2.0 * abs(p - 0.5)
    return float(0.7 * confidently_wrong + 0.3 * uncertainty)


def generate_curriculum(
    blindspot: BlindSpot,
    cfg_sim: dict,
    tokenizer,
    validator: ScenarioValidator,
    registry: ScenarioRegistry,
    recent_pool_records: List[Dict[str, Any]],
    champion_predict_fn: Callable[[Dict[str, Any]], float],
    seed: int,
    curriculum_cfg: Optional[CurriculumConfig] = None,
) -> Curriculum:
    curriculum_cfg = curriculum_cfg or CurriculumConfig()
    budget = curriculum_cfg.curriculum_budget
    scenarios: List[CurriculumScenario] = []
    rng = np.random.default_rng(seed)

    def make_scenario(record, source, method) -> Optional[CurriculumScenario]:
        validation_result = validator.validate(record)
        provenance = ScenarioProvenance(
            scenario_id=f"S{len(scenarios)}_{blindspot.blindspot_id}_{int(time.time()*1000)%100000}",
            source=source, parent_blindspot=blindspot.blindspot_id, generation_method=method,
            simulator_version=SIMULATOR_VERSION, random_seed=seed,
            validation_result=validation_result, intended_use=IntendedUse.TRAIN.value,
        )
        scenario = CurriculumScenario(provenance=provenance, record=record)
        if validation_result != "VALID":
            return scenario  # kept for reporting rejection rate, but caller must filter
        if not registry.register(scenario.scenario_hash(), IntendedUse.TRAIN.value):
            provenance.validation_result = "REJECTED:train_certification_contamination"
            return scenario
        scenario.information_value = _information_value(champion_predict_fn, record)
        return scenario

    # ---- A. hard real examples ----
    n_hard_real = int(budget * curriculum_cfg.hard_real_fraction)
    matching_real = [r for r in recent_pool_records if _matches_blindspot(r, blindspot)]
    rng.shuffle(matching_real)
    all_attempted: List[CurriculumScenario] = []
    for r in matching_real[: n_hard_real * 2]:  # oversample slightly, filter below
        s = make_scenario(r, source="hard_real_example", method="nearest_match_recent_pool")
        all_attempted.append(s)

    # ---- B. controlled simulator perturbations (rejection sampling) ----
    n_perturb = int(budget * curriculum_cfg.perturbation_fraction)
    sim = IndiaPaymentSim(cfg_sim, seed=seed + 12345)
    policy = BehaviorPolicy(
        BehaviorPolicyConfig(cfg_sim["simulator"]["behavior_policy"]["exploration_rate"]),
        np.random.default_rng(seed + 12346),
    )
    max_attempts = n_perturb * curriculum_cfg.max_perturbation_attempts_multiplier
    n_found = 0
    attempts = 0
    while n_found < n_perturb and attempts < max_attempts:
        attempts += 1
        obs = sim.current_observation()
        actions = sim.candidate_actions()
        action, propensity = policy.select_action(obs, actions)
        outcome = sim.step(obs, action)
        rec = _flatten_observation(obs)
        rec.update(
            action_route_id=action.route_id, action_rail=action.rail, action_gateway=action.gateway,
            propensity=propensity, outcome_success=outcome.success, outcome_latency_ms=outcome.latency_ms,
            outcome_processing_cost=outcome.processing_cost, outcome_fraud_loss=outcome.fraud_loss,
            outcome_abandoned=outcome.abandoned, outcome_error_type=outcome.error_type.value,
        )
        if _matches_blindspot(rec, blindspot):
            s = make_scenario(rec, source="simulator_perturbation", method="rejection_sampling_behavior_policy")
            all_attempted.append(s)
            n_found += 1

    # ---- C. boundary / hard-example search: rank ALL validated candidates
    # by information value, keep the top slice within the remaining budget ----
    valid_candidates = [s for s in all_attempted if s.provenance.validation_result == "VALID"]
    valid_candidates.sort(key=lambda s: s.information_value, reverse=True)

    scenarios = valid_candidates[:budget]
    rejected = [s for s in all_attempted if s.provenance.validation_result != "VALID"]

    curriculum = Curriculum(
        curriculum_id=f"curriculum_{blindspot.blindspot_id}_{int(time.time())}",
        parent_blindspot=blindspot.blindspot_id, budget=budget, scenarios=scenarios,
    )
    metadata = {
        "n_attempted": len(all_attempted),
        "n_rejected": len(rejected),
        "rejected_examples": [
            {"scenario_id": s.provenance.scenario_id, "reason": s.provenance.validation_result}
            for s in rejected[:20]
        ],
        "validity_rate": len(valid_candidates) / len(all_attempted) if all_attempted else 0.0,
        "source_breakdown": {
            src: sum(1 for s in scenarios if s.provenance.source == src)
            for src in ("hard_real_example", "simulator_perturbation")
        },
    }
    return curriculum, metadata
