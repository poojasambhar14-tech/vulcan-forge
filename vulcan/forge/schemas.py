"""
Shared schemas for the Vulcan Forge pipeline (vulcan/forge/).

    Champion -> monitor -> degradation/drift -> FAILURE DIAGNOSER
    -> BLIND-SPOT MINER -> CURRICULUM GENERATOR -> SCENARIO VALIDATOR
    -> HEALING POLICY -> train challenger -> historical+recent evaluation
    -> RED-TEAM EXAMINER -> failure-memory regression suite
    -> promotion gate -> PROMOTE / REJECT -> rollback if later degradation

Every dataclass here is a plain, JSON-serializable structure so it can be
persisted verbatim into artifacts/runs/*.json manifests, following the
project's existing reproducibility convention (see vulcan/common/config.py).
"""
from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


class FailureType(str, enum.Enum):
    LOCAL_COVARIATE_SHIFT = "LOCAL_COVARIATE_SHIFT"
    GLOBAL_COVARIATE_SHIFT = "GLOBAL_COVARIATE_SHIFT"
    LABEL_SHIFT = "LABEL_SHIFT"
    CALIBRATION_DRIFT = "CALIBRATION_DRIFT"
    LOCAL_BLIND_SPOT = "LOCAL_BLIND_SPOT"
    OOD_REGION = "OOD_REGION"
    TRANSIENT_SPIKE = "TRANSIENT_SPIKE"
    SCHEMA_OR_PIPELINE_FAILURE = "SCHEMA_OR_PIPELINE_FAILURE"
    CATASTROPHIC_FORGETTING = "CATASTROPHIC_FORGETTING"
    UNKNOWN = "UNKNOWN"


class HealingFamily(str, enum.Enum):
    TARGETED_ADAPTER = "TARGETED_ADAPTER"
    RECALIBRATION = "RECALIBRATION"
    BROADER_CHALLENGER = "BROADER_CHALLENGER"
    ABSTAIN_AND_COLLECT = "ABSTAIN_AND_COLLECT"
    NO_ACTION_CONTINUE_MONITORING = "NO_ACTION_CONTINUE_MONITORING"
    BLOCK_AND_SURFACE_INFRA_FAILURE = "BLOCK_AND_SURFACE_INFRA_FAILURE"
    REJECT_NO_ADAPTATION = "REJECT_NO_ADAPTATION"
    NO_AUTOMATIC_PROMOTION = "NO_AUTOMATIC_PROMOTION"


class IntendedUse(str, enum.Enum):
    TRAIN = "TRAIN"
    CERTIFICATION = "CERTIFICATION"


def stable_hash(obj: Any) -> str:
    """Deterministic hash of any JSON-serializable object."""
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class FailureDiagnosis:
    failure_type: str
    severity: float
    evidence: Dict[str, Any]
    affected_region: Dict[str, Any]
    recommended_healing_family: str
    diagnosis_hash: str = ""

    def __post_init__(self):
        if not self.diagnosis_hash:
            self.diagnosis_hash = stable_hash({
                "failure_type": self.failure_type, "severity": round(self.severity, 4),
                "evidence": self.evidence, "affected_region": self.affected_region,
            })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BlindSpot:
    blindspot_id: str
    dimensions: Dict[str, Any]
    sample_count: int
    model_error: float
    model_error_ci95: float
    uncertainty: float
    financial_exposure: float
    novelty: float
    priority_score: float
    component_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioProvenance:
    scenario_id: str
    source: str                 # "hard_real_example" | "simulator_perturbation" | "boundary_search"
    parent_blindspot: str
    generation_method: str
    simulator_version: str
    random_seed: int
    validation_result: str      # "VALID" | "REJECTED:<reason>"
    intended_use: str           # TRAIN | CERTIFICATION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurriculumScenario:
    provenance: ScenarioProvenance
    record: Dict[str, Any]
    information_value: float = 0.0

    def scenario_hash(self) -> str:
        return stable_hash(self.record)

    def to_dict(self) -> Dict[str, Any]:
        return {"provenance": self.provenance.to_dict(), "record": self.record,
                "information_value": self.information_value}


@dataclass
class Curriculum:
    curriculum_id: str
    parent_blindspot: str
    budget: int
    scenarios: List[CurriculumScenario]
    curriculum_hash: str = ""

    def __post_init__(self):
        if not self.curriculum_hash:
            self.curriculum_hash = stable_hash([s.scenario_hash() for s in self.scenarios])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curriculum_id": self.curriculum_id, "parent_blindspot": self.parent_blindspot,
            "budget": self.budget, "curriculum_hash": self.curriculum_hash,
            "curriculum_size": len(self.scenarios),
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


@dataclass
class HealingDecision:
    diagnosis: FailureDiagnosis
    selected_strategy: str
    evidence: Dict[str, Any]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"diagnosis": self.diagnosis.to_dict(), "selected_strategy": self.selected_strategy,
                "evidence": self.evidence, "reason": self.reason}


@dataclass
class FailureMemoryEntry:
    failure_id: str
    discovered_against: str
    scenario: Dict[str, Any]
    failure_metric: str
    champion_result: float
    challenger_result: float
    severity: float
    first_seen_run: str
    scenario_hash: str = ""
    # ROUTE INVARIANT. The red team discovers ROUTE-level negative flips:
    # the challenger sends a payment to a materially worse route than the
    # champion would. This field stores the invariant that must hold for
    # every future challenger, so "Remember" tests exactly the property
    # "Attack" discovered:
    #   oracle_route_p_success -- evaluator-only per-route true success probs
    #   champion_route         -- what the certified champion chose
    #   bad_challenger_route   -- what the rejected challenger chose
    #   champion_regret        -- champion's regret vs the oracle-best route
    #   max_allowed_regret     -- ceiling any future challenger must stay under
    # Kept OUT of `scenario` so it can never reach training data; `scenario`
    # remains a clean, trainable record.
    route_invariant: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.scenario_hash:
            self.scenario_hash = stable_hash(self.scenario)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionCriterion:
    name: str
    passed: bool
    value: Any
    threshold: Any
    detail: str = ""


@dataclass
class PromotionResult:
    decision: str  # PROMOTE | REJECT
    criteria: List[PromotionCriterion]

    def to_dict(self) -> Dict[str, Any]:
        return {"decision": self.decision, "criteria": [asdict(c) for c in self.criteria]}
