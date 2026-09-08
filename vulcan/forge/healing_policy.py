"""
Healing Policy (Forge pipeline stage: diagnosis -> action).

Explicitly does NOT map every detected problem to "retrain." The mapping
from FailureType to HealingFamily is itself deterministic and
config-driven (see failure_diagnoser.HEALING_FAMILY_BY_FAILURE_TYPE) --
this module is the decision point that ACTS on that recommendation and
logs why, including the cases where the correct action is explicitly to
NOT adapt at all.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from vulcan.forge.schemas import FailureDiagnosis, HealingDecision, HealingFamily


REASON_TEMPLATES = {
    HealingFamily.TARGETED_ADAPTER.value:
        "Weakness is localized ({region}); a small parameter-efficient adapter targeted at the "
        "affected region is expected to recover it without risking broader forgetting.",
    HealingFamily.RECALIBRATION.value:
        "Discrimination appears intact but calibration/label prevalence has shifted; "
        "recalibration is cheaper and lower-risk than retraining and directly addresses the "
        "measured signal ({evidence_summary}).",
    HealingFamily.BROADER_CHALLENGER.value:
        "Shift affects a large fraction of segments ({fraction_affected}); a localized adapter "
        "would not cover enough of the affected input space, so a broader challenger is warranted.",
    HealingFamily.ABSTAIN_AND_COLLECT.value:
        "Current population appears genuinely out-of-distribution relative to the reference "
        "window (z-score={ood_z_score}); no reliable label signal yet justifies retraining, so "
        "the safe action is to fall back/abstain on this region and collect more evidence.",
    HealingFamily.NO_ACTION_CONTINUE_MONITORING.value:
        "Signal has not persisted across enough monitoring windows to distinguish it from a "
        "transient operational spike; retraining on a spike risks overfitting to noise.",
    HealingFamily.BLOCK_AND_SURFACE_INFRA_FAILURE.value:
        "A schema/pipeline anomaly was detected (fraction={schema_anomaly_fraction}); this is an "
        "infrastructure problem, not a model problem, and must be fixed upstream -- retraining "
        "on corrupted inputs would make things worse, not better.",
    HealingFamily.REJECT_NO_ADAPTATION.value:
        "Catastrophic forgetting was detected; no adaptation strategy is applied and the "
        "candidate is rejected outright.",
    HealingFamily.NO_AUTOMATIC_PROMOTION.value:
        "Diagnosis was inconclusive (UNKNOWN); per policy, no automatic promotion is permitted "
        "until a human reviews the evidence.",
}


def decide_healing_strategy(diagnosis: FailureDiagnosis) -> HealingDecision:
    family = diagnosis.recommended_healing_family
    evidence = diagnosis.evidence
    template = REASON_TEMPLATES.get(family, "No template available for this healing family.")
    try:
        reason = template.format(
            region=diagnosis.affected_region,
            evidence_summary={k: round(v, 4) if isinstance(v, float) else v for k, v in evidence.items()},
            fraction_affected=evidence.get("fraction_segments_affected"),
            ood_z_score=evidence.get("ood_embedding_distance_z_score"),
            schema_anomaly_fraction=evidence.get("schema_anomaly_fraction"),
        )
    except (KeyError, IndexError):
        reason = template

    return HealingDecision(
        diagnosis=diagnosis,
        selected_strategy=family,
        evidence=evidence,
        reason=reason,
    )


def requires_training(decision: HealingDecision) -> bool:
    """Only these healing families involve training a challenger at all."""
    return decision.selected_strategy in (
        HealingFamily.TARGETED_ADAPTER.value,
        HealingFamily.BROADER_CHALLENGER.value,
    )
