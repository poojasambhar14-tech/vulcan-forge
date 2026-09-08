"""
Failure Diagnoser (Forge pipeline stage 1).

DriftSafe (vulcan/drift/driftsafe.py) answers "did something change?" via
multi-signal consensus. This module answers the next question: "why did
performance degrade, and what kind of healing is appropriate?"

HARD CONSTRAINT: this module NEVER reads the simulator's hidden regime/shift
label. Every classification is a deterministic function of MEASURED,
OBSERVABLE signals (PSI, embedding-distance drift, calibration delta,
residual drift, per-segment error breakdowns, label prevalence, and a
schema/range-anomaly check) -- reusing vulcan/drift/detectors.py's existing,
already-tested primitives rather than reimplementing them. Hidden simulator
labels may be used AFTERWARD, by a separate caller, purely to score this
diagnosis's accuracy (see scripts/forge_benchmark.py) -- never inside this
module's decision procedure.

This is a deterministic RULE-BASED classifier over real statistics, not a
learned model -- consistent with DriftSafe's own design philosophy
elsewhere in this project. It is reproducible from observable metrics by
construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from vulcan.drift.detectors import (
    input_drift_psi, representation_drift_mmd, calibration_drift,
    residual_drift_page_hinkley, expected_calibration_error, population_stability_index,
)
from vulcan.forge.schemas import FailureDiagnosis, FailureType, HealingFamily


@dataclass
class DiagnoserConfig:
    schema_anomaly_fraction_threshold: float = 0.05   # fraction of records with out-of-range/NaN critical fields
    global_shift_segment_fraction: float = 0.6        # if this fraction of segments show elevated error -> GLOBAL
    local_min_segment_error_delta: float = 0.08       # min error increase to call a segment "affected"
    label_shift_threshold: float = 0.05               # change in mean success label
    calibration_drift_threshold: float = 0.03
    transient_persistence_windows: int = 2            # if signal not sustained this many windows -> TRANSIENT_SPIKE
    ood_embedding_distance_z_threshold: float = 3.0    # z-scored mean embedding distance from reference


HEALING_FAMILY_BY_FAILURE_TYPE = {
    FailureType.LOCAL_BLIND_SPOT: HealingFamily.TARGETED_ADAPTER,
    FailureType.LOCAL_COVARIATE_SHIFT: HealingFamily.TARGETED_ADAPTER,
    FailureType.LABEL_SHIFT: HealingFamily.RECALIBRATION,
    FailureType.CALIBRATION_DRIFT: HealingFamily.RECALIBRATION,
    FailureType.GLOBAL_COVARIATE_SHIFT: HealingFamily.BROADER_CHALLENGER,
    FailureType.OOD_REGION: HealingFamily.ABSTAIN_AND_COLLECT,
    FailureType.TRANSIENT_SPIKE: HealingFamily.NO_ACTION_CONTINUE_MONITORING,
    FailureType.SCHEMA_OR_PIPELINE_FAILURE: HealingFamily.BLOCK_AND_SURFACE_INFRA_FAILURE,
    FailureType.CATASTROPHIC_FORGETTING: HealingFamily.REJECT_NO_ADAPTATION,
    FailureType.UNKNOWN: HealingFamily.NO_AUTOMATIC_PROMOTION,
}


def _schema_anomaly_fraction(records: List[Dict[str, Any]], amount_valid_range=(0.0, 1_000_000.0)) -> float:
    """Fraction of records with a critical field missing, NaN, or outside a
    generously wide valid range -- a real, measurable proxy for pipeline
    breakage, not a simulated label."""
    if not records:
        return 0.0
    n_bad = 0
    for r in records:
        amount = r.get("amount")
        if amount is None or not np.isfinite(amount) or not (amount_valid_range[0] <= amount <= amount_valid_range[1]):
            n_bad += 1
            continue
        if r.get("rail") is None or r.get("issuer") is None:
            n_bad += 1
    return n_bad / len(records)


def _segment_error_rates(records: List[Dict[str, Any]], dims: List[str], min_sample_size: int = 15) -> Dict[str, Dict[str, float]]:
    """Per-segment error rate (fraction with outcome_success == False) for
    each value of each candidate slicing dimension, subject to a minimum
    sample size (so a segment with 3 records doesn't masquerade as a
    finding)."""
    out: Dict[str, Dict[str, float]] = {}
    for dim in dims:
        buckets: Dict[Any, List[bool]] = {}
        for r in records:
            key = r.get(dim)
            buckets.setdefault(key, []).append(not bool(r.get("outcome_success", True)))
        out[dim] = {
            str(k): float(np.mean(v)) for k, v in buckets.items() if len(v) >= min_sample_size
        }
    return out


def diagnose(
    reference_window: Dict[str, np.ndarray],
    current_window: Dict[str, np.ndarray],
    reference_records: List[Dict[str, Any]],
    current_records: List[Dict[str, Any]],
    recent_window_history_consensus: Optional[List[bool]] = None,
    cfg: Optional[DiagnoserConfig] = None,
    slice_dims: Optional[List[str]] = None,
) -> FailureDiagnosis:
    """
    reference_window / current_window: dicts with 'features' (np.ndarray),
        'z' (np.ndarray [N,d]), 'probs' (np.ndarray), 'labels' (np.ndarray) --
        same shape DriftSafe already uses.
    reference_records / current_records: the raw flattened transaction
        records (for segment-level and schema-anomaly analysis).
    recent_window_history_consensus: optional list of booleans, one per
        recent monitoring window, True if that window's DriftSafe consensus
        was triggered -- used to distinguish TRANSIENT_SPIKE from a
        persistent shift. If not provided, persistence cannot be assessed
        and TRANSIENT_SPIKE is not considered.
    """
    cfg = cfg or DiagnoserConfig()
    slice_dims = slice_dims or ["issuer", "rail", "merchant_category", "device_class"]
    evidence: Dict[str, Any] = {}

    # ---- 0. schema/pipeline anomaly check (highest priority: a broken
    # pipeline can masquerade as any other kind of "shift") ----
    schema_frac = _schema_anomaly_fraction(current_records)
    evidence["schema_anomaly_fraction"] = schema_frac
    if schema_frac > cfg.schema_anomaly_fraction_threshold:
        return FailureDiagnosis(
            failure_type=FailureType.SCHEMA_OR_PIPELINE_FAILURE.value,
            severity=min(1.0, schema_frac / (cfg.schema_anomaly_fraction_threshold * 4)),
            evidence=evidence,
            affected_region={"scope": "pipeline"},
            recommended_healing_family=HEALING_FAMILY_BY_FAILURE_TYPE[FailureType.SCHEMA_OR_PIPELINE_FAILURE].value,
        )

    # ---- 1. global signals (reuse DriftSafe's own tested primitives) ----
    psi = input_drift_psi(reference_window["features"], current_window["features"]).value
    mmd = representation_drift_mmd(reference_window["z"], current_window["z"]).value
    calib = calibration_drift(reference_window["probs"], reference_window["labels"],
                               current_window["probs"], current_window["labels"]).value
    residuals = list(np.abs(current_window["probs"] - current_window["labels"]))
    residual_signal = residual_drift_page_hinkley(residuals).value

    evidence.update({"psi": psi, "mmd_proxy": mmd, "calibration_ece_delta": calib, "residual_page_hinkley": residual_signal})

    label_shift = float(current_window["labels"].mean() - reference_window["labels"].mean())
    evidence["label_prevalence_delta"] = label_shift

    # ---- 2. transient-spike check (needs external persistence history) ----
    if recent_window_history_consensus is not None and len(recent_window_history_consensus) >= 1:
        n_triggered = sum(recent_window_history_consensus)
        evidence["recent_windows_triggered"] = n_triggered
        evidence["recent_windows_total"] = len(recent_window_history_consensus)
        if n_triggered < cfg.transient_persistence_windows and not recent_window_history_consensus[-1]:
            return FailureDiagnosis(
                failure_type=FailureType.TRANSIENT_SPIKE.value,
                severity=0.2,
                evidence=evidence,
                affected_region={"scope": "temporary"},
                recommended_healing_family=HEALING_FAMILY_BY_FAILURE_TYPE[FailureType.TRANSIENT_SPIKE].value,
            )

    # ---- 3. segment-level error breakdown ----
    ref_segment_errors = _segment_error_rates(reference_records, slice_dims)
    cur_segment_errors = _segment_error_rates(current_records, slice_dims)

    affected_segments = []
    for dim in slice_dims:
        for value, cur_err in cur_segment_errors.get(dim, {}).items():
            ref_err = ref_segment_errors.get(dim, {}).get(value)
            if ref_err is None:
                continue
            delta = cur_err - ref_err
            if delta >= cfg.local_min_segment_error_delta:
                affected_segments.append({"dim": dim, "value": value, "ref_error": ref_err, "cur_error": cur_err, "delta": delta})

    total_segments_checked = sum(len(v) for v in cur_segment_errors.values())
    fraction_affected = len(affected_segments) / total_segments_checked if total_segments_checked else 0.0
    evidence["fraction_segments_affected"] = fraction_affected
    evidence["n_segments_affected"] = len(affected_segments)
    evidence["n_segments_checked"] = total_segments_checked

    # ---- 4. OOD check: is the current population embedding far outside the
    # reference embedding's own natural variation (not just shifted mean,
    # but genuinely novel/low-density)? ----
    ref_z_std_norm = float(np.linalg.norm(reference_window["z"].std(axis=0))) + 1e-6
    ref_z_mean = reference_window["z"].mean(axis=0)
    cur_dists = np.linalg.norm(current_window["z"] - ref_z_mean, axis=1)
    ref_dists = np.linalg.norm(reference_window["z"] - ref_z_mean, axis=1)
    ood_z_score = float((cur_dists.mean() - ref_dists.mean()) / (ref_dists.std() + 1e-6))
    evidence["ood_embedding_distance_z_score"] = ood_z_score

    # ---- decision procedure (deterministic, ordered) ----
    if abs(label_shift) >= cfg.label_shift_threshold and fraction_affected < 0.3:
        failure_type = FailureType.LABEL_SHIFT
        severity = min(1.0, abs(label_shift) / (cfg.label_shift_threshold * 4))
        region = {"scope": "global", "label_prevalence_delta": label_shift}
    elif ood_z_score >= cfg.ood_embedding_distance_z_threshold:
        failure_type = FailureType.OOD_REGION
        severity = min(1.0, ood_z_score / (cfg.ood_embedding_distance_z_threshold * 2))
        region = {"scope": "novel_region", "ood_z_score": ood_z_score}
    elif calib >= cfg.calibration_drift_threshold and fraction_affected < 0.3 and abs(label_shift) < cfg.label_shift_threshold:
        failure_type = FailureType.CALIBRATION_DRIFT
        severity = min(1.0, calib / (cfg.calibration_drift_threshold * 4))
        region = {"scope": "global", "calibration_ece_delta": calib}
    elif fraction_affected >= cfg.global_shift_segment_fraction:
        failure_type = FailureType.GLOBAL_COVARIATE_SHIFT
        severity = min(1.0, fraction_affected)
        region = {"scope": "global", "affected_segments": affected_segments}
    elif affected_segments:
        # distinguish LOCAL_COVARIATE_SHIFT (that segment's OWN input
        # distribution also moved) from LOCAL_BLIND_SPOT (persistent
        # weakness, input distribution for that segment is unchanged)
        worst = max(affected_segments, key=lambda s: s["delta"])
        segment_records_cur = [r for r in current_records if str(r.get(worst["dim"])) == worst["value"]]
        segment_records_ref = [r for r in reference_records if str(r.get(worst["dim"])) == worst["value"]]
        if len(segment_records_cur) >= 15 and len(segment_records_ref) >= 15:
            seg_psi = population_stability_index(
                np.array([r["amount"] for r in segment_records_ref]),
                np.array([r["amount"] for r in segment_records_cur]),
            )
        else:
            seg_psi = 0.0
        evidence["worst_segment_psi"] = seg_psi
        if seg_psi > 0.15:
            failure_type = FailureType.LOCAL_COVARIATE_SHIFT
        else:
            failure_type = FailureType.LOCAL_BLIND_SPOT
        severity = min(1.0, worst["delta"] / (cfg.local_min_segment_error_delta * 3))
        region = {"scope": "segment", "dim": worst["dim"], "value": worst["value"], "delta": worst["delta"]}
    else:
        failure_type = FailureType.UNKNOWN
        severity = 0.3
        region = {"scope": "unknown"}

    return FailureDiagnosis(
        failure_type=failure_type.value,
        severity=float(severity),
        evidence=evidence,
        affected_region=region,
        recommended_healing_family=HEALING_FAMILY_BY_FAILURE_TYPE[failure_type].value,
    )
