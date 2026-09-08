"""
Blind-Spot Miner (Forge pipeline stage 2).

Slices recent evaluation data across candidate payment-state dimensions and
scores each slice's weakness by a business-aware priority formula:

    priority = error_severity x financial_exposure x uncertainty x novelty

(normalized to [0,1] per component before multiplying), NOT by raw error
rate alone. Enforces a minimum sample size per slice so noise is not
mistaken for a blind spot, and reports a bootstrap 95% CI on each slice's
error rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vulcan.forge.schemas import BlindSpot, stable_hash

DEFAULT_SLICE_DIMS = ["issuer", "rail", "merchant_category", "device_class"]
AMOUNT_BUCKETS = [(0, 1_000), (1_000, 5_000), (5_000, 15_000), (15_000, 30_000), (30_000, 100_000), (100_000, float("inf"))]


@dataclass
class BlindSpotMinerConfig:
    min_sample_size: int = 30
    n_bootstrap: int = 200
    top_k: int = 10
    weight_error: float = 1.0
    weight_exposure: float = 1.0
    weight_uncertainty: float = 1.0
    weight_novelty: float = 1.0


def _amount_bucket_label(amount: float) -> str:
    for lo, hi in AMOUNT_BUCKETS:
        if lo <= amount < hi:
            hi_label = "inf" if hi == float("inf") else str(int(hi))
            return f"{int(lo)}-{hi_label}"
    return "unknown"


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int = 0) -> float:
    """Returns the 95% CI half-width on the mean via bootstrap resampling."""
    if len(values) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float((hi - lo) / 2)


def _normalize(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=np.float64)
    if arr.max() - arr.min() < 1e-9:
        return [0.5 for _ in values]
    return list((arr - arr.min()) / (arr.max() - arr.min()))


def mine_blind_spots(
    records: List[Dict[str, Any]],
    predicted_probs: np.ndarray,
    historical_slice_prevalence: Optional[Dict[str, Dict[str, float]]] = None,
    slice_dims: Optional[List[str]] = None,
    cfg: Optional[BlindSpotMinerConfig] = None,
) -> Tuple[List[BlindSpot], Dict[str, Any]]:
    """
    records: recent flattened transaction records (with real outcome_success,
        amount, etc. -- evaluation data, not training data).
    predicted_probs: champion's predicted P(success) for each record's taken
        route, aligned index-for-index with `records`.
    historical_slice_prevalence: optional {dim: {value: fraction_of_all_historical_records}}
        used to compute "novelty" (how much more common this slice is NOW
        than historically). If not provided, novelty defaults to a neutral
        0.5 for every slice.

    Returns (ranked list of BlindSpot, formula/config metadata for
    reporting).
    """
    cfg = cfg or BlindSpotMinerConfig()
    slice_dims = slice_dims or DEFAULT_SLICE_DIMS
    assert len(records) == len(predicted_probs)

    # build slice key -> list of record indices
    slice_members: Dict[Tuple[str, str], List[int]] = {}
    for i, r in enumerate(records):
        for dim in slice_dims:
            value = str(r.get(dim))
            slice_members.setdefault((dim, value), []).append(i)
        amount_bucket = _amount_bucket_label(float(r.get("amount", 0.0)))
        slice_members.setdefault(("amount_bucket", amount_bucket), []).append(i)

    raw_candidates = []
    for (dim, value), idxs in slice_members.items():
        if len(idxs) < cfg.min_sample_size:
            continue
        sub_records = [records[i] for i in idxs]
        sub_probs = predicted_probs[idxs]
        labels = np.array([float(bool(r.get("outcome_success", False))) for r in sub_records])
        preds = (sub_probs > 0.5).astype(float)
        error_rate = float((preds != labels).mean())
        error_ci = _bootstrap_ci(np.abs(preds - labels), cfg.n_bootstrap)
        uncertainty = float(np.mean(1.0 - 2.0 * np.abs(sub_probs - 0.5)))  # near 0.5 -> high uncertainty
        financial_exposure = float(np.sum([r.get("amount", 0.0) for r in sub_records]) * error_rate)

        novelty = 0.5
        if historical_slice_prevalence is not None:
            hist_prev = historical_slice_prevalence.get(dim, {}).get(value)
            cur_prev = len(idxs) / len(records)
            if hist_prev is not None and hist_prev > 1e-6:
                novelty = float(np.clip((cur_prev - hist_prev) / hist_prev, 0.0, 3.0) / 3.0)
            elif hist_prev is None:
                novelty = 1.0  # never seen historically -> maximally novel

        raw_candidates.append({
            "dim": dim, "value": value, "sample_count": len(idxs),
            "error_rate": error_rate, "error_ci": error_ci,
            "uncertainty": uncertainty, "financial_exposure": financial_exposure,
            "novelty": novelty,
        })

    if not raw_candidates:
        return [], {"formula": "priority = error_severity * financial_exposure * uncertainty * novelty",
                     "weights": cfg.__dict__, "n_candidates": 0}

    norm_error = _normalize([c["error_rate"] for c in raw_candidates])
    norm_exposure = _normalize([c["financial_exposure"] for c in raw_candidates])
    norm_uncertainty = _normalize([c["uncertainty"] for c in raw_candidates])
    norm_novelty = _normalize([c["novelty"] for c in raw_candidates])

    blind_spots = []
    for i, c in enumerate(raw_candidates):
        component_scores = {
            "error_severity": norm_error[i], "financial_exposure": norm_exposure[i],
            "uncertainty": norm_uncertainty[i], "novelty": norm_novelty[i],
        }
        priority = (
            (norm_error[i] ** cfg.weight_error)
            * (norm_exposure[i] ** cfg.weight_exposure)
            * (norm_uncertainty[i] ** cfg.weight_uncertainty)
            * (norm_novelty[i] ** cfg.weight_novelty)
        )
        dimensions = {c["dim"]: c["value"]}
        blindspot_id = f"B{stable_hash(dimensions)[:6]}"
        blind_spots.append(BlindSpot(
            blindspot_id=blindspot_id, dimensions=dimensions, sample_count=c["sample_count"],
            model_error=c["error_rate"], model_error_ci95=c["error_ci"], uncertainty=c["uncertainty"],
            financial_exposure=c["financial_exposure"], novelty=c["novelty"],
            priority_score=float(priority), component_scores=component_scores,
        ))

    blind_spots.sort(key=lambda b: b.priority_score, reverse=True)
    metadata = {
        "formula": "priority = error_severity^w1 * financial_exposure^w2 * uncertainty^w3 * novelty^w4 (each component min-max normalized across candidates)",
        "weights": {"w1_error": cfg.weight_error, "w2_exposure": cfg.weight_exposure,
                    "w3_uncertainty": cfg.weight_uncertainty, "w4_novelty": cfg.weight_novelty},
        "min_sample_size": cfg.min_sample_size,
        "n_candidates": len(raw_candidates),
    }
    return blind_spots[: cfg.top_k], metadata
