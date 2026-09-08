"""
Recalibration healing action (LABEL_SHIFT / CALIBRATION_DRIFT -> RECALIBRATION).

Single-parameter temperature scaling (Guo et al., 2017), fit by minimising
NLL and applied on top of the champion's existing raw probabilities. No
backbone or head weights change, which makes this a deliberately cheaper and
lower-risk intervention than training a challenger -- matching what the
healing policy's reasoning specifies for this failure family.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from vulcan.drift.detectors import expected_calibration_error, brier_score


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(probs: np.ndarray, labels: np.ndarray, t_grid: np.ndarray = None) -> float:
    """Fits a single temperature T minimizing NLL via grid search (avoids
    pulling in a full optimizer dependency for a 1-D convex-ish problem;
    verified against a finer grid to confirm the coarse grid doesn't miss
    the minimum by a meaningful margin -- see
    tests/forge/test_recalibration.py)."""
    if t_grid is None:
        t_grid = np.concatenate([np.linspace(0.2, 3.0, 57), np.linspace(3.2, 8.0, 25)])
    logits = _logit(probs)
    best_t, best_nll = 1.0, float("inf")
    eps = 1e-7
    for t in t_grid:
        calibrated = _sigmoid(logits / t)
        nll = -np.mean(labels * np.log(calibrated + eps) + (1 - labels) * np.log(1 - calibrated + eps))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


@dataclass
class RecalibrationResult:
    temperature: float
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    accuracy_before: float
    accuracy_after: float
    n_calibration_examples: int
    n_fit_examples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def recalibrate(champion_predict_fn: Callable[[Dict[str, Any]], float],
                 records: List[Dict[str, Any]],
                 fit_fraction: float = 0.65) -> RecalibrationResult:
    """Fits temperature scaling on a FIT split and reports its effect on a
    disjoint, later CERTIFY split.

    Previously the temperature was fit on all `records` and then scored on
    those same records, so the reported ECE/Brier improvement was measured
    on the data the parameter had just been tuned to -- optimistic by
    construction, and not evidence the recalibration generalizes. The split
    is CHRONOLOGICAL (not random) to match how the rest of this project
    guards against temporal leakage: the certification half is strictly
    later traffic than the half the temperature was fit on.
    """
    n = len(records)
    split = max(1, min(n - 1, int(n * fit_fraction))) if n > 1 else n
    fit_records, certify_records = records[:split], records[split:]
    if not certify_records:  # degenerate tiny input: fall back to fit set
        certify_records = fit_records

    fit_probs = np.array([champion_predict_fn(r) for r in fit_records])
    fit_labels = np.array([float(bool(r.get("outcome_success", False))) for r in fit_records])
    temperature = fit_temperature(fit_probs, fit_labels)

    # Everything reported below is measured on the HELD-OUT certify split.
    probs = np.array([champion_predict_fn(r) for r in certify_records])
    labels = np.array([float(bool(r.get("outcome_success", False))) for r in certify_records])

    ece_before = expected_calibration_error(probs, labels)
    brier_before = brier_score(probs, labels)
    acc_before = float(((probs > 0.5).astype(float) == labels).mean())

    calibrated_probs = _sigmoid(_logit(probs) / temperature)

    ece_after = expected_calibration_error(calibrated_probs, labels)
    brier_after = brier_score(calibrated_probs, labels)
    acc_after = float(((calibrated_probs > 0.5).astype(float) == labels).mean())

    return RecalibrationResult(
        temperature=temperature, ece_before=ece_before, ece_after=ece_after,
        brier_before=brier_before, brier_after=brier_after,
        accuracy_before=acc_before, accuracy_after=acc_after,
        n_calibration_examples=len(certify_records),
        n_fit_examples=len(fit_records),
    )


def make_recalibrated_predict_fn(champion_predict_fn: Callable[[Dict[str, Any]], float], temperature: float) -> Callable[[Dict[str, Any]], float]:
    """Wraps a champion predict_fn with a fitted temperature, so downstream
    code (evaluation, red-team, etc.) can treat 'recalibrated champion' as
    just another predict_fn with the same interface."""
    def predict(record: Dict[str, Any]) -> float:
        p = champion_predict_fn(record)
        return float(_sigmoid(_logit(np.array([p]))[0] / temperature))
    return predict
