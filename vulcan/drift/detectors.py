"""
Individual DriftSafe signals (master spec section 11). Each detector reports
a scalar value against a reference distribution/baseline plus a boolean
'exceeded' flag against a configured threshold. DriftSafe itself (see
driftsafe.py) combines these via a consensus rule rather than acting on any
single one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SignalReading:
    name: str
    value: float
    threshold: float
    exceeded: bool


def population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Standard PSI over a shared binning derived from the reference sample."""
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    quantiles = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=quantiles)
    cur_counts, _ = np.histogram(current, bins=quantiles)

    ref_pct = ref_counts / max(1, ref_counts.sum())
    cur_pct = cur_counts / max(1, cur_counts.sum())
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def input_drift_psi(reference_features: np.ndarray, current_features: np.ndarray, threshold: float = 0.2) -> SignalReading:
    """reference_features / current_features: [N] a single observable
    continuous feature (e.g. amount, or rolling route success)."""
    psi = population_stability_index(reference_features, current_features)
    return SignalReading("input_drift_psi", psi, threshold, psi > threshold)


def representation_drift_mmd(reference_z: np.ndarray, current_z: np.ndarray, threshold: float = 0.15) -> SignalReading:
    """Lightweight linear-kernel MMD^2 proxy: squared distance between mean
    embeddings, normalized by reference embedding scale. A full RBF-kernel
    MMD is more sensitive but far more expensive; this proxy is adequate for
    detecting mean-shift in the shared representation z_t."""
    ref_mean = reference_z.mean(axis=0)
    cur_mean = current_z.mean(axis=0)
    ref_scale = np.linalg.norm(reference_z.std(axis=0)) + 1e-6
    dist = np.linalg.norm(cur_mean - ref_mean) / ref_scale
    return SignalReading("representation_drift_mmd_proxy", float(dist), threshold, float(dist) > threshold)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probs - labels) ** 2))


def calibration_drift(
    reference_probs: np.ndarray, reference_labels: np.ndarray,
    current_probs: np.ndarray, current_labels: np.ndarray,
    threshold: float = 0.05,
) -> SignalReading:
    ref_ece = expected_calibration_error(reference_probs, reference_labels)
    cur_ece = expected_calibration_error(current_probs, current_labels)
    delta = cur_ece - ref_ece
    return SignalReading("calibration_drift_ece_delta", float(delta), threshold, float(delta) > threshold)


class PageHinkley:
    """Page-Hinkley test for detecting a shift in the mean of a residual
    (prediction error) stream. Used for prediction-residual drift."""

    def __init__(self, delta: float = 0.005, lam: float = 15.0):
        self.delta = delta
        self.lam = lam
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.n = 0
        self.sum_ph = 0.0
        self.min_ph = 0.0

    def update(self, value: float) -> bool:
        """Returns True if a change point is flagged on this update."""
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.sum_ph += value - self.mean - self.delta
        self.min_ph = min(self.min_ph, self.sum_ph)
        return (self.sum_ph - self.min_ph) > self.lam


def residual_drift_page_hinkley(residuals: List[float], delta: float = 0.005, lam: float = 15.0) -> SignalReading:
    ph = PageHinkley(delta=delta, lam=lam)
    flagged_at = None
    for i, r in enumerate(residuals):
        if ph.update(r):
            flagged_at = i
    value = (ph.sum_ph - ph.min_ph)
    return SignalReading("residual_drift_page_hinkley", float(value), lam, flagged_at is not None)
