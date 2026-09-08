"""
DriftSafe (master spec sections 11-12).

Consensus rule: a drift incident starts only when >=2 independent signals
exceed their thresholds for >=2 consecutive windows. This alone rules out
single-window noise, but does not yet distinguish a brief operational
outage from a persistent regime shift that should trigger model adaptation.

That distinction is handled by `IncidentTracker`: an incident must persist
for >= `persistence_windows` consecutive windows (with a minimum sample
count per window) before it is escalated to an "ADAPTATION_CANDIDATE"
state. If the underlying signals recover before that, the incident is
closed with no model update -- only the (separate, immediate) planner
reaction to unhealthy routes applies.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from vulcan.drift.detectors import (
    SignalReading,
    input_drift_psi,
    representation_drift_mmd,
    calibration_drift,
    residual_drift_page_hinkley,
)


class IncidentState(str, enum.Enum):
    NORMAL = "NORMAL"
    SHIFT_DETECTED = "SHIFT_DETECTED"
    PERSISTENCE_CHECK = "PERSISTENCE_CHECK"
    ADAPTATION_CANDIDATE = "ADAPTATION_CANDIDATE"
    RECOVERED = "RECOVERED"


@dataclass
class WindowReport:
    window_index: int
    signals: List[SignalReading]
    n_exceeded: int
    consensus_triggered: bool


@dataclass
class DriftSafeConfig:
    consensus_min_signals: int = 2
    consensus_min_consecutive_windows: int = 2
    persistence_windows: int = 3
    min_sample_count: int = 30
    cooldown_windows: int = 5
    input_drift_threshold: float = 0.2
    representation_drift_threshold: float = 0.15
    calibration_drift_threshold: float = 0.05
    residual_drift_lambda: float = 15.0


class DriftSafe:
    def __init__(self, cfg: DriftSafeConfig, reference_window: dict):
        """reference_window: dict with keys 'features' (np.ndarray [N]),
        'z' (np.ndarray [N, d]), 'probs' (np.ndarray [N]),
        'labels' (np.ndarray [N]) -- captured from a known-stable period."""
        self.cfg = cfg
        self.reference = reference_window
        self.state = IncidentState.NORMAL
        self.consecutive_consensus = 0
        self.persistent_windows = 0
        self.cooldown_remaining = 0
        self.window_history: List[WindowReport] = []
        self.window_idx = 0

    def evaluate_window(self, current_window: dict) -> WindowReport:
        """current_window: same shape as reference_window, for one window
        of recent traffic. Returns the per-window signal report and updates
        the incident state machine."""
        n = len(current_window["labels"])
        signals = []

        if n >= self.cfg.min_sample_count:
            signals.append(
                input_drift_psi(
                    self.reference["features"], current_window["features"], self.cfg.input_drift_threshold
                )
            )
            signals.append(
                representation_drift_mmd(
                    self.reference["z"], current_window["z"], self.cfg.representation_drift_threshold
                )
            )
            signals.append(
                calibration_drift(
                    self.reference["probs"], self.reference["labels"],
                    current_window["probs"], current_window["labels"],
                    self.cfg.calibration_drift_threshold,
                )
            )
            residuals = list(np.abs(current_window["probs"] - current_window["labels"]))
            signals.append(residual_drift_page_hinkley(residuals, lam=self.cfg.residual_drift_lambda))

        n_exceeded = sum(1 for s in signals if s.exceeded)
        consensus = n_exceeded >= self.cfg.consensus_min_signals

        report = WindowReport(
            window_index=self.window_idx, signals=signals, n_exceeded=n_exceeded, consensus_triggered=consensus
        )
        self.window_history.append(report)
        self.window_idx += 1
        self._update_state(consensus)
        return report

    def _update_state(self, consensus_this_window: bool) -> None:
        # Cooldown SUPPRESSES state transitions -- it does not merely count
        # down alongside them. This gives a freshly promoted champion time to
        # affect live traffic before it can be re-diagnosed for the incident
        # it just healed.
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.consecutive_consensus = 0
            return

        if consensus_this_window:
            self.consecutive_consensus += 1
        else:
            self.consecutive_consensus = 0
            if self.state in (IncidentState.SHIFT_DETECTED, IncidentState.PERSISTENCE_CHECK):
                self.state = IncidentState.RECOVERED
                self.persistent_windows = 0
                self.cooldown_remaining = self.cfg.cooldown_windows
                return

        if self.state == IncidentState.NORMAL:
            if self.consecutive_consensus >= self.cfg.consensus_min_consecutive_windows:
                self.state = IncidentState.SHIFT_DETECTED
                self.persistent_windows = 1
        elif self.state == IncidentState.SHIFT_DETECTED:
            self.state = IncidentState.PERSISTENCE_CHECK
            self.persistent_windows += 1
        elif self.state == IncidentState.PERSISTENCE_CHECK:
            self.persistent_windows += 1
            if self.persistent_windows >= self.cfg.persistence_windows:
                self.state = IncidentState.ADAPTATION_CANDIDATE
        elif self.state == IncidentState.RECOVERED:
            self.state = IncidentState.NORMAL
            self.persistent_windows = 0
        # ADAPTATION_CANDIDATE holds until externally reset by the caller
        # once a challenger has been trained (see vulcan/adaptation).

    def reset_after_adaptation(self) -> None:
        self.state = IncidentState.NORMAL
        self.consecutive_consensus = 0
        self.persistent_windows = 0
        self.cooldown_remaining = self.cfg.cooldown_windows

    def rebase(self, new_reference_window: dict) -> None:
        """Replaces the reference distribution with one captured from the
        newly-installed champion on genuinely fresh, post-promotion traffic.

        Without this, every signal (PSI / representation MMD / calibration
        ECE / Page-Hinkley residuals) keeps comparing new traffic against
        the ORIGINAL pre-repair champion's representation and calibration,
        even after that champion has been replaced. Part of any measured
        "drift" after a promotion would then be an artifact of the repair
        itself changing the representation/calibration, not a real change
        in the environment -- inflating false-positive risk on the very
        first incident after a promotion. Call this once, right after a
        promotion is durably installed (paired with `reset_after_adaptation`
        so the incident state machine and the reference distribution move
        together), never mid-incident.
        """
        self.reference = new_reference_window
