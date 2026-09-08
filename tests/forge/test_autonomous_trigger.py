"""
Autonomous trigger.

BEFORE: scripts/demo_forge.py collected a fixed 2000 transactions and then
called run_forge_cycle() unconditionally. The DriftSafe state machine
existed, was well-designed (multi-signal consensus + persistence +
cooldown), and was NEVER CONSULTED -- `grep -n DriftSafe scripts/demo_forge.py`
returned nothing. A human had hardcoded the moment of intervention, so the
"self" in self-healing was doing unearned work.

AFTER: traffic is consumed in windows, each is scored by DriftSafe, and
run_forge_cycle() is reached only if the state machine escalates to
ADAPTATION_CANDIDATE on its own. If the shift is transient the machine
returns to NORMAL and no healing happens -- reported honestly as
NO_INCIDENT rather than healing anyway.

These tests assert the wiring cannot silently regress.
"""
from __future__ import annotations

import inspect

from vulcan.drift.driftsafe import DriftSafe, DriftSafeConfig, IncidentState
import scripts.demo_forge as demo


def test_demo_consults_driftsafe_before_healing():
    src = inspect.getsource(demo.run_forge_demo_stream)
    assert "DriftSafe(" in src, "demo must instantiate the DriftSafe monitor"
    assert "IncidentState.ADAPTATION_CANDIDATE" in src, (
        "healing must be gated on the state machine escalating on its own"
    )
    # The no-incident path must exist: the system is allowed to decide NOT to heal.
    assert "NO_INCIDENT" in src, (
        "demo must be able to conclude that no healing is warranted, otherwise "
        "the trigger is decorative"
    )


def test_driftsafe_requires_persistence_before_escalating():
    """A single anomalous window must NOT trigger healing -- escalation
    requires sustained consensus. This is what stops a transient gateway
    blip from causing an unnecessary model change."""
    cfg = DriftSafeConfig()
    import numpy as np
    ref = {
        "features": np.random.default_rng(0).normal(100, 10, 300),
        "z": np.random.default_rng(1).normal(0, 1, (300, 8)),
        "probs": np.full(300, 0.8),
        "labels": np.ones(300),
    }
    ds = DriftSafe(cfg, ref)
    # One heavily shifted window
    shifted = {
        "features": np.random.default_rng(2).normal(400, 10, 300),
        "z": np.random.default_rng(3).normal(6, 1, (300, 8)),
        "probs": np.full(300, 0.2),
        "labels": np.ones(300),
    }
    ds.evaluate_window(shifted)
    assert ds.state != IncidentState.ADAPTATION_CANDIDATE, (
        "DriftSafe escalated to ADAPTATION_CANDIDATE after a single window; "
        "persistence requirements are not being enforced"
    )


def test_reset_after_adaptation_returns_monitor_to_normal():
    """After a healing cycle the monitor must return to NORMAL, otherwise the
    loop would re-trigger forever on the same incident."""
    cfg = DriftSafeConfig()
    import numpy as np
    ref = {
        "features": np.random.default_rng(0).normal(100, 10, 200),
        "z": np.random.default_rng(1).normal(0, 1, (200, 8)),
        "probs": np.full(200, 0.8),
        "labels": np.ones(200),
    }
    ds = DriftSafe(cfg, ref)
    ds.state = IncidentState.ADAPTATION_CANDIDATE
    ds.reset_after_adaptation()
    assert ds.state == IncidentState.NORMAL


def test_driftsafe_escalates_on_a_partial_subgroup_shift():
    """A shift confined to one subgroup must still escalate.

    Aggregate drift signals are weakest exactly when a shift is localized --
    a single issuer is a minority of traffic, so population-level statistics
    move less than under a global shift. If the state machine could only
    escalate on broad shifts, the autonomous trigger would be blind to the
    localized failures this system exists to catch.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    n = 400
    ref = {
        "features": rng.normal(100, 12, n),
        "z": rng.normal(0, 1, (n, 8)),
        "probs": np.clip(rng.normal(0.80, 0.05, n), 0.01, 0.99),
        "labels": (rng.random(n) < 0.80).astype(float),
    }
    # ~20% of traffic degraded; the rest unchanged.
    k = int(n * 0.2)
    feats = ref["features"].copy(); feats[:k] += 90
    z = ref["z"].copy(); z[:k] += 2.5
    probs = ref["probs"].copy(); probs[:k] = 0.35
    labels = ref["labels"].copy(); labels[:k] = 0.0
    shifted = {"features": feats, "z": z, "probs": probs, "labels": labels}

    ds = DriftSafe(DriftSafeConfig(), ref)
    for _ in range(6):
        ds.evaluate_window(shifted)
        if ds.state == IncidentState.ADAPTATION_CANDIDATE:
            break
    assert ds.state == IncidentState.ADAPTATION_CANDIDATE, (
        f"state machine stalled at {ds.state} under a partial subgroup shift; "
        f"the autonomous trigger would miss localized failures"
    )
