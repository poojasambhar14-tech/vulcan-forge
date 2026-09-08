import numpy as np
import pytest

from vulcan.drift.driftsafe import DriftSafe, DriftSafeConfig, IncidentState


def _make_window(n, feature_mean, z_mean, p_success_mean, d=8, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(feature_mean, 1.0, size=n)
    z = rng.normal(z_mean, 1.0, size=(n, d))
    labels = (rng.random(n) < p_success_mean).astype(float)
    probs = np.clip(rng.normal(p_success_mean, 0.05, size=n), 0.01, 0.99)
    return {"features": features, "z": z, "probs": probs, "labels": labels}


@pytest.fixture
def reference():
    return _make_window(500, feature_mean=0.0, z_mean=0.0, p_success_mean=0.9, seed=1)


def test_normal_regime_low_false_alarm_rate(reference):
    cfg = DriftSafeConfig()
    ds = DriftSafe(cfg, reference)
    n_adaptation_candidates = 0
    for i in range(20):
        window = _make_window(60, feature_mean=0.0, z_mean=0.0, p_success_mean=0.9, seed=100 + i)
        ds.evaluate_window(window)
        if ds.state == IncidentState.ADAPTATION_CANDIDATE:
            n_adaptation_candidates += 1
    assert n_adaptation_candidates == 0
    assert ds.state == IncidentState.NORMAL


def test_persistent_shift_triggers_adaptation_candidate(reference):
    cfg = DriftSafeConfig(persistence_windows=3, consensus_min_consecutive_windows=2)
    ds = DriftSafe(cfg, reference)

    # stable for a few windows
    for i in range(3):
        ds.evaluate_window(_make_window(60, 0.0, 0.0, 0.9, seed=200 + i))
    assert ds.state == IncidentState.NORMAL

    # persistent regime shift: mean shifts hard and STAYS shifted
    reached_candidate = False
    for i in range(10):
        w = _make_window(60, feature_mean=3.0, z_mean=2.5, p_success_mean=0.4, seed=300 + i)
        ds.evaluate_window(w)
        if ds.state == IncidentState.ADAPTATION_CANDIDATE:
            reached_candidate = True
            break
    assert reached_candidate


def test_temporary_shock_does_not_trigger_adaptation(reference):
    cfg = DriftSafeConfig(persistence_windows=4, consensus_min_consecutive_windows=2)
    ds = DriftSafe(cfg, reference)

    for i in range(3):
        ds.evaluate_window(_make_window(60, 0.0, 0.0, 0.9, seed=400 + i))

    # short shock: only 2 bad windows (less than persistence_windows=4), then recovery
    ds.evaluate_window(_make_window(60, 3.0, 2.5, 0.4, seed=500))
    ds.evaluate_window(_make_window(60, 3.0, 2.5, 0.4, seed=501))
    assert ds.state != IncidentState.ADAPTATION_CANDIDATE

    # recovers
    for i in range(5):
        ds.evaluate_window(_make_window(60, 0.0, 0.0, 0.9, seed=600 + i))
    assert ds.state in (IncidentState.NORMAL, IncidentState.RECOVERED)
    assert ds.state != IncidentState.ADAPTATION_CANDIDATE
