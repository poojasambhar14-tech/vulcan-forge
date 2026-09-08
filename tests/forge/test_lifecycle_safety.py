"""
Lifecycle-safety fixes found by external code audit and independently
reproduced here before fixing. Each test pins a bug that was real.

1. COOLDOWN DID NOT SUPPRESS. DriftSafe._update_state decremented
   cooldown_remaining and then fell through into the transition logic, so
   a "cooldown" counted down while transitions proceeded normally.
   Reproduced: immediately after reset_after_adaptation() (cooldown=5), two
   high-consensus windows escalated straight back to SHIFT_DETECTED. Effect:
   a freshly-promoted champion could be re-diagnosed for the incident it had
   just healed, before the repair could affect traffic.

2. ROLLBACK HAD NO PARENT. The demo built an empty ModelRegistry and the
   first entry ever written was the REPAIRED model. Reproduced:
   registry.rollback() returned None and current_champion() became None --
   so the code comment claiming promotion was "rollback-capable" was false.
   V1 is now registered as version 1 before monitoring starts.

3. PROMOTION WAS NOT ATOMIC. The runtime champion pointers were reassigned
   BEFORE registry.promote() was attempted, so a persistence failure left
   the process serving an unregistered, unrecoverable model. Order is now
   persist-then-swap, with an explicit abort path.
"""
from __future__ import annotations

import inspect
import tempfile

import numpy as np

from vulcan.drift.driftsafe import DriftSafe, DriftSafeConfig, IncidentState
from vulcan.registry.model_registry import ModelRegistry
from vulcan.models.downstream_heads import DownstreamHeads
import scripts.demo_forge as demo


def _windows():
    ref = {"features": np.random.default_rng(0).normal(100, 10, 300),
           "z": np.random.default_rng(1).normal(0, 1, (300, 8)),
           "probs": np.full(300, 0.8), "labels": np.ones(300)}
    shifted = {"features": np.random.default_rng(2).normal(400, 10, 300),
               "z": np.random.default_rng(3).normal(6, 1, (300, 8)),
               "probs": np.full(300, 0.2), "labels": np.ones(300)}
    return ref, shifted


def test_cooldown_suppresses_state_transitions():
    ref, shifted = _windows()
    ds = DriftSafe(DriftSafeConfig(), ref)
    ds.state = IncidentState.ADAPTATION_CANDIDATE
    ds.reset_after_adaptation()
    assert ds.cooldown_remaining > 0
    for _ in range(3):
        ds.evaluate_window(shifted)
    assert ds.state == IncidentState.NORMAL, (
        f"DriftSafe escalated to {ds.state} while still in cooldown -- cooldown "
        f"must suppress transitions, not merely count down beside them"
    )


def test_cooldown_expires_and_detection_resumes():
    """The suppression must be temporary. A cooldown that never lets
    detection resume would be just as broken in the other direction."""
    ref, shifted = _windows()
    cfg = DriftSafeConfig()
    ds = DriftSafe(cfg, ref)
    ds.state = IncidentState.ADAPTATION_CANDIDATE
    ds.reset_after_adaptation()
    for _ in range(cfg.cooldown_windows + cfg.consensus_min_consecutive_windows + 2):
        ds.evaluate_window(shifted)
    assert ds.state != IncidentState.NORMAL, (
        "detection never resumed after cooldown expired -- the monitor would be "
        "permanently deaf following its first incident"
    )


def test_rollback_has_a_real_parent_to_restore():
    registry = ModelRegistry(registry_dir=tempfile.mkdtemp())
    h1, h2 = DownstreamHeads(32, 4), DownstreamHeads(32, 4)
    registry.promote(None, h1, {"version_label": "V1", "role": "initial_champion"})
    registry.promote(None, h2, {"version_label": "V2"})
    restored = registry.rollback("guardrail breach", "live_success_rate")
    assert restored is not None, "rollback returned None -- no parent champion was registered"
    assert restored.promotion_manifest.get("version_label") == "V1"
    assert registry.current_champion() is not None
    assert registry.current_champion().version == 1


def test_demo_registers_initial_champion_before_monitoring():
    src = inspect.getsource(demo.run_forge_demo_stream)
    assert "initial_champion" in src, (
        "the demo must register V1 before monitoring, otherwise the first "
        "promotion becomes registry version 1 and rollback has no parent"
    )


def test_promotion_is_atomic_registry_before_pointer_swap():
    """The registry write must precede the runtime pointer swap, so a
    persistence failure cannot leave an unregistered model serving."""
    src = inspect.getsource(demo.run_forge_demo_stream)
    idx_promote = src.find("entry = registry.promote(")
    idx_swap = src.find("champion_adapter = promoted_candidate.adapter")
    assert idx_promote != -1 and idx_swap != -1
    assert idx_promote < idx_swap, (
        "runtime champion pointers are swapped before registry.promote() succeeds; "
        "a registry failure would leave an unregistered model live"
    )
    assert "install_aborted" in src, (
        "there must be an explicit abort path that keeps the previous champion "
        "live when registration fails"
    )
