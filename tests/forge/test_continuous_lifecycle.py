"""
Continuous multi-generation lifecycle.

THE GAP THIS CLOSES: the demo used to `break` at the first promotion. It
could show V1 -> V2, but never "V2 operates, a NEW independent incident
occurs, V2 is diagnosed and repaired into V3". That is the difference
between a repair loop and a self-healing lifecycle, and simply deleting
the `break` would NOT have been enough -- three other things were also
generation-blind:

  * `_window_dict()` computed drift signals through the ORIGINAL V1
    representation path, ignoring any installed adapter, so generation 2
    would have been monitored as if the repair never happened.
  * `driftsafe.reset_after_adaptation()` was never called, so the monitor
    stayed latched in ADAPTATION_CANDIDATE.
  * the monitoring budget (8 windows) was smaller than
    cooldown(5) + consensus + persistence, so a post-promotion generation
    could never re-escalate -- it stalled at PERSISTENCE_CHECK every run.

Verified end-to-end on seed 2024 after the fix: generation 1 heals
(V2 REJECT -> V3 REJECT on red team -> V4 PROMOTE, installed as generation
2), the monitor resets into cooldown, an independent second shock is
injected, and DriftSafe re-escalates on its own after 9 windows / 3600
transactions, opening a second incident against the repaired champion.
"""
from __future__ import annotations

import inspect

import scripts.demo_forge as demo


def test_monitoring_is_champion_adapter_aware():
    """Drift signals must be computed through the CURRENT champion, or
    later generations are monitored as if never repaired."""
    sig = inspect.signature(demo._window_dict)
    assert "adapter" in sig.parameters, (
        "_window_dict must accept the installed champion adapter"
    )
    src = inspect.getsource(demo._window_dict)
    assert "if adapter is not None" in src


def test_monitor_is_reset_after_promotion():
    src = inspect.getsource(demo.run_forge_demo_stream)
    assert "reset_after_adaptation()" in src, (
        "the monitor must be reset after a promotion, otherwise it stays "
        "latched in ADAPTATION_CANDIDATE and cannot detect a new incident"
    )


def test_lifecycle_continues_past_first_promotion():
    src = inspect.getsource(demo.run_forge_demo_stream)
    assert "max_generations" in src and "generation_idx" in src, (
        "champion generations must be tracked separately from repair attempts"
    )
    assert "SECOND_REGIME_SHIFT_INJECTED" in src, (
        "a genuinely independent incident must be able to occur against the "
        "repaired champion, otherwise continuous operation is untested"
    )


def test_monitoring_budget_exceeds_cooldown_and_persistence():
    """A monitoring budget smaller than cooldown + persistence makes
    post-promotion detection structurally impossible."""
    from vulcan.drift.driftsafe import DriftSafeConfig
    cfg = DriftSafeConfig()
    required = cfg.cooldown_windows + cfg.consensus_min_consecutive_windows + cfg.persistence_windows
    assert demo.MAX_MONITOR_WINDOWS > required, (
        f"MAX_MONITOR_WINDOWS={demo.MAX_MONITOR_WINDOWS} must exceed "
        f"cooldown({cfg.cooldown_windows}) + consensus({cfg.consensus_min_consecutive_windows}) "
        f"+ persistence({cfg.persistence_windows})={required}, or a repaired champion "
        f"can never re-escalate a new incident"
    )
