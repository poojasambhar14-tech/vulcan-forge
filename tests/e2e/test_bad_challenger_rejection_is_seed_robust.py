"""
Fix 2 (seed-robustness of the induced-forgetting demo scenario).

The Part 3 demo (scripts/demo_part3.py) trains a deliberately BAD candidate
(narrow recent drift slice, no replay buffer, high capacity/LR, zero weight
decay) and asserts the champion/challenger gate REJECTs it due to historical
regression. That assertion must hold up across seeds, not just the one
shipped as configs/tiny.yaml's default -- otherwise a config change elsewhere
(more transactions, different epochs, etc.) could silently reintroduce a
crash on the flagship safety demo with no test catching it.

This test runs the full STAGE 1-6 scenario across 10 seeds and requires the
REJECT decision in at least 9 of them. It is slower than the rest of the
suite (each seed trains a full candidate) -- expect ~10s/seed on CPU.
"""
from __future__ import annotations

import pytest

from vulcan.common.config import load_config
from scripts.demo_part3 import run_demo


SEEDS = list(range(10))
MIN_REJECT_COUNT = 9


@pytest.mark.slow
def test_bad_challenger_rejection_is_seed_robust():
    cfg = load_config("configs/tiny.yaml")

    outcomes = {}
    for seed in SEEDS:
        manifest = run_demo(cfg, seed, verbose=False)
        outcomes[seed] = manifest["bad_challenger"]["shadow_eval"]["decision"]

    n_reject = sum(1 for d in outcomes.values() if d == "REJECT")

    assert n_reject >= MIN_REJECT_COUNT, (
        f"Bad-challenger REJECT rate too low: {n_reject}/{len(SEEDS)} "
        f"(need >= {MIN_REJECT_COUNT}). Per-seed outcomes: {outcomes}. "
        f"The induced-forgetting scenario in scripts/demo_part3.py's "
        f"BAD_CHALLENGER_* constants needs to be made more reliably "
        f"forgetting-inducing -- see reports/failures.md item 2 addendum."
    )
