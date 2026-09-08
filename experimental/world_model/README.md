# Experimental: Action-Conditioned World Model

**The action-conditioned world model remains an exploratory research module
and is not required for Vulcan Forge promotion decisions.**

This module (`dynamics.py`, `planner.py`, `train_world_model.py`) is
preserved for historical continuity and further research, but it is no
longer part of Vulcan Forge's default product path, headline benchmark,
dashboard home screen, or README primary claim. Forge's promotion pipeline
(`vulcan/forge/`) does not depend on anything in this directory.

## Why it moved here

Across every configuration tested (tiny scale, base scale, with and without
pretraining, with and without an uncertainty gate, before and after a
confirmed temporal-leakage fix), the action-conditioned world model did not
beat the simpler XGBoost or direct Mini-Vulcan-heads baselines on mean
regret. See, in order:
- `reports/failures.md` — original tiny-scale negative result
- `reports/base_scale_result.md` — base-scale confirmation + ablations
- `reports/leakage_fix_result.md` — a real, confirmed leakage bug was found
  and fixed, substantially improving the model's regret at both scales, but
  it still did not beat the baselines on point estimates, and its
  route-preferences still did not track the oracle's true best route.

None of this means the world model is "broken" in some unfixable way — it
means the evidence for it as a product differentiator is not there yet.
Demoting it from the headline position reflects that evidence honestly,
per the same anti-fabrication discipline the rest of this project follows.

**Do not claim the world model beats conventional routing models unless
future measured evidence actually proves it.** If you resume work here,
re-run `experimental/world_model/train_world_model.py` and compare against
`vulcan/models/downstream_heads.py`'s direct-heads baseline using the
existing `scripts/benchmark.py` methodology before making any claim.

## What still works here

Nothing was deleted. `train_world_model.py` runs standalone
(`python -m experimental.world_model.train_world_model --config configs/tiny.yaml`),
`scripts/benchmark.py` still includes Model C/D as comparison arms (now
explicitly framed as a secondary, non-headline comparison — see
`README.md`), and all prior test coverage
(`tests/e2e/test_world_model_and_planner.py`,
`tests/data/test_no_temporal_leakage.py`) still passes against these files.
