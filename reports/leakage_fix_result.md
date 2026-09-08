# Temporal Target Leakage Fix — Result

Cross-linked from `README.md` and `reports/base_scale_result.md`. This is a
new, dated addition; it does not delete or edit the numbers in
`reports/base_scale_result.md` or `reports/failures.md` -- those remain the
honest record of what was true before this fix. This report documents what
changed and what did not.

## Confirmed mechanism

Traced to the code and confirmed still present before this fix, exactly as
described:

1. `vulcan/data/generate.py` logs every training record AFTER calling
   `sim.step()`, so `action_route_id`, `outcome_success`,
   `outcome_fraud_loss`, `outcome_latency_ms`, `outcome_abandoned`, and
   `outcome_error_type` all hold the REAL, realized values for that event.
2. `train/train_world_model.py` built windows where the last event
   (`w[-1]`) was one of these fully-realized records, and
   `vulcan/data/windowing.py::encode_windows` encoded every field of every
   position, including the last, with no masking.
3. `vulcan/models/mini_vulcan.py::MiniVulcanBackbone.current_state()` runs a
   **bidirectional** (non-causal) Transformer and returns `z[:, -1, :]`.
   Self-attention at that position could see that same position's own
   action/outcome fields — the world model was predicting an outcome it had
   already been shown.
4. Confirmed asymmetry: `scripts/benchmark.py`'s live rollout path was NOT
   affected the same way, because `obs_record = _flatten_observation(obs)`
   at decision time genuinely never has action/outcome fields (the action
   hasn't been taken yet) — training saw a "current position" that always
   had real outcome values; live inference saw the same position type with
   those values always absent.

## The fix

`vulcan/data/windowing.py::encode_windows_for_world_model` encodes every
window exactly as `encode_windows` does, EXCEPT the last position has
`action_route_id`, `action_rail`, `action_gateway`, `propensity`,
`outcome_success`, `outcome_latency_ms`, `outcome_processing_cost`,
`outcome_fraud_loss`, `outcome_abandoned`, and `outcome_error_type` removed
from the record entirely before encoding (not set to a hand-picked
placeholder) — so `FieldTokenizer`'s existing `.get(name, default)` /
UNK-token fallback engages exactly the same way it already does for a live
`obs_record` that never had those keys. `propensity` was added to the
stripped-field set beyond what was originally listed, because it is present
in a full training record but absent from `_flatten_observation`'s output —
without stripping it too, the exact-equality test below would fail on a
real, distinct discrepancy.

`train/train_world_model.py` now calls `encode_windows_for_world_model`
instead of `encode_windows` for the window-embedding step (both train and
validation). The candidate action still enters only through the existing
`ActionEmbedding` pathway — nothing was added to the state to compensate.

**Scope note:** this fix was applied only to the world model
(`train/train_world_model.py`), as instructed. `train/train_baseline_heads.py`
(Baseline B) uses the identical `encode_windows` + `current_state`
mechanism and — by the same reasoning — very likely has the identical leak,
unaddressed by this pass. Baseline B's regret numbers in the tables below
are therefore NOT necessarily a leak-free comparison point; they are
reported as-is (unchanged from `reports/base_scale_result.md`) because
fixing them was out of scope for this pass.

## Regression test: `tests/data/test_no_temporal_leakage.py`

Three tests, all passing:
1. `test_current_position_encoding_excludes_outcome_and_action_fields` —
   directly inspects encoded tensors; every masked field at the last
   position equals the tokenizer's UNK/default encoding, not the real
   logged value.
2. `test_training_and_inference_current_state_encoding_match` — the key
   test. It positively demonstrates BOTH sides of the before/after
   contrast in a single run (no git history exists in this repository to
   revert against, so the old function's failure is demonstrated directly
   rather than via version control): encoding a real training record's last
   position with the OLD `encode_windows` does **not** match a genuine live
   `obs_record`'s encoding (asserted explicitly — this assertion is itself
   proof the leak was real); encoding the same record with the NEW
   `encode_windows_for_world_model` matches it exactly
   (`torch.equal`/`torch.allclose`, not merely "close").
3. `test_history_positions_are_not_masked` — confirms the fix does not
   over-mask; genuine past-history positions retain their real values.

Full suite: `pytest tests/ -q` → 36 passed (33 previously existing + 3 new).

## Step 3.4: TRUE oracle action-effect sizes, measured fresh

Computed directly from `IndiaPaymentSim.oracle_outcome_distribution` (never
from any prior write-up's numbers, which were not assumed correct):

| Config | Oracle mean \|pairwise effect\| | Oracle mean range effect (max−min route) |
|---|---|---|
| Tiny (num_routes=3) | 0.0282 (first 1,000-state sample) / 0.0698 (2,247-window realistic-trajectory sample used for the before/after comparison below) | 0.0424 / 0.1047 |
| Base (num_routes=4) | 0.0401 | 0.0758 / 0.0523 (realistic-trajectory sample) |

The two sampling methodologies give visibly different numbers for the same
config — the oracle's true effect size has real sampling variance depending
on which states are sampled (which routes' rolling-health features have
been exercised). The realistic-trajectory sample (built the same way as the
model's own training/eval windows, with genuine rolling history) is the one
used as the calibration reference for the before/after comparison
immediately below, since it is the apples-to-apples reference for that
specific comparison.

## Step 4.1: action sensitivity before vs. after, tiny config

Same trained-checkpoint process both times (backbone frozen at its
pretrained weights, world model retrained fresh), same held-out
realistic-trajectory evaluation windows (247 windows), oracle mean
\|pairwise effect\| = 0.0698 for this specific sample:

| | val success acc | predicted mean \|pairwise effect\| | MAE vs. oracle | Pearson r | Spearman r | best-route ranking acc (chance=0.333) |
|---|---|---|---|---|---|---|
| BEFORE (old, leaky `encode_windows`) | 1.0000 | 0.0866 | 0.0898 | -0.030 | -0.058 | 0.0000 |
| AFTER (new, `encode_windows_for_world_model`) | 0.7790 | 0.0121 | 0.0577 | -0.082 | -0.118 | 0.0000 |

**Honest reading of this table:** the pre-fix `val_success_acc=1.0000` is
itself the clearest single symptom of the leak — a model that perfectly
predicts an outcome it was shown is not learning to predict anything. After
the fix, accuracy drops to a plausible 0.7790, which is the expected
signature of a genuinely harder, non-leaked task. The pre-fix "action
sensitivity" number (0.0866) is **not** a real effect — it is inflated by
the SAME leak also being present in the evaluation windows when using the
old encoding function, and it is essentially uncorrelated with the true
oracle structure (Pearson -0.03, ranking accuracy 0%, i.e. worse than
chance). Post-fix, the predicted effect size (0.0121) is smaller in
absolute terms but is no longer a leak artifact; however, it is still only
~17% of the oracle's true effect size for this sample (0.0121 / 0.0698),
and it remains uncorrelated with which route the oracle actually considers
best (ranking accuracy still 0.0000, worse than the 33% chance rate).
Diagnostic root cause of the 0% ranking accuracy: for this specific
evaluation trajectory, the oracle's best route is route 1 in 247/247 cases
(a near-constant target, a property of this simulator seed's structural
route-quality values), while the model's predicted best route is route 2
in 237/247 cases and route 0 in the remaining 10 — the model has a
different, and wrong, constant preference, not a state-dependent one that
merely disagrees with the oracle sometimes.

Retraining the actual `checkpoints/world_model.pt` (not the standalone
diagnostic copy) reproduced this consistently: `val_success_acc=0.7790`
matching to 4 decimal places, confirming the diagnostic and the real
pipeline agree.

## Step 4.1 (base config)

Retraining the real base-scale world model with the fix:

| | val success acc | val success bce | predicted mean \|pairwise effect\| | oracle mean \|pairwise effect\| | MAE | Pearson r | Spearman r | best-route ranking acc (chance=0.25) |
|---|---|---|---|---|---|---|---|---|
| AFTER fix (base) | 0.8116 | 0.4813 | 0.0350 | 0.0523 | 0.0180 | 0.023 | 0.007 | 0.1230 |

At base scale, the predicted effect size (0.0350) is much closer to the
oracle's (0.0523) than at tiny scale — about 67% of the oracle magnitude,
genuinely in the same order of magnitude, not just "an improvement over
near-zero." Correlation and ranking accuracy remain weak (ranking accuracy
12.3%, below the 25% chance rate for 4 routes) — the same qualitative
finding as tiny: the fix produces a real, non-trivial, correctly-scaled
action-dependent signal, but that signal still does not reliably track
which route is actually best.

## Step 4.2 / 4.3: full benchmark, before vs. after, both scales

**Tiny** (seeds 1,2,3, 500 steps/seed):

| Policy | Regret BEFORE fix | Regret AFTER fix |
|---|---|---|
| Baseline A (XGBoost) | 0.0203 ± 0.0288 | 0.0203 ± 0.0288 (unchanged, not affected by this fix) |
| Baseline B (Mini-Vulcan direct) | 0.0364 ± 0.0291 | 0.0347 ± 0.0287 (unchanged mechanism; small run-to-run variation from fresh XGBoost/data generation) |
| Model C (world model) | 0.0530 ± 0.0340 | **0.0326 ± 0.0271** |
| Model D (world model + gate) | 0.0341 ± 0.0428 (0.2% fallback) | 0.0326 ± 0.0271 (0.0% fallback) |

**Base** (seeds 1,2,3,4,5, 500 steps/seed):

| Policy | Regret BEFORE fix | Regret AFTER fix |
|---|---|---|
| Baseline A (XGBoost) | 0.0184 ± 0.0200 | 0.0184 ± 0.0200 (unchanged) |
| Baseline B (Mini-Vulcan direct) | 0.0199 ± 0.0193 | 0.0199 ± 0.0193 (unchanged) |
| Model C (world model) | 0.0554 ± 0.0274 | **0.0282 ± 0.0138** |
| Model D (world model + gate) | 0.0554 ± 0.0274 (0% fallback) | 0.0282 ± 0.0138 (0% fallback) |

Full manifests: `artifacts/runs/benchmark_1787743272.json` (tiny, after),
`artifacts/runs/benchmark_1787743799.json` (base, after).

## Does the world model now beat, tie, or lose to Baseline A?

Stating this precisely and separately from "the bug is fixed," as
instructed:

- **The bug was real, and fixing it produced a large, genuine improvement
  in regret at both scales** — roughly halving Model C's regret at tiny
  scale (0.0530 → 0.0326) and roughly halving it again in relative terms at
  base scale (0.0554 → 0.0282).
- **The world model still does not beat Baseline A (XGBoost) at either
  scale**, comparing point estimates: 0.0326 vs. 0.0203 at tiny, 0.0282 vs.
  0.0184 at base. Model C's regret is numerically higher (worse) in both
  cases.
- **At base scale, the remaining gap is no longer clearly resolvable given
  sampling noise**: Model C's regret (0.0282 ± 0.0138, range
  [0.0144, 0.0420]) and Baseline A's regret (0.0184 ± 0.0200, range
  [-0.0016, 0.0384]) now have substantially overlapping confidence
  intervals. Before the fix, the gap (0.0554 vs. 0.0184) was large relative
  to either interval and not plausibly attributable to noise. This is a
  genuinely different, weaker conclusion than before the fix — "loses,
  clearly" has become "loses on point estimate, but the two are not
  clearly distinguishable at this sample size" — but it is not the same
  claim as "ties" or "wins," and is reported here as exactly what it is.
- **At tiny scale, the gap remains clearer**: 0.0326 ± 0.0271 vs.
  0.0203 ± 0.0288 — intervals overlap here too (tiny-scale regret estimates
  are noisy relative to their means throughout this project), so the same
  caveat about statistical distinguishability applies, but the point-estimate
  gap did not close as much proportionally as at base scale.

## Is the leak the whole story, or is something else still the bottleneck?

Both, in different proportions at different scales. The leak was real and
fixing it materially improved decision quality — this is not a case where
"the mechanism is now correct but nothing changed." At the same time, per
the oracle-calibrated action-sensitivity results above, the model's
route-preferences still do not track the oracle's true best-route pattern
at either scale (near-zero or negative correlation, ranking accuracy at or
below chance both times). This means at least one other factor — plausibly
the planner's fixed utility weights (`PlannerWeights` defaults, never
tuned), the training objective's balance between success/fraud/abandon/
latency/next-state terms, or the behavior policy's route-coverage pattern
(the training data's action distribution is not the same as the uniform
counterfactual coverage the oracle comparison implicitly assumes) — remains
a real, distinct, and still-unresolved contributor to why the world model
does not yet win outright. This is a narrower, more specific finding than
"the world model doesn't work": the representation-and-decoding pipeline
now produces a real, correctly-scaled (at least at base scale)
action-dependent signal; it does not yet produce a CORRECTLY-DIRECTED one.

## What was not done in this pass

- Baseline B's identical, unaddressed leak (see "Scope note" above) — its
  numbers should not be read as a clean comparison point until this is
  investigated separately.
- No changes to `vulcan/planner/planner.py`'s utility weights, the world
  model's training objective, or the behavior policy's coverage pattern —
  all three are named above as plausible remaining bottlenecks but were out
  of scope for this pass, which was limited to the one confirmed leakage
  bug.
- Only 3 seeds (tiny) / 5 seeds (base) were used, matching prior
  methodology exactly for a clean before/after comparison, not because more
  seeds would not be worthwhile.
