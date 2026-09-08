# Failure Analysis

This document records genuine negative/mixed results observed while building
and running this prototype. Per the project's own acceptance gate,
scientific credibility takes priority over a manufactured win.

## 1. On the `tiny` config, XGBoost (Baseline A) matched or beat the
   Mini-Vulcan world model (Model C) on success rate and regret

A real benchmark run (`artifacts/runs/benchmark_*.json`, 3 seeds x 400 steps)
showed:

| Policy | Success rate | Mean regret |
|---|---|---|
| Baseline A (XGBoost) | ~0.81 | ~0.020 |
| Baseline B (Mini-Vulcan direct heads) | ~0.80 | ~0.036 |
| Model C (world model + planner) | ~0.79 | ~0.052 |

The action-conditioned world model did **not** outperform the much simpler
flat XGBoost baseline at `tiny` scale. Plausible reasons, in order of
likely contribution:

- **Model capacity vs. data.** The `tiny` config trains a ~370K-parameter
  Transformer backbone on 20K synthetic transactions for only 3 pretraining
  epochs — an XGBoost model with explicit engineered features (including
  the same rolling-route-success signals) is a very strong baseline at this
  scale precisely because the underlying task's signal is mostly captured
  by those hand-engineered features already.
- **Planner myopia.** The planner's utility function and the world model's
  single-step (horizon=1) prediction may be less well calibrated early in
  training than XGBoost's directly-optimized classification objective.
- **Frozen backbone.** Both Model B and Model C reuse the same frozen,
  lightly-pretrained backbone; if backbone pretraining under-converged,
  every downstream head built on it inherits that ceiling.

This is exactly the kind of result section 40 of the build spec anticipates:
*"If the world model fails to beat Mini-Vulcan: report that result
accurately, analyse why, and keep DriftSafe fully functional."* DriftSafe,
adaptation, and rollback (Part 2/3) do not depend on the world model beating
XGBoost, and all pass their mandatory acceptance tests independently of this
result. A `base`-scale run with more pretraining epochs would be a natural
next experiment before drawing broader conclusions.

**Update: the base-scale experiment has now been run.** See
`reports/base_scale_result.md` for the full writeup. Short version: moving
from `tiny` to `base` scale did **not** close the gap (if anything it
widened marginally), and a follow-up ablation found no measurable benefit
from self-supervised pretraining on this benchmark either. The "model
capacity vs. data" hypothesis above is therefore not supported by the
evidence — worth recording here since this section originally proposed it
as the most likely explanation.

## 2. Naive accuracy alone does not catch catastrophic forgetting

During Part 3 demo tuning, the first version of the champion/challenger gate
used **accuracy only** to guard against historical-regime regression. A
deliberately bad candidate (adapter trained on drift-only data, no replay
buffer, many epochs) was *incorrectly promoted* by an accuracy-only gate,
because the underlying success label is imbalanced (~85% positive), so an
overconfident model that always leans "success" still scores well on
accuracy even when badly miscalibrated on the historical regime.

Adding a **log-loss (BCE) regression gate** on the historical canary set
exposed the problem and correctly rejected the bad candidate. This mirrors
the general warning in the NVIDIA transaction-model blueprint and the
project's own evaluation section: **accuracy/ROC-AUC-style metrics are
insufficient for imbalanced outcomes; calibration-sensitive metrics like
log-loss or PR-AUC are needed.** The final gate configuration
(`vulcan/adaptation/champion_challenger.py`) reflects this fix.

### Addendum: the induced-forgetting scenario was itself seed-sensitive

The log-loss gate above was correct, but the *bad-challenger training
procedure* used to demonstrate it (train an adapter on the whole post-shock
drift buffer, no replay, 120 epochs, lr=5e-3) turned out to be
**seed-sensitive**: with `configs/tiny.yaml`'s shipped default seed (42),
that procedure sometimes produced a candidate whose historical-canary
log-loss *improved* rather than regressed, so the gate correctly
`PROMOTE`d it — which is the right decision given that candidate's actual
behavior, but defeats the point of the demo (which is supposed to show a
genuinely bad candidate being caught).

Root cause: the whole post-shock drift buffer still contains substantial
normal-like signal (only 2 of 5 issuers were degraded by the injected
shock), so "no replay buffer" alone was not a strong enough perturbation to
reliably induce forgetting — overfitting to a buffer that's still ~60%
normal-looking data often just reproduces normal-regime behavior.

Fix: `scripts/demo_part3.py`'s bad-challenger step was changed to train on
only the most recent **25%** of the drift buffer (a narrower, less diverse,
more heavily-degraded-issuer-weighted slice), with higher adapter capacity
(`bottleneck_dim=64` vs. 16), more epochs (300 vs. 120), a higher learning
rate (8e-3 vs. 5e-3), and zero weight decay — all pushing toward genuine
overfitting to a small, non-representative slice rather than learning
something that generalizes. The historical canary evaluation set was also
widened (using 80% of collected normal-regime traffic instead of 50%) to
give a real forgetting effect more statistical surface area to be detected
against.

This was verified empirically, not assumed: `tests/e2e/test_bad_challenger_rejection_is_seed_robust.py`
runs the full scenario across seeds 0-9 and requires the REJECT decision in
at least 9 of them. **Result after the fix: 10/10 seeds correctly REJECT**
the bad challenger. The shipped default (`configs/tiny.yaml`, seed 42) was
re-verified end-to-end after this fix and passes without any config change.

## 3. Drift-window sizing is sensitive to config

Early attempts at the Part 3 demo used a 60-transaction observation window
per DriftSafe evaluation step, which produced too few sliding windows
(`history_len=32` with a 4-way stride) to pass the `min_sample_count`
threshold — so DriftSafe silently never evaluated any signals and never
detected the injected regime shift. Increasing to 400 transactions per
window fixed this. This is noted here as an integration pitfall for anyone
re-running or re-configuring the demo at different scales.

## 4. World-model uncertainty is currently a rough proxy

The planner's confidence gate uses `latency_log_logvar` as an uncertainty
proxy for the whole decision. This is a simplification: a more complete
implementation would aggregate uncertainty across all predicted heads
(success/fraud/abandon/latency), not just latency. Documented here rather
than silently left as if it were a deliberate design choice.
