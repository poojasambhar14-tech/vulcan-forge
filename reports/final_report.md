# Vulcan 2.0 — Research Report

This report is generated from persisted run manifests under `artifacts/runs/`. No numbers below are hand-typed. Re-generate with `python -m scripts.generate_report` after re-running the pipeline.

## 1. Hypothesis

Can a transaction foundation model move beyond directly predicting the best payment route and instead learn action-conditioned payment dynamics, compare possible payment futures, detect when its own predictions become miscalibrated, and safely adapt without silently degrading previously learned behavior? See `docs/novelty_boundary.md` for the precise claim boundary.

## 2. Self-supervised pretraining (Part 1)

- Run: `pretrain_1787725534`
- Transactions: 100,000, windows: train=4372, val=934
- Parameters: 2,755,747
- Final train loss: 1.6882, train masked-cat-acc: 0.4815
- Validation masked-cat-acc: 0.4868, continuous MSE: 0.2332, next-event-acc: 0.7658
- Checkpoint SHA256: `97e6add3018c20a1...`

## 3. Baseline B: Mini-Vulcan direct heads (Part 2)

- Run: `baseline_heads_1787726098`
- Validation success BCE: 0.0490, accuracy: 0.9936

## 4. Action-conditioned world model (Part 4)

- Run: `world_model_1787733362`
- Validation success BCE: 0.0024, accuracy: 0.9989
- Next-state MSE: 0.0009
- Action sensitivity (|ΔP(success)| across actions): 0.0044 (> 0 confirms genuine action-conditioning)

## 5. DriftSafe + adaptation + shadow evaluation (Part 3 mandatory checkpoint)

- Run: `part3_demo_1787712033366`, drift detected at window 3
- Good challenger (1.227% of backbone trainable, replay_buffer=True): **PROMOTE**
  - Recent-drift accuracy delta=+0.0506 within tolerance (champion=0.7975 challenger=0.8481)
  - Historical-canary accuracy delta=+0.0435 within tolerance (champion=0.7993 challenger=0.8428)
  - Calibration ECE delta=-0.0896 within tolerance
  - Historical-canary log-loss delta=-0.0390 within tolerance
- Bad challenger (replay_buffer=False): **REJECT**
  - Recent-drift accuracy regressed by 0.1646 (champion=0.7975 challenger=0.6329), exceeds max_recent_regression=0.02
  - Historical-canary accuracy delta=-0.0017 within tolerance (champion=0.7993 challenger=0.7977)
  - Calibration ECE delta=-0.0021 within tolerance
  - Historical-canary log-loss (BCE) regressed by 1.8661 (champion=0.4020 challenger=2.2680), exceeds max_historical_bce_regression=0.03. Accuracy alone did not catch this because success labels are imbalanced; log-loss exposes the challenger's overconfident, miscalibrated predictions on the historical (non-drift) regime -- a sign of catastrophic forgetting.

## 6. Benchmark (Part 5) — tiny vs. base scale, 4 policy arms

### Tiny config (`benchmark_1787733595`, seeds=[1, 2, 3], steps/seed=500)

| Policy | Success rate | Mean regret | p95 latency (ms) | Abandon rate | Fallback rate |
|---|---|---|---|---|---|
| baseline_a_xgboost | 0.8133 ± 0.0307 | 0.0203 ± 0.0288 | 2500.0 | 0.0227 | 0.0% |
| baseline_b_mini_vulcan | 0.8027 ± 0.0323 | 0.0364 ± 0.0291 | 2500.0 | 0.0200 | 0.0% |
| model_c_world_model | 0.7887 ± 0.0378 | 0.0530 ± 0.0340 | 2500.0 | 0.0220 | 24.9% |
| model_d_world_model_plus_gate | 0.8013 ± 0.0490 | 0.0341 ± 0.0428 | 2500.0 | 0.0213 | 0.2% |

### Base config (`benchmark_1787733825`, seeds=[1, 2, 3, 4, 5], steps/seed=500)

| Policy | Success rate | Mean regret | p95 latency (ms) | Abandon rate | Fallback rate |
|---|---|---|---|---|---|
| baseline_a_xgboost | 0.8240 ± 0.0308 | 0.0184 ± 0.0200 | 2500.0 | 0.0192 | 0.0% |
| baseline_b_mini_vulcan | 0.8184 ± 0.0354 | 0.0199 ± 0.0193 | 2500.0 | 0.0188 | 0.0% |
| model_c_world_model | 0.7828 ± 0.0176 | 0.0554 ± 0.0274 | 2500.0 | 0.0212 | 0.0% |
| model_d_world_model_plus_gate | 0.7828 ± 0.0176 | 0.0554 ± 0.0274 | 2500.0 | 0.0212 | 0.0% |

**Honest interpretation:** the world model (Model C) did not beat the XGBoost or Mini-Vulcan-direct baselines at either scale. Full discussion, including why this does not support the "model capacity vs. data" hypothesis, is in `reports/base_scale_result.md`.

## 7. Ablations (Part 5/P2)

Two ablations, both isolating a single factor while holding everything else fixed at base scale (see `reports/base_scale_result.md` for full methodology):

### 7.1 Pretrained vs. randomly-initialized backbone (Model C, base scale)

| Backbone | Success rate | Mean regret |
|---|---|---|
| Pretrained | 0.7828 ± 0.0176 | 0.0554 ± 0.0274 |
| Random init | 0.7912 ± 0.0490 | 0.0504 ± 0.0327 |

The random-init variant's regret is nominally *lower* (better) than the pretrained variant's, though the confidence intervals overlap substantially. Honest reading: **no measurable benefit from self-supervised pretraining** on this benchmark, not a claim that pretraining actively hurts.

### 7.2 World model with vs. without the uncertainty gate (Model C vs. Model D)

**Tiny:** Model C regret=0.0530, Model D regret=0.0341 (Model D deferred to XGBoost on 0.2% of decisions)
**Base:** Model C regret=0.0554, Model D regret=0.0554 (Model D deferred to XGBoost on 0.0% of decisions)

At tiny scale, the uncertainty gate provides a real, partial improvement (Model D's regret is closer to the XGBoost baseline than Model C's). At base scale, the gate never fires (0% fallback), so Model D collapses to Model C -- a concrete, measured limitation of the current fixed-threshold uncertainty proxy, not a fixed one. See `reports/base_scale_result.md` P1.

## 8. Limitations

See `docs/limitations.md`.

## 9. Failure analysis

See `reports/failures.md` and `reports/base_scale_result.md`.
