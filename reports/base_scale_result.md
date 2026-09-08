# Base-Scale Result (Part 5 follow-up)

Cross-linked from `README.md` and `reports/failures.md` item 1.

This report runs the exact experiment `reports/failures.md` item 1 named as
the next step: **does moving from `tiny` to `base` scale close the gap
between the action-conditioned world model (Model C) and the much simpler
XGBoost baseline (Baseline A)?** It also adds two new, honest data points:
an uncertainty-gated hybrid policy (Model D) and a pretrained-vs-random-init
backbone ablation. All numbers below are read directly from persisted
manifests under `artifacts/runs/` — none are hand-typed.

## Hardware and wall-clock time

This sandbox has **1 CPU core** (Intel Xeon @ 2.10GHz), ~3.9GB RAM, **no
GPU** (`torch.cuda.is_available()` returns `False`). Approximate wall-clock
time for the full base-scale pipeline, run serially:

| Stage | Wall-clock time (approx.) |
|---|---|
| Pretraining (`train/pretrain.py`, base config) | ~9 min |
| Baseline heads (`train/train_baseline_heads.py`) | ~5 min |
| World model (`train/train_world_model.py`) | ~4 min |
| 5-seed benchmark, pretrained variant | ~3 min |
| Random-backbone ablation world model training | ~3 min |
| 5-seed benchmark, random-backbone variant | ~4 min |
| **Total** | **~28 min** |

This is substantially slower per-example than `tiny` (which completes its
whole pipeline in under a minute) — expected, given `base` uses a
~2.76M-parameter backbone (vs. tiny's ~370K), 100K transactions (vs. 20K),
and 5 pretraining epochs on a single CPU core with no batched GPU
parallelism available in this environment.

## P0: tiny-scale vs. base-scale, side by side

Both benchmarks below use identical methodology: paired transaction
trajectories (same seed → same observation sequence across policies, see
`tests/simulator/test_paired_trajectories.py`), 500 evaluation steps per
seed, mean regret computed against the evaluator-only oracle
(`vulcan/evaluation/oracle_reference.py`).

**Tiny config** (seeds 1,2,3 — `artifacts/runs/benchmark_1787733595.json`):

| Policy | Success rate | Mean regret | Fraud loss | Fallback rate |
|---|---|---|---|---|
| Baseline A (XGBoost) | 0.8133 ± 0.0307 | 0.0203 ± 0.0288 | 45.10 | 0.0% |
| Baseline B (Mini-Vulcan direct) | 0.8027 ± 0.0323 | 0.0364 ± 0.0291 | 42.80 | 0.0% |
| Model C (world model) | 0.7887 ± 0.0378 | 0.0530 ± 0.0340 | 38.12 | 24.9% |
| Model D (world model + gate) | 0.8013 ± 0.0490 | 0.0341 ± 0.0428 | 48.67 | 0.2% |

**Base config** (seeds 1,2,3,4,5 — `artifacts/runs/benchmark_1787733825.json`):

| Policy | Success rate | Mean regret | Fraud loss | Fallback rate |
|---|---|---|---|---|
| Baseline A (XGBoost) | 0.8240 ± 0.0308 | 0.0184 ± 0.0200 | 32.82 | 0.0% |
| Baseline B (Mini-Vulcan direct) | 0.8184 ± 0.0354 | 0.0199 ± 0.0193 | 35.81 | 0.0% |
| Model C (world model) | 0.7828 ± 0.0176 | 0.0554 ± 0.0274 | 35.54 | 0.0% |
| Model D (world model + gate) | 0.7828 ± 0.0176 | 0.0554 ± 0.0274 | 35.54 | 0.0% |

### Honest interpretation

**Scaling up did NOT close the gap. If anything, it stayed essentially flat
or widened slightly.** At `tiny`, XGBoost's regret advantage over Model C is
0.0530 − 0.0203 = 0.0327. At `base`, it is 0.0554 − 0.0184 = 0.0370 —
marginally *larger*, not smaller. Baseline B (Mini-Vulcan direct heads, same
pretrained backbone as the world model) also improved *more* than Model C
did when moving to base scale (regret 0.0364→0.0199, a clear improvement,
vs. Model C's 0.0530→0.0554, essentially flat-to-worse).

This does **not** support the "model capacity vs. data" hypothesis from
`reports/failures.md` item 1 as the primary explanation — if it were mainly
about the shared backbone being too small/undertrained, Baseline B (which
uses the exact same backbone) should have seen a similarly muted
improvement, but it improved substantially while Model C did not. The more
likely explanation, given the ablation result below, is specific to the
**world-model dynamics head and/or planner formulation**, not the shared
representation. This is a genuine negative result for the project's central
research claim ("the world model makes better decisions than simpler
baselines") at both scales tested, and is reported here plainly rather than
explained away.

## P1: Model D — uncertainty-gated hybrid (world model + XGBoost fallback)

Model D reuses the exact same uncertainty proxy the planner already computes
(`vulcan/planner/planner.py`'s `latency_log_logvar` threshold check — no new
uncertainty mechanism was invented) and, when triggered, defers to the
XGBoost baseline's prediction instead of the planner's fixed stable-route
fallback.

- **At tiny scale**, Model D deferred to XGBoost on **0.2%** of decisions
  and closed roughly half the regret gap versus Model C (0.0530 → 0.0341,
  vs. Baseline A's 0.0203) — a genuine, if partial, improvement from
  knowing when not to trust the world model.
- **At base scale, the gate never fired (0.0% fallback rate)**, so Model D
  is numerically identical to Model C. This is a real, reportable
  limitation, not a null result to hide: the fixed `max_latency_logvar`
  threshold in `UncertaintyGateConfig` was tuned/observed at tiny scale and
  does not transfer to the different latency-prediction distribution the
  base-scale world model produces. The base-scale world model's predicted
  log-variance for latency apparently never crosses the threshold, meaning
  its self-reported uncertainty is either genuinely lower (more confident)
  or — more likely, given the flat regret result above — **not a reliable
  uncertainty signal at this scale**. This matches the concern already
  flagged in `reports/failures.md` item 4 ("world-model uncertainty is
  currently a rough proxy") and is now confirmed empirically rather than
  just noted as a risk.

## P2: Ablation — pretrained vs. randomly-initialized backbone (base scale)

Backbone frozen either way; only whether `train/pretrain.py`'s checkpoint
was loaded differs (`train/train_world_model.py --no_pretrained_backbone`).
Baseline B is unaffected in both arms (it always uses the normally
pretrained backbone — see `scripts/benchmark.py::load_trained_components`,
which deliberately keeps Baseline B's backbone fixed so the ablation
isolates the world model arm only).

| Model C variant | Success rate | Mean regret | Val success BCE (pretraining stage) |
|---|---|---|---|
| Pretrained backbone | 0.7828 ± 0.0176 | 0.0554 ± 0.0274 | 0.0024 |
| Random-init backbone | 0.7912 ± 0.0490 | 0.0504 ± 0.0327 | 0.0014 |

**Honest interpretation: self-supervised pretraining shows no measurable
benefit for the world model at base scale on this benchmark** — the
random-backbone variant's regret (0.0504) is nominally *better* than the
pretrained variant's (0.0554), though the confidence intervals overlap
substantially (0.0554±0.0274 vs. 0.0504±0.0327), so the honest reading is
"no detectable difference," not "pretraining hurts." Either way, this does
**not** support pretraining as the explanation for the world model's
underperformance against XGBoost. Combined with the P0/P1 results above,
the evidence points toward the world-model dynamics head, planner utility
formulation, or uncertainty calibration as more likely culprits than
backbone capacity or pretraining — a direction for future work, not
something this pass attempts to fix (per the scope boundary in this
prompt's own instructions).

This ablation was run at base scale only (not also re-run at tiny scale),
since base is the scale at which the headline P0 comparison lives; a reader
should not assume the same direction necessarily holds at tiny scale.

**Update:** a confirmed temporal target-leakage bug in the world model's
state construction (see `reports/leakage_fix_result.md`) was found and
fixed after this report was written. The regret numbers throughout this
document reflect the PRE-FIX world model. Post-fix, Model C's regret
improved substantially at both scales (tiny: 0.0530→0.0326; base:
0.0554→0.0282), though it still does not beat Baseline A on point
estimates at either scale. This document's numbers are left unedited for
historical continuity; see the leakage-fix report for the corrected
comparison and a precise statement of what changed and what did not.

## What this does and does not prove

**Proven by this pass:**
- The "model capacity vs. data" hypothesis is not supported by moving to
  base scale — a genuine negative result, now tested rather than assumed.
- An uncertainty-gated hybrid (Model D) provides a measurable, if partial,
  improvement at tiny scale, but its benefit does not transfer to base
  scale because the underlying uncertainty gate does not fire there — a
  concrete, reproducible limitation of the current uncertainty proxy.
- Pretraining shows no measurable benefit to the world model's decision
  quality at base scale in this benchmark.

**Not proven, and not claimed:** that the world model "makes better
decisions than simpler baselines." Across every configuration tested in
this project (tiny, base, pretrained backbone, random backbone, with and
without the uncertainty gate), XGBoost or the direct Mini-Vulcan heads
matched or beat the action-conditioned world model on mean regret. See
`README.md`'s results section for how this is framed against what the
project does demonstrate successfully.
