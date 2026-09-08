# Final Acceptance Gate — Status

**Point-in-time document, pre-Forge.** This checklist was written against
the original "Vulcan 2.0" acceptance gate (master spec section 39), before
Vulcan Forge existed. It is kept as historical record and is not
retroactively rewritten to describe Forge — see `README.md` and
`reports/forge_final_report.md` for Forge's own, current acceptance
evidence. Where a line below is now known to be stale (e.g. test counts),
it is annotated in place rather than silently edited.

Per master spec section 39. Checked items are backed by a real, re-runnable
command and/or a passing test; unchecked items are honestly reported as not
done, not silently skipped.

- [x] simulator is stochastic and reproducible — `tests/simulator/test_environment.py`
- [x] counterfactual oracle is isolated from training — `tests/simulator/test_leakage_prevention.py` (AST-checked)
- [x] chronological splits verified — `tests/unit/test_chronological_split.py`
- [x] XGBoost baseline genuinely runs — `vulcan/evaluation/baselines.py`; protected by `tests/evaluation/test_baselines.py` (imports `XGBoostBaseline` directly, so a clean environment missing the dependency fails collection immediately with `ModuleNotFoundError`; also asserts trained predictions are valid probabilities and non-constant). Exercised end-to-end in `scripts/benchmark.py`.
- [x] Mini-Vulcan genuinely trains — `train/train_baseline_heads.py`, real loss curve in manifest
- [x] self-supervised pretraining genuinely runs — `train/pretrain.py`, real loss curve in manifest
- [x] action-conditioned world model genuinely trains — `train/train_world_model.py`
- [x] changing action can change predicted outcome — asserted in `train/train_world_model.py` (`action_sensitivity > 1e-4`) and `tests/e2e/test_world_model_and_planner.py`
- [x] planner evaluates multiple candidate actions — `vulcan/planner/planner.py`, structured `candidates` list in every decision trace
- [x] chosen prediction is compared against actual simulator outcome — `scripts/benchmark.py` (`compute_regret` against oracle, per real transaction)
- [x] calibration metrics update online — `vulcan/drift/detectors.py` (ECE/Brier), consumed by `vulcan/drift/driftsafe.py`
- [x] persistent drift can be detected — `tests/drift/test_driftsafe.py::test_persistent_shift_triggers_adaptation_candidate`, real detection in `scripts/demo_part3.py`
- [x] short transient shock does not automatically cause retraining — `tests/drift/test_driftsafe.py::test_temporary_shock_does_not_trigger_adaptation`
- [x] parameter-efficient challenger can train — `vulcan/adaptation/candidate_trainer.py` (bottleneck adapter, ~1.2% of backbone trainable)
- [x] challenger runs against champion — `vulcan/adaptation/champion_challenger.py`
- [x] bad challenger is genuinely rejected — `scripts/demo_part3.py` STAGE 6, asserted in the script itself; robustness across seeds verified by `tests/e2e/test_bad_challenger_rejection_is_seed_robust.py` (10/10 seeds reject as of the last run — see `reports/failures.md` addendum)
- [x] successful challenger can genuinely promote — `scripts/demo_part3.py` STAGE 4, asserted
- [x] rollback restores previous model — `tests/e2e/test_rollback_and_adapter.py::test_rollback_restores_exact_previous_checkpoint_hash`
- [x] benchmark is generated from real runs — `scripts/benchmark.py`, no hand-typed numbers
- [x] multiple seeds supported — `scripts/benchmark.py --seeds 1,2,3`
- [x] confidence intervals calculated — 95% CI in `scripts/benchmark.py::summarize`
- [x] frontend has no hardcoded result values — `frontend/index.html` fetches all data from `/api/*`
- [x] README contains no fabricated result — `README.md` reports real numbers or points at manifests
- [x] all tests pass — 30/30 at the time this acceptance gate was originally written, `pytest tests/ -q`. **Stale as of Vulcan Forge**: the suite has since grown to 53 (36 pre-Forge + 17 Forge-specific); see `README.md`'s Tests section for the current count. This line is left as historical record of the original acceptance pass rather than silently updated, per this document's own status as a point-in-time checklist.
- [ ] frontend lint passes — N/A: frontend is static HTML/JS, not a lint-checked framework project (see README's scope note)
- [ ] frontend production build passes — N/A: no build step (static dashboard, not Next.js — see README)
- [x] one-command demo works — `python -m scripts.demo_part3 --config configs/tiny.yaml` runs end-to-end and asserts its own success criteria
- [x] final experiment manifests verify correctly — every `artifacts/runs/*.json` includes a checkpoint SHA256; `ModelRegistry.verify_rollback_hash_matches` tested

## Explicitly not attempted in this build (see README "What is NOT included")

- Full ablation suite (spec section 21) — **partially done**: two of six pairs
  (pretrained-vs-random backbone; world-model-with-vs-without-uncertainty-gate)
  were completed in a follow-up pass, see `reports/final_report.md` section 7
  and `reports/base_scale_result.md`. The remaining four pairs are still not
  attempted.
- TabFormer external validation benchmark (spec section 36)
- Premium Next.js frontend (spec section 24-27) — substituted with a static dashboard
- Windows `.ps1` scripts are written but not executed on Windows (built/tested on Linux only)

## Follow-up research pass (base-scale experiment, Model D, ablations)

Beyond the original acceptance gate, a follow-up pass ran the base-scale
experiment `reports/failures.md` item 1 proposed, added a fourth benchmarked
policy arm (Model D, uncertainty-gated hybrid), and ran two ablations. Full
results: `reports/base_scale_result.md`. Headline honest finding: the
world model still does not beat the simpler baselines at base scale, and
pretraining shows no measurable benefit to it either — a negative result
now confirmed at two scales with an ablation, not just asserted at one.
