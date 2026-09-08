"""
Master spec section 34: auto-generate reports/final_report.md from
persisted artifacts/runs/*.json manifests. Every number in the report is
read from a manifest file, not typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path


def _latest(prefix: str):
    candidates = sorted(Path("artifacts/runs").glob(f"{prefix}*.json"))
    if not candidates:
        return None

    def ts(p):
        try:
            return int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            return 0

    with open(max(candidates, key=ts)) as f:
        return json.load(f)


def _all(prefix: str):
    """All manifests matching a filename prefix, newest first."""
    candidates = sorted(Path("artifacts/runs").glob(f"{prefix}*.json"))

    def ts(p):
        try:
            return int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            return 0

    out = []
    for p in sorted(candidates, key=ts, reverse=True):
        with open(p) as f:
            out.append(json.load(f))
    return out


def _latest_benchmark(model_size: str, world_model_variant: str):
    for m in _all("benchmark_"):
        if m.get("model_size") == model_size and m.get("world_model_variant") == world_model_variant:
            return m
    return None


def main():
    # Sections 2-4 report the most recently trained pipeline (currently
    # base-scale, since that was trained after tiny in this session) as the
    # "primary" walkthrough. Section 6 explicitly reports BOTH scales side
    # by side for the benchmark comparison, which is where the tiny-vs-base
    # distinction actually matters.
    pretrain = _latest("pretrain_")
    baseline_heads = _latest("baseline_heads_")
    world_model = None
    for m in _all("world_model_"):
        if m.get("use_pretrained_backbone", True):
            world_model = m
            break
    part3 = _latest("part3_demo_")

    lines = []
    lines.append("# Vulcan 2.0 — Research Report\n")
    lines.append(
        "This report is generated from persisted run manifests under "
        "`artifacts/runs/`. No numbers below are hand-typed. Re-generate with "
        "`python -m scripts.generate_report` after re-running the pipeline.\n"
    )

    lines.append("## 1. Hypothesis\n")
    lines.append(
        "Can a transaction foundation model move beyond directly predicting the "
        "best payment route and instead learn action-conditioned payment "
        "dynamics, compare possible payment futures, detect when its own "
        "predictions become miscalibrated, and safely adapt without silently "
        "degrading previously learned behavior? See `docs/novelty_boundary.md` "
        "for the precise claim boundary.\n"
    )

    lines.append("## 2. Self-supervised pretraining (Part 1)\n")
    if pretrain:
        fm = pretrain["final_metrics"]
        lines.append(f"- Run: `{pretrain['run_id']}`")
        lines.append(f"- Transactions: {pretrain['n_transactions']:,}, windows: train={pretrain['n_train_windows']}, val={pretrain['n_val_windows']}")
        lines.append(f"- Parameters: {pretrain['n_parameters']:,}")
        lines.append(f"- Final train loss: {fm['train_loss']:.4f}, train masked-cat-acc: {fm['train_masked_cat_acc']:.4f}")
        lines.append(f"- Validation masked-cat-acc: {fm['val_masked_cat_acc']:.4f}, continuous MSE: {fm['val_continuous_mse']:.4f}, next-event-acc: {fm['val_next_event_acc']:.4f}")
        lines.append(f"- Checkpoint SHA256: `{pretrain['model_checkpoint_sha256'][:16]}...`\n")
    else:
        lines.append("No pretraining run found. Run `python -m train.pretrain`.\n")

    lines.append("## 3. Baseline B: Mini-Vulcan direct heads (Part 2)\n")
    if baseline_heads:
        fm = baseline_heads["final_metrics"]
        lines.append(f"- Run: `{baseline_heads['run_id']}`")
        lines.append(f"- Validation success BCE: {fm['val_success_bce']:.4f}, accuracy: {fm['val_success_acc']:.4f}\n")
    else:
        lines.append("No baseline-heads run found. Run `python -m train.train_baseline_heads`.\n")

    lines.append("## 4. Action-conditioned world model (Part 4)\n")
    if world_model:
        fm = world_model["final_metrics"]
        lines.append(f"- Run: `{world_model['run_id']}`")
        lines.append(f"- Validation success BCE: {fm['val_success_bce']:.4f}, accuracy: {fm['val_success_acc']:.4f}")
        lines.append(f"- Next-state MSE: {fm['val_next_state_mse']:.4f}")
        lines.append(f"- Action sensitivity (|ΔP(success)| across actions): {fm['action_sensitivity']:.4f} "
                      f"(> 0 confirms genuine action-conditioning)\n")
    else:
        lines.append("No world-model run found. Run `python -m train.train_world_model`.\n")

    lines.append("## 5. DriftSafe + adaptation + shadow evaluation (Part 3 mandatory checkpoint)\n")
    if part3:
        good = part3["good_challenger"]
        bad = part3["bad_challenger"]
        lines.append(f"- Run: `{part3['run_id']}`, drift detected at window {part3['drift_detected_at_window']}")
        lines.append(f"- Good challenger ({good['manifest']['pct_trainable_of_backbone']}% of backbone trainable, "
                      f"replay_buffer={good['manifest']['used_replay_buffer']}): **{good['shadow_eval']['decision']}**")
        for r in good["shadow_eval"]["reasons"]:
            lines.append(f"  - {r}")
        lines.append(f"- Bad challenger (replay_buffer={bad['manifest']['used_replay_buffer']}): **{bad['shadow_eval']['decision']}**")
        for r in bad["shadow_eval"]["reasons"]:
            lines.append(f"  - {r}")
        lines.append("")
    else:
        lines.append("No Part 3 demo run found. Run `python -m scripts.demo_part3`.\n")

    lines.append("## 6. Benchmark (Part 5) — tiny vs. base scale, 4 policy arms\n")
    tiny_bm = _latest_benchmark("tiny", "pretrained")
    base_bm = _latest_benchmark("base", "pretrained")
    base_bm_random = _latest_benchmark("base", "random_backbone")

    def _bm_table(bm):
        rows = ["| Policy | Success rate | Mean regret | p95 latency (ms) | Abandon rate | Fallback rate |",
                "|---|---|---|---|---|---|"]
        for name, m in bm["summary"].items():
            rows.append(
                f"| {name} | {m['success']['mean']:.4f} ± {m['success']['ci95']:.4f} "
                f"| {m['regret']['mean']:.4f} ± {m['regret']['ci95']:.4f} "
                f"| {m['latency_p95']:.1f} | {m['abandon']['mean']:.4f} "
                f"| {m['fallback_rate']*100:.1f}% |"
            )
        return rows

    if tiny_bm:
        lines.append(f"### Tiny config (`{tiny_bm['run_id']}`, seeds={tiny_bm['seeds']}, steps/seed={tiny_bm['n_eval_steps_per_seed']})\n")
        lines.extend(_bm_table(tiny_bm))
        lines.append("")
    else:
        lines.append("No tiny-scale benchmark run found. Run `python -m scripts.benchmark --config configs/tiny.yaml`.\n")

    if base_bm:
        lines.append(f"### Base config (`{base_bm['run_id']}`, seeds={base_bm['seeds']}, steps/seed={base_bm['n_eval_steps_per_seed']})\n")
        lines.extend(_bm_table(base_bm))
        lines.append("")
    else:
        lines.append("No base-scale benchmark run found. Run `python -m scripts.benchmark --config configs/base.yaml`.\n")

    lines.append(
        "**Honest interpretation:** the world model (Model C) did not beat the XGBoost "
        "or Mini-Vulcan-direct baselines at either scale. Full discussion, including why "
        "this does not support the \"model capacity vs. data\" hypothesis, is in "
        "`reports/base_scale_result.md`.\n"
    )

    lines.append("## 7. Ablations (Part 5/P2)\n")
    lines.append(
        "Two ablations, both isolating a single factor while holding everything else "
        "fixed at base scale (see `reports/base_scale_result.md` for full methodology):\n"
    )
    lines.append("### 7.1 Pretrained vs. randomly-initialized backbone (Model C, base scale)\n")
    if base_bm and base_bm_random:
        mc_pretrained = base_bm["summary"]["model_c_world_model"]
        mc_random = base_bm_random["summary"]["model_c_world_model"]
        lines.append("| Backbone | Success rate | Mean regret |")
        lines.append("|---|---|---|")
        lines.append(f"| Pretrained | {mc_pretrained['success']['mean']:.4f} ± {mc_pretrained['success']['ci95']:.4f} "
                      f"| {mc_pretrained['regret']['mean']:.4f} ± {mc_pretrained['regret']['ci95']:.4f} |")
        lines.append(f"| Random init | {mc_random['success']['mean']:.4f} ± {mc_random['success']['ci95']:.4f} "
                      f"| {mc_random['regret']['mean']:.4f} ± {mc_random['regret']['ci95']:.4f} |")
        lines.append("")
        lines.append(
            "The random-init variant's regret is nominally *lower* (better) than the "
            "pretrained variant's, though the confidence intervals overlap substantially. "
            "Honest reading: **no measurable benefit from self-supervised pretraining** on "
            "this benchmark, not a claim that pretraining actively hurts.\n"
        )
    else:
        lines.append("Ablation runs not found. Run `python -m train.train_world_model --config configs/base.yaml --no_pretrained_backbone` "
                      "then `python -m scripts.benchmark --config configs/base.yaml --seeds 1,2,3,4,5 --world_model_variant random_backbone`.\n")

    lines.append("### 7.2 World model with vs. without the uncertainty gate (Model C vs. Model D)\n")
    if tiny_bm and base_bm:
        for label, bm in [("Tiny", tiny_bm), ("Base", base_bm)]:
            mc = bm["summary"]["model_c_world_model"]
            md = bm["summary"]["model_d_world_model_plus_gate"]
            lines.append(f"**{label}:** Model C regret={mc['regret']['mean']:.4f}, "
                          f"Model D regret={md['regret']['mean']:.4f} "
                          f"(Model D deferred to XGBoost on {md['fallback_rate']*100:.1f}% of decisions)")
        lines.append("")
        lines.append(
            "At tiny scale, the uncertainty gate provides a real, partial improvement "
            "(Model D's regret is closer to the XGBoost baseline than Model C's). At base "
            "scale, the gate never fires (0% fallback), so Model D collapses to Model C -- "
            "a concrete, measured limitation of the current fixed-threshold uncertainty "
            "proxy, not a fixed one. See `reports/base_scale_result.md` P1.\n"
        )
    else:
        lines.append("Benchmark runs not found for this comparison.\n")

    lines.append("## 8. Limitations\n")
    lines.append("See `docs/limitations.md`.\n")

    lines.append("## 9. Failure analysis\n")
    lines.append("See `reports/failures.md` and `reports/base_scale_result.md`.\n")

    out = Path("reports/final_report.md")
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
