"""
SILENT REGRESSION CHALLENGE
===========================

A dedicated adversarial-certification benchmark answering the one question
a technical reviewer will actually ask about Forge:

    "How do you know your red team adds value over ordinary validation?"

Rather than hoping a lucky demo seed produces a dramatic catch, this
benchmark DELIBERATELY constructs the conditions under which silent
regressions occur, then measures -- across seeds, with ground truth --
how often ordinary gates miss them and how often red team catches them.

GROUNDING IN THE LITERATURE
---------------------------
The failure mode being measured is the "negative flip" of Yan et al.,
*Positive-Congruent Training: Towards Regression-Free Model Updates*
(CVPR 2021): a case the OLD model handled correctly that the NEW model
handles incorrectly. Their central empirical finding is that the negative
flip rate stays positive even when average error improves -- i.e. average
metrics are structurally blind to this failure. Forge's red team is an
independent search for exactly those cases.

For a payment ROUTING model the decision is not "will this succeed"
(measured and found degenerate in this simulator: oracle p_success > 0.5
for 100% of sampled transactions, so no binary decision can ever flip) but
"WHICH ROUTE should carry this". So a negative flip here is a ROUTE flip:
the challenger sends a transaction to a route with materially lower true
success probability than the route the champion would have chosen.

EXPERIMENTAL DESIGN
-------------------
Two challenger construction arms per seed, both trained by the SAME real
`candidate_trainer.train_candidate` used in production Forge:

  RISKY  -- adapter trained on a NARROW slice curriculum with NO historical
            replay and a high epoch count. This is a realistic (not
            contrived) failure mode of targeted PEFT repair: the adapter
            over-specializes to the target region and quietly degrades
            routing elsewhere.
  SAFE   -- adapter trained on a BROAD curriculum WITH historical replay at
            a moderate epoch count. Expected not to regress.

Neither the slice used to build the RISKY curriculum, nor the fact that an
arm is "risky", is ever communicated to the red team. Red team searches a
fresh, independently-seeded simulator trajectory and must discover any
regression on its own. This is what makes the recall number meaningful.

GROUND TRUTH
------------
"Does a silent regression actually exist?" is decided independently of both
the standard gates and the red team, by measuring the route-level negative
flip rate on a large held-out probe trajectory (a third, separate seed)
using the simulator's oracle. A challenger is deemed to carry a real
regression if its probe NFR exceeds `GROUND_TRUTH_NFR_THRESHOLD` at a
regret magnitude at least `GROUND_TRUTH_MIN_REGRET`.

REPORTED METRICS
----------------
  silent_regression_rate      how often the adaptation created a real regression
  standard_gate_miss_rate     of those, how often ordinary gates said PROMOTE
  redteam_detection_recall    of those, how often red team found it
  redteam_false_positive_rate on challengers with NO real regression, how
                              often red team wrongly flagged one
  memory_replay_pass_rate     whether a repaired successor passes the stored cases

Run:
    python -m scripts.silent_regression_benchmark --seeds 1,2,3,4,5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from vulcan.common.config import load_config
from vulcan.common.seeding import set_global_seed
from vulcan.data.generate import generate_training_data, _flatten_observation
from vulcan.data.windowing import build_windows
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.models.mini_vulcan import MiniVulcanBackbone, MiniVulcanConfig
from vulcan.models.downstream_heads import DownstreamHeads
from vulcan.adaptation.candidate_trainer import train_candidate
from vulcan.adaptation.champion_challenger import evaluate_model_on_slice
from vulcan.forge.forge_loop import make_predict_fn, wrap_record_as_window
from vulcan.forge.redteam_examiner import run_redteam_search, RedTeamConfig
from vulcan.forge.scenario_validator import ScenarioValidator, ScenarioRegistry
from vulcan.simulator.environment import IndiaPaymentSim

# Ground-truth thresholds, calibrated from the OBSERVED regret distribution
# under realistic operating conditions rather than picked a priori. Measured
# on a 120-transaction realistic probe with a genuinely divergent challenger:
# route-flip regret increases ranged up to ~0.013 oracle success-probability
# units, with the bulk between 0.003 and 0.013. An earlier value of 0.02 was
# set before that distribution was measured and sat ABOVE the entire observed
# range, which is why both ground truth and red team reported zero material
# flips -- the bar was outside the data. 0.005 (half a percentage point of
# true success probability) is inside the observed range, is materially
# costly at payment volume, and still excludes numerical-noise ties.
# The red team's own reporting threshold (RedTeamConfig.severity_threshold_to_report)
# is set to the same value below so that "a regression exists" and "a
# regression is worth reporting" are the same bar.
GROUND_TRUTH_MIN_REGRET = 0.005
GROUND_TRUTH_NFR_THRESHOLD = 0.05   # >5% of probe transactions materially mis-routed
PROBE_SIZE = 120
REDTEAM_SEVERITY_THRESHOLD = 0.005


def _load_champion(cfg):
    records = generate_training_data(cfg, seed=cfg["seed"], n_transactions=4000)
    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(records)
    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])

    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    backbone.load_state_dict(torch.load("checkpoints/mini_vulcan_pretrained.pt", weights_only=False)["backbone_state_dict"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    heads = DownstreamHeads(model_cfg.d_model, cfg["simulator"]["num_routes"])
    heads.load_state_dict(torch.load("checkpoints/baseline_heads.pt", weights_only=False)["heads_state_dict"])
    heads.eval()
    return records, tokenizer, model_cfg, backbone, heads


def measure_route_nfr(cfg, champ_fn, chall_fn, background, probe_seed: int) -> Dict[str, Any]:
    """GROUND TRUTH. Independent probe trajectory; for each transaction,
    ask both models which route they'd pick and compare the true oracle
    success probability of those routes.

    NOTE ON THE STEPPING POLICY: an earlier version of this probe advanced
    the simulator with `sim.step(obs, actions[0])` -- i.e. it routed every
    single probe transaction down route 0. That drove the simulator into a
    degenerate state (route 0's health/congestion collapsing while the
    others stayed fresh), which inflated the spread between routes and
    therefore inflated measured regret by roughly 5x versus a realistic
    trajectory. It reported NFR=0.625 where a realistically-stepped
    trajectory reports near zero. Ground truth must be measured under
    realistic operating conditions, so this now advances the simulator with
    the same exploration behavior policy the rest of the project uses."""
    from vulcan.simulator.behavior_policy import BehaviorPolicy, BehaviorPolicyConfig

    sim = IndiaPaymentSim(cfg, seed=probe_seed)
    policy = BehaviorPolicy(
        BehaviorPolicyConfig(cfg["simulator"]["behavior_policy"]["exploration_rate"]),
        np.random.default_rng(probe_seed + 7),
    )
    flips, n = 0, 0
    regret_gains: List[float] = []
    all_gains: List[float] = []
    examples: List[Dict[str, Any]] = []

    for _ in range(PROBE_SIZE):
        obs = sim.current_observation()
        actions = sim.candidate_actions()
        dist = sim.oracle_outcome_distribution(obs)
        rec = _flatten_observation(obs)
        routes = sorted(dist.keys())

        def choose(fn):
            best_r, best_p = routes[0], -1.0
            for r in routes:
                probe = dict(rec)
                probe["action_route_id"] = r
                p = fn(probe)
                if p > best_p:
                    best_p, best_r = p, r
            return best_r

        champ_r, chall_r = choose(champ_fn), choose(chall_fn)
        best_r = max(routes, key=lambda r: dist[r]["p_success"])
        champ_regret = dist[best_r]["p_success"] - dist[champ_r]["p_success"]
        chall_regret = dist[best_r]["p_success"] - dist[chall_r]["p_success"]
        gain = chall_regret - champ_regret
        all_gains.append(gain)

        if champ_r != chall_r and gain >= GROUND_TRUTH_MIN_REGRET:
            flips += 1
            regret_gains.append(gain)
            if len(examples) < 3:
                examples.append({
                    "amount": rec.get("amount"), "issuer": rec.get("issuer"), "rail": rec.get("rail"),
                    "champion_route": champ_r, "challenger_route": chall_r,
                    "regret_increase": round(gain, 4),
                })
        n += 1
        chosen, _prop = policy.select_action(obs, actions)
        sim.step(obs, chosen)

    nfr = flips / max(1, n)
    return {
        "route_negative_flip_rate": nfr,
        "n_probe": n,
        "n_material_flips": flips,
        "mean_regret_increase": float(np.mean(regret_gains)) if regret_gains else 0.0,
        "max_regret_increase": float(np.max(regret_gains)) if regret_gains else 0.0,
        "max_regret_gain_any": float(np.max(all_gains)) if all_gains else 0.0,
        "has_real_regression": bool(nfr >= GROUND_TRUTH_NFR_THRESHOLD),
        "examples": examples,
    }


def standard_gate_verdict(backbone, tokenizer, champion_heads, candidate,
                           recent_windows, historical_windows, weak_windows) -> Dict[str, Any]:
    """The ordinary (non-red-team) validation a conventional ML pipeline
    would run: aggregate accuracy on recent / historical / target-slice
    holdouts, plus calibration. Deliberately EXCLUDES red team and failure
    memory -- this is the baseline red team is being compared against."""
    champ_recent = evaluate_model_on_slice(backbone, None, champion_heads, tokenizer, recent_windows)
    champ_hist = evaluate_model_on_slice(backbone, None, champion_heads, tokenizer, historical_windows)
    champ_weak = evaluate_model_on_slice(backbone, None, champion_heads, tokenizer, weak_windows)
    ch_recent = evaluate_model_on_slice(backbone, candidate.adapter, candidate.heads, tokenizer, recent_windows)
    ch_hist = evaluate_model_on_slice(backbone, candidate.adapter, candidate.heads, tokenizer, historical_windows)
    ch_weak = evaluate_model_on_slice(backbone, candidate.adapter, candidate.heads, tokenizer, weak_windows)

    criteria = {
        "recent_performance": (ch_recent["accuracy"] or 0) - (champ_recent["accuracy"] or 0) >= -0.02,
        "historical_retention": (ch_hist["accuracy"] or 0) - (champ_hist["accuracy"] or 0) >= -0.03,
        "weak_region_recovery": (ch_weak["accuracy"] or 0) - (champ_weak["accuracy"] or 0) >= 0.0,
        "calibration": (ch_recent["ece"] or 0) - (champ_recent["ece"] or 0) <= 0.05,
    }
    passed = all(criteria.values())
    return {
        "criteria": {k: bool(v) for k, v in criteria.items()},
        "standard_gate_says_promote": bool(passed),
        "champion": {"recent": champ_recent["accuracy"], "historical": champ_hist["accuracy"], "weak": champ_weak["accuracy"]},
        "challenger": {"recent": ch_recent["accuracy"], "historical": ch_hist["accuracy"], "weak": ch_weak["accuracy"]},
    }


def build_arm(arm: str, cfg, backbone, tokenizer, heads, model_cfg,
              records, history_len, seed):
    """RISKY = narrow slice, no replay, high epochs. SAFE = broad + replay,
    moderate epochs. Both use the real production trainer."""
    target_issuer = sorted({r.get("issuer") for r in records if r.get("issuer")})[0]

    if arm == "RISKY":
        slice_records = [r for r in records if r.get("issuer") == target_issuer]
        drift = [wrap_record_as_window(r, records, history_len) for r in slice_records[:120]]
        replay: List = []            # deliberately none -- this is the failure mode
        epochs = 120
    else:
        broad = records[: 400]
        drift = [wrap_record_as_window(r, records, history_len) for r in broad[:120]]
        replay = build_windows(records, history_len, stride=max(1, history_len // 8))[:120]
        epochs = 25

    candidate = train_candidate(
        backbone, tokenizer, heads.state_dict(),
        drift_windows=drift, replay_windows=replay,
        num_routes=cfg["simulator"]["num_routes"], d_model=model_cfg.d_model,
        epochs=epochs, seed=seed,
    )
    return candidate, target_issuer


def run_seed(cfg, seed: int) -> List[Dict[str, Any]]:
    records, tokenizer, model_cfg, backbone, heads = _load_champion(cfg)
    history_len = cfg["model"]["history_len"]

    recent_windows = build_windows(records[-1500:], history_len, stride=max(1, history_len // 4))[:250]
    historical_windows = build_windows(records[:2000], history_len, stride=max(1, history_len // 4))[:250]

    champ_fn = make_predict_fn(backbone, heads, None, tokenizer, records, history_len)
    results = []

    for arm in ("RISKY", "SAFE"):
        t0 = time.time()
        candidate, target_issuer = build_arm(arm, cfg, backbone, tokenizer, heads, model_cfg,
                                              records, history_len, seed)
        weak_records = [r for r in records if r.get("issuer") == target_issuer]
        weak_windows = [wrap_record_as_window(r, records, history_len) for r in weak_records[:200]]

        chall_fn = make_predict_fn(backbone, candidate.heads, candidate.adapter, tokenizer, records, history_len)

        # --- GROUND TRUTH (independent probe seed, unseen by both gates) ---
        truth = measure_route_nfr(cfg, champ_fn, chall_fn, records, probe_seed=seed + 90001)

        # --- ORDINARY VALIDATION ---
        gate = standard_gate_verdict(backbone, tokenizer, heads, candidate,
                                      recent_windows, historical_windows, weak_windows)

        # --- RED TEAM (told nothing about arm or target slice) ---
        validator = ScenarioValidator(tokenizer, cfg)
        registry = ScenarioRegistry()
        discovered, rt_meta = run_redteam_search(
            champ_fn, chall_fn, cfg, validator, registry, seed=seed + 4242,
            cfg=RedTeamConfig(seed_pool_size=250, top_k_to_refine=20,
                              severity_threshold_to_report=REDTEAM_SEVERITY_THRESHOLD),
        )

        results.append({
            "seed": seed, "arm": arm, "target_issuer": target_issuer,
            "ground_truth": truth,
            "standard_gate": gate,
            "redteam": {
                "n_discovered": len(discovered),
                "detected": bool(len(discovered) > 0),
                "metadata": rt_meta,
                "examples": [
                    {k: d.get(k) for k in ("champion_route", "challenger_route", "champion_regret",
                                            "challenger_regret", "regression_severity")}
                    for d in discovered[:3]
                ],
            },
            "wall_time_sec": round(time.time() - t0, 1),
        })
        print(f"  [{arm}] real_regression={truth['has_real_regression']} "
              f"(NFR={truth['route_negative_flip_rate']:.3f}) | "
              f"standard_gate_promote={gate['standard_gate_says_promote']} | "
              f"redteam_found={len(discovered)}  ({results[-1]['wall_time_sec']}s)")

    return results


def summarize(all_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    with_regression = [r for r in all_runs if r["ground_truth"]["has_real_regression"]]
    without_regression = [r for r in all_runs if not r["ground_truth"]["has_real_regression"]]

    risky = [r for r in all_runs if r["arm"] == "RISKY"]
    risky_with_reg = [r for r in risky if r["ground_truth"]["has_real_regression"]]

    missed_by_standard = [r for r in with_regression if r["standard_gate"]["standard_gate_says_promote"]]
    caught_by_redteam = [r for r in with_regression if r["redteam"]["detected"]]
    caught_among_missed = [r for r in missed_by_standard if r["redteam"]["detected"]]
    false_positives = [r for r in without_regression if r["redteam"]["detected"]]

    def rate(num, den):
        return (len(num) / len(den)) if den else None

    return {
        "n_runs": len(all_runs),
        "n_risky_arm": len(risky),
        "silent_regression_creation_rate_risky_arm": rate(risky_with_reg, risky),
        "n_with_real_regression": len(with_regression),
        "n_without_real_regression": len(without_regression),
        "standard_gate_miss_rate": rate(missed_by_standard, with_regression),
        "redteam_detection_recall": rate(caught_by_redteam, with_regression),
        "redteam_recall_on_cases_standard_gate_missed": rate(caught_among_missed, missed_by_standard),
        "redteam_false_positive_rate": rate(false_positives, without_regression),
        "n_caught_by_redteam_that_standard_gate_would_have_shipped": len(caught_among_missed),
    }


def main(config_path: str, seeds: List[int]):
    cfg = load_config(config_path)
    set_global_seed(cfg["seed"])
    all_runs: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n=== SEED {seed} ===")
        all_runs.extend(run_seed(cfg, seed))

    summary = summarize(all_runs)
    print("\n" + "=" * 78)
    print("SILENT REGRESSION CHALLENGE — SUMMARY")
    print("=" * 78)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<58} {v:.3f}")
        else:
            print(f"  {k:<58} {v}")
    print("=" * 78)

    run_id = f"silent_regression_{int(time.time())}"
    Path("artifacts/runs").mkdir(parents=True, exist_ok=True)
    out = Path(f"artifacts/runs/{run_id}.json")
    with open(out, "w") as f:
        json.dump({
            "run_id": run_id, "config_path": config_path, "seeds": seeds,
            "ground_truth_thresholds": {
                "min_regret": GROUND_TRUTH_MIN_REGRET,
                "nfr_threshold": GROUND_TRUTH_NFR_THRESHOLD,
                "probe_size": PROBE_SIZE,
            },
            "runs": all_runs, "summary": summary,
        }, f, indent=2, default=str)
    print(f"\nManifest: {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/tiny.yaml")
    ap.add_argument("--seeds", type=str, default="1,2,3")
    args = ap.parse_args()
    main(args.config, [int(s) for s in args.seeds.split(",")])
