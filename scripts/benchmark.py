"""
Part 5: benchmark (master spec sections 17-21).

Compares, on IDENTICAL paired transaction sequences (same seed, same
observations, per test_paired_trajectories):

  Baseline A -- XGBoost, flat engineered features, greedy on P(success)
  Baseline B -- Mini-Vulcan direct per-route heads, greedy utility
  Model   C -- action-conditioned world model + deterministic planner
  Oracle    -- evaluator-only best action (upper bound, never used in training)

Metrics: payment success rate, fraud loss, p95 latency, abandonment rate,
mean regret vs oracle. Run across multiple seeds; report mean/std/95% CI.
Every number here comes from executing the policy against the simulator and
reading back `sim.step()`'s realized outcome -- nothing is hand-entered.

Usage:
    python -m scripts.benchmark --config configs/tiny.yaml --seeds 1,2,3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vulcan.common.config import load_config, git_commit
from vulcan.common.checkpoint_naming import ckpt_name
from vulcan.common.seeding import set_global_seed
from vulcan.simulator.environment import IndiaPaymentSim, RouteAction
from vulcan.simulator.behavior_policy import BehaviorPolicy, BehaviorPolicyConfig
from vulcan.data.generate import generate_training_data, _flatten_observation
from vulcan.data.chronological_split import chronological_split
from vulcan.data.windowing import build_windows, encode_windows_for_world_model
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.models.mini_vulcan import MiniVulcanBackbone, MiniVulcanConfig
from vulcan.models.downstream_heads import DownstreamHeads, greedy_utility_select
from experimental.world_model.dynamics import ActionConditionedWorldModel
from experimental.world_model.planner import plan_decision, PlannerWeights, PlannerConstraints, UncertaintyGateConfig
from vulcan.evaluation.baselines import XGBoostBaseline
from vulcan.evaluation.oracle_reference import oracle_best_action_value, compute_regret


def _utility_fn(weights: PlannerWeights):
    def fn(route_id, dist_entry):
        return (
            weights.w_success * dist_entry["p_success"]
            - weights.w_fraud * dist_entry["p_fraud"]
        )
    return fn


def load_trained_components(cfg, train_records, world_model_variant: str = "pretrained"):
    """world_model_variant:
      'pretrained' -- normal Model C/D, backbone loaded from train/pretrain.py's checkpoint
      'random_backbone' -- P2 ablation: ONLY the world model's backbone is
          left at random init; Baseline B (Mini-Vulcan direct heads) still
          uses the normally pretrained backbone, since its heads were
          trained against that specific embedding space and would produce
          meaningless output if fed embeddings from an unrelated random
          backbone. Two separate backbone instances are returned so the
          ablation isolates the world model arm only, per spec item P2.1.
    """
    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(train_records)
    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])
    num_routes = cfg["simulator"]["num_routes"]
    size = cfg["model"]["size"]

    # backbone used by baseline_a/baseline_b -- ALWAYS the normally
    # pretrained one, regardless of world_model_variant.
    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    bstate = torch.load(Path("checkpoints") / ckpt_name("mini_vulcan_pretrained", size), weights_only=False)
    backbone.load_state_dict(bstate["backbone_state_dict"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # backbone used by model_c/model_d -- swappable for the P2 ablation.
    if world_model_variant == "pretrained":
        world_model_backbone = backbone
        world_model_ckpt_name = ckpt_name("world_model", size)
    elif world_model_variant == "random_backbone":
        world_model_backbone = MiniVulcanBackbone(tokenizer, model_cfg)
        rand_backbone_path = Path("checkpoints") / ckpt_name("mini_vulcan_backbone_random", size)
        rstate = torch.load(rand_backbone_path, weights_only=False)
        world_model_backbone.load_state_dict(rstate["backbone_state_dict"])
        world_model_backbone.eval()
        for p in world_model_backbone.parameters():
            p.requires_grad = False
        world_model_ckpt_name = ckpt_name("world_model_random_backbone", size)
    else:
        raise ValueError(f"unknown world_model_variant: {world_model_variant}")

    heads = DownstreamHeads(model_cfg.d_model, num_routes)
    hstate = torch.load(Path("checkpoints") / ckpt_name("baseline_heads", size), weights_only=False)
    heads.load_state_dict(hstate["heads_state_dict"])
    heads.eval()

    world_model = ActionConditionedWorldModel(model_cfg.d_model, num_routes)
    wstate = torch.load(Path("checkpoints") / world_model_ckpt_name, weights_only=False)
    world_model.load_state_dict(wstate["world_model_state_dict"])
    world_model.eval()

    return tokenizer, backbone, world_model_backbone, heads, world_model, model_cfg, num_routes


def run_one_seed(cfg, seed: int, n_eval_steps: int, xgb_model: XGBoostBaseline,
                  tokenizer, backbone, world_model_backbone, heads, world_model, model_cfg, num_routes,
                  history_buffer_source, policy_names):
    results = {name: {"success": [], "latency": [], "fraud": [], "abandon": [], "regret": [], "used_fallback": []}
               for name in policy_names}

    weights = PlannerWeights()
    planner_constraints = PlannerConstraints()
    uncertainty_cfg = UncertaintyGateConfig()

    for policy_name in policy_names:
        sim = IndiaPaymentSim(cfg, seed=seed)
        rolling_history = list(history_buffer_source)  # warm start history for windowing
        for _ in range(n_eval_steps):
            obs = sim.current_observation()
            obs_record = _flatten_observation(obs)
            route_specs = [
                {"route_id": r, "rail": sim.route_rail[r], "gateway": sim.route_gateway[r]}
                for r in range(num_routes)
            ]
            used_fallback = False

            if policy_name == "baseline_a_xgboost":
                chosen_route = xgb_model.choose_route(obs_record, route_specs)
            elif policy_name == "baseline_b_mini_vulcan":
                window = (rolling_history[-(model_cfg.history_len - 1):] + [obs_record])
                window = window[-model_cfg.history_len:]
                if len(window) < model_cfg.history_len:
                    window = [window[0]] * (model_cfg.history_len - len(window)) + window
                cat, cont = encode_windows_for_world_model(tokenizer, [window])
                cont_full = torch.cat([cont, torch.zeros_like(cont)], dim=-1)
                with torch.no_grad():
                    z = backbone.current_state(cat, cont_full)
                    out = heads(z)
                    chosen_route = int(greedy_utility_select(out, {"w_success": weights.w_success, "w_fraud": weights.w_fraud,
                                                                     "w_latency": weights.w_latency, "w_abandon": weights.w_abandon})[0])
            elif policy_name in ("model_c_world_model", "model_d_world_model_plus_gate"):
                window = (rolling_history[-(model_cfg.history_len - 1):] + [obs_record])
                window = window[-model_cfg.history_len:]
                if len(window) < model_cfg.history_len:
                    window = [window[0]] * (model_cfg.history_len - len(window)) + window
                cat, cont = encode_windows_for_world_model(tokenizer, [window])
                cont_full = torch.cat([cont, torch.zeros_like(cont)], dim=-1)
                with torch.no_grad():
                    z = world_model_backbone.current_state(cat, cont_full)
                decision = plan_decision(
                    world_model, z, num_routes, obs.network.rolling_route_success,
                    obs.transaction.retry_count, weights, planner_constraints, uncertainty_cfg,
                )
                if policy_name == "model_c_world_model":
                    # plain Model C: uses the planner's own (fixed-route)
                    # fallback if triggered, same as always.
                    chosen_route = decision["chosen_action"]
                    used_fallback = decision["fallback_triggered"]
                else:
                    # Model D (spec section 10): reuse the SAME uncertainty
                    # decision the planner already computed
                    # (`fallback_triggered`, driven by the world model's own
                    # latency-logvar uncertainty proxy) but redirect the
                    # fallback to the XGBoost baseline's prediction instead
                    # of a fixed stable route -- "direct baseline as
                    # fallback" per spec section 10's second listed option.
                    if decision["fallback_triggered"]:
                        chosen_route = xgb_model.choose_route(obs_record, route_specs)
                        used_fallback = True
                    else:
                        chosen_route = decision["chosen_action"]
                        used_fallback = False
            else:
                raise ValueError(f"unknown policy_name: {policy_name}")

            action = RouteAction(chosen_route, sim.route_rail[chosen_route], sim.route_gateway[chosen_route])
            regret = compute_regret(sim, obs, chosen_route, _utility_fn(weights))
            outcome = sim.step(obs, action)

            results[policy_name]["success"].append(float(outcome.success))
            results[policy_name]["latency"].append(outcome.latency_ms)
            results[policy_name]["fraud"].append(outcome.fraud_loss)
            results[policy_name]["abandon"].append(float(outcome.abandoned))
            results[policy_name]["regret"].append(regret)
            results[policy_name]["used_fallback"].append(bool(used_fallback))

            rec = _flatten_observation(obs)
            rec["action_route_id"] = chosen_route
            rolling_history.append(rec)

    return results


def summarize(results_per_seed: dict) -> dict:
    summary = {}
    for policy_name, seed_results in results_per_seed.items():
        metrics = {}
        for metric in ["success", "latency", "fraud", "abandon", "regret"]:
            per_seed_means = [np.mean(r[metric]) for r in seed_results]
            arr = np.array(per_seed_means)
            mean = float(arr.mean())
            std = float(arr.std())
            if len(arr) > 1:
                ci95 = float(1.96 * std / np.sqrt(len(arr)))
            else:
                ci95 = 0.0
            metrics[metric] = {"mean": mean, "std": std, "ci95": ci95, "per_seed": per_seed_means}
        # p95 latency computed pooled across seeds too
        pooled_latency = np.concatenate([r["latency"] for r in seed_results])
        metrics["latency_p95"] = float(np.percentile(pooled_latency, 95))
        # fallback rate: fraction of decisions where this policy deferred to
        # a fallback (Model C: fixed stable route; Model D: XGBoost baseline)
        pooled_fallback = np.concatenate([np.array(r["used_fallback"], dtype=float) for r in seed_results])
        metrics["fallback_rate"] = float(pooled_fallback.mean())
        summary[policy_name] = metrics
    return summary


def main(config_path: str, seeds: list[int], n_eval_steps: int, world_model_variant: str = "pretrained"):
    cfg = load_config(config_path)
    set_global_seed(cfg["seed"])

    print("Generating training data for baselines / feature fitting...")
    train_records = generate_training_data(cfg, seed=cfg["seed"], n_transactions=cfg["data"]["n_transactions"])
    train, val, test = chronological_split(train_records, cfg["data"]["train_frac"], cfg["data"]["val_frac"], cfg["data"]["test_frac"])

    print("Training Baseline A (XGBoost)...")
    xgb_model = XGBoostBaseline(num_routes=cfg["simulator"]["num_routes"])
    xgb_model.fit(train)

    print(f"Loading trained Mini-Vulcan components (world_model_variant={world_model_variant})...")
    tokenizer, backbone, world_model_backbone, heads, world_model, model_cfg, num_routes = load_trained_components(cfg, train, world_model_variant=world_model_variant)

    history_buffer_source = train[-model_cfg.history_len:]

    policy_names = ["baseline_a_xgboost", "baseline_b_mini_vulcan", "model_c_world_model", "model_d_world_model_plus_gate"]

    print(f"Running benchmark over seeds={seeds}, n_eval_steps={n_eval_steps} per seed per policy...")
    results_per_seed = {name: [] for name in policy_names}
    for seed in seeds:
        print(f"  seed={seed}...")
        r = run_one_seed(cfg, seed, n_eval_steps, xgb_model, tokenizer, backbone, world_model_backbone, heads, world_model, model_cfg, num_routes, history_buffer_source, policy_names)
        for name in results_per_seed:
            results_per_seed[name].append(r[name])

    summary = summarize(results_per_seed)

    print("\n" + "=" * 110)
    print(f"{'Policy':<32}{'Success Rate':<18}{'Fraud Loss':<16}{'p95 Latency':<16}{'Abandon Rate':<14}{'Mean Regret':<16}{'Fallback Rate'}")
    print("-" * 110)
    for name, m in summary.items():
        print(
            f"{name:<32}"
            f"{m['success']['mean']:.4f}±{m['success']['ci95']:.4f}  "
            f"{m['fraud']['mean']:.2f}±{m['fraud']['ci95']:.2f}      "
            f"{m['latency_p95']:.1f}ms          "
            f"{m['abandon']['mean']:.4f}      "
            f"{m['regret']['mean']:.4f}±{m['regret']['ci95']:.4f}  "
            f"{m['fallback_rate']*100:.1f}%"
        )
    print("=" * 110)

    manifest = {
        "run_id": f"benchmark_{int(time.time())}",
        "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "config_path": config_path,
        "model_size": cfg["model"]["size"],
        "world_model_variant": world_model_variant,
        "seeds": seeds,
        "n_eval_steps_per_seed": n_eval_steps,
        "summary": summary,
    }
    Path("artifacts/runs").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"artifacts/runs/{manifest['run_id']}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nFull benchmark manifest saved to {out_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tiny.yaml")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--n_eval_steps", type=int, default=500)
    parser.add_argument("--world_model_variant", type=str, default="pretrained", choices=["pretrained", "random_backbone"])
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    main(args.config, seeds, args.n_eval_steps, args.world_model_variant)
