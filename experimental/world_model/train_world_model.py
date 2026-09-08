"""
Part 4: train the action-conditioned payment world model.

Core learning target:
    P(outcome, next_state | current_payment_state, candidate_action)

Trained on bandit feedback (only the taken route's outcome is observed per
example), same data-leakage discipline as everywhere else: only
`sim.step()`-realized outcomes are used, never the evaluator-only oracle.

Usage:
    python -m train.train_world_model --config configs/tiny.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vulcan.common.config import load_config, config_hash, file_sha256, git_commit
from vulcan.common.checkpoint_naming import ckpt_name
from vulcan.common.batched_inference import batched_current_state
from vulcan.common.seeding import set_global_seed
from vulcan.data.generate import generate_training_data, assert_no_leaked_columns, _flatten_observation
from vulcan.data.chronological_split import chronological_split
from vulcan.data.windowing import build_windows_with_next, encode_windows_for_world_model
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.models.mini_vulcan import MiniVulcanBackbone, MiniVulcanConfig
from experimental.world_model.dynamics import ActionConditionedWorldModel, gaussian_nll_loss
from vulcan.simulator.environment import IndiaPaymentSim


def run(config_path: str, quick: bool = False, use_pretrained_backbone: bool = True):
    cfg = load_config(config_path)
    seed = cfg["seed"]
    set_global_seed(seed)
    num_routes = cfg["simulator"]["num_routes"]
    history_len = cfg["model"]["history_len"]

    n_txn = cfg["data"]["n_transactions"] if not quick else 2000
    records = generate_training_data(cfg, seed=seed, n_transactions=n_txn)
    assert_no_leaked_columns(records)
    train, val, test = chronological_split(records, cfg["data"]["train_frac"], cfg["data"]["val_frac"], cfg["data"]["test_frac"])

    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(train)

    train_windows, train_next = build_windows_with_next(train, history_len, stride=max(1, history_len // 8))
    val_windows, val_next = build_windows_with_next(val, history_len, stride=max(1, history_len // 4))

    # drop trailing windows with no "next record" (needed for next-state target)
    def filter_valid(windows, nexts):
        pairs = [(w, n) for w, n in zip(windows, nexts) if n is not None]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    train_windows, train_next = filter_valid(train_windows, train_next)
    val_windows, val_next = filter_valid(val_windows, val_next)

    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])
    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    ckpt_path = Path("checkpoints") / ckpt_name("mini_vulcan_pretrained", cfg["model"]["size"])
    if use_pretrained_backbone and ckpt_path.exists():
        state = torch.load(ckpt_path, weights_only=False)
        backbone.load_state_dict(state["backbone_state_dict"])
        print(f"Loaded pretrained backbone from {ckpt_path}")
    elif not use_pretrained_backbone:
        # P2 ablation: backbone stays at its fresh random initialization
        # (never loaded from train/pretrain.py's checkpoint). Save it so
        # scripts/benchmark.py can load the EXACT same random backbone this
        # world model was trained against, rather than trying to
        # reconstruct a matching random init from a seed.
        rand_backbone_path = Path("checkpoints") / ckpt_name("mini_vulcan_backbone_random", cfg["model"]["size"])
        rand_backbone_path.parent.mkdir(exist_ok=True)
        torch.save({"backbone_state_dict": backbone.state_dict()}, rand_backbone_path)
        print(f"[ABLATION] backbone left at random initialization, saved to {rand_backbone_path}")
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    def prep(windows, next_records):
        cat, cont = encode_windows_for_world_model(tokenizer, windows)
        cont_full = torch.cat([cont, torch.zeros_like(cont)], dim=-1)
        with torch.no_grad():
            z = batched_current_state(backbone, cat, cont_full)
        route = torch.tensor([w[-1]["action_route_id"] for w in windows], dtype=torch.long)
        success = torch.tensor([float(bool(w[-1]["outcome_success"])) for w in windows], dtype=torch.float32)
        fraud = torch.tensor([float(w[-1]["outcome_fraud_loss"] > 0) for w in windows], dtype=torch.float32)
        abandon = torch.tensor([float(bool(w[-1]["outcome_abandoned"])) for w in windows], dtype=torch.float32)
        log_latency = torch.tensor([np.log1p(w[-1]["outcome_latency_ms"]) for w in windows], dtype=torch.float32)
        next_state = torch.tensor(
            [n[f"route_{int(route[i])}_rolling_success"] for i, n in enumerate(next_records)], dtype=torch.float32
        )
        return z, route, success, fraud, abandon, log_latency, next_state

    z_train, route_train, success_train, fraud_train, abandon_train, loglat_train, nextstate_train = prep(train_windows, train_next)
    z_val, route_val, success_val, fraud_val, abandon_val, loglat_val, nextstate_val = prep(val_windows, val_next)

    world_model = ActionConditionedWorldModel(model_cfg.d_model, num_routes)
    optimizer = torch.optim.AdamW(world_model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 1 if quick else 15
    batch_size = 64
    n = z_train.shape[0]
    history = []
    for epoch in range(epochs):
        world_model.train()
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            out = world_model(z_train[idx], route_train[idx])
            loss_success = torch.nn.functional.binary_cross_entropy_with_logits(out["success_logit"], success_train[idx])
            loss_fraud = torch.nn.functional.binary_cross_entropy_with_logits(out["fraud_logit"], fraud_train[idx])
            loss_abandon = torch.nn.functional.binary_cross_entropy_with_logits(out["abandon_logit"], abandon_train[idx])
            loss_latency = gaussian_nll_loss(out["latency_log_mean"], out["latency_log_logvar"], loglat_train[idx])
            loss_next_state = torch.nn.functional.mse_loss(out["next_state_pred"], nextstate_train[idx])
            loss = loss_success + loss_fraud + loss_abandon + 0.1 * loss_latency + 0.5 * loss_next_state

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = sum(losses) / len(losses)
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"epoch {epoch+1}/{epochs} loss={mean_loss:.4f}")

    world_model.eval()
    with torch.no_grad():
        out = world_model(z_val, route_val)
        val_success_bce = torch.nn.functional.binary_cross_entropy_with_logits(out["success_logit"], success_val).item()
        val_success_acc = ((torch.sigmoid(out["success_logit"]) > 0.5).float() == success_val).float().mean().item()
        val_next_state_mse = torch.nn.functional.mse_loss(out["next_state_pred"], nextstate_val).item()

        # weak sanity floor (kept as a cheap smoke-test): changing the
        # action must change the predicted outcome AT ALL. This alone does
        # NOT establish the model learned anything meaningful -- a randomly
        # initialized action embedding would also pass it. See the real,
        # oracle-calibrated check below.
        r0 = torch.zeros_like(route_val)
        r1 = torch.ones_like(route_val) % num_routes
        out_r0 = world_model(z_val, r0)
        out_r1 = world_model(z_val, r1) if num_routes > 1 else out_r0
        action_sensitivity = float((torch.sigmoid(out_r0["success_logit"]) - torch.sigmoid(out_r1["success_logit"])).abs().mean())

    print(f"VAL success_bce={val_success_bce:.4f} success_acc={val_success_acc:.4f} next_state_mse={val_next_state_mse:.4f}")
    print(f"Action sensitivity (weak sanity check, |P(success|route=0) - P(success|route=1)| mean): {action_sensitivity:.4f}")
    assert action_sensitivity > 1e-4, "World model predictions do not change with the action -- not action-conditioned!"

    # ---- real, oracle-calibrated action-sensitivity check ----
    # Builds a fresh, realistic evaluation trajectory (own simulator
    # instance + behavior policy, EVALUATOR-ONLY oracle queries), and
    # compares the model's predicted pairwise route-effect sizes against
    # the simulator's TRUE (oracle) pairwise route-effect sizes for the
    # SAME states. This is the check that actually tells you whether the
    # model has learned something real, not just "non-zero."
    from vulcan.evaluation.action_sensitivity import (
        oracle_pairwise_effects, predicted_pairwise_effects, compare_predicted_vs_oracle,
    )
    from vulcan.simulator.behavior_policy import BehaviorPolicy, BehaviorPolicyConfig

    eval_sim = IndiaPaymentSim(cfg, seed=seed + 999)
    eval_policy = BehaviorPolicy(
        BehaviorPolicyConfig(cfg["simulator"]["behavior_policy"]["exploration_rate"]),
        np.random.default_rng(seed + 998),
    )
    eval_records, eval_obs_states = [], []
    n_eval_txns = max(2000, history_len * 30)
    for _ in range(n_eval_txns):
        obs = eval_sim.current_observation()
        actions = eval_sim.candidate_actions()
        action, propensity = eval_policy.select_action(obs, actions)
        outcome = eval_sim.step(obs, action)
        rec = _flatten_observation(obs)
        rec.update(
            action_route_id=action.route_id, action_rail=action.rail, action_gateway=action.gateway,
            propensity=propensity, outcome_success=outcome.success, outcome_latency_ms=outcome.latency_ms,
            outcome_processing_cost=outcome.processing_cost, outcome_fraud_loss=outcome.fraud_loss,
            outcome_abandoned=outcome.abandoned, outcome_error_type=outcome.error_type.value,
        )
        eval_records.append(rec)
        eval_obs_states.append(obs)  # EVALUATOR-ONLY

    eval_stride = max(1, history_len // 4)
    eval_windows, eval_windows_last_obs = [], []
    for start in range(0, len(eval_records) - history_len + 1, eval_stride):
        eval_windows.append(eval_records[start:start + history_len])
        eval_windows_last_obs.append(eval_obs_states[start + history_len - 1])

    oracle_result = oracle_pairwise_effects(eval_sim, eval_windows_last_obs)
    eval_cat, eval_cont = encode_windows_for_world_model(tokenizer, eval_windows)
    eval_cont_full = torch.cat([eval_cont, torch.zeros_like(eval_cont)], dim=-1)
    with torch.no_grad():
        z_eval = batched_current_state(backbone, eval_cat, eval_cont_full)
        predicted_result = predicted_pairwise_effects(world_model, z_eval, num_routes)
    oracle_comparison = compare_predicted_vs_oracle(oracle_result, predicted_result)

    print(
        f"ORACLE-CALIBRATED action sensitivity: predicted_mean_effect={oracle_comparison['predicted_mean_abs_pairwise_effect']:.4f} "
        f"oracle_mean_effect={oracle_comparison['oracle_mean_abs_pairwise_effect']:.4f} "
        f"MAE={oracle_comparison['mae']:.4f} pearson_r={oracle_comparison['pearson_r']:.4f} "
        f"spearman_r={oracle_comparison['spearman_r']:.4f} "
        f"best_route_ranking_acc={oracle_comparison['best_route_ranking_accuracy']:.4f} "
        f"(chance={oracle_comparison['chance_accuracy']:.4f})"
    )

    ckpt_base_name = "world_model" if use_pretrained_backbone else "world_model_random_backbone"
    ckpt_out = Path("checkpoints") / ckpt_name(ckpt_base_name, cfg["model"]["size"])
    torch.save({"world_model_state_dict": world_model.state_dict(), "num_routes": num_routes}, ckpt_out)
    ckpt_hash = file_sha256(ckpt_out)

    manifest = {
        "run_id": f"world_model_{'random_backbone_' if not use_pretrained_backbone else ''}{int(time.time())}",
        "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "seed": seed,
        "config_hash": config_hash(cfg),
        "use_pretrained_backbone": use_pretrained_backbone,
        "checkpoint_sha256": ckpt_hash,
        "history": history,
        "final_metrics": {
            "val_success_bce": val_success_bce,
            "val_success_acc": val_success_acc,
            "val_next_state_mse": val_next_state_mse,
            "action_sensitivity": action_sensitivity,
            "oracle_calibrated_action_sensitivity": oracle_comparison,
        },
    }
    Path("artifacts/runs").mkdir(parents=True, exist_ok=True)
    with open(f"artifacts/runs/{manifest['run_id']}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {ckpt_out} and manifest.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tiny.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no_pretrained_backbone", action="store_true",
                         help="P2 ablation: skip loading the pretrained backbone checkpoint, "
                              "leaving it at random initialization (frozen, same as the normal path).")
    args = parser.parse_args()
    run(args.config, quick=args.quick, use_pretrained_backbone=not args.no_pretrained_backbone)
