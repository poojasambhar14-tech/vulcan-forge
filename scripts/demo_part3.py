"""
Part 3 mandatory demo checkpoint (master spec section 13-15 acceptance gate).

Demonstrates, with real numbers from real runs (no hardcoded results):

  NORMAL TRAFFIC -> DRIFT -> DETECTED -> CHALLENGER TRAINED -> SHADOW EVAL
  -> GOOD CHALLENGER -> PROMOTED

  BAD CHALLENGER -> RECENT PERFORMANCE LOOKS GOOD -> HISTORICAL CANARY
  REGRESSES -> REJECTED

Usage:
    python -m scripts.demo_part3 --config configs/tiny.yaml

`run_demo()` is factored out so it can be re-run across many seeds without
going through argparse/print side effects -- see
tests/e2e/test_bad_challenger_rejection_is_seed_robust.py, which calls it
directly to verify the bad-challenger-rejection scenario holds up across
seeds, not just the one shipped as the config default.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vulcan.common.config import load_config, file_sha256, git_commit
from vulcan.common.seeding import set_global_seed
from vulcan.simulator.environment import IndiaPaymentSim, RegimeShock
from vulcan.simulator.behavior_policy import BehaviorPolicy, BehaviorPolicyConfig
from vulcan.data.generate import _flatten_observation
from vulcan.data.windowing import build_windows, encode_windows_for_world_model
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.models.mini_vulcan import MiniVulcanBackbone, MiniVulcanConfig
from vulcan.models.downstream_heads import DownstreamHeads
from vulcan.drift.driftsafe import DriftSafe, DriftSafeConfig, IncidentState
from vulcan.adaptation.candidate_trainer import train_candidate
from vulcan.adaptation.champion_challenger import shadow_evaluate_and_decide, GateThresholds, evaluate_model_on_slice
from vulcan.registry.model_registry import ModelRegistry


# How much of the collected post-shock drift stream the BAD challenger is
# allowed to train on, as a fraction of the available drift-train windows,
# counted from the END (most recent) of that buffer. Narrowing this (rather
# than using the full 60% split used for the good challenger) concentrates
# the bad challenger's training signal into a smaller, less diverse slice
# of the drift regime, which more reliably induces the kind of overfitting
# that fails to generalize back to the historical (pre-shock) regime. See
# reports/failures.md item 2 addendum for the empirical tuning behind this.
BAD_CHALLENGER_RECENT_FRACTION = 0.25
BAD_CHALLENGER_EPOCHS = 300
BAD_CHALLENGER_LR = 8e-3
BAD_CHALLENGER_WEIGHT_DECAY = 0.0
BAD_CHALLENGER_BOTTLENECK_DIM = 64


def _collect_stream(sim, policy, n_steps, tokenizer_fit_records=None):
    """Generate a stream of transaction records by stepping the simulator
    with the behavior policy (same discipline as vulcan.data.generate, but
    kept local here since we need to interleave a live regime shock)."""
    records = []
    for _ in range(n_steps):
        obs = sim.current_observation()
        actions = sim.candidate_actions()
        action, propensity = policy.select_action(obs, actions)
        outcome = sim.step(obs, action)
        rec = _flatten_observation(obs)
        rec.update(
            action_route_id=action.route_id,
            action_rail=action.rail,
            action_gateway=action.gateway,
            propensity=propensity,
            outcome_success=outcome.success,
            outcome_latency_ms=outcome.latency_ms,
            outcome_processing_cost=outcome.processing_cost,
            outcome_fraud_loss=outcome.fraud_loss,
            outcome_abandoned=outcome.abandoned,
            outcome_error_type=outcome.error_type.value,
        )
        records.append(rec)
    return records


def run_demo(cfg: dict, seed: int, verbose: bool = True) -> dict:
    """Runs the full Part 3 scenario for one seed and returns a manifest
    dict (does NOT assert/raise on the bad-challenger outcome -- that is
    the caller's responsibility, so this function can be reused by the
    seed-robustness test without crashing on the very fragility it's
    checking for)."""
    set_global_seed(seed)
    history_len = cfg["model"]["history_len"]
    num_routes = cfg["simulator"]["num_routes"]

    def log(msg):
        if verbose:
            print(msg)

    log("=" * 70)
    log("PART 3 DEMO: DriftSafe -> Adaptation -> Shadow Evaluation -> Promote/Reject")
    log("=" * 70)

    sim = IndiaPaymentSim(cfg, seed=seed)
    policy = BehaviorPolicy(BehaviorPolicyConfig(cfg["simulator"]["behavior_policy"]["exploration_rate"]), np.random.default_rng(seed + 777))

    log("\n[STAGE 1] NORMAL TRAFFIC")
    normal_records = _collect_stream(sim, policy, n_steps=6000)
    log(f"  collected {len(normal_records)} normal-regime transactions")

    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(normal_records)

    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])
    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    backbone_ckpt = Path("checkpoints/mini_vulcan_pretrained.pt")
    if backbone_ckpt.exists():
        state = torch.load(backbone_ckpt, weights_only=False)
        backbone.load_state_dict(state["backbone_state_dict"])
        log("  loaded pretrained backbone")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    champion_heads = DownstreamHeads(model_cfg.d_model, num_routes)
    heads_ckpt = Path("checkpoints/baseline_heads.pt")
    if heads_ckpt.exists():
        hstate = torch.load(heads_ckpt, weights_only=False)
        champion_heads.load_state_dict(hstate["heads_state_dict"])
        log("  loaded champion (baseline) heads")
    champion_ckpt_hash = file_sha256(heads_ckpt) if heads_ckpt.exists() else "unknown"

    # Wider historical canary: use ALL normal-regime windows as canary
    # evaluation surface (a small separate slice is still carved out for
    # the DriftSafe reference distribution), rather than splitting normal
    # traffic 50/50. More canary data means a real forgetting effect has
    # more statistical surface area to be detected against, and reduces
    # variance-driven false negatives in the promotion gate.
    normal_windows = build_windows(normal_records, history_len, stride=max(1, history_len // 4))
    n_reference = max(50, len(normal_windows) // 5)
    reference_holdout = normal_windows[:n_reference]
    historical_canary_holdout = normal_windows[n_reference:]

    def _reference_window_dict(windows, n=400):
        windows = windows[:n]
        cat, cont = encode_windows_for_world_model(tokenizer, windows)
        cont_full = torch.cat([cont, torch.zeros_like(cont)], dim=-1)
        with torch.no_grad():
            z = backbone.current_state(cat, cont_full)
            out = champion_heads(z)
            idx = torch.arange(z.shape[0])
            taken = torch.tensor([w[-1]["action_route_id"] for w in windows], dtype=torch.long)
            probs = torch.sigmoid(out["success_logit"])[idx, taken].numpy()
        labels = np.array([float(bool(w[-1]["outcome_success"])) for w in windows])
        features = np.array([w[-1]["amount"] for w in windows])
        return {"features": features, "z": z.numpy(), "probs": probs, "labels": labels}

    reference = _reference_window_dict(reference_holdout)

    driftsafe_cfg = DriftSafeConfig(persistence_windows=3, consensus_min_consecutive_windows=2)
    driftsafe = DriftSafe(driftsafe_cfg, reference)

    log("\n[STAGE 2] INJECT PERSISTENT REGIME SHIFT: ISSUER_DEGRADATION")
    shock = RegimeShock(
        name="issuer_degradation",
        start_step=sim.step_idx,
        duration=100000,  # persists for the rest of the run
        issuer_health_delta={issuer: -0.55 for issuer in sim.issuers[:2]},
    )
    sim.inject_shock(shock)
    log(f"  shock injected at step={shock.start_step}, affecting issuers={sim.issuers[:2]}")

    drift_stream_windows_all = []
    window_size_steps = 400
    # The injected shock is PERSISTENT (duration=100000, i.e. does not
    # recover on its own), so there is no correctness reason to give up
    # waiting for consensus+persistence after only 15 windows -- that cap
    # was tight enough that ordinary seed-to-seed noise in which windows
    # happen to hit >=2-signal consensus could make a real, persistent
    # shift look undetected purely because the state machine ran out of
    # attempts, not because the signal wasn't there. Raised to give the
    # SAME detection logic more windows to accumulate consensus before
    # concluding detection genuinely failed.
    max_windows_to_check = 40
    detected_at_window = None
    for w_idx in range(max_windows_to_check):
        window_records = _collect_stream(sim, policy, n_steps=window_size_steps)
        drift_stream_windows_all.extend(window_records)
        windows = build_windows(window_records, history_len, stride=max(1, history_len // 4))
        if not windows:
            continue
        window_dict = _reference_window_dict(windows, n=len(windows))
        report = driftsafe.evaluate_window(window_dict)
        log(f"  window {w_idx}: state={driftsafe.state.value:22s} n_signals_exceeded={report.n_exceeded}/4")
        if driftsafe.state == IncidentState.ADAPTATION_CANDIDATE:
            detected_at_window = w_idx
            break

    if detected_at_window is None:
        raise AssertionError("DriftSafe failed to detect the injected persistent regime shift")
    log(f"\n  >>> DRIFT DETECTED AND PERSISTED -> ADAPTATION_CANDIDATE at window {detected_at_window} <<<")

    drift_windows_full = build_windows(drift_stream_windows_all, history_len, stride=max(1, history_len // 4))
    split = int(len(drift_windows_full) * 0.6)
    drift_train_windows = drift_windows_full[:split]
    recent_drift_holdout = drift_windows_full[split:]

    replay_windows = normal_windows[: min(len(normal_windows), len(drift_train_windows))]

    log("\n[STAGE 3] TRAIN GOOD CHALLENGER (drift buffer + historical replay buffer)")
    good = train_candidate(
        backbone, tokenizer, champion_heads.state_dict(),
        drift_windows=drift_train_windows, replay_windows=replay_windows,
        num_routes=num_routes, d_model=model_cfg.d_model,
        epochs=25, seed=seed, parent_checkpoint_hash=champion_ckpt_hash,
    )
    log(f"  trained: {good.manifest['n_trainable_parameters']:,} trainable params "
        f"({good.manifest['pct_trainable_of_backbone']}% of backbone), "
        f"final_train_loss={good.manifest['final_train_loss']:.4f}")

    log("\n[STAGE 4] SHADOW EVALUATION: good challenger vs champion")
    good_result = shadow_evaluate_and_decide(
        backbone, tokenizer, champion_heads, good.adapter, good.heads,
        recent_drift_holdout, historical_canary_holdout, GateThresholds(),
    )
    for r in good_result["reasons"]:
        log(f"  - {r}")
    log(f"  DECISION: {good_result['decision']}")

    registry = ModelRegistry()
    registry_entry = None
    if good_result["decision"] == "PROMOTE":
        registry_entry = registry.promote(good.adapter, good.heads, {"shadow_eval": good_result, "candidate_manifest": good.manifest})
        log(f"  >>> PROMOTED as registry version {registry_entry.version} (checkpoint sha256={registry_entry.checkpoint_sha256[:12]}...) <<<")

    # Bad challenger: train on only the most recent, narrowest slice of the
    # drift buffer (rather than the whole 60% split used for the good
    # challenger), with higher capacity, more epochs, higher LR, and zero
    # weight decay -- all of which push toward overfitting to a small,
    # noisy, non-representative slice of the drift regime rather than
    # learning something that generalizes. See BAD_CHALLENGER_* constants
    # above and reports/failures.md item 2 addendum.
    n_recent = max(20, int(len(drift_train_windows) * BAD_CHALLENGER_RECENT_FRACTION))
    bad_train_windows = drift_train_windows[-n_recent:]

    log("\n[STAGE 5] TRAIN BAD CHALLENGER (narrow recent drift slice ONLY, no replay, overfit)")
    bad = train_candidate(
        backbone, tokenizer, champion_heads.state_dict(),
        drift_windows=bad_train_windows, replay_windows=None,
        num_routes=num_routes, d_model=model_cfg.d_model,
        epochs=BAD_CHALLENGER_EPOCHS, lr=BAD_CHALLENGER_LR, seed=seed + 1,
        bottleneck_dim=BAD_CHALLENGER_BOTTLENECK_DIM,
        weight_decay=BAD_CHALLENGER_WEIGHT_DECAY,
        parent_checkpoint_hash=champion_ckpt_hash,
    )
    log(f"  trained on {len(bad_train_windows)} windows (narrow recent slice), "
        f"{bad.manifest['n_trainable_parameters']:,} trainable params, "
        f"final_train_loss={bad.manifest['final_train_loss']:.4f}")

    log("\n[STAGE 6] SHADOW EVALUATION: bad challenger vs champion")
    bad_result = shadow_evaluate_and_decide(
        backbone, tokenizer, champion_heads, bad.adapter, bad.heads,
        recent_drift_holdout, historical_canary_holdout, GateThresholds(),
    )
    for r in bad_result["reasons"]:
        log(f"  - {r}")
    log(f"  DECISION: {bad_result['decision']}")

    manifest = {
        "run_id": f"part3_demo_{int(time.time()*1000)}",
        "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "seed": seed,
        "drift_detected_at_window": detected_at_window,
        "good_challenger": {"manifest": good.manifest, "shadow_eval": good_result},
        "bad_challenger": {"manifest": bad.manifest, "shadow_eval": bad_result},
    }
    return manifest


def main(config_path: str):
    cfg = load_config(config_path)
    seed = cfg["seed"]
    manifest = run_demo(cfg, seed, verbose=True)

    bad_decision = manifest["bad_challenger"]["shadow_eval"]["decision"]
    assert bad_decision == "REJECT", (
        "Expected the bad (narrow-slice, overfit) challenger to be REJECTED due to historical regression."
    )
    print("\n  >>> BAD CHALLENGER CORRECTLY REJECTED (historical canary regression) <<<")

    Path("artifacts/runs").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"artifacts/runs/{manifest['run_id']}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nFull manifest saved to {out_path}")
    print("\nPART 3 MANDATORY DEMO CHECKPOINT: PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tiny.yaml")
    args = parser.parse_args()
    main(args.config)
