"""
Forge benchmark (master spec section 15).

Compares four arms across multiple hidden regime-shift scenarios and
seeds, all under the SAME maximum data/compute budget where applicable:

  A. STATIC          -- no adaptation at all
  B. RANDOM_RETRAIN   -- adapter trained on an equal-size RANDOM sample of
                         recent+historical data (same budget as Forge's
                         curriculum)
  C. DRIFT_RETRAIN    -- DriftSafe-triggered adaptation using the EXISTING
                         (pre-Forge) champion/challenger gate from
                         vulcan/adaptation/champion_challenger.py, trained
                         on all available recent drift data up to the same
                         budget cap
  D. FORGE            -- diagnose -> blind spot -> targeted curriculum ->
                         train -> red-team -> certify (vulcan/forge/forge_loop.py)

SCALE DISCLOSURE: master spec section 15 asks for >=10 regime-shift
scenarios and preferably 5 seeds. Given this environment's 1-CPU-core
constraint (verified earlier in this project's history -- see
reports/base_scale_result.md's hardware section), this benchmark runs
6 scenarios x 3 seeds x 4 arms = 72 full runs, which already takes several
minutes on this hardware. This is fully disclosed here and in
reports/forge_final_report.md, not hidden. The 6 scenarios were chosen to
cover the categories master spec section 15 lists as examples (issuer mix
shift, gateway degradation, rail congestion, fraud prevalence shift, a
compound shift, and a transient spike) rather than an arbitrary subset.

The injected shift's identity is hidden from every arm's diagnosis/decision
logic during the run. It is used ONLY after the fact, by this script, to
score diagnosis accuracy -- never fed into Forge or any baseline.
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
from vulcan.data.windowing import build_windows, encode_windows_for_world_model
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.models.mini_vulcan import MiniVulcanBackbone, MiniVulcanConfig
from vulcan.models.downstream_heads import DownstreamHeads
from vulcan.adaptation.candidate_trainer import train_candidate
from vulcan.adaptation.champion_challenger import evaluate_model_on_slice, shadow_evaluate_and_decide, GateThresholds
from vulcan.drift.driftsafe import DriftSafe, DriftSafeConfig, IncidentState
from vulcan.simulator.environment import IndiaPaymentSim, RegimeShock
from vulcan.simulator.behavior_policy import BehaviorPolicy, BehaviorPolicyConfig
from vulcan.forge.forge_loop import run_forge_cycle, make_predict_fn
from vulcan.forge.curriculum_generator import CurriculumConfig
from vulcan.forge.failure_memory import FailureMemory
from vulcan.forge.schemas import FailureType


BUDGET = 300  # equal max training-example budget for every adaptive arm

SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "issuer_mix_shift",
        "true_failure_type": FailureType.LOCAL_BLIND_SPOT.value,
        "shock": lambda: RegimeShock(name="issuer_degradation", start_step=0, duration=100000,
                                      issuer_health_delta={"HDFC": -0.7}),
    },
    {
        "name": "gateway_degradation",
        "true_failure_type": FailureType.GLOBAL_COVARIATE_SHIFT.value,
        "shock": lambda: RegimeShock(name="gateway_degradation", start_step=0, duration=100000,
                                      gateway_health_delta={"GW_A": -0.6, "GW_B": -0.6, "GW_C": -0.6}),
    },
    {
        "name": "rail_congestion",
        "true_failure_type": FailureType.GLOBAL_COVARIATE_SHIFT.value,
        "shock": lambda: RegimeShock(name="rail_congestion", start_step=0, duration=100000,
                                      congestion_delta=0.5),
    },
    {
        "name": "fraud_prevalence_shift",
        "true_failure_type": FailureType.LABEL_SHIFT.value,
        "shock": lambda: RegimeShock(name="fraud_shift", start_step=0, duration=100000,
                                      fraud_rate_multiplier=6.0),
    },
    {
        "name": "compound_issuer_and_gateway_shift",
        "true_failure_type": FailureType.GLOBAL_COVARIATE_SHIFT.value,
        "shock": lambda: RegimeShock(name="compound", start_step=0, duration=100000,
                                      issuer_health_delta={"HDFC": -0.6}, gateway_health_delta={"GW_A": -0.4}),
    },
    {
        "name": "transient_spike",
        "true_failure_type": FailureType.TRANSIENT_SPIKE.value,
        "shock": lambda: RegimeShock(name="temporary_outage", start_step=0, duration=150,  # SHORT -- recovers
                                      gateway_health_delta={"GW_A": -0.9}),
    },
]


def _window_dict(records, tokenizer, backbone, heads, history_len, n=300):
    # Masked encoding, consistent with the rest of the pipeline.
    windows = build_windows(records, history_len, stride=max(1, history_len // 4))[:n]
    cat, cont = encode_windows_for_world_model(tokenizer, windows)
    cont_full = torch.cat([cont, torch.zeros_like(cont)], dim=-1)
    with torch.no_grad():
        z = backbone.current_state(cat, cont_full)
        out = heads(z)
        idx = torch.arange(z.shape[0])
        taken = torch.tensor([w[-1]["action_route_id"] for w in windows], dtype=torch.long)
        probs = torch.sigmoid(out["success_logit"])[idx, taken].numpy()
    labels = np.array([float(bool(w[-1]["outcome_success"])) for w in windows])
    features = np.array([w[-1]["amount"] for w in windows])
    return {"features": features, "z": z.numpy(), "probs": probs, "labels": labels}


def _collect_stream(sim, policy, n_steps):
    records = []
    for _ in range(n_steps):
        obs = sim.current_observation()
        actions = sim.candidate_actions()
        action, propensity = policy.select_action(obs, actions)
        outcome = sim.step(obs, action)
        rec = _flatten_observation(obs)
        rec.update(action_route_id=action.route_id, action_rail=action.rail, action_gateway=action.gateway,
                   propensity=propensity, outcome_success=outcome.success, outcome_latency_ms=outcome.latency_ms,
                   outcome_processing_cost=outcome.processing_cost, outcome_fraud_loss=outcome.fraud_loss,
                   outcome_abandoned=outcome.abandoned, outcome_error_type=outcome.error_type.value)
        records.append(rec)
    return records


def run_static_arm(champ_recent_eval) -> Dict[str, Any]:
    t0 = time.time()
    return {
        "recovered_accuracy": champ_recent_eval["accuracy"],
        "historical_retention_delta": 0.0,
        "curriculum_size": 0,
        "training_time_sec": time.time() - t0,
        "decision": "NO_ADAPTATION",
    }


def run_random_retrain_arm(backbone, tokenizer, champion_heads, historical_records, recent_records,
                            recent_windows, historical_windows, num_routes, history_len, seed) -> Dict[str, Any]:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    pool = historical_records + recent_records
    idx = rng.choice(len(pool), size=min(BUDGET, len(pool)), replace=False)
    sampled = [pool[i] for i in idx]
    from vulcan.forge.forge_loop import wrap_record_as_window
    train_windows = [wrap_record_as_window(r, historical_records, history_len) for r in sampled]
    replay_windows = build_windows(historical_records, history_len, stride=max(1, history_len // 8))[: len(train_windows)]

    candidate = train_candidate(
        backbone, tokenizer, champion_heads.state_dict(), drift_windows=train_windows, replay_windows=replay_windows,
        num_routes=num_routes, d_model=backbone.cfg.d_model, epochs=25, seed=seed,
    )
    chall_recent = evaluate_model_on_slice(backbone, candidate.adapter, candidate.heads, tokenizer, recent_windows)
    chall_hist = evaluate_model_on_slice(backbone, candidate.adapter, candidate.heads, tokenizer, historical_windows)
    champ_hist = evaluate_model_on_slice(backbone, None, champion_heads, tokenizer, historical_windows)

    return {
        "recovered_accuracy": chall_recent["accuracy"],
        "historical_retention_delta": (chall_hist["accuracy"] or 0) - (champ_hist["accuracy"] or 0),
        "curriculum_size": len(train_windows),
        "training_time_sec": time.time() - t0,
        "decision": "RETRAINED",
    }


def run_drift_retrain_arm(cfg, backbone, tokenizer, champion_heads, historical_records, recent_records,
                           reference_window, current_window, recent_windows, historical_windows,
                           num_routes, history_len, seed) -> Dict[str, Any]:
    t0 = time.time()
    # DriftSafe is a SEQUENTIAL state machine: it escalates
    # NORMAL -> SHIFT_DETECTED -> PERSISTENCE_CHECK -> ADAPTATION_CANDIDATE
    # across consecutive windows, and its consensus/persistence requirements
    # only have meaning when it is fed a stream. Scoring one aggregate window
    # and reading `consensus_triggered` measures a different, weaker quantity
    # -- a single-shot threshold check -- and systematically under-reports
    # detection on shifts that build over time or affect a subset of traffic.
    # The recent stream is therefore replayed window-by-window, exactly as the
    # live monitoring path does, and the arm triggers when the state machine
    # reaches ADAPTATION_CANDIDATE on its own.
    driftsafe = DriftSafe(DriftSafeConfig(), reference_window)
    monitor_window_size = max(1, len(recent_records) // 4)
    triggered = False
    for w in range(4):
        chunk = recent_records[w * monitor_window_size:(w + 1) * monitor_window_size]
        if len(chunk) < 10:
            break
        driftsafe.evaluate_window(
            _window_dict(chunk, tokenizer, backbone, champion_heads, history_len))
        if driftsafe.state == IncidentState.ADAPTATION_CANDIDATE:
            triggered = True
            break

    if not triggered:
        return {
            "recovered_accuracy": None, "historical_retention_delta": 0.0,
            "curriculum_size": 0, "training_time_sec": time.time() - t0, "decision": "NO_DRIFT_DETECTED",
        }

    from vulcan.forge.forge_loop import wrap_record_as_window
    train_records = recent_records[:BUDGET]
    train_windows = [wrap_record_as_window(r, historical_records, history_len) for r in train_records]
    replay_windows = build_windows(historical_records, history_len, stride=max(1, history_len // 8))[: len(train_windows)]

    candidate = train_candidate(
        backbone, tokenizer, champion_heads.state_dict(), drift_windows=train_windows, replay_windows=replay_windows,
        num_routes=num_routes, d_model=backbone.cfg.d_model, epochs=25, seed=seed,
    )
    result = shadow_evaluate_and_decide(
        backbone, tokenizer, champion_heads, candidate.adapter, candidate.heads,
        recent_windows, historical_windows, GateThresholds(),
    )
    chall_recent = evaluate_model_on_slice(backbone, candidate.adapter, candidate.heads, tokenizer, recent_windows)
    return {
        "recovered_accuracy": chall_recent["accuracy"] if result["decision"] == "PROMOTE" else None,
        "historical_retention_delta": result["challenger"]["historical"]["accuracy"] - result["champion"]["historical"]["accuracy"],
        "curriculum_size": len(train_windows),
        "training_time_sec": time.time() - t0,
        "decision": result["decision"],
    }


def run_error_triggered_retrain_arm(backbone, tokenizer, champion_heads, historical_records, recent_records,
                                     recent_windows, historical_windows, num_routes, history_len, seed,
                                     champ_recent_accuracy, champ_historical_accuracy,
                                     error_threshold: float = 0.03) -> Dict[str, Any]:
    """A responsive-but-undiagnosed baseline (spec review item 4): triggers
    ONLY if recent accuracy has dropped more than `error_threshold` below
    historical accuracy -- no diagnosis, no blind-spot mining, no targeted
    curriculum, no red-team. Just: is performance down? If so, retrain on
    an equal-budget random sample and deploy via ordinary (accuracy-only)
    validation. This exists so DriftSafe-never-triggering (Finding 1 in
    reports/forge_final_report.md) doesn't leave Forge looking like it beat
    a baseline that was simply asleep -- this one wakes up reliably."""
    t0 = time.time()
    accuracy_drop = (champ_historical_accuracy or 0) - (champ_recent_accuracy or 0)
    if accuracy_drop < error_threshold:
        return {
            "recovered_accuracy": None, "historical_retention_delta": 0.0,
            "curriculum_size": 0, "training_time_sec": time.time() - t0, "decision": "NO_ERROR_TRIGGER",
        }

    rng = np.random.default_rng(seed + 9999)
    pool = historical_records + recent_records
    idx = rng.choice(len(pool), size=min(BUDGET, len(pool)), replace=False)
    sampled = [pool[i] for i in idx]
    from vulcan.forge.forge_loop import wrap_record_as_window
    train_windows = [wrap_record_as_window(r, historical_records, history_len) for r in sampled]
    replay_windows = build_windows(historical_records, history_len, stride=max(1, history_len // 8))[: len(train_windows)]

    candidate = train_candidate(
        backbone, tokenizer, champion_heads.state_dict(), drift_windows=train_windows, replay_windows=replay_windows,
        num_routes=num_routes, d_model=backbone.cfg.d_model, epochs=25, seed=seed,
    )
    chall_recent = evaluate_model_on_slice(backbone, candidate.adapter, candidate.heads, tokenizer, recent_windows)
    chall_hist = evaluate_model_on_slice(backbone, candidate.adapter, candidate.heads, tokenizer, historical_windows)

    # ordinary validation: accuracy must improve on recent, must not
    # regress badly on historical -- no calibration/redteam/failure-memory checks
    recent_improved = (chall_recent["accuracy"] or 0) >= (champ_recent_accuracy or 0)
    hist_ok = (chall_hist["accuracy"] or 0) >= (champ_historical_accuracy or 0) - 0.05
    decision = "PROMOTE" if (recent_improved and hist_ok) else "REJECT"

    return {
        "recovered_accuracy": chall_recent["accuracy"] if decision == "PROMOTE" else None,
        "historical_retention_delta": (chall_hist["accuracy"] or 0) - (champ_historical_accuracy or 0),
        "curriculum_size": len(train_windows),
        "training_time_sec": time.time() - t0,
        "decision": decision,
    }


def run_forge_arm(cfg, backbone, tokenizer, champion_heads, historical_records, recent_records,
                   reference_window, current_window, seed) -> Dict[str, Any]:
    t0 = time.time()
    fm = FailureMemory(memory_dir=f"artifacts/forge/benchmark_failure_memory_{seed}")
    manifest = run_forge_cycle(
        cfg, backbone, champion_heads, tokenizer, recent_records, historical_records,
        reference_window, current_window, historical_records[:2000], fm, seed=seed,
        curriculum_cfg=CurriculumConfig(curriculum_budget=BUDGET), verbose=False,
    )
    result = {
        "training_time_sec": time.time() - t0,
        "decision": manifest["outcome"],
        "diagnosed_failure_type": manifest["diagnosis"]["failure_type"],
        "diagnosis_severity": manifest["diagnosis"]["severity"],
        "n_blind_spots_found": len(manifest["blind_spots"]),
        "top_blindspot_dims": manifest["blind_spots"][0]["dimensions"] if manifest["blind_spots"] else None,
    }
    if manifest["outcome"] in ("PROMOTE", "REJECT") and manifest.get("curriculum") is not None:
        result["curriculum_size"] = manifest["curriculum"]["curriculum_size"]
        result["curriculum_validity_rate"] = manifest["curriculum_metadata"]["validity_rate"]
        result["recovered_accuracy"] = manifest["evaluation"]["challenger"]["recent"]["accuracy"] if manifest["outcome"] == "PROMOTE" else None
        result["historical_retention_delta"] = (
            manifest["evaluation"]["challenger"]["historical"]["accuracy"] - manifest["evaluation"]["champion"]["historical"]["accuracy"]
        )
        result["weak_region_before"] = manifest["evaluation"]["champion"]["weak_region"]["accuracy"]
        result["weak_region_after"] = manifest["evaluation"]["challenger"]["weak_region"]["accuracy"]
        result["n_redteam_failures"] = len(manifest["redteam"]["discovered_failures"])
        result["n_new_failure_memories"] = manifest["n_new_failure_memories"]
    elif manifest["outcome"] in ("PROMOTE", "REJECT") and manifest.get("recalibration") is not None:
        # RECALIBRATION path: no curriculum/challenger/red-team, just a
        # temperature-scaling artifact. Report its own metrics under the
        # same field names where meaningful, 0/None where not applicable.
        recal = manifest["recalibration"]
        result["curriculum_size"] = 0  # no training data used at all
        result["curriculum_validity_rate"] = None
        result["recovered_accuracy"] = recal["accuracy_after"] if manifest["outcome"] == "PROMOTE" else None
        result["historical_retention_delta"] = 0.0  # no weight changes -> historical performance is untouched by construction
        result["weak_region_before"] = None
        result["weak_region_after"] = None
        result["n_redteam_failures"] = 0  # red-team does not run for recalibration-only fixes
        result["n_new_failure_memories"] = 0
        result["recalibration_temperature"] = recal["temperature"]
        result["recalibration_ece_before"] = recal["ece_before"]
        result["recalibration_ece_after"] = recal["ece_after"]
    else:
        result["curriculum_size"] = 0
        result["recovered_accuracy"] = None
        result["historical_retention_delta"] = 0.0
    return result


def run_one_scenario_seed(cfg, tokenizer, backbone, champion_heads, historical_records, scenario, seed) -> Dict[str, Any]:
    num_routes = cfg["simulator"]["num_routes"]
    history_len = cfg["model"]["history_len"]

    reference_window = _window_dict(historical_records[:2000], tokenizer, backbone, champion_heads, history_len)
    historical_windows = build_windows(historical_records[:2000], history_len, stride=max(1, history_len // 4))

    sim = IndiaPaymentSim(cfg, seed=seed + 1)
    policy = BehaviorPolicy(BehaviorPolicyConfig(cfg["simulator"]["behavior_policy"]["exploration_rate"]), np.random.default_rng(seed + 2))
    shock = scenario["shock"]()
    sim.inject_shock(shock)
    recent_records = _collect_stream(sim, policy, 1500)
    current_window = _window_dict(recent_records, tokenizer, backbone, champion_heads, history_len)
    recent_windows = build_windows(recent_records, history_len, stride=max(1, history_len // 4))

    champ_recent_eval = evaluate_model_on_slice(backbone, None, champion_heads, tokenizer, recent_windows)
    champ_hist_eval = evaluate_model_on_slice(backbone, None, champion_heads, tokenizer, historical_windows)

    results = {}
    results["static"] = run_static_arm(champ_recent_eval)
    results["random_retrain"] = run_random_retrain_arm(
        backbone, tokenizer, champion_heads, historical_records, recent_records,
        recent_windows, historical_windows, num_routes, history_len, seed,
    )
    results["error_triggered_retrain"] = run_error_triggered_retrain_arm(
        backbone, tokenizer, champion_heads, historical_records, recent_records,
        recent_windows, historical_windows, num_routes, history_len, seed,
        champ_recent_eval["accuracy"], champ_hist_eval["accuracy"],
    )
    results["drift_retrain"] = run_drift_retrain_arm(
        cfg, backbone, tokenizer, champion_heads, historical_records, recent_records,
        reference_window, current_window, recent_windows, historical_windows, num_routes, history_len, seed,
    )
    results["forge"] = run_forge_arm(cfg, backbone, tokenizer, champion_heads, historical_records, recent_records,
                                       reference_window, current_window, seed)

    return {
        "scenario": scenario["name"], "true_failure_type": scenario["true_failure_type"],
        "champion_recent_accuracy_before": champ_recent_eval["accuracy"],
        "champion_historical_accuracy": champ_hist_eval["accuracy"],
        "results": results,
    }


def main(config_path: str, seeds: List[int]):
    cfg = load_config(config_path)
    set_global_seed(cfg["seed"])

    print("Generating training data / champion...")
    historical_records = generate_training_data(cfg, seed=cfg["seed"], n_transactions=cfg["data"]["n_transactions"])
    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(historical_records)
    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])

    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    backbone.load_state_dict(torch.load("checkpoints/mini_vulcan_pretrained.pt", weights_only=False)["backbone_state_dict"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    champion_heads = DownstreamHeads(model_cfg.d_model, cfg["simulator"]["num_routes"])
    champion_heads.load_state_dict(torch.load("checkpoints/baseline_heads.pt", weights_only=False)["heads_state_dict"])
    champion_heads.eval()

    all_runs = []
    for scenario in SCENARIOS:
        for seed in seeds:
            print(f"  scenario={scenario['name']} seed={seed}...")
            t0 = time.time()
            run_result = run_one_scenario_seed(cfg, tokenizer, backbone, champion_heads, historical_records, scenario, seed)
            run_result["wall_time_sec"] = time.time() - t0
            all_runs.append(run_result)
            print(f"    done in {run_result['wall_time_sec']:.1f}s -- "
                  f"static_acc={run_result['results']['static']['recovered_accuracy']:.3f} "
                  f"random={run_result['results']['random_retrain']['decision']} "
                  f"error_trig={run_result['results']['error_triggered_retrain']['decision']} "
                  f"drift={run_result['results']['drift_retrain']['decision']} "
                  f"forge={run_result['results']['forge']['decision']}"
                  f" (diagnosed={run_result['results']['forge']['diagnosed_failure_type']}, true={run_result['true_failure_type']})")

    summary = summarize(all_runs)
    print("\n" + "=" * 100)
    print_summary(summary)
    print("=" * 100)

    manifest = {
        "run_id": f"forge_benchmark_{int(time.time())}",
        "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": config_path,
        "seeds": seeds,
        "scenarios": [s["name"] for s in SCENARIOS],
        "budget": BUDGET,
        "scale_note": "6 scenarios x %d seeds, reduced from spec's suggested 10 scenarios x 5 seeds due to 1-CPU-core hardware constraints (see reports/forge_final_report.md)" % len(seeds),
        "runs": all_runs,
        "summary": summary,
    }
    Path("artifacts/runs").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"artifacts/runs/{manifest['run_id']}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nFull manifest saved to {out_path}")
    return manifest


def summarize(all_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    arms = ["static", "random_retrain", "error_triggered_retrain", "drift_retrain", "forge"]
    summary: Dict[str, Any] = {}

    def ci95(arr):
        arr = np.array(arr)
        return float(1.96 * arr.std() / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

    for arm in arms:
        # ---- SYSTEM-LEVEL DEPLOYED ACCURACY (the headline metric) ----
        # For EVERY one of the 18 scenarios, not just the ones where this
        # arm happened to train something: if this arm deployed a change
        # (decision in PROMOTE/RETRAINED and a recovered_accuracy was
        # measured), use that deployed accuracy. Otherwise -- rejected, no
        # trigger, no diagnosis-driven action -- the system-level outcome
        # IS the champion's own recent accuracy, because that's what stays
        # deployed. This replaces an earlier version of this benchmark that
        # reported Forge's mean_recovered_accuracy over only the subset of
        # cases it trained-and-promoted, compared against STATIC's full
        # 18-case mean -- an apples-to-oranges comparison a technical
        # reviewer correctly flagged. See reports/forge_final_report.md
        # section 3 for the full account of this fix.
        system_level = []
        for r in all_runs:
            arm_result = r["results"][arm]
            champ_baseline = r["champion_recent_accuracy_before"]
            deployed_something = arm_result["decision"] in ("PROMOTE", "RETRAINED") and arm_result.get("recovered_accuracy") is not None
            system_level.append(arm_result["recovered_accuracy"] if deployed_something else champ_baseline)

        recoveries = [r["results"][arm].get("recovered_accuracy") for r in all_runs if r["results"][arm].get("recovered_accuracy") is not None]
        retentions = [r["results"][arm]["historical_retention_delta"] for r in all_runs]
        curr_sizes = [r["results"][arm]["curriculum_size"] for r in all_runs]
        times = [r["results"][arm]["training_time_sec"] for r in all_runs]
        n_deployed = sum(1 for r in all_runs if r["results"][arm]["decision"] in ("PROMOTE", "RETRAINED"))
        n_rejected = sum(1 for r in all_runs if r["results"][arm]["decision"] == "REJECT")
        n_no_action = sum(1 for r in all_runs if r["results"][arm]["decision"] in
                           ("NO_ADAPTATION", "NO_DRIFT_DETECTED", "NO_ERROR_TRIGGER", "NO_TRAINING_PERFORMED"))

        summary[arm] = {
            "n_runs": len(all_runs),
            "mean_system_level_accuracy": float(np.mean(system_level)),
            "system_level_accuracy_ci95": ci95(system_level),
            "n_recovered_measurements": len(recoveries),
            "mean_recovered_accuracy_when_deployed": float(np.mean(recoveries)) if recoveries else None,
            "recovered_accuracy_ci95": ci95(recoveries) if recoveries else None,
            "mean_historical_retention_delta": float(np.mean(retentions)),
            "retention_ci95": ci95(retentions),
            "mean_curriculum_size": float(np.mean(curr_sizes)),
            "mean_training_time_sec": float(np.mean(times)),
            "n_deployed": n_deployed,
            "n_rejected": n_rejected,
            "n_no_action": n_no_action,
            "rejection_rate": n_rejected / len(all_runs),
            "no_action_rate": n_no_action / len(all_runs),
        }
        if arm == "forge":
            recoveries_per_example = [
                ((r["results"]["forge"].get("recovered_accuracy") or 0) - (r["champion_recent_accuracy_before"] or 0)) / max(1, r["results"]["forge"]["curriculum_size"])
                for r in all_runs if r["results"]["forge"]["curriculum_size"] > 0
            ]
            diagnosis_correct = sum(
                1 for r in all_runs
                if r["results"]["forge"]["diagnosed_failure_type"] == r["true_failure_type"]
            )
            weak_region_recoveries = [
                (r["results"]["forge"].get("weak_region_after") or 0) - (r["results"]["forge"].get("weak_region_before") or 0)
                for r in all_runs if r["results"]["forge"].get("weak_region_after") is not None
            ]
            redteam_discovery_rate = np.mean([
                1 if r["results"]["forge"].get("n_redteam_failures", 0) > 0 else 0 for r in all_runs
            ])
            summary[arm]["diagnosis_accuracy"] = diagnosis_correct / len(all_runs)
            summary[arm]["mean_recovery_per_training_example"] = float(np.mean(recoveries_per_example)) if recoveries_per_example else None
            summary[arm]["mean_weak_region_recovery"] = float(np.mean(weak_region_recoveries)) if weak_region_recoveries else None
            summary[arm]["redteam_found_something_rate"] = float(redteam_discovery_rate)

    return summary


def print_summary(summary: Dict[str, Any]):
    print("HEADLINE: system-level deployed accuracy across ALL 18 scenarios")
    print("(promoted/retrained -> deployed accuracy; rejected/no-trigger/no-action -> champion retained)\n")
    print(f"{'Arm':<24}{'SystemAcc':<16}{'HistRetention':<16}{'CurrSize':<10}{'TrainTime':<12}{'Deployed':<10}{'Rejected':<10}{'NoAction'}")
    for arm, s in summary.items():
        print(f"{arm:<24}{s['mean_system_level_accuracy']:.4f}±{s['system_level_accuracy_ci95']:.4f}  "
              f"{s['mean_historical_retention_delta']:+.4f}        "
              f"{s['mean_curriculum_size']:<10.0f}{s['mean_training_time_sec']:<12.1f}"
              f"{s['n_deployed']:<10}{s['n_rejected']:<10}{s['n_no_action']}")
    if "diagnosis_accuracy" in summary.get("forge", {}):
        f = summary["forge"]
        print(f"\nForge diagnosis accuracy: {f['diagnosis_accuracy']:.2%}")
        print(f"Forge mean weak-region recovery (on cases it adapted): {f.get('mean_weak_region_recovery')}")
        print(f"Forge red-team found-something rate: {f.get('redteam_found_something_rate'):.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tiny.yaml")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    main(args.config, seeds)
