"""
Vulcan Forge 5-minute demo.

    python -m scripts.demo_forge --seed 2024

Runs a REAL, reproducible end-to-end Forge cycle. Every number printed
comes from an actual computation on this run -- there are no predetermined
frontend strings.

--------------------------------------------------------------------------
STREAMING REFACTOR NOTE (for the Control Room UI, api/forge_demo_api.py):

`run_forge_demo_stream()` below is the SINGLE source of truth for the
demo's orchestration (setup, shift injection, degradation measurement, the
round loop calling vulcan.forge.forge_loop.run_forge_cycle). It is a
generator that yields one real StageEvent per completed stage. `main()`
(the CLI) is now a thin consumer of this same generator -- it does not
duplicate any orchestration logic, so the CLI and the API/UI can never
drift apart.

IMPORTANT HONESTY NOTE on event timing: `vulcan/forge/forge_loop.py`'s
`run_forge_cycle()` is a single blocking call that computes all nine
stages internally and returns one manifest containing every stage's real
output -- it was NOT modified to yield incrementally (out of scope for
this pass: "do not modify any file under vulcan/forge/"). So the events
below for DIAGNOSE through PROMOTION_GATE are all decomposed from that one
returned manifest and yielded in rapid succession immediately after the
(real) computation finishes, not truly streamed as each stage completes
internally. Every event's DATA is 100% real (sliced directly from the
real manifest, nothing invented) -- only the *emission timing* is
batched rather than incremental. This is exactly why Step 3's client-side
minimum-dwell pacing exists: it paces real, already-computed data for
human readability, not a fabricated delay standing in for missing
computation.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

from vulcan.common.config import load_config
from vulcan.common.seeding import set_global_seed
from vulcan.data.generate import generate_training_data, _flatten_observation
from vulcan.data.windowing import build_windows, encode_windows_for_world_model
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.models.mini_vulcan import MiniVulcanBackbone, MiniVulcanConfig
from vulcan.models.downstream_heads import DownstreamHeads
from vulcan.simulator.environment import IndiaPaymentSim, RegimeShock
from vulcan.simulator.behavior_policy import BehaviorPolicy, BehaviorPolicyConfig
from vulcan.forge.forge_loop import run_forge_cycle, wrap_record_as_window
from vulcan.forge.curriculum_generator import CurriculumConfig
from vulcan.forge.failure_memory import FailureMemory
from vulcan.drift.driftsafe import DriftSafe, DriftSafeConfig, IncidentState
from vulcan.adaptation.champion_challenger import evaluate_model_on_slice


# Autonomous-monitoring cadence: DriftSafe evaluates one window at a time.
MONITOR_WINDOW_SIZE = 400
# Monitoring budget must exceed cooldown + consensus + persistence, or a
# post-promotion generation can never re-escalate: with cooldown=5 the old
# budget of 8 windows left only 3 for detection, and generation 2 stalled at
# PERSISTENCE_CHECK every time. This is a monitoring-duration parameter, not
# a detection threshold -- it does not make drift easier to declare, it just
# gives the state machine enough windows to reach a verdict.
MAX_MONITOR_WINDOWS = 14
# How far the live success rate may fall below the pre-heal baseline before a
# freshly promoted champion is automatically rolled back.
GUARDRAIL_TOLERANCE = 0.02

STAGE_NAMES = [
    "DIAGNOSE", "BLIND_SPOT_MINER", "HEALING_POLICY", "CURRICULUM",
    "TRAIN_CHALLENGER", "EVALUATE", "RED_TEAM", "FAILURE_MEMORY", "PROMOTION_GATE",
]


def _window_dict(records, tokenizer, backbone, heads, history_len, n=300, adapter=None):
    # Builds the reference/current windows the failure diagnoser computes
    # PSI / MMD / ECE / Page-Hinkley signals from, under masked encoding.
    windows = build_windows(records, history_len, stride=max(1, history_len // 4))[:n]
    cat, cont = encode_windows_for_world_model(tokenizer, windows)
    cont_full = torch.cat([cont, torch.zeros_like(cont)], dim=-1)
    with torch.no_grad():
        z = backbone.current_state(cat, cont_full)
        # CHAMPION-AWARE MONITORING: once a repaired champion is installed,
        # the drift signals (calibration ECE, representation MMD, residuals)
        # must be computed through the CURRENT champion's representation
        # path, not the original V1 one. Without this, generation 2+ would be
        # monitored as if the repair had never happened.
        if adapter is not None:
            z = adapter(z)
        out = heads(z)
        idx = torch.arange(z.shape[0])
        taken = torch.tensor([w[-1]["action_route_id"] for w in windows], dtype=torch.long)
        probs = torch.sigmoid(out["success_logit"])[idx, taken].numpy()
    labels = np.array([float(bool(w[-1]["outcome_success"])) for w in windows])
    features = np.array([w[-1]["amount"] for w in windows])
    return {"features": features, "z": z.numpy(), "probs": probs, "labels": labels}


def _champion_route_selector(backbone, heads, adapter, tokenizer, background, history_len, cfg):
    """Returns a function that picks a route using the CHAMPION MODEL's own
    multi-objective utility (success, fraud, latency, abandonment) via
    greedy_utility_select -- the same decision rule Baseline B uses.

    WHY THIS EXISTS: previously the simulated 'live' traffic in this demo was
    routed entirely by BehaviorPolicy, meaning the champion model never
    actually decided anything. Promoting V2 changed model parameters but did
    NOT change which routes the payment system took, so post-promotion
    'improvement' could only ever be shadow evaluation on data the old policy
    generated. With this, installing a new champion causally changes real
    executed routes and therefore real outcomes -- which is what makes
    post-heal measurement meaningful rather than offline classification."""
    import torch as _torch
    from vulcan.models.downstream_heads import greedy_utility_select
    weights = cfg.get("planner", {}).get("utility_weights", {}) or {
        "w_success": 1.0, "w_fraud": 1.0, "w_latency": 0.3, "w_abandon": 0.5,
    }

    def select(obs, actions):
        rec = _flatten_observation(obs)
        window = wrap_record_as_window(rec, background, history_len)
        cat, cont = encode_windows_for_world_model(tokenizer, [window])
        cont_full = _torch.cat([cont, _torch.zeros_like(cont)], dim=-1)
        with _torch.no_grad():
            z = backbone.current_state(cat, cont_full)
            if adapter is not None:
                z = adapter(z)
            out = heads(z)
            chosen_route = int(greedy_utility_select(out, weights)[0])
        for a in actions:
            if a.route_id == chosen_route:
                return a
        return actions[0]  # chosen route unavailable this step

    return select


def _collect_traffic(sim, n_steps, route_selector, policy, exploration_rate, rng):
    """Executes real traffic. Routes are chosen by `route_selector` (the
    champion model) except for an epsilon fraction kept on the exploration
    policy, so the behaviour log retains the coverage the trainer needs."""
    out_records = []
    for _ in range(n_steps):
        obs = sim.current_observation()
        actions = sim.candidate_actions()
        if route_selector is not None and rng.random() > exploration_rate:
            action = route_selector(obs, actions)
            propensity = 1.0 - exploration_rate
        else:
            action, propensity = policy.select_action(obs, actions)
        outcome = sim.step(obs, action)
        rec = _flatten_observation(obs)
        rec.update(
            action_route_id=action.route_id, action_rail=action.rail, action_gateway=action.gateway,
            propensity=propensity, outcome_success=outcome.success, outcome_latency_ms=outcome.latency_ms,
            outcome_processing_cost=outcome.processing_cost, outcome_fraud_loss=outcome.fraud_loss,
            outcome_abandoned=outcome.abandoned, outcome_error_type=outcome.error_type.value,
        )
        out_records.append(rec)
    return out_records


def _monitor_until_incident(sim, driftsafe, selector, policy, cfg, tokenizer, backbone,
                             champion_heads, champion_adapter, history_len, seed_base,
                             generation, max_windows, window_size):
    """Consumes traffic in windows under the CURRENT champion and returns
    (records, monitor_reports, triggered). Reusable across generations, which
    is what makes continuous operation possible -- the previous version
    inlined this once, so only the first champion could ever be monitored."""
    records, reports = [], []
    for w in range(max_windows):
        chunk = _collect_traffic(
            sim, window_size, selector, policy,
            cfg["simulator"]["behavior_policy"]["exploration_rate"],
            np.random.default_rng(seed_base + w))
        records.extend(chunk)
        wd = _window_dict(chunk, tokenizer, backbone, champion_heads, history_len,
                          adapter=champion_adapter)
        report = driftsafe.evaluate_window(wd)
        reports.append({
            "generation": generation, "window": w, "state": driftsafe.state.value,
            "n_signals_exceeded": report.n_exceeded, "consensus": report.consensus_triggered,
            "signals": {s.name: {"value": s.value, "threshold": s.threshold, "exceeded": s.exceeded}
                         for s in report.signals},
        })
        if driftsafe.state == IncidentState.ADAPTATION_CANDIDATE:
            break
    return records, reports, driftsafe.state == IncidentState.ADAPTATION_CANDIDATE


def _event(event: str, stage: Optional[str], data: Dict[str, Any], round_idx: Optional[int] = None) -> Dict[str, Any]:
    return {"event": event, "stage": stage, "round": round_idx, "data": data, "timestamp": time.time()}


def run_forge_demo_stream(config_path: str, seed: int, max_rounds: int,
                           failure_memory_dir: Optional[str] = None,
                           historical_replay_ratio: float = 1.0) -> Iterator[Dict[str, Any]]:
    """The single source of truth for the demo's orchestration. Yields one
    real StageEvent dict per completed stage."""
    cfg = load_config(config_path)
    set_global_seed(cfg["seed"])

    yield _event("SETUP", "CONFIG_LOADED", {"config_path": config_path, "seed": seed, "max_rounds": max_rounds})

    historical_records = generate_training_data(cfg, seed=cfg["seed"], n_transactions=4000)
    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(historical_records)
    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])

    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    backbone_ckpt = Path("checkpoints/mini_vulcan_pretrained.pt")
    if not backbone_ckpt.exists():
        raise SystemExit("No pretrained backbone found. Run: python -m train.pretrain --config configs/tiny.yaml")
    backbone.load_state_dict(torch.load(backbone_ckpt, weights_only=False)["backbone_state_dict"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    heads = DownstreamHeads(model_cfg.d_model, cfg["simulator"]["num_routes"])
    heads_ckpt = Path("checkpoints/baseline_heads.pt")
    if not heads_ckpt.exists():
        raise SystemExit("No champion heads found. Run: python -m train.train_baseline_heads --config configs/tiny.yaml")
    heads.load_state_dict(torch.load(heads_ckpt, weights_only=False)["heads_state_dict"])
    heads.eval()

    reference_window = _window_dict(historical_records[:2000], tokenizer, backbone, heads, model_cfg.history_len)
    yield _event("SETUP", "CHAMPION_LOADED", {"healthy_success_rate": float(reference_window["labels"].mean())})

    sim = IndiaPaymentSim(cfg, seed=seed + 1)
    policy = BehaviorPolicy(BehaviorPolicyConfig(cfg["simulator"]["behavior_policy"]["exploration_rate"]),
                            np.random.default_rng(seed + 2))
    shock = RegimeShock(name="issuer_degradation", start_step=0, duration=100000,
                        issuer_health_delta={"HDFC": -0.8, "ICICI": -0.5})
    sim.inject_shock(shock)
    yield _event("SETUP", "REGIME_SHIFT_INJECTED", {"shock_name": shock.name})

    # LIVE TRAFFIC IS ROUTED BY THE CHAMPION MODEL (with epsilon exploration),
    # not by BehaviorPolicy alone. This is what makes promotion causal: when a
    # new champion is installed, the routes actually executed change, so the
    # realized outcomes change too. Without it, "post-heal improvement" could
    # only ever be offline re-scoring of traffic the old policy produced.
    v1_selector = _champion_route_selector(
        backbone, heads, None, tokenizer, historical_records, model_cfg.history_len, cfg)

    # ---- AUTONOMOUS TRIGGER ----
    # Traffic is consumed in sequential windows and each one is fed to
    # DriftSafe's multi-signal state machine (PSI / representation MMD /
    # calibration ECE delta / Page-Hinkley residual drift), which requires
    # signal CONSENSUS sustained over consecutive windows before escalating
    # NORMAL -> SHIFT_DETECTED -> PERSISTENCE_CHECK -> ADAPTATION_CANDIDATE.
    #
    # This replaces the previous flow, where the demo script itself decided
    # when to heal: it collected a fixed 2000 transactions and then called
    # run_forge_cycle() unconditionally. Under that design the "self" in
    # self-healing was doing unearned work -- a human had hardcoded the
    # moment of intervention. Now the SYSTEM decides, from measured signals
    # alone, and if the shift is transient the state machine returns to
    # NORMAL and no healing is triggered at all.
    driftsafe = DriftSafe(DriftSafeConfig(), reference_window)
    recent_records = []
    monitor_windows = []
    for w in range(MAX_MONITOR_WINDOWS):
        chunk = _collect_traffic(
            sim, MONITOR_WINDOW_SIZE, v1_selector, policy,
            cfg["simulator"]["behavior_policy"]["exploration_rate"],
            np.random.default_rng(seed + 4242 + w))
        recent_records.extend(chunk)
        wd = _window_dict(chunk, tokenizer, backbone, heads, model_cfg.history_len)
        report = driftsafe.evaluate_window(wd)
        monitor_windows.append({
            "window": w,
            "state": driftsafe.state.value,
            "n_signals_exceeded": report.n_exceeded,
            "consensus": report.consensus_triggered,
            "signals": {s.name: {"value": s.value, "threshold": s.threshold, "exceeded": s.exceeded}
                         for s in report.signals},
        })
        yield _event("MONITOR_WINDOW", "MONITOR_WINDOW", monitor_windows[-1])
        if driftsafe.state == IncidentState.ADAPTATION_CANDIDATE:
            break

    triggered = driftsafe.state == IncidentState.ADAPTATION_CANDIDATE
    yield _event("DRIFTSAFE_DECISION", "DRIFTSAFE_DECISION", {
        "triggered": triggered,
        "final_state": driftsafe.state.value,
        "n_windows_monitored": len(monitor_windows),
        "n_transactions_monitored": len(recent_records),
    })
    if not triggered:
        # The system genuinely decided no healing was warranted. Reported
        # honestly rather than healing anyway.
        yield _event("DEMO_COMPLETE", None, {
            "run_id": f"demo_forge_{int(time.time())}", "manifest_path": "(no cycle run)",
            "final_outcome": "NO_INCIDENT", "n_rounds": 0,
            "failure_memory_stats": {},
            "story": "DriftSafe did not escalate to ADAPTATION_CANDIDATE; no healing triggered.",
        })
        return

    current_window = _window_dict(recent_records, tokenizer, backbone, heads, model_cfg.history_len)
    # Pre-heal baseline, measured the SAME way the post-heal number will be
    # (champion-routed, same simulator, same regime) so the comparison is
    # like-for-like rather than policy-vs-model.
    pre_heal_rate = float(np.mean([bool(r["outcome_success"]) for r in recent_records]))

    healthy_rate = float(reference_window["labels"].mean())
    current_rate = float(current_window["labels"].mean())
    yield _event("MEASURABLE_DEGRADATION", "MEASURABLE_DEGRADATION", {
        "healthy_success_rate": healthy_rate, "current_success_rate": current_rate,
        "delta": current_rate - healthy_rate,
    })

    # Additive, presentation-supporting computation (NOT part of Forge's
    # decision logic -- vulcan/forge/ is untouched by this). Real
    # per-issuer success rate in the healthy baseline window vs the
    # current (post-shift) window, computed directly from the same
    # historical_records / recent_records already in scope. This does not
    # feed back into any diagnosis or promotion decision; it exists only
    # so the UI can show which issuers are actually affected, using real
    # numbers, instead of only the single top-ranked blind spot.
    issuer_breakdown = []
    for issuer in cfg["simulator"]["issuers"]:
        hist_recs = [r for r in historical_records[:2000] if r.get("issuer") == issuer]
        rec_recs = [r for r in recent_records if r.get("issuer") == issuer]
        if len(hist_recs) < 10 or len(rec_recs) < 10:
            continue
        baseline_rate = sum(1 for r in hist_recs if r["outcome_success"]) / len(hist_recs)
        current_issuer_rate = sum(1 for r in rec_recs if r["outcome_success"]) / len(rec_recs)
        issuer_breakdown.append({
            "issuer": issuer, "baseline_success_rate": baseline_rate,
            "current_success_rate": current_issuer_rate,
            "delta": current_issuer_rate - baseline_rate,
            "n_baseline": len(hist_recs), "n_current": len(rec_recs),
        })
    issuer_breakdown.sort(key=lambda x: x["delta"])
    yield _event("ISSUER_BREAKDOWN", "ISSUER_BREAKDOWN", {"issuers": issuer_breakdown})

    run_id_suffix = int(time.time())
    fm_dir = failure_memory_dir or f"artifacts/forge/demo_failure_memory_seed{seed}_{run_id_suffix}"
    fm = FailureMemory(memory_dir=fm_dir)
    yield _event("SETUP", "FAILURE_MEMORY_INITIALIZED", {"memory_dir": fm_dir, "fresh": failure_memory_dir is None})

    budgets = [150] + [2000] * (max_rounds - 1)
    all_manifests = []
    final_outcome = None

    # CLOSED-LOOP STATE. `champion_adapter` starts as None (V1 is the bare
    # pretrained backbone + baseline heads). When a challenger is PROMOTED,
    # it is INSTALLED here and in the ModelRegistry, and the next cycle
    # diagnoses, repairs and certifies against that new champion. This is
    # what makes Forge a closed self-healing loop rather than a repair layer
    # that only ever proposes fixes to a fixed model.
    from vulcan.registry.model_registry import ModelRegistry
    registry = ModelRegistry(registry_dir=f"artifacts/registry/demo_{run_id_suffix}")
    # Register V1 before monitoring begins so every later promotion has a
    # predecessor to roll back to.
    try:
        _v1_entry = registry.promote(
            None, heads,
            promotion_manifest={"version_label": "V1", "role": "initial_champion",
                                 "note": "baseline champion registered before monitoring "
                                         "so later promotions have a rollback parent"},
        )
        v1_registry_info = {"registered": True, "version": getattr(_v1_entry, "version", None),
                             "checkpoint_sha256": getattr(_v1_entry, "checkpoint_sha256", None)}
    except Exception as e:
        v1_registry_info = {"registered": False, "error": f"{type(e).__name__}: {e}"}
    yield _event("SETUP", "INITIAL_CHAMPION_REGISTERED", v1_registry_info)
    champion_adapter = None
    champion_heads = heads
    champion_generation = 1
    # Champion runtime state also carries a calibration temperature, so
    # recalibration healing is a first-class installed artifact rather than
    # a certified-but-discarded result. 1.0 == identity (no recalibration).
    champion_temperature = 1.0
    # Outer-lifecycle bookkeeping. `generation_idx` counts CHAMPION
    # generations (V1 -> V2 -> V3), which is distinct from the repair
    # attempts inside a single incident. The audit was right that calling
    # every candidate "V2, V3, V4" conflated the two: before promotion those
    # are attempts against the SAME champion.
    generation_idx = 0
    max_generations = 2

    for round_idx in range(max_rounds):
        round_num = round_idx + 1
        round_seed = seed + round_idx
        budget = budgets[round_idx] if round_idx < len(budgets) else 2000
        version_label = f"V{round_idx + 2}"

        yield _event("ROUND_START", None, {"version_label": version_label, "curriculum_budget": budget, "seed": round_seed}, round_idx=round_num)

        manifest = run_forge_cycle(
            cfg, backbone, champion_heads, tokenizer,
            recent_records=recent_records, historical_records=historical_records,
            reference_window=reference_window, current_window=current_window,
            reference_records=historical_records[:2000],
            failure_memory=fm, seed=round_seed,
            curriculum_cfg=CurriculumConfig(curriculum_budget=budget),
            verbose=False,
            champion_adapter=champion_adapter,
            historical_replay_ratio=historical_replay_ratio,
        )
        manifest["version_label"] = version_label
        promoted_candidate = manifest.pop("_candidate", None)
        all_manifests.append(manifest)
        final_outcome = manifest["outcome"]

        yield _event("STAGE_COMPLETE", "DIAGNOSE", manifest["diagnosis"], round_idx=round_num)
        yield _event("STAGE_COMPLETE", "BLIND_SPOT_MINER", {
            "blind_spots": manifest["blind_spots"], "metadata": manifest["blindspot_miner_metadata"],
        }, round_idx=round_num)
        yield _event("STAGE_COMPLETE", "HEALING_POLICY", manifest["healing_decision"], round_idx=round_num)

        if manifest["outcome"] == "NO_TRAINING_PERFORMED":
            yield _event("ROUND_COMPLETE", None, {"version_label": version_label, "outcome": manifest["outcome"]}, round_idx=round_num)
            break

        yield _event("STAGE_COMPLETE", "CURRICULUM", {
            **(manifest.get("curriculum") or {}),
            "metadata": manifest.get("curriculum_metadata"),
        }, round_idx=round_num)

        if manifest.get("recalibration") is not None:
            yield _event("STAGE_COMPLETE", "TRAIN_CHALLENGER", {"recalibration": manifest["recalibration"]}, round_idx=round_num)
            yield _event("STAGE_COMPLETE", "PROMOTION_GATE", manifest["promotion"], round_idx=round_num)
            if manifest["outcome"] == "PROMOTE":
                # INSTALL the recalibration as real champion state. Previously
                # this branch certified a temperature and then `break`-ed
                # without ever applying it, so recalibration healing was
                # approved-but-never-deployed -- unlike adapter healing, which
                # genuinely installs. Temperature now becomes part of the
                # champion runtime and every later prediction goes through it.
                champion_temperature = manifest["recalibration"]["temperature"]
                champion_generation += 1
                yield _event("CHAMPION_RECALIBRATED", "CHAMPION_RECALIBRATED", {
                    "version_label": version_label,
                    "generation": champion_generation,
                    "temperature": champion_temperature,
                    "ece_before": manifest["recalibration"]["ece_before"],
                    "ece_after": manifest["recalibration"]["ece_after"],
                    "n_fit_examples": manifest["recalibration"].get("n_fit_examples"),
                    "n_certify_examples": manifest["recalibration"].get("n_calibration_examples"),
                }, round_idx=round_num)
            yield _event("ROUND_COMPLETE", None, {"version_label": version_label, "outcome": manifest["outcome"]}, round_idx=round_num)
            if manifest["outcome"] == "PROMOTE":
                break
            continue

        yield _event("STAGE_COMPLETE", "TRAIN_CHALLENGER", manifest.get("challenger_training"), round_idx=round_num)
        yield _event("STAGE_COMPLETE", "EVALUATE", manifest.get("evaluation"), round_idx=round_num)
        yield _event("STAGE_COMPLETE", "RED_TEAM", manifest.get("redteam"), round_idx=round_num)
        yield _event("STAGE_COMPLETE", "FAILURE_MEMORY", manifest.get("failure_memory_check"), round_idx=round_num)
        yield _event("STAGE_COMPLETE", "PROMOTION_GATE", manifest.get("promotion"), round_idx=round_num)

        if manifest["outcome"] == "PROMOTE":
            # ---- CLOSE THE LOOP ----
            # Actually INSTALL the promoted challenger as the new champion,
            # register it (hash-verified, rollback-capable), then RE-MEASURE
            # the newly-installed champion against fresh post-shift traffic.
            # Previously this branch just `break`-ed: the challenger was
            # declared promoted and then discarded, so the model never
            # actually healed. Now generation N+1 becomes the thing the next
            # cycle monitors, diagnoses and repairs.
            if promoted_candidate is not None:
                # ATOMIC PROMOTION. Persist and verify the artifact first;
                # the active champion pointers are swapped only once
                # registration has succeeded, so a failed write can never
                # leave an unregistered model serving. On failure the
                # previous champion stays live.
                try:
                    entry = registry.promote(
                        promoted_candidate.adapter, promoted_candidate.heads,
                        promotion_manifest={
                            "version_label": version_label,
                            "round": round_num,
                            "criteria": manifest["promotion"]["criteria"],
                        },
                    )
                    registry_info = {"registered": True,
                                      "checkpoint_sha256": getattr(entry, "checkpoint_sha256", None),
                                      "checkpoint_path": getattr(entry, "checkpoint_path", None),
                                      "version": getattr(entry, "version", None)}
                    # ---- pointer swap happens ONLY after durable registration ----
                    champion_adapter = promoted_candidate.adapter
                    champion_heads = promoted_candidate.heads
                    champion_generation += 1
                except Exception as e:
                    # Registry failed: do NOT install. V(current) stays live.
                    registry_info = {"registered": False, "error": f"{type(e).__name__}: {e}",
                                      "install_aborted": True}
                    yield _event("PROMOTION_ABORTED", "PROMOTION_ABORTED", {
                        "version_label": version_label,
                        "reason": registry_info["error"],
                        "active_champion_generation": champion_generation,
                    }, round_idx=round_num)
                    yield _event("ROUND_COMPLETE", None, {
                        "version_label": version_label, "outcome": "PROMOTION_ABORTED"}, round_idx=round_num)
                    break

                # POST-PROMOTION MEASUREMENT ON FRESH, CHAMPION-ROUTED TRAFFIC.
                # This previously re-scored `recent_records` -- traffic that was
                # generated BEFORE the new champion existed, with routes chosen
                # by BehaviorPolicy. That could only ever be shadow evaluation:
                # it could not show that the repair healed the deployed system,
                # only how the new weights score on old data.
                #
                # Now: continue the SAME live simulator forward, with the newly
                # installed champion actually choosing the routes, and measure
                # the realized outcomes it produces. Because the pre-heal
                # baseline below is measured the same way (champion-routed, same
                # simulator, same regime), the comparison is like-for-like.
                post_selector = _champion_route_selector(
                    backbone, champion_heads, champion_adapter, tokenizer,
                    historical_records, model_cfg.history_len, cfg)
                post_records = _collect_traffic(
                    sim, 600, post_selector, policy,
                    cfg["simulator"]["behavior_policy"]["exploration_rate"],
                    np.random.default_rng(round_seed + 777))
                post_rate = float(np.mean([bool(r["outcome_success"]) for r in post_records]))
                champion_eval = evaluate_model_on_slice(
                    backbone, champion_adapter, champion_heads, tokenizer,
                    build_windows(post_records, model_cfg.history_len,
                                  stride=max(1, model_cfg.history_len // 4))[:300],
                )
                # ---- POST-PROMOTION GUARDRAIL + AUTOMATIC ROLLBACK ----
                # The promotion gate certifies a repair BEFORE deployment.
                # This guardrail checks whether it holds up once serving: if
                # the installed champion's live success rate falls materially
                # below the pre-heal baseline it was meant to improve, the
                # install is automatically reverted to the previous champion,
                # with on-disk hash verification.
                guardrail_breach = post_rate < (pre_heal_rate - GUARDRAIL_TOLERANCE)
                if guardrail_breach:
                    rb = registry.rollback_runtime(
                        reason=f"post-promotion live success {post_rate:.4f} fell "
                               f"more than {GUARDRAIL_TOLERANCE:.3f} below pre-heal {pre_heal_rate:.4f}",
                        adapter=champion_adapter, heads=champion_heads,
                        triggering_metric="live_success_rate")
                    # Adopt the authoritative restored state. restored_adapter
                    # of None means the restored champion runs WITHOUT an
                    # adapter and the current one must be uninstalled.
                    if rb.get("restored"):
                        champion_adapter = rb.get("restored_adapter")
                        if rb.get("restored_heads") is not None:
                            champion_heads = rb["restored_heads"]
                        champion_generation -= 1
                    yield _event("ROLLBACK", "ROLLBACK", {
                        "triggered": True,
                        "restored": rb.get("restored", False),
                        "restored_version": rb.get("restored_version"),
                        "restored_label": rb.get("restored_label"),
                        "adapter_uninstalled": rb.get("adapter_must_be_uninstalled", False),
                        "hash_verified": rb.get("load_report", {}).get("hash_verified"),
                        "pre_heal_live_success_rate": pre_heal_rate,
                        "post_promotion_live_success_rate": post_rate,
                        "tolerance": GUARDRAIL_TOLERANCE,
                    }, round_idx=round_num)

                    # The candidate that triggered this guardrail was NOT
                    # kept live -- it was gate-passed, deployed, measured on
                    # fresh traffic, found to regress live success beyond
                    # tolerance, and reverted. Reporting it as "installed"/
                    # "promoted" from here on would contradict the rollback
                    # that just happened. Overwrite the manifest's outcome
                    # (already appended to all_manifests, so this mutation
                    # is what the FINAL STORY / full_manifest reflect) and
                    # emit an honest, distinct terminal event instead of the
                    # normal CHAMPION_UPDATED + ROUND_COMPLETE(PROMOTE) pair.
                    manifest["outcome"] = "PROMOTED_THEN_ROLLED_BACK"
                    final_outcome = manifest["outcome"]
                    yield _event("ROUND_COMPLETE", None, {
                        "version_label": version_label,
                        "outcome": manifest["outcome"],
                        "restored_label": rb.get("restored_label"),
                        "restored_version": rb.get("restored_version"),
                        "live_success_rate_delta": post_rate - pre_heal_rate,
                    }, round_idx=round_num)
                    # The incident that triggered this repair attempt is NOT
                    # resolved -- the fix that passed the gate failed once
                    # serving. Do not proceed into the "new independent
                    # incident" continuous-operation flow below, which
                    # assumes the prior incident was genuinely healed; that
                    # would misrepresent an unresolved incident as resolved.
                    break

                yield _event("CHAMPION_UPDATED", "CHAMPION_UPDATED", {
                    "version_label": version_label,
                    "generation": champion_generation,
                    "registry": registry_info,
                    "post_promotion_accuracy": champion_eval["accuracy"],
                    "post_promotion_ece": champion_eval["ece"],
                    "live_success_rate": post_rate,
                    "pre_heal_live_success_rate": pre_heal_rate,
                    "live_success_rate_delta": post_rate - pre_heal_rate,
                    "n_executed_transactions": len(post_records),
                    "traffic_routed_by": "champion_model_greedy_utility",
                    "n_evaluated": champion_eval["n"],
                }, round_idx=round_num)

            yield _event("ROUND_COMPLETE", None, {"version_label": version_label, "outcome": "PROMOTE"}, round_idx=round_num)

            # ---- CONTINUOUS OPERATION (outer lifecycle loop) ----
            # Previously this was a bare `break`: the run ended at the first
            # promotion, so the system could show V1 -> V2 but never
            # "V2 operates, a NEW independent incident occurs, V2 is diagnosed
            # and repaired into V3". That is the difference between a repair
            # loop and a self-healing lifecycle.
            #
            # Now: reset the monitor (which enters a genuine cooldown, so the
            # freshly-installed champion is not immediately re-diagnosed for
            # the incident it just healed), then resume monitoring the NEW
            # champion on fresh traffic it routes itself. A second, different
            # regime shift is injected to create a genuinely independent
            # incident -- Forge is not told about it, exactly as with the
            # first one.
            if generation_idx + 1 >= max_generations:
                break

            # REBASE the reference distribution onto the newly-installed
            # champion, using `post_records` -- the fresh, champion-routed
            # traffic already collected above for the guardrail check (not
            # re-used training/curriculum data). Without this, DriftSafe
            # keeps comparing future windows against the ORIGINAL pre-repair
            # champion's representation/calibration, so part of any signal
            # after this point would be an artifact of the repair itself
            # rather than a genuine new incident.
            new_reference = _window_dict(
                post_records, tokenizer, backbone, champion_heads,
                model_cfg.history_len, adapter=champion_adapter)
            driftsafe.rebase(new_reference)
            driftsafe.reset_after_adaptation()
            yield _event("MONITOR_RESET", "MONITOR_RESET", {
                "generation": champion_generation,
                "state": driftsafe.state.value,
                "cooldown_windows": driftsafe.cooldown_remaining,
                "reference_rebased": True,
            })

            second_shock = RegimeShock(
                name="gateway_degradation_wave_2", start_step=0, duration=100000,
                gateway_health_delta={"GW_B": -0.7, "GW_C": -0.5})
            sim.inject_shock(second_shock)
            yield _event("SETUP", "SECOND_REGIME_SHIFT_INJECTED", {
                "shock_name": second_shock.name,
                "note": "independent incident against the repaired champion; Forge is not told",
            })

            new_selector = _champion_route_selector(
                backbone, champion_heads, champion_adapter, tokenizer,
                historical_records, model_cfg.history_len, cfg)
            gen_records, gen_reports, gen_triggered = _monitor_until_incident(
                sim, driftsafe, new_selector, policy, cfg, tokenizer, backbone,
                champion_heads, champion_adapter, model_cfg.history_len,
                seed + 90000 + generation_idx * 100, champion_generation,
                MAX_MONITOR_WINDOWS, MONITOR_WINDOW_SIZE)
            for r in gen_reports:
                yield _event("MONITOR_WINDOW", "MONITOR_WINDOW", r)
            yield _event("DRIFTSAFE_DECISION", "DRIFTSAFE_DECISION", {
                "triggered": gen_triggered, "final_state": driftsafe.state.value,
                "generation": champion_generation,
                "n_windows_monitored": len(gen_reports),
                "n_transactions_monitored": len(gen_records),
            })
            if not gen_triggered:
                break

            # A genuinely new incident against the repaired champion: rebuild
            # the diagnosis inputs from ITS traffic and continue healing.
            recent_records = gen_records
            pre_heal_rate = float(np.mean([bool(r["outcome_success"]) for r in gen_records]))
            current_window = _window_dict(gen_records, tokenizer, backbone, champion_heads,
                                           model_cfg.history_len, adapter=champion_adapter)
            generation_idx += 1
            continue
        elif manifest["outcome"] == "REJECT":
            reasons = [c["name"] for c in manifest["promotion"]["criteria"] if not c["passed"]]
            yield _event("ROUND_COMPLETE", None, {
                "version_label": version_label, "outcome": "REJECT", "reasons": reasons,
                "n_new_failure_memories": manifest["n_new_failure_memories"],
            }, round_idx=round_num)
        else:
            yield _event("ROUND_COMPLETE", None, {"version_label": version_label, "outcome": manifest["outcome"]}, round_idx=round_num)
            break

    out_dir = Path("artifacts/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"demo_forge_{int(time.time())}"
    out_path = out_dir / f"{run_id}.json"
    full_manifest = {"run_id": run_id, "seed": seed, "rounds": all_manifests, "final_outcome": final_outcome}
    with open(out_path, "w") as f:
        json.dump(full_manifest, f, indent=2, default=str)

    yield _event("DEMO_COMPLETE", None, {
        "run_id": run_id, "manifest_path": str(out_path), "final_outcome": final_outcome,
        "n_rounds": len(all_manifests), "failure_memory_stats": fm.stats(),
        "story": " -> ".join(m["outcome"] for m in all_manifests),
    })


def main(config_path: str, seed: int, max_rounds: int, failure_memory_dir: str = None,
         historical_replay_ratio: float = 1.0):
    """Thin CLI consumer of run_forge_demo_stream -- no orchestration logic
    of its own."""
    print("=" * 78)
    print("VULCAN FORGE — 5-minute demo")
    print(f"config={config_path}  seed={seed}  max_rounds={max_rounds}")
    print("=" * 78)

    for evt in run_forge_demo_stream(config_path, seed, max_rounds, failure_memory_dir,
                                      historical_replay_ratio):
        stage, data, round_idx = evt["stage"], evt["data"], evt["round"]

        if stage == "CHAMPION_LOADED":
            print("\n[1/8] CHAMPION — loading healthy, already-trained champion...")
            print(f"  champion loaded. reference (healthy) window: mean success rate = {data['healthy_success_rate']:.4f}")
        elif stage == "REGIME_SHIFT_INJECTED":
            print("\n[2/8] HIDDEN REGIME SHIFT — injecting issuer degradation (Forge is NOT told this)...")
        elif stage == "MEASURABLE_DEGRADATION":
            print("\n[3/8] MEASURABLE DEGRADATION")
            print(f"  healthy success rate = {data['healthy_success_rate']:.4f}")
            print(f"  current success rate = {data['current_success_rate']:.4f}  (delta = {data['delta']:+.4f})")
        elif stage == "FAILURE_MEMORY_INITIALIZED":
            print(f"  (failure memory for this run: {data['memory_dir']}{' [fresh]' if data['fresh'] else ' [reused, explicitly requested]'})")
        elif evt["event"] == "ROUND_START":
            print(f"\n{'='*78}\nROUND {round_idx} — training challenger {data['version_label']} "
                  f"(curriculum budget={data['curriculum_budget']}, seed={data['seed']})\n{'='*78}")
        elif stage == "DIAGNOSE":
            print(f"[FORGE] Stage 1: FAILURE DIAGNOSER")
            print(f"  failure_type={data['failure_type']} severity={data['severity']:.3f} recommended_healing={data['recommended_healing_family']}")
        elif stage == "BLIND_SPOT_MINER":
            print(f"[FORGE] Stage 2: BLIND-SPOT MINER")
            bs = data["blind_spots"]
            if bs:
                top = bs[0]
                print(f"  top blind spot: {top['blindspot_id']} dims={top['dimensions']} priority={top['priority_score']:.3f} n={top['sample_count']}")
            else:
                print("  no blind spot met the minimum sample size threshold")
        elif stage == "HEALING_POLICY":
            print(f"[FORGE] Stage 3: HEALING POLICY")
            print(f"  strategy={data['selected_strategy']}")
            print(f"  reason: {data['reason']}")
        elif stage == "CURRICULUM":
            print(f"[FORGE] Stage 4: CURRICULUM GENERATOR + SCENARIO VALIDATOR")
            if data:
                meta = data.get("metadata") or {}
                if meta.get("n_regions_covered"):
                    print(f"  BROADER_CHALLENGER: curriculum spans {meta['n_regions_covered']} regions "
                          f"{meta.get('region_blindspot_ids')}")
                print(f"  curriculum size={data['curriculum_size']} (budget={data['budget']})")
        elif stage == "TRAIN_CHALLENGER":
            print(f"[FORGE] Stage 5: TRAIN CHALLENGER")
            if data and "recalibration" in data:
                r = data["recalibration"]
                print(f"  recalibration: temperature={r['temperature']:.3f} ECE {r['ece_before']:.4f} -> {r['ece_after']:.4f}")
            elif data:
                print(f"  trained: {data['n_trainable_parameters']:,} trainable params ({data['pct_trainable_of_backbone']}% of backbone)")
        elif stage == "EVALUATE":
            print(f"[FORGE] Stage 6: EVALUATE")
            if data:
                print(f"  recent   champ_acc={data['champion']['recent']['accuracy']:.4f} chall_acc={data['challenger']['recent']['accuracy']:.4f}")
        elif stage == "RED_TEAM":
            print(f"[FORGE] Stage 7: RED-TEAM EXAMINER")
            if data:
                print(f"  seed_pool={data['metadata']['seed_pool_size']} discovered_failures={len(data['discovered_failures'])}")
        elif stage == "FAILURE_MEMORY":
            print(f"[FORGE] Stage 8: FAILURE-MEMORY REGRESSION CHECK")
            if data:
                print(f"  failure memory: {data['n_cases']} known cases, {data['n_failing']} still failing (pass rate={data['pass_rate']:.3f})")
        elif stage == "PROMOTION_GATE":
            print(f"[FORGE] Stage 9: PROMOTION GATE")
            if data:
                for c in data["criteria"]:
                    print(f"  {c['name']}: {'PASS' if c['passed'] else 'FAIL'} (value={c['value']}, threshold={c['threshold']})")
                print(f"  FINAL: {data['decision']}")
        elif stage == "MONITOR_RESET":
            print(f"\n[MONITOR RESET] champion generation {data['generation']} now live; "
                  f"state={data['state']}, cooldown={data['cooldown_windows']} windows "
                  f"(prevents re-diagnosing the incident just healed)")
        elif stage == "SECOND_REGIME_SHIFT_INJECTED":
            print(f"\n[NEW INCIDENT] injecting '{data['shock_name']}' against the REPAIRED champion "
                  f"— Forge is not told.")
        elif stage == "MONITOR_WINDOW":
            print(f"  [monitor w{data['window']}] state={data['state']:<21} "
                  f"signals_exceeded={data['n_signals_exceeded']} consensus={data['consensus']}")
        elif stage == "DRIFTSAFE_DECISION":
            if data["triggered"]:
                print(f"\n[AUTONOMOUS TRIGGER] DriftSafe escalated to {data['final_state']} after "
                      f"{data['n_windows_monitored']} windows / {data['n_transactions_monitored']} transactions.")
                print("  Forge was NOT told to heal — the monitor decided.")
            else:
                print(f"\n[AUTONOMOUS TRIGGER] DriftSafe did NOT escalate (final state={data['final_state']}). "
                      f"No healing triggered.")
        elif stage == "CHAMPION_RECALIBRATED":
            print(f"\n[CLOSED LOOP] recalibration INSTALLED as champion state "
                  f"(generation {data['generation']})")
            print(f"  temperature={data['temperature']:.3f}  "
                  f"ECE {data['ece_before']:.4f} -> {data['ece_after']:.4f} "
                  f"(fit n={data['n_fit_examples']}, held-out certify n={data['n_certify_examples']})")
        elif stage == "ROLLBACK":
            print(f"\n[GUARDRAIL BREACH] live success {data['post_promotion_live_success_rate']:.4f} "
                  f"vs pre-heal {data['pre_heal_live_success_rate']:.4f} "
                  f"(tolerance {data['tolerance']:.3f})")
            print(f"  AUTOMATIC ROLLBACK -> {data['restored_label']} (registry v{data['restored_version']}), "
                  f"restored={data['restored']}, hash_verified={data['hash_verified']}, "
                  f"adapter_uninstalled={data['adapter_uninstalled']}")
        elif stage == "CHAMPION_UPDATED":
            print(f"\n[CLOSED LOOP] {data['version_label']} INSTALLED as champion "
                  f"(generation {data['generation']})")
            print(f"  registry: {data['registry']}")
            print(f"  {data['n_executed_transactions']} NEW transactions executed, routed by the champion model")
            print(f"  live success rate: {data['pre_heal_live_success_rate']:.4f} (pre-heal) -> "
                  f"{data['live_success_rate']:.4f} (post-heal)  delta={data['live_success_rate_delta']:+.4f}")
            print(f"  deployed champion: accuracy={data['post_promotion_accuracy']:.4f} "
                  f"ece={data['post_promotion_ece']:.4f} (n={data['n_evaluated']})")
        elif evt["event"] == "ROUND_COMPLETE":
            if data["outcome"] == "PROMOTE":
                print(f"\n>>> {data['version_label']} PROMOTED. <<<")
            elif data["outcome"] == "PROMOTED_THEN_ROLLED_BACK":
                print(f"\n>>> {data['version_label']} PASSED THE GATE BUT FAILED LIVE "
                      f"(delta={data['live_success_rate_delta']:+.4f}). "
                      f"AUTOMATICALLY ROLLED BACK to {data.get('restored_label', 'previous champion')} "
                      f"(registry v{data.get('restored_version')}). Incident NOT resolved. <<<")
            elif data["outcome"] == "REJECT":
                print(f"\n>>> {data['version_label']} REJECTED. Reason(s): {data['reasons']}")
                print(f">>> {data['n_new_failure_memories']} new failure(s) stored permanently. Escalating...")
            else:
                print(f"\n>>> No training performed this round ({data['outcome']}).")
        elif evt["event"] == "DEMO_COMPLETE":
            print(f"\n{'='*78}\nFINAL STORY: {data['story']}\n{'='*78}")
            print(f"\nFull manifest saved to {data['manifest_path']}")
            print(f"Failure memory stats: {data['failure_memory_stats']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tiny.yaml")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--failure_memory_dir", type=str, default=None)
    parser.add_argument("--historical_replay_ratio", type=float, default=1.0,
                         help="How much generic historical data anchors the adapter, as a fraction "
                              "of curriculum size. 1.0=conservative repair, 0.0=aggressive targeted "
                              "repair (realistic under incident pressure, and the configuration most "
                              "likely to cause silent collateral regression).")
    args = parser.parse_args()
    main(args.config, args.seed, args.max_rounds, args.failure_memory_dir,
         args.historical_replay_ratio)
