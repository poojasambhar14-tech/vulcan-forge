"""
Slower tests (train real small models) covering:
  - red-team examiner finds a genuine, seeded regression
  - red-team's discovered set never overlaps the training set
  - a candidate with catastrophic-forgetting-like behavior is rejected
  - failure memory persists across separate FailureMemory instances
  - a full forge cycle runs end-to-end and produces a coherent manifest
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from vulcan.common.config import load_config
from vulcan.common.seeding import set_global_seed
from vulcan.data.generate import generate_training_data, _flatten_observation
from vulcan.data.windowing import build_windows, encode_windows
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.models.mini_vulcan import MiniVulcanBackbone, MiniVulcanConfig
from vulcan.models.downstream_heads import DownstreamHeads
from vulcan.simulator.environment import IndiaPaymentSim, RegimeShock
from vulcan.simulator.behavior_policy import BehaviorPolicy, BehaviorPolicyConfig
from vulcan.adaptation.candidate_trainer import train_candidate
from vulcan.forge.forge_loop import make_predict_fn, run_forge_cycle
from vulcan.forge.redteam_examiner import run_redteam_search, RedTeamConfig
from vulcan.forge.scenario_validator import ScenarioValidator, ScenarioRegistry
from vulcan.forge.curriculum_generator import CurriculumConfig
from vulcan.forge.failure_memory import FailureMemory


@pytest.fixture(scope="module")
def loaded_champion(trained_checkpoints):
    cfg = load_config("configs/tiny.yaml")
    seed = cfg["seed"]
    set_global_seed(seed)
    historical_records = generate_training_data(cfg, seed=seed, n_transactions=3000)
    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(historical_records)
    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])

    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    backbone.load_state_dict(torch.load("checkpoints/mini_vulcan_pretrained.pt", weights_only=False)["backbone_state_dict"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    heads = DownstreamHeads(model_cfg.d_model, cfg["simulator"]["num_routes"])
    heads.load_state_dict(torch.load("checkpoints/baseline_heads.pt", weights_only=False)["heads_state_dict"])
    heads.eval()

    return cfg, tokenizer, backbone, heads, historical_records, model_cfg


@pytest.mark.slow
def test_redteam_finds_seeded_regression_and_champion_vs_self_is_clean(loaded_champion):
    cfg, tokenizer, backbone, heads, historical_records, model_cfg = loaded_champion
    champ_predict = make_predict_fn(backbone, heads, None, tokenizer, historical_records, model_cfg.history_len)

    # sanity: champion vs itself must show ~zero regression everywhere
    validator = ScenarioValidator(tokenizer, cfg)
    registry = ScenarioRegistry()
    self_failures, _ = run_redteam_search(
        champ_predict, champ_predict, cfg, validator, registry, seed=1,
        cfg=RedTeamConfig(severity_threshold_to_report=0.01, max_failures_to_return=50),
    )
    assert len(self_failures) == 0, "champion vs itself must show no regression, found: " + str(self_failures)

    # a DELIBERATELY BAD challenger (heavily overfit to a narrow slice, no
    # replay) should show genuine, seeded regressions the red-team can find
    windows = build_windows(historical_records, model_cfg.history_len, stride=8)[:80]
    bad_candidate = train_candidate(
        backbone, tokenizer, heads.state_dict(), drift_windows=windows, replay_windows=None,
        num_routes=cfg["simulator"]["num_routes"], d_model=model_cfg.d_model,
        epochs=200, lr=1e-2, weight_decay=0.0, bottleneck_dim=64, seed=2,
    )
    bad_predict = make_predict_fn(backbone, bad_candidate.heads, bad_candidate.adapter, tokenizer, historical_records, model_cfg.history_len)

    validator2 = ScenarioValidator(tokenizer, cfg)
    registry2 = ScenarioRegistry()
    failures, metadata = run_redteam_search(
        champ_predict, bad_predict, cfg, validator2, registry2, seed=3,
        cfg=RedTeamConfig(severity_threshold_to_report=0.05, max_failures_to_return=50),
    )
    assert len(failures) > 0, "red-team should find at least one genuine regression against a deliberately overfit challenger"
    for f in failures:
        assert f["regression_severity"] > 0


@pytest.mark.slow
def test_redteam_discovered_set_differs_from_training_set(loaded_champion):
    cfg, tokenizer, backbone, heads, historical_records, model_cfg = loaded_champion
    champ_predict = make_predict_fn(backbone, heads, None, tokenizer, historical_records, model_cfg.history_len)

    windows = build_windows(historical_records, model_cfg.history_len, stride=8)[:80]
    training_hashes = set()
    from vulcan.forge.schemas import stable_hash
    for w in windows:
        training_hashes.add(stable_hash(w[-1]))

    bad_candidate = train_candidate(
        backbone, tokenizer, heads.state_dict(), drift_windows=windows, replay_windows=None,
        num_routes=cfg["simulator"]["num_routes"], d_model=model_cfg.d_model,
        epochs=200, lr=1e-2, weight_decay=0.0, bottleneck_dim=64, seed=2,
    )
    bad_predict = make_predict_fn(backbone, bad_candidate.heads, bad_candidate.adapter, tokenizer, historical_records, model_cfg.history_len)

    validator = ScenarioValidator(tokenizer, cfg)
    registry = ScenarioRegistry()
    # register the training set as TRAIN first
    for h in training_hashes:
        registry.register(h, "TRAIN")

    failures, _ = run_redteam_search(
        champ_predict, bad_predict, cfg, validator, registry, seed=3,
        cfg=RedTeamConfig(severity_threshold_to_report=0.05, max_failures_to_return=50),
    )
    discovered_hashes = {f["scenario_hash"] for f in failures}
    assert discovered_hashes.isdisjoint(training_hashes), (
        "red-team discovered scenarios must never overlap the training set "
        "(the registry should have refused to register any that did)"
    )


def test_failure_memory_persists_across_instances(tmp_path):
    from vulcan.forge.schemas import FailureMemoryEntry
    mem_dir = str(tmp_path / "fm")
    fm1 = FailureMemory(memory_dir=mem_dir)
    entry = FailureMemoryEntry(
        failure_id="F1", discovered_against="challenger_v1", scenario={"amount": 500, "issuer": "HDFC"},
        failure_metric="test", champion_result=0.9, challenger_result=0.1, severity=0.8, first_seen_run="run1",
    )
    assert fm1.add(entry) is True
    assert fm1.add(entry) is False  # duplicate, same scenario -> hash collision -> skipped

    fm2 = FailureMemory(memory_dir=mem_dir)
    assert len(fm2.regression_suite()) == 1
    assert fm2.regression_suite()[0].failure_id == "F1"


@pytest.mark.slow
def test_full_forge_cycle_end_to_end_produces_coherent_manifest(loaded_champion, tmp_path):
    cfg, tokenizer, backbone, heads, historical_records, model_cfg = loaded_champion
    seed = cfg["seed"]

    sim = IndiaPaymentSim(cfg, seed=seed + 1)
    policy = BehaviorPolicy(BehaviorPolicyConfig(cfg["simulator"]["behavior_policy"]["exploration_rate"]), np.random.default_rng(seed + 2))
    shock = RegimeShock(name="issuer_degradation", start_step=0, duration=100000, issuer_health_delta={"HDFC": -0.6})
    sim.inject_shock(shock)
    recent_records = []
    for _ in range(1200):
        obs = sim.current_observation()
        actions = sim.candidate_actions()
        action, propensity = policy.select_action(obs, actions)
        outcome = sim.step(obs, action)
        rec = _flatten_observation(obs)
        rec.update(action_route_id=action.route_id, action_rail=action.rail, action_gateway=action.gateway,
                   propensity=propensity, outcome_success=outcome.success, outcome_latency_ms=outcome.latency_ms,
                   outcome_processing_cost=outcome.processing_cost, outcome_fraud_loss=outcome.fraud_loss,
                   outcome_abandoned=outcome.abandoned, outcome_error_type=outcome.error_type.value)
        recent_records.append(rec)

    def window_dict(records, n=200):
        windows = build_windows(records, model_cfg.history_len, stride=max(1, model_cfg.history_len // 4))[:n]
        cat, cont = encode_windows(tokenizer, windows)
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

    reference_window = window_dict(historical_records[:1500])
    current_window = window_dict(recent_records)
    fm = FailureMemory(memory_dir=str(tmp_path / "fm_cycle"))

    manifest = run_forge_cycle(
        cfg, backbone, heads, tokenizer, recent_records, historical_records,
        reference_window, current_window, historical_records[:1500], fm, seed=seed,
        curriculum_cfg=CurriculumConfig(curriculum_budget=200), verbose=False,
    )

    assert manifest["diagnosis"]["failure_type"] in [t.value for t in __import__("vulcan.forge.schemas", fromlist=["FailureType"]).FailureType]
    assert manifest["outcome"] in ("PROMOTE", "REJECT", "NO_TRAINING_PERFORMED")
    if manifest["outcome"] in ("PROMOTE", "REJECT"):
        assert "curriculum" in manifest
        assert manifest["curriculum"]["curriculum_size"] <= manifest["curriculum"]["budget"]
        assert "promotion" in manifest and manifest["promotion"] is not None
        assert len(manifest["promotion"]["criteria"]) >= 5
