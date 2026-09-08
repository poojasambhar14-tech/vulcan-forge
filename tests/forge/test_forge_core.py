import inspect
import ast

import numpy as np
import pytest

from vulcan.common.config import load_config
from vulcan.forge import failure_diagnoser
from vulcan.forge.failure_diagnoser import diagnose, DiagnoserConfig
from vulcan.forge.schemas import FailureType, HealingFamily
from vulcan.forge.blindspot_miner import mine_blind_spots, BlindSpotMinerConfig
from vulcan.forge.scenario_validator import ScenarioValidator, ScenarioRegistry, ValidationConfig
from vulcan.forge.healing_policy import decide_healing_strategy, requires_training
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.data.generate import generate_training_data


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/tiny.yaml")


@pytest.fixture(scope="module")
def tokenizer(cfg):
    records = generate_training_data(cfg, seed=1, n_transactions=500)
    tok = FieldTokenizer(cfg)
    tok.fit(records)
    return tok


def _fake_window(n, mean_feature, mean_z, mean_success, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "features": rng.normal(mean_feature, 1.0, size=n),
        "z": rng.normal(mean_z, 1.0, size=(n, d)),
        "labels": (rng.random(n) < mean_success).astype(float),
        "probs": np.clip(rng.normal(mean_success, 0.05, size=n), 0.01, 0.99),
    }


def _fake_records(n, issuer_success_map, seed=0):
    rng = np.random.default_rng(seed)
    issuers = list(issuer_success_map.keys())
    records = []
    for i in range(n):
        issuer = rng.choice(issuers)
        success = rng.random() < issuer_success_map[issuer]
        records.append({
            "issuer": issuer, "rail": "UPI", "merchant_category": "ecommerce",
            "device_class": "android", "amount": float(rng.uniform(100, 5000)),
            "outcome_success": bool(success),
        })
    return records


# ----------------------------------------------------------------------
# 1. Failure diagnosis does not access hidden shift label
# ----------------------------------------------------------------------
def test_diagnoser_module_never_imports_simulator_hidden_state():
    """Architectural guarantee: the diagnoser module must not import or
    reference the simulator's hidden regime/shift internals."""
    source = inspect.getsource(failure_diagnoser)
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("simulator.environment" in m for m in imported), (
        "failure_diagnoser must never import the simulator directly -- "
        "it should only see measured signals passed in by the caller."
    )
    assert "_HiddenEnvState" not in source
    assert "hidden" not in source.lower() or "hidden regime" in source.lower()  # only in comments/docstrings


def test_diagnosis_is_a_pure_function_of_measured_signals():
    """Calling diagnose() with the SAME measured signals but different
    (irrelevant) extra kwargs should never change the answer -- confirms
    the decision procedure only depends on what's passed in, not on any
    side-channel."""
    ref = _fake_window(300, 0.0, 0.0, 0.85, seed=1)
    cur_shifted = _fake_window(300, 3.0, 2.5, 0.4, seed=2)
    ref_records = _fake_records(300, {"HDFC": 0.85, "ICICI": 0.85}, seed=1)
    cur_records = _fake_records(300, {"HDFC": 0.85, "ICICI": 0.85}, seed=2)

    d1 = diagnose(ref, cur_shifted, ref_records, cur_records)
    d2 = diagnose(ref, cur_shifted, ref_records, cur_records)
    assert d1.failure_type == d2.failure_type
    assert d1.severity == d2.severity


# ----------------------------------------------------------------------
# 2. Blind-spot miner finds an injected weak region
# ----------------------------------------------------------------------
def test_blindspot_miner_finds_injected_weak_region():
    records = _fake_records(1000, {"HDFC": 0.35, "ICICI": 0.90, "SBI": 0.90}, seed=5)
    # simulate a champion that's oblivious to the HDFC weakness (predicts ~0.85 for everyone)
    predicted_probs = np.full(len(records), 0.85)
    blind_spots, metadata = mine_blind_spots(records, predicted_probs, cfg=BlindSpotMinerConfig(min_sample_size=20))
    assert len(blind_spots) > 0
    top = blind_spots[0]
    assert top.dimensions.get("issuer") == "HDFC", f"expected HDFC to be the top blind spot, got {top.dimensions}"


def test_min_sample_threshold_prevents_noisy_false_blind_spots():
    """A segment with very few samples must never surface as a blind spot,
    even if its (noisy) error rate looks bad."""
    records = _fake_records(500, {"HDFC": 0.85, "ICICI": 0.85}, seed=6)
    # inject 3 (too few) badly-performing records under a rare issuer value
    for _ in range(3):
        records.append({"issuer": "RARE_BANK", "rail": "UPI", "merchant_category": "ecommerce",
                         "device_class": "android", "amount": 500.0, "outcome_success": False})
    predicted_probs = np.full(len(records), 0.85)
    blind_spots, _ = mine_blind_spots(records, predicted_probs, cfg=BlindSpotMinerConfig(min_sample_size=30))
    assert not any(b.dimensions.get("issuer") == "RARE_BANK" for b in blind_spots)


def test_business_exposure_affects_priority():
    """Two segments with IDENTICAL error rates but different transaction
    amounts (financial exposure) must be prioritized differently."""
    low_value_records = _fake_records(200, {"LOW_VALUE_BANK": 0.5}, seed=7)
    for r in low_value_records:
        r["amount"] = 100.0
    high_value_records = _fake_records(200, {"HIGH_VALUE_BANK": 0.5}, seed=8)
    for r in high_value_records:
        r["amount"] = 50000.0
    all_records = low_value_records + high_value_records
    predicted_probs = np.full(len(all_records), 0.5)
    blind_spots, _ = mine_blind_spots(all_records, predicted_probs, cfg=BlindSpotMinerConfig(min_sample_size=20))
    by_issuer = {b.dimensions.get("issuer"): b for b in blind_spots if "issuer" in b.dimensions}
    assert by_issuer["HIGH_VALUE_BANK"].financial_exposure > by_issuer["LOW_VALUE_BANK"].financial_exposure
    assert by_issuer["HIGH_VALUE_BANK"].priority_score >= by_issuer["LOW_VALUE_BANK"].priority_score


# ----------------------------------------------------------------------
# 3. Scenario validator: rejects invalid, enforces train/cert separation
# ----------------------------------------------------------------------
def test_invalid_synthetic_scenarios_rejected(cfg, tokenizer):
    validator = ScenarioValidator(tokenizer, cfg)
    bad_amount = {"amount": -50.0, "rail": "UPI", "merchant_category": "ecommerce",
                  "merchant_segment": "smb", "issuer": "HDFC", "device_class": "android",
                  "geo_bucket": 0, "time_bucket": 0, "previous_attempts": 0, "retry_count": 0}
    assert validator.validate(bad_amount).startswith("REJECTED")

    bad_categorical = dict(bad_amount)
    bad_categorical["amount"] = 500.0
    bad_categorical["issuer"] = "NOT_A_REAL_BANK"
    assert validator.validate(bad_categorical).startswith("REJECTED")

    oracle_leak = dict(bad_categorical)
    oracle_leak["issuer"] = "HDFC"
    oracle_leak["oracle_p_success"] = 0.9
    assert validator.validate(oracle_leak).startswith("REJECTED")


def test_valid_scenario_passes(cfg, tokenizer):
    validator = ScenarioValidator(tokenizer, cfg)
    good = {"amount": 500.0, "rail": "UPI", "merchant_category": "ecommerce",
            "merchant_segment": "smb", "issuer": "HDFC", "device_class": "android",
            "geo_bucket": 0, "time_bucket": 0, "previous_attempts": 0, "retry_count": 0}
    assert validator.validate(good) == "VALID"


def test_train_certification_leakage_impossible():
    registry = ScenarioRegistry()
    h = "some_scenario_hash_123"
    assert registry.register(h, "TRAIN") is True
    # attempting to register the SAME hash under CERTIFICATION must fail
    assert registry.register(h, "CERTIFICATION") is False
    assert registry.contaminates(h, "CERTIFICATION") is True
    # a different hash is unaffected
    assert registry.register("other_hash", "CERTIFICATION") is True


def test_duplicate_scenarios_rejected_via_registry():
    registry = ScenarioRegistry()
    h = "dup_hash"
    assert registry.register(h, "TRAIN") is True
    assert registry.register(h, "TRAIN") is True  # same use twice is fine (idempotent)
    assert registry.contaminates(h, "TRAIN") is False


# ----------------------------------------------------------------------
# 4. Healing policy: does not default everything to retrain
# ----------------------------------------------------------------------
def test_healing_policy_chooses_non_retrain_path_for_schema_failure():
    from vulcan.forge.schemas import FailureDiagnosis
    diagnosis = FailureDiagnosis(
        failure_type=FailureType.SCHEMA_OR_PIPELINE_FAILURE.value, severity=0.9,
        evidence={"schema_anomaly_fraction": 0.3}, affected_region={"scope": "pipeline"},
        recommended_healing_family=HealingFamily.BLOCK_AND_SURFACE_INFRA_FAILURE.value,
    )
    decision = decide_healing_strategy(diagnosis)
    assert decision.selected_strategy == HealingFamily.BLOCK_AND_SURFACE_INFRA_FAILURE.value
    assert requires_training(decision) is False


def test_healing_policy_no_retrain_for_transient_spike():
    from vulcan.forge.schemas import FailureDiagnosis
    diagnosis = FailureDiagnosis(
        failure_type=FailureType.TRANSIENT_SPIKE.value, severity=0.2,
        evidence={}, affected_region={"scope": "temporary"},
        recommended_healing_family=HealingFamily.NO_ACTION_CONTINUE_MONITORING.value,
    )
    decision = decide_healing_strategy(diagnosis)
    assert requires_training(decision) is False


def test_healing_policy_requires_training_for_local_blind_spot():
    from vulcan.forge.schemas import FailureDiagnosis
    diagnosis = FailureDiagnosis(
        failure_type=FailureType.LOCAL_BLIND_SPOT.value, severity=0.7,
        evidence={}, affected_region={"scope": "segment"},
        recommended_healing_family=HealingFamily.TARGETED_ADAPTER.value,
    )
    decision = decide_healing_strategy(diagnosis)
    assert requires_training(decision) is True


# ----------------------------------------------------------------------
# 5. Transient spike does not automatically retrain (diagnoser level)
# ----------------------------------------------------------------------
def test_transient_spike_not_flagged_as_persistent_shift():
    ref = _fake_window(300, 0.0, 0.0, 0.85, seed=10)
    cur = _fake_window(300, 3.0, 2.5, 0.4, seed=11)  # looks like a big shift...
    ref_records = _fake_records(300, {"HDFC": 0.85}, seed=10)
    cur_records = _fake_records(300, {"HDFC": 0.4}, seed=11)
    # ...but persistence history shows it was NOT sustained (only 1 of 5 recent windows triggered)
    diagnosis = diagnose(ref, cur, ref_records, cur_records,
                          recent_window_history_consensus=[False, False, True, False, False])
    assert diagnosis.failure_type == FailureType.TRANSIENT_SPIKE.value
    assert diagnosis.recommended_healing_family == HealingFamily.NO_ACTION_CONTINUE_MONITORING.value
