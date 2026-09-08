"""
Protects vulcan/evaluation/baselines.py (Baseline A, XGBoost).

Importing xgboost/sklearn here means a clean environment missing either
dependency fails the whole suite immediately with a clear ImportError,
instead of silently passing (see docs/acceptance_gate_status.md).
"""
from vulcan.common.config import load_config
from vulcan.data.generate import generate_training_data
from vulcan.evaluation.baselines import XGBoostBaseline


def test_xgboost_baseline_trains_and_predicts_valid_probabilities():
    cfg = load_config("configs/tiny.yaml")
    num_routes = cfg["simulator"]["num_routes"]

    records = generate_training_data(cfg, seed=1, n_transactions=300)
    train_records, holdout_records = records[:250], records[250:]

    model = XGBoostBaseline(num_routes=num_routes)
    model.fit(train_records)

    route_specs = [
        {"route_id": r, "rail": model_route_rail(cfg, r), "gateway": f"GW_{chr(65+r)}"}
        for r in range(num_routes)
    ]

    probs = []
    for rec in holdout_records:
        for spec in route_specs:
            p = model.predict_success_proba_for_route(rec, spec["route_id"], spec["rail"], spec["gateway"])
            assert 0.0 <= p <= 1.0, f"predicted probability {p} out of [0,1] range"
            probs.append(p)

    # the model must have actually learned something -- not just returning
    # a single constant probability for every (state, route) pair
    assert len(set(round(p, 6) for p in probs)) > 1, "XGBoost baseline returned a constant prediction (did not learn)"


def test_xgboost_baseline_choose_route_returns_valid_route_id():
    cfg = load_config("configs/tiny.yaml")
    num_routes = cfg["simulator"]["num_routes"]

    records = generate_training_data(cfg, seed=2, n_transactions=250)
    model = XGBoostBaseline(num_routes=num_routes)
    model.fit(records[:200])

    route_specs = [
        {"route_id": r, "rail": model_route_rail(cfg, r), "gateway": f"GW_{chr(65+r)}"}
        for r in range(num_routes)
    ]
    for rec in records[200:220]:
        chosen = model.choose_route(rec, route_specs)
        assert chosen in range(num_routes)


def model_route_rail(cfg, route_id):
    rails = ["UPI", "CARD", "NETBANKING"]
    return rails[route_id % len(rails)]
