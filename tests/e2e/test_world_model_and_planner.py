import torch
import pytest

from experimental.world_model.dynamics import ActionConditionedWorldModel
from experimental.world_model.planner import (
    plan_decision, PlannerWeights, PlannerConstraints, UncertaintyGateConfig,
)


@pytest.fixture
def world_model():
    torch.manual_seed(0)
    return ActionConditionedWorldModel(d_model=32, num_routes=3)


def test_action_changes_prediction(world_model):
    z = torch.randn(1, 32)
    out0 = world_model(z, torch.tensor([0]))
    out1 = world_model(z, torch.tensor([1]))
    # different (randomly initialized) action embeddings must produce
    # different outputs for the same state
    assert not torch.allclose(out0["success_logit"], out1["success_logit"])


def test_predict_all_routes_shape(world_model):
    z = torch.randn(2, 32)
    preds = world_model.predict_all_routes(z, num_routes=3)
    assert set(preds.keys()) == {0, 1, 2}
    for r, p in preds.items():
        assert p["success_logit"].shape == (2,)


def test_planner_returns_full_decision_trace(world_model):
    z = torch.randn(1, 32)
    result = plan_decision(
        world_model, z, num_routes=3,
        observable_route_health={0: 0.9, 1: 0.9, 2: 0.9},
        retry_count=0,
        weights=PlannerWeights(),
        constraints=PlannerConstraints(),
        uncertainty_cfg=UncertaintyGateConfig(max_latency_logvar=100.0),  # effectively disable gate
    )
    assert len(result["candidates"]) == 3
    assert result["chosen_action"] in (0, 1, 2)
    assert result["confidence_status"] in ("CONFIDENT", "UNSURE")
    for c in result["candidates"]:
        assert "utility_components" in c
        assert "utility" in c


def test_planner_excludes_unhealthy_routes(world_model):
    z = torch.randn(1, 32)
    result = plan_decision(
        world_model, z, num_routes=3,
        observable_route_health={0: 0.01, 1: 0.9, 2: 0.9},  # route 0 unhealthy
        retry_count=0,
        weights=PlannerWeights(),
        constraints=PlannerConstraints(min_route_health=0.15, max_fraud_ceiling=0.99),
        uncertainty_cfg=UncertaintyGateConfig(max_latency_logvar=100.0),
    )
    route0 = next(c for c in result["candidates"] if c["route_id"] == 0)
    assert route0["eligible"] is False
    assert result["chosen_action"] != 0


def test_uncertainty_gate_triggers_fallback(world_model):
    z = torch.randn(1, 32)
    result = plan_decision(
        world_model, z, num_routes=3,
        observable_route_health={0: 0.9, 1: 0.9, 2: 0.9},
        retry_count=0,
        weights=PlannerWeights(),
        constraints=PlannerConstraints(),
        uncertainty_cfg=UncertaintyGateConfig(max_latency_logvar=-999.0),  # impossible to satisfy -> always falls back
    )
    assert result["fallback_triggered"] is True
    assert result["confidence_status"] == "UNSURE"
    assert result["chosen_action"] == 0
