import numpy as np
import pytest

from vulcan.common.config import load_config
from vulcan.simulator.environment import IndiaPaymentSim, RouteAction


@pytest.fixture
def cfg():
    return load_config("configs/tiny.yaml")


def test_deterministic_under_same_seed(cfg):
    sim1 = IndiaPaymentSim(cfg, seed=123)
    sim2 = IndiaPaymentSim(cfg, seed=123)

    for _ in range(50):
        obs1 = sim1.current_observation()
        obs2 = sim2.current_observation()
        action = RouteAction(route_id=0, rail=sim1.route_rail[0], gateway=sim1.route_gateway[0])
        out1 = sim1.step(obs1, action)
        out2 = sim2.step(obs2, action)
        assert out1.success == out2.success
        assert out1.latency_ms == out2.latency_ms
        assert out1.error_type == out2.error_type


def test_different_seeds_diverge(cfg):
    sim1 = IndiaPaymentSim(cfg, seed=1)
    sim2 = IndiaPaymentSim(cfg, seed=2)

    successes_1, successes_2 = [], []
    for _ in range(200):
        obs1 = sim1.current_observation()
        obs2 = sim2.current_observation()
        action = RouteAction(route_id=0, rail=sim1.route_rail[0], gateway=sim1.route_gateway[0])
        successes_1.append(sim1.step(obs1, action).success)
        successes_2.append(sim2.step(obs2, action).success)

    assert successes_1 != successes_2


def test_probabilities_valid(cfg):
    sim = IndiaPaymentSim(cfg, seed=7)
    obs = sim.current_observation()
    dist = sim.oracle_outcome_distribution(obs)
    for route, stats in dist.items():
        assert 0.0 <= stats["p_success"] <= 1.0
        assert 0.0 <= stats["p_fraud"] <= 1.0
        assert stats["expected_latency_ms"] > 0


def test_route_masking_and_action_space(cfg):
    sim = IndiaPaymentSim(cfg, seed=3)
    actions = sim.candidate_actions()
    assert len(actions) == cfg["simulator"]["num_routes"]
    route_ids = {a.route_id for a in actions}
    assert route_ids == set(range(cfg["simulator"]["num_routes"]))


def test_stochastic_outcome_reproducible_with_fresh_rng(cfg):
    """Same seed + same sequence of actions -> identical trajectory, even
    across independently constructed simulator instances (regression guard
    against accidental global RNG state usage)."""
    def trajectory(seed):
        sim = IndiaPaymentSim(cfg, seed=seed)
        results = []
        for _ in range(30):
            obs = sim.current_observation()
            a = sim.candidate_actions()[obs.step % sim.num_routes]
            out = sim.step(obs, a)
            results.append((out.success, round(out.latency_ms, 4), out.error_type))
        return results

    assert trajectory(99) == trajectory(99)
