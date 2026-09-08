from vulcan.common.config import load_config
from vulcan.simulator.environment import IndiaPaymentSim, RouteAction


def test_same_seed_same_observation_sequence_regardless_of_actions():
    """Critical for fair benchmark comparisons (master spec section 20):
    two simulator instances with the same seed must show the identical
    sequence of transactions/observations, even if driven by different
    action-selection policies (which consume the *outcome* RNG stream
    differently)."""
    cfg = load_config("configs/tiny.yaml")

    sim_a = IndiaPaymentSim(cfg, seed=55)
    sim_b = IndiaPaymentSim(cfg, seed=55)

    obs_seq_a = []
    obs_seq_b = []
    for i in range(100):
        obs_a = sim_a.current_observation()
        obs_b = sim_b.current_observation()
        obs_seq_a.append((obs_a.transaction.amount, obs_a.transaction.issuer, obs_a.transaction.merchant_category))
        obs_seq_b.append((obs_b.transaction.amount, obs_b.transaction.issuer, obs_b.transaction.merchant_category))

        # policy A always picks route 0; policy B always picks route (i % num_routes)
        action_a = RouteAction(0, sim_a.route_rail[0], sim_a.route_gateway[0])
        route_b = i % sim_b.num_routes
        action_b = RouteAction(route_b, sim_b.route_rail[route_b], sim_b.route_gateway[route_b])

        sim_a.step(obs_a, action_a)
        sim_b.step(obs_b, action_b)

    assert obs_seq_a == obs_seq_b
