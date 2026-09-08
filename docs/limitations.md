# Limitations

- **Synthetic environment.** All experiments run against `IndiaPaymentSim`, a
  stochastic simulator with hand-specified structural mechanisms. It is
  designed to make action-conditioned counterfactual evaluation possible
  (which real historical logs do not allow), but its dynamics are not
  validated against real UPI/card network data.
- **Scale.** Model sizes (tiny ~2-4M params, base ~8-12M params) are far
  smaller than any production payments foundation model. This is a
  research-scale prototype, not a production system.
- **No RL.** The planner is a transparent deterministic multi-objective
  scorer, not a learned policy. This is a deliberate scope decision (see
  `docs/novelty_boundary.md` and the master spec, section 37).
- **Compounding rollout error.** World-model rollouts beyond horizon=1 are
  expected to degrade in accuracy, consistent with general world-model
  literature; horizon=1 is the primary evaluated setting.
- **Simulated network health signals.** Rolling success/latency/load
  features are computed from the simulator's own observable history, not
  from an independent real-world monitoring system.
- This document will be updated with concrete negative results once
  benchmark runs (Part 5) are complete; see `reports/failures.md` when
  available.
