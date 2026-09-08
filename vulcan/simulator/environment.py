"""
IndiaPaymentSim: a controlled stochastic payment environment.

Design goal (see docs/research_basis.md and master spec section 3-4):
Real historical payment logs only tell us what happened under the action
that was actually taken. They do NOT tell us what would have happened under
actions that were not taken. To evaluate an action-conditioned world model
against genuine counterfactuals, we need an environment where the *evaluator*
knows the true structural transition probabilities for every action, while
the *model under training* only ever observes (state, action_taken, outcome)
tuples.

This module deliberately separates:
  - `EnvironmentState` (HIDDEN): ground-truth regime variables never exposed
    to the model.
  - `ObservableState`: realistic signals a payment model could plausibly see.
  - `PaymentOutcome`: the stochastic result of taking one action.
  - `oracle_outcome_distribution(...)`: an evaluator-only function that can
    compute outcome probabilities for ANY candidate action, including ones
    not taken. This must never be called from training-data generation code.

Everything is driven by a seeded `numpy.random.Generator` for reproducibility.
"""
from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


class ErrorType(str, enum.Enum):
    NONE = "none"
    ISSUER_DECLINE = "issuer_decline"
    GATEWAY_TIMEOUT = "gateway_timeout"
    NETWORK_ERROR = "network_error"
    FRAUD_BLOCK = "fraud_block"
    USER_ABANDON = "user_abandon"


@dataclass
class RouteAction:
    route_id: int
    rail: str          # e.g. "UPI", "CARD", "NETBANKING"
    gateway: str        # e.g. "GW_A", "GW_B", "GW_C"


@dataclass
class RegimeShock:
    """A named, time-boxed perturbation to hidden environment state.
    Populated fully in Part 3; NORMAL regime uses the identity shock."""
    name: str = "normal"
    start_step: int = 0
    duration: int = 0
    issuer_health_delta: Dict[str, float] = field(default_factory=dict)
    gateway_health_delta: Dict[str, float] = field(default_factory=dict)
    congestion_delta: float = 0.0
    fraud_rate_multiplier: float = 1.0

    def active_at(self, step: int) -> bool:
        return self.start_step <= step < self.start_step + self.duration


@dataclass
class _HiddenEnvState:
    """Ground-truth regime variables. NEVER exposed to the model directly."""
    issuer_health: Dict[str, float]
    gateway_health: Dict[str, float]
    rail_congestion: Dict[str, float]
    fraud_regime_level: float
    time_of_day_load: float
    recovering_from_outage: bool = False
    outage_active: bool = False


@dataclass
class ObservableTransaction:
    """Fields a real payment model could plausibly observe."""
    amount: float
    rail: str
    merchant_category: str
    merchant_segment: str
    issuer: str
    device_class: str
    geo_bucket: int
    time_bucket: int
    previous_attempts: int
    retry_count: int
    customer_tenure_days: int


@dataclass
class NetworkObservation:
    """Observable network-health signals (derived from recent HISTORY only,
    never from the hidden ground truth directly)."""
    rolling_route_success: Dict[int, float]
    route_latency_ms: Dict[int, float]
    recent_timeout_rate: Dict[int, float]
    route_load: Dict[int, float]
    recent_issuer_decline_rate: float
    gateway_availability_signal: Dict[int, float]


@dataclass
class ObservableState:
    transaction: ObservableTransaction
    network: NetworkObservation
    step: int


@dataclass
class PaymentOutcome:
    success: bool
    latency_ms: float
    processing_cost: float
    fraud_loss: float
    abandoned: bool
    error_type: ErrorType
    next_observable_state: Optional[ObservableState] = None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class IndiaPaymentSim:
    """Stochastic payment environment with hidden regime state.

    All randomness flows through `self.rng` (a numpy Generator constructed
    from an explicit seed), so two instances constructed with the same seed
    and driven with the same sequence of actions produce identical outcomes.
    """

    ROUTE_BASE_QUALITY_SEED_OFFSET = 1000

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.seed = seed
        # Two independent RNG streams: `obs_rng` drives transaction sampling
        # (so the SAME sequence of states can be replayed across different
        # policies for fair, paired benchmark comparisons -- master spec
        # section 20), and `outcome_rng` drives the stochastic outcome of
        # whatever action a policy actually takes. Keeping them separate
        # means one policy's choices never desync the state sequence shown
        # to a different policy evaluated under the same seed.
        self.rng = np.random.default_rng(seed)          # legacy alias, used by behavior policy sampling elsewhere
        self.obs_rng = np.random.default_rng(seed)
        self.outcome_rng = np.random.default_rng(seed + 500_000)

        sim_cfg = cfg["simulator"]
        self.num_routes = sim_cfg["num_routes"]
        self.issuers = sim_cfg["issuers"]
        self.merchant_segments = sim_cfg["merchant_segments"]
        self.device_classes = sim_cfg["device_classes"]
        self.time_buckets = sim_cfg["time_buckets"]
        self.geo_buckets = sim_cfg["geo_buckets"]

        self.rails = ["UPI", "CARD", "NETBANKING"]
        self.gateways = [f"GW_{chr(65+i)}" for i in range(self.num_routes)]
        self.merchant_categories = [
            "ecommerce", "food_delivery", "travel", "utilities", "gaming", "subscriptions"
        ]

        # Deterministic per-route base quality, derived from seed (not RNG
        # draws, so it never desyncs from the outcome RNG stream).
        base_rng = np.random.default_rng(seed + self.ROUTE_BASE_QUALITY_SEED_OFFSET)
        self.route_base_quality = {
            r: float(base_rng.uniform(-0.3, 0.6)) for r in range(self.num_routes)
        }
        self.route_rail = {r: self.rails[r % len(self.rails)] for r in range(self.num_routes)}
        self.route_gateway = {r: self.gateways[r] for r in range(self.num_routes)}

        # merchant-route affinity: some merchants prefer some routes
        self.merchant_route_affinity = {
            (m, r): float(base_rng.uniform(-0.15, 0.15))
            for m in self.merchant_categories
            for r in range(self.num_routes)
        }

        self._reset_hidden_state()

        self.step_idx = 0
        self.active_shocks: List[RegimeShock] = []

        # rolling history buffers per route, used to construct the
        # OBSERVABLE network signals (never derived from hidden state
        # directly, to avoid smuggling ground truth into observations)
        self._history_window = 200
        self._route_outcomes: Dict[int, List[bool]] = {r: [] for r in range(self.num_routes)}
        self._route_latencies: Dict[int, List[float]] = {r: [] for r in range(self.num_routes)}
        self._route_timeouts: Dict[int, List[bool]] = {r: [] for r in range(self.num_routes)}
        self._issuer_declines: List[bool] = []

    # ------------------------------------------------------------------
    # Hidden state management
    # ------------------------------------------------------------------
    def _reset_hidden_state(self) -> None:
        self.hidden = _HiddenEnvState(
            issuer_health={i: 1.0 for i in self.issuers},
            gateway_health={g: 1.0 for g in self.gateways},
            rail_congestion={r: 0.0 for r in self.rails},
            fraud_regime_level=0.05,
            time_of_day_load=0.0,
        )

    def inject_shock(self, shock: RegimeShock) -> None:
        """Register a regime shock. Hidden identity is never exposed to the
        model; only its downstream effects on observable outcomes are."""
        shock.start_step = self.step_idx if shock.start_step == 0 else shock.start_step
        self.active_shocks.append(shock)

    def _apply_active_shocks(self) -> _HiddenEnvState:
        """Compute effective hidden state at the current step, applying any
        active shocks on top of the baseline hidden state. Does not mutate
        self.hidden permanently for outage-type shocks (those recover)."""
        issuer_health = dict(self.hidden.issuer_health)
        gateway_health = dict(self.hidden.gateway_health)
        congestion = dict(self.hidden.rail_congestion)
        fraud_mult = 1.0
        outage_active = False

        still_active = []
        for shock in self.active_shocks:
            if shock.active_at(self.step_idx):
                still_active.append(shock)
                for issuer, delta in shock.issuer_health_delta.items():
                    issuer_health[issuer] = float(np.clip(issuer_health.get(issuer, 1.0) + delta, 0.0, 1.0))
                for gw, delta in shock.gateway_health_delta.items():
                    gateway_health[gw] = float(np.clip(gateway_health.get(gw, 1.0) + delta, 0.0, 1.0))
                for rail in congestion:
                    congestion[rail] = float(np.clip(congestion[rail] + shock.congestion_delta, 0.0, 1.0))
                fraud_mult *= shock.fraud_rate_multiplier
                if shock.name == "temporary_outage":
                    outage_active = True
            elif shock.start_step + shock.duration <= self.step_idx:
                pass  # expired, drop it
            else:
                still_active.append(shock)
        self.active_shocks = still_active

        return _HiddenEnvState(
            issuer_health=issuer_health,
            gateway_health=gateway_health,
            rail_congestion=congestion,
            fraud_regime_level=self.hidden.fraud_regime_level * fraud_mult,
            time_of_day_load=self.hidden.time_of_day_load,
            outage_active=outage_active,
        )

    # ------------------------------------------------------------------
    # Observation construction (model-visible)
    # ------------------------------------------------------------------
    def _sample_transaction(self) -> ObservableTransaction:
        merchant_category = self.obs_rng.choice(self.merchant_categories)
        merchant_segment = self.obs_rng.choice(self.merchant_segments)
        issuer = self.obs_rng.choice(self.issuers)
        device_class = self.obs_rng.choice(self.device_classes)
        amount = float(np.round(self.obs_rng.lognormal(mean=6.5, sigma=1.1), 2))
        geo_bucket = int(self.obs_rng.integers(0, self.geo_buckets))
        time_bucket = int(self.obs_rng.integers(0, self.time_buckets))
        previous_attempts = int(self.obs_rng.poisson(0.3))
        retry_count = 0
        customer_tenure_days = int(self.obs_rng.exponential(180))

        return ObservableTransaction(
            amount=amount,
            rail=self.obs_rng.choice(self.rails),
            merchant_category=merchant_category,
            merchant_segment=merchant_segment,
            issuer=issuer,
            device_class=device_class,
            geo_bucket=geo_bucket,
            time_bucket=time_bucket,
            previous_attempts=previous_attempts,
            retry_count=retry_count,
            customer_tenure_days=customer_tenure_days,
        )

    def _network_observation(self) -> NetworkObservation:
        def _rolling_mean(buf: List[float], default: float) -> float:
            if not buf:
                return default
            window = buf[-self._history_window:]
            return float(np.mean(window))

        rolling_success = {r: _rolling_mean([float(x) for x in self._route_outcomes[r]], 0.85) for r in range(self.num_routes)}
        latency = {r: _rolling_mean(self._route_latencies[r], 400.0) for r in range(self.num_routes)}
        timeout_rate = {r: _rolling_mean([float(x) for x in self._route_timeouts[r]], 0.02) for r in range(self.num_routes)}
        load = {r: float(np.clip(len(self._route_outcomes[r][-50:]) / 50.0, 0.0, 1.0)) for r in range(self.num_routes)}
        recent_decline = _rolling_mean([float(x) for x in self._issuer_declines], 0.05)
        availability = {r: 1.0 - timeout_rate[r] for r in range(self.num_routes)}

        return NetworkObservation(
            rolling_route_success=rolling_success,
            route_latency_ms=latency,
            recent_timeout_rate=timeout_rate,
            route_load=load,
            recent_issuer_decline_rate=recent_decline,
            gateway_availability_signal=availability,
        )

    def current_observation(self) -> ObservableState:
        txn = self._sample_transaction()
        net = self._network_observation()
        return ObservableState(transaction=txn, network=net, step=self.step_idx)

    # ------------------------------------------------------------------
    # Structural success-probability model (INTERNAL — parameters never
    # exposed to the model)
    # ------------------------------------------------------------------
    def _success_probability(self, obs: ObservableState, action: RouteAction, hidden: _HiddenEnvState) -> float:
        txn = obs.transaction
        route = action.route_id
        base_quality = self.route_base_quality[route]
        issuer_health = hidden.issuer_health.get(txn.issuer, 1.0)
        gateway_health = hidden.gateway_health.get(action.gateway, 1.0)
        congestion = hidden.rail_congestion.get(action.rail, 0.0)
        retry_penalty = 0.15 * txn.retry_count
        affinity = self.merchant_route_affinity.get((txn.merchant_category, route), 0.0)

        # amount effect: very large amounts are marginally riskier
        amount_effect = -0.05 * float(np.tanh(txn.amount / 50000.0))

        logit = (
            base_quality
            + 1.4 * (issuer_health - 0.5)
            + 1.2 * (gateway_health - 0.5)
            - 1.6 * congestion
            - retry_penalty
            + affinity
            + amount_effect
        )
        return float(_sigmoid(logit))

    def oracle_outcome_distribution(self, obs: ObservableState, candidate_routes: Optional[List[int]] = None) -> Dict[int, Dict[str, float]]:
        """EVALUATOR-ONLY. Returns the true structural success probability
        (and related expected values) for every candidate route, including
        ones not actually taken. This function must NEVER be called by
        training-data generation code (`vulcan/data/generate.py` calls
        `.step()` only, and enforces this — see tests/simulator for a
        leakage-prevention regression test)."""
        hidden = self._apply_active_shocks()
        routes = candidate_routes if candidate_routes is not None else list(range(self.num_routes))
        out = {}
        for r in routes:
            action = RouteAction(route_id=r, rail=self.route_rail[r], gateway=self.route_gateway[r])
            p_success = self._success_probability(obs, action, hidden)
            out[r] = {
                "p_success": p_success,
                "expected_latency_ms": self._latency_mean(action, hidden),
                "p_fraud": self._fraud_probability(obs, action, hidden),
            }
        return out

    def _latency_mean(self, action: RouteAction, hidden: _HiddenEnvState) -> float:
        congestion = hidden.rail_congestion.get(action.rail, 0.0)
        gateway_health = hidden.gateway_health.get(action.gateway, 1.0)
        base = 250.0 + 400.0 * congestion + 150.0 * (1.0 - gateway_health)
        return float(base)

    def _fraud_probability(self, obs: ObservableState, action: RouteAction, hidden: _HiddenEnvState) -> float:
        base = hidden.fraud_regime_level
        amount_boost = 0.02 * float(np.tanh(obs.transaction.amount / 100000.0))
        return float(np.clip(base + amount_boost, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Core stepping API (used by TRAINING data generation)
    # ------------------------------------------------------------------
    def step(self, obs: ObservableState, action: RouteAction) -> PaymentOutcome:
        """Take one action in the environment and return the realized,
        stochastic outcome. This is the ONLY function training-data
        generation is allowed to call for outcomes."""
        hidden = self._apply_active_shocks()
        p_success = self._success_probability(obs, action, hidden)
        p_fraud = self._fraud_probability(obs, action, hidden)
        lat_mean = self._latency_mean(action, hidden)

        success_roll = self.outcome_rng.random()
        success = success_roll < p_success

        error_type = ErrorType.NONE
        abandoned = False
        fraud_loss = 0.0
        processing_cost = float(obs.transaction.amount) * 0.012  # flat MDR-like cost

        latency_ms = float(max(50.0, self.outcome_rng.normal(loc=lat_mean, scale=lat_mean * 0.2)))

        if not success:
            r = self.outcome_rng.random()
            if r < 0.45:
                error_type = ErrorType.ISSUER_DECLINE
            elif r < 0.75:
                error_type = ErrorType.GATEWAY_TIMEOUT
                latency_ms = max(latency_ms, 2500.0)
            elif r < 0.90:
                error_type = ErrorType.NETWORK_ERROR
            else:
                error_type = ErrorType.USER_ABANDON
                abandoned = True

        fraud_roll = self.outcome_rng.random()
        if success and fraud_roll < p_fraud:
            fraud_loss = float(obs.transaction.amount)
            if error_type == ErrorType.NONE:
                error_type = ErrorType.FRAUD_BLOCK

        # update rolling history buffers (OBSERVABLE side effects only)
        route = action.route_id
        self._route_outcomes[route].append(success)
        self._route_latencies[route].append(latency_ms)
        self._route_timeouts[route].append(error_type == ErrorType.GATEWAY_TIMEOUT)
        self._issuer_declines.append(error_type == ErrorType.ISSUER_DECLINE)
        for buf in (self._route_outcomes[route], self._route_latencies[route], self._route_timeouts[route]):
            if len(buf) > self._history_window * 2:
                del buf[: len(buf) - self._history_window * 2]
        if len(self._issuer_declines) > self._history_window * 2:
            del self._issuer_declines[: len(self._issuer_declines) - self._history_window * 2]

        self.step_idx += 1
        next_obs = self.current_observation()

        return PaymentOutcome(
            success=success,
            latency_ms=latency_ms,
            processing_cost=processing_cost,
            fraud_loss=fraud_loss,
            abandoned=abandoned,
            error_type=error_type,
            next_observable_state=next_obs,
        )

    def candidate_actions(self) -> List[RouteAction]:
        return [
            RouteAction(route_id=r, rail=self.route_rail[r], gateway=self.route_gateway[r])
            for r in range(self.num_routes)
        ]
