"""
Scenario Validator (Forge pipeline stage 4).

The curriculum generator (and the red-team examiner) must NEVER be trusted
to validate their own generated data. This module is a deterministic,
independent gate: every generated scenario passes through here before it
can be used for training or certification, and it can reject invalid
scenarios for concrete, logged reasons.

Also enforces the hard invariant that a scenario used for TRAIN can never
appear in CERTIFICATION (and vice versa) via a persistent registry keyed by
scenario hash -- see ScenarioRegistry below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np

from vulcan.forge.schemas import stable_hash, IntendedUse

FORBIDDEN_LEAKED_PREFIXES = ("oracle_", "hidden_", "regime_name")


@dataclass
class ValidationConfig:
    amount_min: float = 1.0
    amount_max: float = 2_000_000.0
    max_retry_count: int = 10
    max_previous_attempts: int = 20
    near_duplicate_hash_len: int = 12   # shorter hash prefix = coarser near-dup bucketing


class ScenarioValidator:
    def __init__(self, tokenizer, sim_cfg: dict, cfg: Optional[ValidationConfig] = None):
        """tokenizer: a fitted vulcan.tokenization.tokenizer.FieldTokenizer,
        used as the source of truth for valid categorical vocabularies and
        valid route/rail/gateway relationships (never re-derived ad hoc)."""
        self.tokenizer = tokenizer
        self.cfg = cfg or ValidationConfig()
        self.num_routes = sim_cfg["simulator"]["num_routes"]
        self.valid_rails = {"UPI", "CARD", "NETBANKING"}
        self.valid_gateways = {f"GW_{chr(65+i)}" for i in range(self.num_routes)}
        # route_id -> (rail, gateway) is deterministic in IndiaPaymentSim:
        # route_rail[r] = rails[r % len(rails)], route_gateway[r] = gateways[r]
        rails_cycle = ["UPI", "CARD", "NETBANKING"]
        self.valid_route_rail = {r: rails_cycle[r % len(rails_cycle)] for r in range(self.num_routes)}
        self.valid_route_gateway = {r: f"GW_{chr(65+r)}" for r in range(self.num_routes)}

    def _reject(self, reason: str) -> str:
        return f"REJECTED:{reason}"

    def validate(self, record: Dict[str, Any]) -> str:
        """Returns 'VALID' or 'REJECTED:<reason>'. Never raises."""
        # 1. oracle leakage
        for key in record:
            if key.startswith(FORBIDDEN_LEAKED_PREFIXES):
                return self._reject(f"oracle_or_hidden_field_present:{key}")

        # 2. schema: required fields present, correct basic types
        required = ["amount", "rail", "merchant_category", "merchant_segment", "issuer",
                    "device_class", "geo_bucket", "time_bucket", "previous_attempts", "retry_count"]
        for field_name in required:
            if field_name not in record:
                return self._reject(f"missing_required_field:{field_name}")

        amount = record["amount"]
        if amount is None or not np.isfinite(amount):
            return self._reject("amount_not_finite")
        if not (self.cfg.amount_min <= amount <= self.cfg.amount_max):
            return self._reject(f"amount_out_of_range:{amount}")

        # 3. domain constraints: categorical values must be in the
        # tokenizer's known vocabulary (closed sets, not scanned from data)
        for field_name in ("rail", "merchant_category", "merchant_segment", "issuer", "device_class"):
            vocab = self.tokenizer.categorical_fields.get(field_name)
            if vocab is None:
                continue
            value = str(record[field_name])
            if value not in vocab.stoi or value in ("<UNK>", "<MASK>"):
                return self._reject(f"invalid_categorical_value:{field_name}={value}")

        # 4. valid issuer/rail/gateway relationships (if an action is
        # attached to this scenario)
        if "action_route_id" in record:
            route = record["action_route_id"]
            if route not in range(self.num_routes):
                return self._reject(f"invalid_route_id:{route}")
            expected_rail = self.valid_route_rail[route]
            expected_gateway = self.valid_route_gateway[route]
            if record.get("action_rail") not in (None, expected_rail):
                return self._reject(f"invalid_route_rail_pairing:route={route} rail={record.get('action_rail')}")
            if record.get("action_gateway") not in (None, expected_gateway):
                return self._reject(f"invalid_route_gateway_pairing:route={route} gateway={record.get('action_gateway')}")

        # 5. impossible state combinations / configured bounds
        if record["retry_count"] < 0 or record["retry_count"] > self.cfg.max_retry_count:
            return self._reject(f"retry_count_out_of_bounds:{record['retry_count']}")
        if record["previous_attempts"] < 0 or record["previous_attempts"] > self.cfg.max_previous_attempts:
            return self._reject(f"previous_attempts_out_of_bounds:{record['previous_attempts']}")
        if record["geo_bucket"] < 0 or record["time_bucket"] < 0:
            return self._reject("negative_bucket_index")

        # 6. temporal consistency (if a step index is present)
        if "step" in record and record["step"] is not None and record["step"] < 0:
            return self._reject("negative_step_index")

        return "VALID"

    def near_duplicate_key(self, record: Dict[str, Any]) -> str:
        """Coarse hash used for near-duplicate detection: rounds continuous
        fields to reduce sensitivity to negligible perturbations."""
        rounded = dict(record)
        if "amount" in rounded:
            rounded["amount"] = round(float(rounded["amount"]) / 100.0) * 100  # nearest 100
        keys = ["amount", "rail", "merchant_category", "issuer", "device_class", "geo_bucket", "time_bucket"]
        projected = {k: rounded.get(k) for k in keys}
        return stable_hash(projected)[: self.cfg.near_duplicate_hash_len]


class ScenarioRegistry:
    """Enforces the hard TRAIN/CERTIFICATION separation invariant across the
    whole Forge run: once a scenario hash is registered under one intended
    use, it can never be registered (or re-used) under the other."""

    def __init__(self):
        self._hash_to_use: Dict[str, str] = {}

    def register(self, scenario_hash: str, intended_use: str) -> bool:
        """Returns True if registration succeeded (no conflict), False if
        this scenario hash was already registered under the OTHER use."""
        existing = self._hash_to_use.get(scenario_hash)
        if existing is not None and existing != intended_use:
            return False
        self._hash_to_use[scenario_hash] = intended_use
        return True

    def contaminates(self, scenario_hash: str, intended_use: str) -> bool:
        existing = self._hash_to_use.get(scenario_hash)
        return existing is not None and existing != intended_use

    def all_hashes_for(self, intended_use: str) -> Set[str]:
        return {h for h, u in self._hash_to_use.items() if u == intended_use}
