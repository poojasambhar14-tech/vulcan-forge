"""
Field tokenizer for payment events.

Turns one flat transaction record (as produced by
`vulcan.data.generate.generate_training_data`) into:
  - a dict of categorical field -> integer id (with a reserved MASK id)
  - a vector of normalized continuous features (with a parallel mask flag
    vector, so the model can distinguish "masked" from "naturally zero")

Categorical vocabularies are built from the simulator config (known, closed
sets), not scanned from data, so there is no vocabulary leakage between
train/val/test splits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

MASK_TOKEN = "<MASK>"
UNK_TOKEN = "<UNK>"


@dataclass
class CategoricalVocab:
    name: str
    values: List[str]
    stoi: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        vocab = [UNK_TOKEN, MASK_TOKEN] + list(self.values)
        self.stoi = {v: i for i, v in enumerate(vocab)}

    @property
    def mask_id(self) -> int:
        return self.stoi[MASK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode(self, value: Any) -> int:
        return self.stoi.get(str(value), self.stoi[UNK_TOKEN])


class FieldTokenizer:
    """Builds fixed categorical vocabularies from config and normalization
    stats for continuous fields (fit on the TRAIN split only)."""

    def __init__(self, cfg: dict):
        sim_cfg = cfg["simulator"]
        num_routes = sim_cfg["num_routes"]
        self.num_routes = num_routes

        error_types = [
            "none", "issuer_decline", "gateway_timeout",
            "network_error", "fraud_block", "user_abandon",
        ]

        self.categorical_fields: Dict[str, CategoricalVocab] = {
            "rail": CategoricalVocab("rail", ["UPI", "CARD", "NETBANKING"]),
            "merchant_category": CategoricalVocab(
                "merchant_category",
                ["ecommerce", "food_delivery", "travel", "utilities", "gaming", "subscriptions"],
            ),
            "merchant_segment": CategoricalVocab("merchant_segment", list(sim_cfg["merchant_segments"])),
            "issuer": CategoricalVocab("issuer", list(sim_cfg["issuers"])),
            "device_class": CategoricalVocab("device_class", list(sim_cfg["device_classes"])),
            "geo_bucket": CategoricalVocab("geo_bucket", [str(i) for i in range(sim_cfg["geo_buckets"])]),
            "time_bucket": CategoricalVocab("time_bucket", [str(i) for i in range(sim_cfg["time_buckets"])]),
            "action_route_id": CategoricalVocab("action_route_id", [str(i) for i in range(num_routes)]),
            "action_rail": CategoricalVocab("action_rail", ["UPI", "CARD", "NETBANKING"]),
            "action_gateway": CategoricalVocab("action_gateway", [f"GW_{chr(65+i)}" for i in range(num_routes)]),
            "outcome_error_type": CategoricalVocab("outcome_error_type", error_types),
            "outcome_success": CategoricalVocab("outcome_success", ["False", "True"]),
            "outcome_abandoned": CategoricalVocab("outcome_abandoned", ["False", "True"]),
        }

        self.continuous_fields: List[str] = (
            [
                "amount_log",
                "previous_attempts",
                "retry_count",
                "customer_tenure_days",
                "propensity",
                "recent_issuer_decline_rate",
                "outcome_latency_ms_log",
                "outcome_processing_cost_log",
                "outcome_fraud_loss_log",
            ]
            + [f"route_{r}_rolling_success" for r in range(num_routes)]
            + [f"route_{r}_latency_ms_log" for r in range(num_routes)]
            + [f"route_{r}_timeout_rate" for r in range(num_routes)]
            + [f"route_{r}_load" for r in range(num_routes)]
            + [f"route_{r}_gw_availability" for r in range(num_routes)]
        )

        self._mean: Dict[str, float] = {}
        self._std: Dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    def _derive_continuous_raw(self, record: Dict[str, Any]) -> Dict[str, float]:
        raw = {
            "amount_log": math.log1p(record["amount"]),
            "previous_attempts": float(record["previous_attempts"]),
            "retry_count": float(record["retry_count"]),
            "customer_tenure_days": float(record["customer_tenure_days"]),
            "propensity": float(record.get("propensity", 0.5)),
            "recent_issuer_decline_rate": float(record.get("recent_issuer_decline_rate", 0.05)),
            "outcome_latency_ms_log": math.log1p(float(record.get("outcome_latency_ms", 0.0))),
            "outcome_processing_cost_log": math.log1p(float(record.get("outcome_processing_cost", 0.0))),
            "outcome_fraud_loss_log": math.log1p(float(record.get("outcome_fraud_loss", 0.0))),
        }
        for r in range(self.num_routes):
            raw[f"route_{r}_rolling_success"] = float(record.get(f"route_{r}_rolling_success", 0.85))
            raw[f"route_{r}_latency_ms_log"] = math.log1p(float(record.get(f"route_{r}_latency_ms", 300.0)))
            raw[f"route_{r}_timeout_rate"] = float(record.get(f"route_{r}_timeout_rate", 0.02))
            raw[f"route_{r}_load"] = float(record.get(f"route_{r}_load", 0.5))
            raw[f"route_{r}_gw_availability"] = float(record.get(f"route_{r}_gw_availability", 0.95))
        return raw

    def fit(self, train_records: List[Dict[str, Any]]) -> None:
        """Fit continuous normalization stats on the TRAIN split only."""
        accum: Dict[str, List[float]] = {f: [] for f in self.continuous_fields}
        for rec in train_records:
            raw = self._derive_continuous_raw(rec)
            for f in self.continuous_fields:
                accum[f].append(raw[f])
        for f in self.continuous_fields:
            arr = np.array(accum[f], dtype=np.float64)
            self._mean[f] = float(arr.mean()) if len(arr) else 0.0
            self._std[f] = float(arr.std()) if len(arr) else 1.0
            if self._std[f] < 1e-6:
                self._std[f] = 1.0
        self._fitted = True

    def encode_categorical(self, record: Dict[str, Any]) -> Dict[str, int]:
        out = {}
        for name, vocab in self.categorical_fields.items():
            value = record.get(name)
            out[name] = vocab.encode(value)
        return out

    def encode_continuous(self, record: Dict[str, Any]) -> np.ndarray:
        assert self._fitted, "FieldTokenizer.fit() must be called before encoding"
        raw = self._derive_continuous_raw(record)
        vec = np.array(
            [(raw[f] - self._mean[f]) / self._std[f] for f in self.continuous_fields],
            dtype=np.float32,
        )
        return vec

    @property
    def n_continuous(self) -> int:
        return len(self.continuous_fields)

    def vocab_sizes(self) -> Dict[str, int]:
        return {name: vocab.size for name, vocab in self.categorical_fields.items()}
