"""
Baseline A (master spec section 17): XGBoost over flat engineered tabular
features, selecting the route with highest predicted success probability.
Deliberately does not model fraud/latency trade-offs -- this is a naive but
standard baseline (see docs/limitations.md), not a hidden strawman.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLS = [
    "rail", "merchant_category", "merchant_segment", "issuer", "device_class",
    "action_rail", "action_gateway",
]
NUMERIC_COLS_BASE = [
    "amount", "geo_bucket", "time_bucket", "previous_attempts", "retry_count",
    "customer_tenure_days", "propensity", "recent_issuer_decline_rate", "action_route_id",
]


class XGBoostBaseline:
    def __init__(self, num_routes: int):
        self.num_routes = num_routes
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_cols: List[str] = []
        self.model: xgb.XGBClassifier | None = None

    def _numeric_cols(self) -> List[str]:
        cols = list(NUMERIC_COLS_BASE)
        for r in range(self.num_routes):
            cols += [f"route_{r}_rolling_success", f"route_{r}_latency_ms", f"route_{r}_timeout_rate", f"route_{r}_load"]
        return cols

    def _build_df(self, records: List[Dict[str, Any]], fit_encoders: bool) -> pd.DataFrame:
        df = pd.DataFrame(records)
        for col in CATEGORICAL_COLS:
            if fit_encoders:
                enc = LabelEncoder()
                df[col + "_enc"] = enc.fit_transform(df[col].astype(str))
                self.encoders[col] = enc
            else:
                enc = self.encoders[col]
                df[col + "_enc"] = df[col].astype(str).map(
                    {cls: i for i, cls in enumerate(enc.classes_)}
                ).fillna(-1)
        feature_cols = [c + "_enc" for c in CATEGORICAL_COLS] + self._numeric_cols()
        if fit_encoders:
            self.feature_cols = feature_cols
        return df.reindex(columns=feature_cols).fillna(0.0)

    def fit(self, records: List[Dict[str, Any]]) -> None:
        X = self._build_df(records, fit_encoders=True)
        y = np.array([float(bool(r["outcome_success"])) for r in records])
        self.model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", n_jobs=2,
        )
        self.model.fit(X, y)

    def predict_success_proba_for_route(self, obs_record: Dict[str, Any], route_id: int, rail: str, gateway: str) -> float:
        rec = dict(obs_record)
        rec["action_route_id"] = route_id
        rec["action_rail"] = rail
        rec["action_gateway"] = gateway
        X = self._build_df([rec], fit_encoders=False)
        return float(self.model.predict_proba(X)[0, 1])

    def choose_route(self, obs_record: Dict[str, Any], route_specs: List[Dict[str, Any]]) -> int:
        """route_specs: list of {route_id, rail, gateway}. Chooses argmax
        predicted success probability."""
        scores = [
            self.predict_success_proba_for_route(obs_record, spec["route_id"], spec["rail"], spec["gateway"])
            for spec in route_specs
        ]
        return route_specs[int(np.argmax(scores))]["route_id"]
