"""
Champion/challenger shadow evaluation + promotion rule (master spec section 14).

The challenger is NEVER deployed directly. It is evaluated against:
  A. recent drift traffic (holdout, not used in candidate training)
  B. a fixed historical canary regime (holdout, pre-shift "normal" data)
  C. a calibration set (ECE)

Promotion requires ALL of:
  - recent-drift utility does not regress vs champion by more than
    `min_recent_improvement_or_tolerance`
  - historical canary accuracy does not regress beyond `max_historical_regression`
  - calibration (ECE) does not regress beyond `max_calibration_regression`

All thresholds are configured (not hand-tuned per scenario) and every
decision returns structured, itemized reasons -- never a bare boolean.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch

from vulcan.drift.detectors import expected_calibration_error


class PromotionDecision(str, enum.Enum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    HOLD = "HOLD"


@dataclass
class GateThresholds:
    max_recent_regression: float = 0.02       # accuracy may drop at most this much on recent-drift set
    max_historical_regression: float = 0.03   # accuracy may drop at most this much on historical canary
    max_historical_bce_regression: float = 0.03  # historical log-loss may rise at most this much
    max_calibration_regression: float = 0.05  # ECE may rise at most this much


def _predict(backbone, adapter, heads, tokenizer, windows):
    from vulcan.data.windowing import encode_windows_for_world_model

    # INVARIANT: every accuracy/BCE/ECE number produced here is computed
    # with the transaction's own outcome masked out of the model input, so
    # scores reflect decision-time information only.
    cat, cont = encode_windows_for_world_model(tokenizer, windows)
    cont_full = torch.cat([cont, torch.zeros_like(cont)], dim=-1)
    with torch.no_grad():
        z = backbone.current_state(cat, cont_full)
        if adapter is not None:
            z = adapter(z)
        out = heads(z)
        idx = torch.arange(z.shape[0])
        taken_route = torch.tensor([w[-1]["action_route_id"] for w in windows], dtype=torch.long)
        p_success = torch.sigmoid(out["success_logit"])[idx, taken_route].numpy()
    labels = np.array([float(bool(w[-1]["outcome_success"])) for w in windows])
    return p_success, labels


def evaluate_model_on_slice(backbone, adapter, heads, tokenizer, windows) -> Dict[str, float]:
    if len(windows) == 0:
        return {"accuracy": None, "bce": None, "ece": None, "n": 0}
    probs, labels = _predict(backbone, adapter, heads, tokenizer, windows)
    preds = (probs > 0.5).astype(float)
    accuracy = float((preds == labels).mean())
    eps = 1e-7
    bce = float(-np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)))
    ece = expected_calibration_error(probs, labels)
    return {"accuracy": accuracy, "bce": bce, "ece": ece, "n": len(windows)}


def shadow_evaluate_and_decide(
    backbone,
    tokenizer,
    champion_heads,
    challenger_adapter,
    challenger_heads,
    recent_drift_holdout: List[list],
    historical_canary_holdout: List[list],
    thresholds: GateThresholds,
) -> Dict[str, Any]:
    champ_recent = evaluate_model_on_slice(backbone, None, champion_heads, tokenizer, recent_drift_holdout)
    chall_recent = evaluate_model_on_slice(backbone, challenger_adapter, challenger_heads, tokenizer, recent_drift_holdout)

    champ_hist = evaluate_model_on_slice(backbone, None, champion_heads, tokenizer, historical_canary_holdout)
    chall_hist = evaluate_model_on_slice(backbone, challenger_adapter, challenger_heads, tokenizer, historical_canary_holdout)

    reasons = []
    decision = PromotionDecision.PROMOTE

    recent_delta = chall_recent["accuracy"] - champ_recent["accuracy"]
    if recent_delta < -thresholds.max_recent_regression:
        decision = PromotionDecision.REJECT
        reasons.append(
            f"Recent-drift accuracy regressed by {-recent_delta:.4f} "
            f"(champion={champ_recent['accuracy']:.4f} challenger={chall_recent['accuracy']:.4f}), "
            f"exceeds max_recent_regression={thresholds.max_recent_regression}"
        )
    else:
        reasons.append(
            f"Recent-drift accuracy delta={recent_delta:+.4f} within tolerance "
            f"(champion={champ_recent['accuracy']:.4f} challenger={chall_recent['accuracy']:.4f})"
        )

    hist_delta = chall_hist["accuracy"] - champ_hist["accuracy"]
    if hist_delta < -thresholds.max_historical_regression:
        decision = PromotionDecision.REJECT
        reasons.append(
            f"Historical-canary accuracy regressed by {-hist_delta:.4f} "
            f"(champion={champ_hist['accuracy']:.4f} challenger={chall_hist['accuracy']:.4f}), "
            f"exceeds max_historical_regression={thresholds.max_historical_regression}. "
            f"Likely catastrophic forgetting."
        )
    else:
        reasons.append(
            f"Historical-canary accuracy delta={hist_delta:+.4f} within tolerance "
            f"(champion={champ_hist['accuracy']:.4f} challenger={chall_hist['accuracy']:.4f})"
        )

    calib_delta = chall_hist["ece"] - champ_hist["ece"]
    if calib_delta > thresholds.max_calibration_regression:
        decision = PromotionDecision.REJECT
        reasons.append(
            f"Historical calibration (ECE) regressed by {calib_delta:.4f}, "
            f"exceeds max_calibration_regression={thresholds.max_calibration_regression}"
        )
    else:
        reasons.append(f"Calibration ECE delta={calib_delta:+.4f} within tolerance")

    hist_bce_delta = chall_hist["bce"] - champ_hist["bce"]
    if hist_bce_delta > thresholds.max_historical_bce_regression:
        decision = PromotionDecision.REJECT
        reasons.append(
            f"Historical-canary log-loss (BCE) regressed by {hist_bce_delta:.4f} "
            f"(champion={champ_hist['bce']:.4f} challenger={chall_hist['bce']:.4f}), "
            f"exceeds max_historical_bce_regression={thresholds.max_historical_bce_regression}. "
            f"Accuracy alone did not catch this because success labels are imbalanced; "
            f"log-loss exposes the challenger's overconfident, miscalibrated predictions "
            f"on the historical (non-drift) regime -- a sign of catastrophic forgetting."
        )
    else:
        reasons.append(f"Historical-canary log-loss delta={hist_bce_delta:+.4f} within tolerance")

    return {
        "decision": decision.value,
        "reasons": reasons,
        "champion": {"recent": champ_recent, "historical": champ_hist},
        "challenger": {"recent": chall_recent, "historical": chall_hist},
        "thresholds": thresholds.__dict__,
    }
