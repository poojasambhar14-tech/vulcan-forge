"""Build fixed-length sliding windows of consecutive events and encode them
into tensors for the Mini-Vulcan backbone."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from vulcan.tokenization.tokenizer import FieldTokenizer


def build_windows(
    records: List[Dict[str, Any]], history_len: int, stride: int | None = None
) -> List[List[Dict[str, Any]]]:
    """Records must already be in chronological order. Returns overlapping
    windows of exactly `history_len` consecutive events."""
    if stride is None:
        stride = max(1, history_len // 4)
    windows = []
    for start in range(0, len(records) - history_len + 1, stride):
        windows.append(records[start : start + history_len])
    return windows


def build_windows_with_next(
    records: List[Dict[str, Any]], history_len: int, stride: int | None = None
):
    """Like build_windows, but also returns the single chronologically-next
    record after each window (or None if the window ends at the last
    record). Used by the world model's next-state prediction target."""
    if stride is None:
        stride = max(1, history_len // 4)
    windows = []
    next_records = []
    for start in range(0, len(records) - history_len + 1, stride):
        windows.append(records[start : start + history_len])
        next_idx = start + history_len
        next_records.append(records[next_idx] if next_idx < len(records) else None)
    return windows, next_records


def encode_windows(tokenizer: FieldTokenizer, windows: List[List[Dict[str, Any]]]):
    """Returns (categorical_ids: dict[name] -> LongTensor [N, K],
                continuous: FloatTensor [N, K, n_continuous]).

    NOTE: this encodes every field of every position in the window,
    including the last one. This is correct for masked-event-modeling
    pretraining (train/pretrain.py) and for downstream-head fine-tuning
    where the last position's OWN outcome is the training label (bandit
    feedback on the taken action -- see vulcan/models/downstream_heads.py).

    It is NOT correct for the action-conditioned world model
    (train/train_world_model.py), whose whole premise is predicting the
    last position's outcome from state alone -- see
    `encode_windows_for_world_model` below and
    tests/data/test_no_temporal_leakage.py.
    """
    field_names = list(tokenizer.categorical_fields.keys())
    N = len(windows)
    K = len(windows[0]) if windows else 0
    C = tokenizer.n_continuous

    cat_arrays = {name: np.zeros((N, K), dtype=np.int64) for name in field_names}
    cont_array = np.zeros((N, K, C), dtype=np.float32)

    for wi, window in enumerate(windows):
        for ei, record in enumerate(window):
            cat = tokenizer.encode_categorical(record)
            for name in field_names:
                cat_arrays[name][wi, ei] = cat[name]
            cont_array[wi, ei, :] = tokenizer.encode_continuous(record)

    categorical_ids = {name: torch.from_numpy(arr) for name, arr in cat_arrays.items()}
    continuous = torch.from_numpy(cont_array)
    return categorical_ids, continuous


# Fields that are only known AFTER an action is taken and its outcome is
# realized. These must never be visible in the "current" (last) position of
# a window that a decision-time model (the action-conditioned world model)
# is meant to predict from -- see docs/research_basis.md and the leakage
# analysis this function's caller cross-references. This set is exactly the
# fields present in a full training record (vulcan.data.generate) but
# ABSENT from a live decision-time observation
# (vulcan.data.generate._flatten_observation) -- computed as that set
# difference, not hand-picked, so it can't silently drift out of sync with
# either function.
CURRENT_POSITION_UNKNOWN_AT_DECISION_TIME_FIELDS = (
    "action_route_id",
    "action_rail",
    "action_gateway",
    "propensity",
    "outcome_success",
    "outcome_latency_ms",
    "outcome_processing_cost",
    "outcome_fraud_loss",
    "outcome_abandoned",
    "outcome_error_type",
)


def _mask_decision_time_unknown_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a shallow copy of `record` with every field in
    CURRENT_POSITION_UNKNOWN_AT_DECISION_TIME_FIELDS removed entirely (not
    set to some hand-picked placeholder). Removing the KEY, rather than
    setting a value, means `FieldTokenizer.encode_categorical`'s
    `record.get(name)` -> `None` -> str(None) -> UNK-token fallback, and
    `FieldTokenizer._derive_continuous_raw`'s `record.get(name, default)`
    fallback, both engage EXACTLY the same code path they already engage
    for a genuine live decision-time record
    (vulcan.data.generate._flatten_observation's output), which never had
    these keys in the first place. This is what makes the encoding
    identical in schema and semantics, not just superficially similar."""
    masked = dict(record)
    for field_name in CURRENT_POSITION_UNKNOWN_AT_DECISION_TIME_FIELDS:
        masked.pop(field_name, None)
    return masked


def encode_windows_for_world_model(tokenizer: FieldTokenizer, windows: List[List[Dict[str, Any]]]):
    """Same as `encode_windows`, EXCEPT the last position of every window
    (the "current state" the world model predicts an action's outcome
    from) has its action/outcome/propensity fields stripped before
    encoding, so the model can never see the answer to the question it is
    being trained to answer. All earlier positions (genuine history) are
    encoded exactly as `encode_windows` would encode them.

    Fixes the temporal target leakage described in
    tests/data/test_no_temporal_leakage.py: MiniVulcanBackbone.current_state()
    returns z[:, -1, :] from a bidirectional (non-causal) Transformer, so
    self-attention at the last position could otherwise see that same
    position's own action_route_id/outcome_success/outcome_latency_ms/
    outcome_fraud_loss/outcome_abandoned fields -- fields that do not exist
    yet at live decision time (see vulcan.data.generate._flatten_observation,
    which scripts/benchmark.py's rollout path uses to build obs_record)."""
    field_names = list(tokenizer.categorical_fields.keys())
    N = len(windows)
    K = len(windows[0]) if windows else 0
    C = tokenizer.n_continuous

    cat_arrays = {name: np.zeros((N, K), dtype=np.int64) for name in field_names}
    cont_array = np.zeros((N, K, C), dtype=np.float32)

    for wi, window in enumerate(windows):
        last_idx = len(window) - 1
        for ei, record in enumerate(window):
            effective_record = _mask_decision_time_unknown_fields(record) if ei == last_idx else record
            cat = tokenizer.encode_categorical(effective_record)
            for name in field_names:
                cat_arrays[name][wi, ei] = cat[name]
            cont_array[wi, ei, :] = tokenizer.encode_continuous(effective_record)

    categorical_ids = {name: torch.from_numpy(arr) for name, arr in cat_arrays.items()}
    continuous = torch.from_numpy(cont_array)
    return categorical_ids, continuous
