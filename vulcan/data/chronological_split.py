"""Chronological (never random) splitting of transaction records."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def chronological_split(
    records: List[Dict[str, Any]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    time_key: str = "step",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "fractions must sum to 1"

    records_sorted = sorted(records, key=lambda r: r[time_key])
    n = len(records_sorted)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = records_sorted[:n_train]
    val = records_sorted[n_train : n_train + n_val]
    test = records_sorted[n_train + n_val :]

    _assert_no_temporal_leakage(train, val, test, time_key)
    return train, val, test


def _assert_no_temporal_leakage(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    time_key: str,
) -> None:
    if not train or not val or not test:
        return
    max_train = max(r[time_key] for r in train)
    min_val = min(r[time_key] for r in val)
    max_val = max(r[time_key] for r in val)
    min_test = min(r[time_key] for r in test)

    assert max_train < min_val, (
        f"Temporal leakage: max(train)={max_train} >= min(val)={min_val}"
    )
    assert max_val < min_test, (
        f"Temporal leakage: max(val)={max_val} >= min(test)={min_test}"
    )
