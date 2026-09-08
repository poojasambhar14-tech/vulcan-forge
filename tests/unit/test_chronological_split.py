import pytest

from vulcan.data.chronological_split import chronological_split


def _fake_records(n):
    return [{"step": i, "value": i * 2} for i in range(n)]


def test_split_is_chronological_and_non_overlapping():
    records = _fake_records(1000)
    train, val, test = chronological_split(records, 0.7, 0.15, 0.15)

    assert max(r["step"] for r in train) < min(r["step"] for r in val)
    assert max(r["step"] for r in val) < min(r["step"] for r in test)
    assert len(train) + len(val) + len(test) == 1000


def test_split_rejects_bad_fractions():
    records = _fake_records(10)
    with pytest.raises(AssertionError):
        chronological_split(records, 0.5, 0.3, 0.3)


def test_split_handles_shuffled_input_by_sorting():
    import random

    records = _fake_records(500)
    random.shuffle(records)
    train, val, test = chronological_split(records, 0.7, 0.15, 0.15)
    assert max(r["step"] for r in train) < min(r["step"] for r in val)
    assert max(r["step"] for r in val) < min(r["step"] for r in test)
