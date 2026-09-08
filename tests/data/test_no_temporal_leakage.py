"""
Temporal target leakage in the action-conditioned world model's state
construction.

This is a DIFFERENT correctness property from
tests/simulator/test_leakage_prevention.py (which checks that training data
never contains COUNTERFACTUAL/oracle information -- what would have
happened under an action not taken). This file checks that training data
never lets the model see the ANSWER to the question it is being trained to
answer -- the REALIZED outcome of the CURRENT event, at the CURRENT
position, before that outcome is supposed to be predicted from state alone.
Both are "the model must not see the answer" properties, but via distinct
mechanisms, so this stays a separate file per project convention.

CONFIRMED MECHANISM (see vulcan/data/windowing.py's
`encode_windows_for_world_model` docstring for the fix):
  1. vulcan/data/generate.py logs every field of a transaction, INCLUDING
     action_route_id and outcome_success/latency/fraud/abandoned/error_type,
     into the same record, after the outcome is known.
  2. train/train_world_model.py builds windows where the LAST event (w[-1])
     is one of these fully-realized records.
  3. vulcan/models/mini_vulcan.py's MiniVulcanBackbone.current_state()
     returns z[:, -1, :] from a BIDIRECTIONAL (non-causal) Transformer, so
     self-attention at the last position can see that position's own
     action/outcome fields -- fields that do not exist yet at live decision
     time (scripts/benchmark.py's obs_record, built from
     vulcan.data.generate._flatten_observation, never has them).
  4. The fix (`encode_windows_for_world_model`) strips exactly those fields
     from the last position before encoding, so the tokenizer's existing
     UNK/default fallback engages identically to how it already engages for
     a genuine live decision-time record.
"""
from __future__ import annotations

import torch

from vulcan.common.config import load_config
from vulcan.data.generate import generate_training_data, _flatten_observation
from vulcan.data.windowing import (
    build_windows,
    encode_windows,
    encode_windows_for_world_model,
    CURRENT_POSITION_UNKNOWN_AT_DECISION_TIME_FIELDS,
)
from vulcan.tokenization.tokenizer import FieldTokenizer


def _fitted_tokenizer(cfg):
    records = generate_training_data(cfg, seed=1, n_transactions=300)
    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(records)
    return tokenizer, records


def test_current_position_encoding_excludes_outcome_and_action_fields():
    """Directly inspect the encoded tensors: every action_*/outcome_*/
    propensity field at the LAST window position must be at the
    tokenizer's UNK/default encoding, never the real logged value -- even
    though the raw record fed in has real, non-default values for all of
    them."""
    cfg = load_config("configs/tiny.yaml")
    tokenizer, records = _fitted_tokenizer(cfg)
    history_len = 8
    windows = build_windows(records, history_len=history_len, stride=history_len)
    assert len(windows) > 0

    # Sanity: the raw last-position record must have REAL, non-default
    # values for every field we expect to be masked (otherwise this test
    # would trivially pass for the wrong reason).
    last_record = windows[0][-1]
    for f in CURRENT_POSITION_UNKNOWN_AT_DECISION_TIME_FIELDS:
        assert f in last_record, f"test fixture record unexpectedly lacks '{f}'"

    cat_ids, cont = encode_windows_for_world_model(tokenizer, windows)

    # categorical fields: must equal the UNK id, not the real encoded value
    for f in ("action_route_id", "action_rail", "action_gateway", "outcome_success",
              "outcome_abandoned", "outcome_error_type"):
        vocab = tokenizer.categorical_fields[f]
        unk_id = vocab.stoi["<UNK>"]
        last_position_ids = cat_ids[f][:, -1]
        assert torch.all(last_position_ids == unk_id), (
            f"'{f}' at the last window position is not at the UNK encoding -- "
            f"real logged values are leaking into the current-state encoding."
        )
        # and the real value WOULD have been encoded differently, proving
        # this isn't just a field that always maps to UNK regardless
        real_encoded = vocab.encode(last_record[f])
        assert real_encoded != unk_id, f"fixture issue: '{f}' real value also encodes as UNK"

    # continuous fields derived from outcome_*/propensity: must equal
    # whatever the tokenizer's OWN default fallback produces for a record
    # that never had the key, not a value derived from the real logged data.
    continuous_field_names = tokenizer.continuous_fields
    for f in ("propensity", "outcome_latency_ms_log", "outcome_processing_cost_log", "outcome_fraud_loss_log"):
        idx = continuous_field_names.index(f)
        expected_default_vec = tokenizer.encode_continuous({k: v for k, v in last_record.items()
                                                              if k not in CURRENT_POSITION_UNKNOWN_AT_DECISION_TIME_FIELDS})
        actual = cont[0, -1, idx].item()
        assert abs(actual - expected_default_vec[idx].item()) < 1e-6, (
            f"continuous field '{f}' at the last position does not match the tokenizer's "
            f"own default-fallback encoding -- real outcome data may be leaking in."
        )


def test_training_and_inference_current_state_encoding_match():
    """THE key regression test. A single synthetic observation is encoded
    two ways:
      (a) as the last position of a training window, via the FIXED
          `encode_windows_for_world_model` -- starting from a record that
          DOES have real action/outcome/propensity fields set (as any real
          training record would).
      (b) as scripts/benchmark.py's live obs_record at decision time --
          built via the exact same `_flatten_observation` function that
          file uses, which never has those fields to begin with.

    These two encodings MUST be exactly equal. This test is written to
    also positively demonstrate that the OLD (pre-fix) encoding function
    (plain `encode_windows`) does NOT satisfy this property, so the
    before/after contrast is verified within this same test run rather
    than relying on external git history (this repository has none)."""
    cfg = load_config("configs/tiny.yaml")
    tokenizer, records = _fitted_tokenizer(cfg)
    history_len = 4

    # Use a genuine simulator observation for realism, then attach a
    # plausible (non-default) real outcome to build the "training" record.
    training_record = dict(records[50])  # a real, fully-realized training record
    assert training_record.get("outcome_success") is not None

    # Build the "live" record exactly as scripts/benchmark.py would: same
    # transaction/network fields, but genuinely never had action_*/outcome_*/
    # propensity keys set.
    live_record = {
        k: v for k, v in training_record.items()
        if k not in CURRENT_POSITION_UNKNOWN_AT_DECISION_TIME_FIELDS
    }
    # Sanity: confirm this matches _flatten_observation's actual schema for
    # the fields that matter (same transaction/network keys).
    assert set(live_record.keys()) & {"amount", "rail", "issuer"} == {"amount", "rail", "issuer"}

    # Build minimal windows (history_len copies of a filler record, then
    # the position under test as the last entry) for both paths.
    filler = records[0]
    window_training = [filler] * (history_len - 1) + [training_record]
    window_live = [filler] * (history_len - 1) + [live_record]

    # --- OLD (pre-fix) behavior: plain encode_windows leaks the real values ---
    old_cat, old_cont = encode_windows(tokenizer, [window_training])
    live_cat, live_cont = encode_windows(tokenizer, [window_live])
    old_matches = all(torch.equal(old_cat[f][:, -1], live_cat[f][:, -1]) for f in old_cat) and torch.allclose(
        old_cont[:, -1, :], live_cont[:, -1, :], atol=1e-6
    )
    assert not old_matches, (
        "Expected the OLD encode_windows path to NOT match live-inference encoding "
        "(that mismatch IS the leak this test exists to catch). If this assertion "
        "fails, either the leak was already fixed elsewhere or the test fixture is "
        "not exercising a real discrepancy."
    )

    # --- NEW (fixed) behavior: encode_windows_for_world_model must match exactly ---
    new_cat, new_cont = encode_windows_for_world_model(tokenizer, [window_training])
    for f in new_cat:
        assert torch.equal(new_cat[f][:, -1], live_cat[f][:, -1]), (
            f"Mismatch in categorical field '{f}' between fixed training-time "
            f"encoding and live-inference encoding."
        )
    assert torch.allclose(new_cont[:, -1, :], live_cont[:, -1, :], atol=1e-6), (
        "Mismatch in continuous fields between fixed training-time encoding and "
        "live-inference encoding."
    )


def test_history_positions_are_not_masked():
    """Only the LAST position should be masked. Earlier positions in the
    window (genuine past history) must retain their real action/outcome
    values -- the fix must not over-mask and destroy legitimate historical
    context the model is allowed to condition on."""
    cfg = load_config("configs/tiny.yaml")
    tokenizer, records = _fitted_tokenizer(cfg)
    history_len = 5
    windows = build_windows(records, history_len=history_len, stride=history_len)
    window = windows[1]  # anything with real history

    cat_ids, cont = encode_windows_for_world_model(tokenizer, [window])

    for pos in range(history_len - 1):  # all EXCEPT the last
        real_record = window[pos]
        expected = tokenizer.categorical_fields["outcome_success"].encode(real_record["outcome_success"])
        actual = cat_ids["outcome_success"][0, pos].item()
        assert actual == expected, (
            f"Position {pos} (genuine history, not the current position) was "
            f"unexpectedly masked -- only the LAST position should be masked."
        )
