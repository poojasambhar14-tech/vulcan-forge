"""
Runtime rollback.

THE GAP: ModelRegistry.rollback() only flipped a `rolled_back` flag and
returned the previous entry's metadata. It never deserialized the previous
adapter/heads back into the running objects. So the registry could report
"V1 restored" while the live process was still serving V2's weights -- the
two could disagree silently, which is the worst failure mode a rollback
mechanism can have.

Also: verify_rollback_hash_matches() compares one stored metadata field to
another. It does NOT re-hash the bytes on disk, so it cannot detect a
corrupted or altered checkpoint.

THE FIX: load_entry() re-hashes the checkpoint currently on disk, refuses
to load on mismatch, and restores weights into supplied live objects.
rollback_runtime() combines that with the registry flip so both ends agree.

The decisive assertion below is BEHAVIOURAL, not bookkeeping: predictions
after rollback must match the pre-promotion champion bit-for-bit.
"""
from __future__ import annotations

import tempfile

import torch

from vulcan.models.downstream_heads import DownstreamHeads
from vulcan.registry.model_registry import ModelRegistry


def _signature(heads: DownstreamHeads, probe: torch.Tensor) -> torch.Tensor:
    """A fixed-canary behavioural signature: same input, same weights =>
    identical output. This is what proves the RUNTIME rolled back, as
    opposed to only the registry's metadata."""
    with torch.no_grad():
        return heads(probe)["success_logit"].clone()


def test_rollback_restores_runtime_weights_and_behaviour():
    torch.manual_seed(0)
    d_model, n_routes = 32, 4
    registry = ModelRegistry(registry_dir=tempfile.mkdtemp())
    probe = torch.randn(6, d_model)

    v1 = DownstreamHeads(d_model, n_routes)
    v1_signature = _signature(v1, probe)
    registry.promote(None, v1, {"version_label": "V1", "role": "initial_champion"})

    # A genuinely different V2.
    v2 = DownstreamHeads(d_model, n_routes)
    with torch.no_grad():
        for p in v2.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    registry.promote(None, v2, {"version_label": "V2"})

    live = v2
    assert not torch.allclose(_signature(live, probe), v1_signature), (
        "test setup invalid: V2 must behave differently from V1"
    )

    result = registry.rollback_runtime("guardrail breach", heads=live,
                                        triggering_metric="live_success_rate")

    assert result["restored"] is True, result
    assert result["restored_label"] == "V1"
    assert result["load_report"]["hash_verified"] is True
    assert result["load_report"]["heads_loaded"] is True

    # THE decisive check: the live object now behaves exactly like V1.
    assert torch.allclose(_signature(live, probe), v1_signature, atol=1e-6), (
        "registry reported a rollback but the runtime object still behaves "
        "like the rolled-back champion"
    )


def test_load_entry_rehashes_disk_and_refuses_corrupted_checkpoint():
    """Hash verification must read the bytes on disk, not just compare two
    metadata fields -- otherwise a corrupted artifact loads silently."""
    torch.manual_seed(1)
    registry = ModelRegistry(registry_dir=tempfile.mkdtemp())
    heads = DownstreamHeads(16, 3)
    entry = registry.promote(None, heads, {"version_label": "V1"})

    # Corrupt the checkpoint on disk, leaving registry metadata untouched.
    with open(entry.checkpoint_path, "ab") as f:
        f.write(b"corruption")

    target = DownstreamHeads(16, 3)
    before = _signature(target, torch.randn(4, 16))
    report = registry.load_entry(entry, heads=target, verify_hash=True)

    assert report["hash_verified"] is False
    assert "error" in report
    assert report["heads_loaded"] is False, "a corrupted checkpoint must not be loaded"
    # metadata-only comparison would have wrongly said this was fine
    assert registry.verify_rollback_hash_matches(entry.checkpoint_sha256) is True


def test_rollback_from_adapter_champion_to_no_adapter_champion():
    """THE TOPOLOGY BUG. Rolling a champion that HAS an adapter back to one
    that does NOT must uninstall the adapter. The previous implementation
    only loaded an adapter when the checkpoint had adapter state, so this
    case silently left the rolled-back adapter installed: the registry said
    V1 was restored while the runtime kept applying V2's repair.

    Reproduced before the fix -- adapter outputs were bit-identical before
    and after 'rollback'. The contract is now that callers adopt
    `restored_adapter`, where None explicitly means 'uninstall yours'."""
    from vulcan.adaptation.adapter import BottleneckAdapter

    torch.manual_seed(2)
    registry = ModelRegistry(registry_dir=tempfile.mkdtemp())
    registry.promote(None, DownstreamHeads(32, 4), {"version_label": "V1"})  # no adapter
    adapter, v2_heads = BottleneckAdapter(32, 16), DownstreamHeads(32, 4)
    registry.promote(adapter, v2_heads, {"version_label": "V2"})             # has adapter

    champion_adapter = adapter  # runtime pointer
    result = registry.rollback_runtime("breach", adapter=adapter, heads=v2_heads)

    assert result["restored"] is True
    assert result["adapter_must_be_uninstalled"] is True, (
        "restoring a champion with no adapter must signal that the current "
        "adapter has to be uninstalled"
    )
    champion_adapter = result["restored_adapter"]
    assert champion_adapter is None, (
        "runtime still holds the rolled-back adapter -- registry and runtime disagree"
    )


def test_rollback_runtime_reports_failure_when_no_parent_exists():
    registry = ModelRegistry(registry_dir=tempfile.mkdtemp())
    heads = DownstreamHeads(8, 2)
    registry.promote(None, heads, {"version_label": "V1"})
    result = registry.rollback_runtime("breach", heads=heads)
    assert result["restored"] is False
    assert "no previous champion" in result["reason"]
