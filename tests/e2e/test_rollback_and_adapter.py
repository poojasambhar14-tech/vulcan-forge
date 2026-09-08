import torch

from vulcan.registry.model_registry import ModelRegistry
from vulcan.models.downstream_heads import DownstreamHeads
from vulcan.adaptation.adapter import BottleneckAdapter


def test_rollback_restores_exact_previous_checkpoint_hash(tmp_path):
    registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))

    heads_v1 = DownstreamHeads(d_model=16, num_routes=3)
    entry_v1 = registry.promote(None, heads_v1, {"note": "v1"})
    hash_v1 = entry_v1.checkpoint_sha256

    heads_v2 = DownstreamHeads(d_model=16, num_routes=3)
    adapter_v2 = BottleneckAdapter(d_model=16)
    entry_v2 = registry.promote(adapter_v2, heads_v2, {"note": "v2"})

    assert registry.current_champion().version == entry_v2.version

    restored = registry.rollback(reason="guardrail breach", triggering_metric="fraud_rate")
    assert restored is not None
    assert restored.version == entry_v1.version
    assert restored.checkpoint_sha256 == hash_v1
    assert registry.verify_rollback_hash_matches(hash_v1)
    assert entry_v2.rolled_back is True


def test_registry_persists_across_instances(tmp_path):
    reg_dir = str(tmp_path / "registry")
    registry1 = ModelRegistry(registry_dir=reg_dir)
    heads = DownstreamHeads(d_model=16, num_routes=3)
    entry = registry1.promote(None, heads, {"note": "v1"})

    registry2 = ModelRegistry(registry_dir=reg_dir)
    assert registry2.current_champion() is not None
    assert registry2.current_champion().checkpoint_sha256 == entry.checkpoint_sha256


def test_bottleneck_adapter_starts_as_identity():
    adapter = BottleneckAdapter(d_model=32, bottleneck_dim=8)
    z = torch.randn(5, 32)
    out = adapter(z)
    assert torch.allclose(z, out, atol=1e-6)  # zero-init up-projection -> identity at init
