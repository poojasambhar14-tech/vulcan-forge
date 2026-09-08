"""
Model registry + rollback (master spec sections 15, 22).

Keeps an append-only history of promoted models. Rollback restores the
exact previous champion checkpoint (verified by SHA256 hash equality).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from vulcan.common.config import file_sha256


@dataclass
class RegistryEntry:
    version: int
    utc_timestamp: str
    checkpoint_path: str
    checkpoint_sha256: str
    promotion_manifest: Dict[str, Any]
    rolled_back: bool = False


class ModelRegistry:
    def __init__(self, registry_dir: str = "artifacts/registry"):
        self.dir = Path(registry_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.history: List[RegistryEntry] = []
        self._load()

    def _index_path(self) -> Path:
        return self.dir / "index.json"

    def _load(self):
        idx = self._index_path()
        if idx.exists():
            with open(idx) as f:
                raw = json.load(f)
            self.history = [RegistryEntry(**e) for e in raw]

    def _save(self):
        with open(self._index_path(), "w") as f:
            json.dump([e.__dict__ for e in self.history], f, indent=2)

    def current_champion(self) -> Optional[RegistryEntry]:
        for entry in reversed(self.history):
            if not entry.rolled_back:
                return entry
        return None

    def promote(self, adapter, heads, promotion_manifest: Dict[str, Any]) -> RegistryEntry:
        version = len(self.history) + 1
        ckpt_path = self.dir / f"champion_v{version}.pt"
        torch.save({"adapter_state_dict": adapter.state_dict() if adapter else None,
                    "heads_state_dict": heads.state_dict()}, ckpt_path)
        ckpt_hash = file_sha256(ckpt_path)
        entry = RegistryEntry(
            version=version,
            utc_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            checkpoint_path=str(ckpt_path),
            checkpoint_sha256=ckpt_hash,
            promotion_manifest=promotion_manifest,
        )
        self.history.append(entry)
        self._save()
        return entry

    def rollback(self, reason: str, triggering_metric: Optional[str] = None) -> Optional[RegistryEntry]:
        """Marks the current champion as rolled back and returns the
        restored (previous) champion entry."""
        current = self.current_champion()
        if current is None:
            return None
        current.rolled_back = True
        current.promotion_manifest["rollback_reason"] = reason
        current.promotion_manifest["rollback_triggering_metric"] = triggering_metric
        current.promotion_manifest["rollback_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._save()
        restored = self.current_champion()
        return restored

    def verify_rollback_hash_matches(self, expected_hash: str) -> bool:
        restored = self.current_champion()
        return restored is not None and restored.checkpoint_sha256 == expected_hash

    def load_entry(self, entry: RegistryEntry, adapter=None, heads=None,
                    verify_hash: bool = True, adapter_factory=None, heads_factory=None) -> Dict[str, Any]:
        """Deserializes a registry entry's checkpoint back into live model
        objects -- the missing half of rollback.

        Returns authoritative restored state rather than relying on
        mutating whatever objects the caller happens to hold. The report's
        `restored_adapter` and `restored_heads` are the source of truth;
        `restored_adapter is None` explicitly means the restored champion
        runs WITHOUT an adapter and the caller must uninstall its own.
        Callers perform an atomic pointer swap from those values.

        `verify_hash` re-hashes the checkpoint bytes CURRENTLY ON DISK, so a
        corrupted or altered artifact is refused rather than loaded.
        """
        ckpt_path = Path(entry.checkpoint_path)
        report: Dict[str, Any] = {
            "version": entry.version,
            "checkpoint_path": str(ckpt_path),
            "exists": ckpt_path.exists(),
            "hash_verified": None,
            "adapter_loaded": False,
            "heads_loaded": False,
            "restored_adapter": None,
            "restored_heads": None,
        }
        if not ckpt_path.exists():
            report["error"] = "checkpoint file missing"
            return report

        if verify_hash:
            actual = file_sha256(ckpt_path)
            report["expected_sha256"] = entry.checkpoint_sha256
            report["actual_sha256"] = actual
            report["hash_verified"] = (actual == entry.checkpoint_sha256)
            if not report["hash_verified"]:
                report["error"] = "checkpoint hash mismatch -- refusing to load"
                return report

        blob = torch.load(ckpt_path, weights_only=False)
        adapter_state = blob.get("adapter_state_dict")
        report["had_adapter_state"] = adapter_state is not None

        # ---- heads ----
        if blob.get("heads_state_dict") is not None:
            target_heads = heads
            if target_heads is None and heads_factory is not None:
                target_heads = heads_factory()
            if target_heads is not None:
                target_heads.load_state_dict(blob["heads_state_dict"])
                report["heads_loaded"] = True
                report["restored_heads"] = target_heads

        # ---- adapter, including the "restored champion has none" case ----
        if adapter_state is not None:
            target_adapter = adapter
            if target_adapter is None and adapter_factory is not None:
                target_adapter = adapter_factory()
            if target_adapter is not None:
                target_adapter.load_state_dict(adapter_state)
                report["adapter_loaded"] = True
                report["restored_adapter"] = target_adapter
        else:
            # Authoritative: the restored champion runs WITHOUT an adapter.
            # The caller must uninstall whatever adapter it currently holds.
            report["restored_adapter"] = None
            report["adapter_must_be_uninstalled"] = True

        return report

    def rollback_runtime(self, reason: str, adapter=None, heads=None,
                          triggering_metric: Optional[str] = None,
                          adapter_factory=None, heads_factory=None) -> Dict[str, Any]:
        """Full rollback: mark the current champion rolled back, then restore
        the previous champion's weights with on-disk hash verification.

        Callers MUST adopt `restored_adapter` / `restored_heads` from the
        result rather than assuming their existing objects were mutated --
        `restored_adapter` of None means the previous champion had no adapter
        and the current one must be uninstalled."""
        restored = self.rollback(reason, triggering_metric)
        if restored is None:
            return {"restored": False, "reason": "no previous champion to roll back to"}
        load_report = self.load_entry(restored, adapter=adapter, heads=heads,
                                       verify_hash=True,
                                       adapter_factory=adapter_factory,
                                       heads_factory=heads_factory)
        ok = bool(load_report.get("hash_verified")) and "error" not in load_report
        return {
            "restored": ok,
            "restored_version": restored.version,
            "restored_label": restored.promotion_manifest.get("version_label"),
            "restored_adapter": load_report.get("restored_adapter"),
            "restored_heads": load_report.get("restored_heads"),
            "adapter_must_be_uninstalled": load_report.get("adapter_must_be_uninstalled", False),
            "load_report": load_report,
        }
