"""
Failure Memory (Forge pipeline stage: after red-team discovery).

Every independently discovered challenger failure becomes a PERMANENT
regression case. Every future challenger must pass the entire accumulated
suite, not just the new red-team run that most recently found something.
Deduplicated by scenario hash so identical scenarios don't inflate the
suite. Follows the same JSON-backed persistence pattern as
vulcan/registry/model_registry.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from vulcan.forge.schemas import FailureMemoryEntry


class FailureMemory:
    def __init__(self, memory_dir: str = "artifacts/forge/failure_memory"):
        self.dir = Path(memory_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[FailureMemoryEntry] = []
        self._load()

    def _index_path(self) -> Path:
        return self.dir / "index.json"

    def _load(self):
        idx = self._index_path()
        if idx.exists():
            with open(idx) as f:
                raw = json.load(f)
            self.entries = [FailureMemoryEntry(**e) for e in raw]

    def _save(self):
        with open(self._index_path(), "w") as f:
            json.dump([e.__dict__ for e in self.entries], f, indent=2, default=str)

    def known_hashes(self) -> set:
        return {e.scenario_hash for e in self.entries}

    def add(self, entry: FailureMemoryEntry) -> bool:
        """Returns True if newly added, False if it was a duplicate of an
        existing entry (by scenario_hash) and was skipped."""
        if entry.scenario_hash in self.known_hashes():
            return False
        self.entries.append(entry)
        self._save()
        return True

    def regression_suite(self) -> List[FailureMemoryEntry]:
        return list(self.entries)

    def version(self) -> str:
        """A stable identifier for 'which failure memory state' a
        certification run was checked against -- for provenance manifests."""
        from vulcan.forge.schemas import stable_hash
        return stable_hash(sorted(e.scenario_hash for e in self.entries))

    def stats(self) -> Dict[str, Any]:
        return {
            "n_entries": len(self.entries),
            "version": self.version(),
            "severity_mean": (sum(e.severity for e in self.entries) / len(self.entries)) if self.entries else 0.0,
        }
