"""
Step 0's core acceptance test: the CLI (scripts/demo_forge.py's main()) and
the new API path (api/forge_demo_api.py, driven by the SAME
run_forge_demo_stream() generator) must produce identical manifests for
the same seed -- proving the refactor did not introduce any divergence
between what a human running the CLI sees and what the Control Room UI
would show.

Both runs train real small models on this environment's single CPU core,
so this test is slow (~60-90s total) -- marked accordingly, same
convention as tests/e2e/test_bad_challenger_rejection_is_seed_robust.py.
"""
from __future__ import annotations

import glob
import json
import math
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

NON_DETERMINISTIC_KEY_SUBSTRINGS = (
    "run_id", "timestamp", "candidate_id", "scenario_id", "curriculum_id",
    "first_seen_run", "discovered_against", "manifest_path", "memory_dir",
)


def _is_nondeterministic_key(key: str) -> bool:
    key_lower = key.lower()
    return any(s in key_lower for s in NON_DETERMINISTIC_KEY_SUBSTRINGS)


def assert_manifests_match(a, b, path="root"):
    assert type(a) == type(b), f"type mismatch at {path}: {type(a)} vs {type(b)}"

    if isinstance(a, dict):
        assert set(a.keys()) == set(b.keys()), f"key mismatch at {path}: {set(a.keys())} vs {set(b.keys())}"
        for k in a:
            if _is_nondeterministic_key(k):
                continue
            assert_manifests_match(a[k], b[k], path=f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"list length mismatch at {path}: {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_manifests_match(x, y, path=f"{path}[{i}]")
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), f"float mismatch at {path}: {a} vs {b}"
    else:
        assert a == b, f"value mismatch at {path}: {a!r} vs {b!r}"


@pytest.mark.slow
def test_cli_and_api_produce_identical_manifests_for_same_seed(trained_checkpoints):
    seed = 2024
    config_path = "configs/tiny.yaml"

    existing_before = set(glob.glob("artifacts/runs/demo_forge_*.json"))
    result = __import__("subprocess").run(
        [sys.executable, "-m", "scripts.demo_forge", "--config", config_path, "--seed", str(seed), "--max_rounds", "4"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, f"CLI run failed:\n{result.stdout}\n{result.stderr}"
    existing_after = set(glob.glob("artifacts/runs/demo_forge_*.json"))
    new_files = existing_after - existing_before
    assert len(new_files) == 1, f"expected exactly one new manifest file, got {new_files}"
    cli_manifest_path = new_files.pop()
    with open(cli_manifest_path) as f:
        cli_manifest = json.load(f)

    client = TestClient(app)
    start_resp = client.post("/api/forge/demo/start", json={"seed": seed, "config": config_path, "max_rounds": 4})
    assert start_resp.status_code == 200
    run_id = start_resp.json()["run_id"]

    status = "running"
    for _ in range(120):
        events_resp = client.get(f"/api/forge/demo/{run_id}/events", params={"since": 0})
        assert events_resp.status_code == 200
        status = events_resp.json()["status"]
        if status == "complete":
            break
        if status == "error":
            pytest.fail(f"API demo run errored: {events_resp.json()['error']}")
        time.sleep(1)
    assert status == "complete", "API demo run did not complete within the poll budget"

    manifest_resp = client.get(f"/api/forge/demo/{run_id}/manifest")
    assert manifest_resp.status_code == 200
    api_manifest = manifest_resp.json()

    assert cli_manifest["final_outcome"] == api_manifest["final_outcome"]
    assert len(cli_manifest["rounds"]) == len(api_manifest["rounds"])
    assert_manifests_match(cli_manifest, api_manifest)
