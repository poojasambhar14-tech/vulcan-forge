"""
Additive API surface for the Forge Control Room UI (frontend/forge_control_room.html).

Does not modify any existing endpoint in api/main.py. Every value this
router ever returns comes from a real event yielded by
scripts.demo_forge.run_forge_demo_stream(), which is the SAME generator
the CLI consumes -- see that module's STREAMING REFACTOR NOTE for the
one honest caveat on event timing (batched-but-real, not incremental).

Runs the (blocking, real, several-seconds) demo in a background thread so
the HTTP request returns immediately; events accumulate in an in-memory
per-run buffer that the UI polls.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from scripts.demo_forge import run_forge_demo_stream

router = APIRouter(prefix="/api/forge/demo", tags=["forge-demo"])

_RUNS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


class StartDemoRequest(BaseModel):
    seed: int = 2024
    config: str = "configs/tiny.yaml"
    max_rounds: int = 4
    failure_memory_dir: Optional[str] = None


def _run_worker(run_id: str, req: StartDemoRequest) -> None:
    try:
        for evt in run_forge_demo_stream(req.config, req.seed, req.max_rounds, req.failure_memory_dir):
            with _LOCK:
                _RUNS[run_id]["events"].append(evt)
        with _LOCK:
            _RUNS[run_id]["status"] = "complete"
    except Exception as e:
        with _LOCK:
            _RUNS[run_id]["status"] = "error"
            _RUNS[run_id]["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


@router.post("/start")
def start_demo(req: StartDemoRequest):
    run_id = f"apidemo_{int(time.time() * 1000)}"
    with _LOCK:
        _RUNS[run_id] = {"events": [], "status": "running", "error": None, "seed": req.seed, "started_at": time.time()}
    thread = threading.Thread(target=_run_worker, args=(run_id, req), daemon=True)
    thread.start()
    return {"run_id": run_id}


@router.get("/{run_id}/events")
def get_events(run_id: str, since: int = 0):
    with _LOCK:
        run = _RUNS.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}")
        events = run["events"][since:]
        next_cursor = len(run["events"])
        status = run["status"]
        error = run["error"]
    return {"events": events, "next_cursor": next_cursor, "status": status, "error": error}


@router.get("/certification-evidence")
def certification_evidence():
    """Serves the most recent Silent Regression Challenge manifest -- the
    multi-seed benchmark measuring whether red-team certification catches
    regressions ordinary validation misses. Read live from
    artifacts/runs/silent_regression_*.json; nothing is hardcoded, and if
    no benchmark has been run this returns available=false rather than
    inventing numbers."""
    import glob
    paths = sorted(glob.glob("artifacts/runs/silent_regression_*.json"))
    if not paths:
        return {"available": False, "reason": "No silent_regression_*.json manifest found in artifacts/runs/. "
                                                "Run: python -m scripts.silent_regression_benchmark --seeds 1,2,3,4,5"}
    with open(paths[-1]) as f:
        data = json.load(f)
    return {"available": True, "manifest_path": paths[-1], **data}


@router.get("/{run_id}/manifest")
def get_manifest(run_id: str):
    with _LOCK:
        run = _RUNS.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}")
        if run["status"] != "complete":
            raise HTTPException(status_code=409, detail=f"Run {run_id} not complete yet (status={run['status']})")
        events = run["events"]

    demo_complete = next((e for e in events if e["event"] == "DEMO_COMPLETE"), None)
    if demo_complete is None:
        raise HTTPException(status_code=500, detail="Run completed without a DEMO_COMPLETE event")

    manifest_path = Path(demo_complete["data"]["manifest_path"])
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Manifest file not found at {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)
