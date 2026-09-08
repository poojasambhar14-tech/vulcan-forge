"""
FastAPI backend (master spec section 23). Serves ONLY data read from
persisted manifests under artifacts/runs and artifacts/registry -- nothing
here is hardcoded. Run with:

    uvicorn api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

RUNS_DIR = Path("artifacts/runs")
REGISTRY_DIR = Path("artifacts/registry")

app = FastAPI(title="Vulcan 2.0 API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Additive: Forge Control Room endpoints (api/forge_demo_api.py). Registered
# before the static-file mount below so these API paths are matched first.
from api.forge_demo_api import router as forge_demo_router  # noqa: E402
app.include_router(forge_demo_router)


def _load_run(run_id: str) -> Dict[str, Any]:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    with open(path) as f:
        return json.load(f)


def _list_runs(prefix: Optional[str] = None) -> List[str]:
    if not RUNS_DIR.exists():
        return []
    ids = [p.stem for p in RUNS_DIR.glob("*.json")]
    if prefix:
        ids = [i for i in ids if i.startswith(prefix)]
    return sorted(ids)


def _latest_run(prefix: str) -> Optional[Dict[str, Any]]:
    ids = _list_runs(prefix)
    if not ids:
        return None
    # run_ids embed a unix timestamp suffix -> lexicographic sort works for
    # same-prefix ids since timestamps are fixed-width-ish; sort numerically instead
    def ts(run_id: str) -> int:
        try:
            return int(run_id.rsplit("_", 1)[-1])
        except ValueError:
            return 0
    latest_id = max(ids, key=ts)
    return _load_run(latest_id)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/runs")
def list_runs():
    return {"run_ids": _list_runs()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return _load_run(run_id)


@app.get("/api/runs/{run_id}/manifest")
def get_run_manifest(run_id: str):
    return _load_run(run_id)


@app.get("/api/benchmark/latest")
def latest_benchmark():
    run = _latest_run("benchmark_")
    if run is None:
        raise HTTPException(status_code=404, detail="No benchmark runs found. Run scripts/benchmark.py first.")
    return run


@app.get("/api/drift/latest")
def latest_drift_demo():
    run = _latest_run("part3_demo_")
    if run is None:
        raise HTTPException(status_code=404, detail="No Part 3 demo runs found. Run scripts/demo_part3.py first.")
    return run


@app.get("/api/pretrain/latest")
def latest_pretrain():
    run = _latest_run("pretrain_")
    if run is None:
        raise HTTPException(status_code=404, detail="No pretraining runs found. Run train/pretrain.py first.")
    return run


@app.get("/api/registry")
def registry_index():
    idx_path = REGISTRY_DIR / "index.json"
    if not idx_path.exists():
        return {"entries": []}
    with open(idx_path) as f:
        return {"entries": json.load(f)}


# Serve the static dashboard (frontend/) at the root, if present.
if Path("frontend").exists():
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
