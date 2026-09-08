# scripts/bootstrap.ps1
# One-command setup. Does NOT require venv activation.
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
}
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pip install -e ".[dev,api]"
.venv\Scripts\python.exe -m pip install xgboost scikit-learn
Write-Host "Bootstrap complete. Use .venv\Scripts\python.exe to run scripts."
