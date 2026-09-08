# scripts/benchmark.ps1
$py = ".venv\Scripts\python.exe"
& $py -m scripts.benchmark --config configs/tiny.yaml --seeds 1,2,3
