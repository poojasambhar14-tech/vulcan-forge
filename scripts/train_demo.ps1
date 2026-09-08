# scripts/train_demo.ps1
$py = ".venv\Scripts\python.exe"
& $py -m train.pretrain --config configs/tiny.yaml
& $py -m train.train_baseline_heads --config configs/tiny.yaml
& $py -m scripts.demo_part3 --config configs/tiny.yaml
& $py -m train.train_world_model --config configs/tiny.yaml
& $py -m scripts.generate_report
Write-Host "Demo checkpoint generated. See artifacts/runs/ and reports/final_report.md"
