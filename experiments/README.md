# Experiments (แยกจาก production `outputs/`)

โฟลเดอร์นี้เก็บ **ผลการทดลอง / จูน / EDA** — ไม่ปนกับ pipeline จริง (`outputs/eval`, `artifacts/`)

## โครงสร้าง

```
experiments/
├── manifests/              # YAML sweep configs
├── results/{run_id}/       # JSON จาก run_experiments.py
└── baseline/               # try_log จาก tune_baseline (legacy)
```

## รันทดลอง

```powershell
python scripts/run_experiments.py experiments/manifests/baseline_sweep.yaml
python scripts/tune_baseline.py --append-try-log
python experiments/xlmr_preprocess_ablation.py --epochs 1
```

รายละเอียด: [docs/experiments.md](../docs/experiments.md)

## Production (คนละโฟลเดอร์)

| งาน | Path |
|-----|------|
| โมเดลที่ใช้จริง | `artifacts/baseline/`, `artifacts/xlmr/`, `artifacts/embedding/` |
| Eval หลัง pipeline | `outputs/eval/eval_report.json` |
| Dashboard HTML | `outputs/reports/eval_report_viz.html` |

Config: [src/rris/config.py](../src/rris/config.py) · เริ่มต้น: [docs/getting-started.md](../docs/getting-started.md)
