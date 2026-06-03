# Experiments (แยกจาก production `outputs/`)

โฟลเดอร์นี้เก็บ **ผลการทดลอง / จูน / EDA** เท่านั้น — ไม่ปนกับ pipeline จริง (`outputs/eval`, `outputs/scores`, `artifacts/`)

## โครงสร้าง

```
experiments/
└── baseline/
    ├── try_log.md           # บันทึกทุก try + ceiling analysis
    ├── tune_log.json        # metrics ทุก candidate จาก tune_baseline.py
    ├── eda_summary.json     # EDA จาก eda_baseline_data.py
    ├── errors/              # error CSV จากช่วงทดลอง
    └── eval/                # eval JSON snapshot ระหว่าง sweep
```

## รันทดลอง

```powershell
python scripts/eda_baseline_data.py
python scripts/tune_baseline.py --append-try-log
python scripts/summarize_tune_ceiling.py
```

## Production (คนละโฟลเดอร์)

| งาน | Path |
|-----|------|
| โมเดลที่ใช้จริง | `artifacts/baseline/` |
| Eval หลัง pipeline | `outputs/eval/eval_report.json` |
| Score CSV | `outputs/scores/` |
| Dashboard HTML | `outputs/reports/eval_report_viz.html` |

รายละเอียด config: [src/rris/config.py](../src/rris/config.py) · คู่มือโปรเจกต: [docs/workflow.md](../docs/workflow.md)
