# Experiments

วิธีรันชุดทดลองจาก manifest YAML แยกจาก pipeline production

---

## Orchestrator

```powershell
pip install pyyaml   # หรือ pip install -e ".[dev]"
python scripts/run_experiments.py experiments/manifests/baseline_sweep.yaml
python scripts/run_experiments.py experiments/manifests/augment_sweep.yaml --run-id my_run
python scripts/run_experiments.py experiments/manifests/xlmr_sweep.yaml --skip-train
```

ผลลัพธ์: `experiments/results/{run_id}/{model}/{variant}.json` — MAE, F1, anomaly metrics บน holdout

---

## Manifests

| ไฟล์ | ทดลอง |
|------|--------|
| `experiments/manifests/baseline_sweep.yaml` | oversample, trunc, 3class, LSA, char TF-IDF |
| `experiments/manifests/xlmr_sweep.yaml` | preprocess strategies, regression vs classification |
| `experiments/manifests/embedding_sweep.yaml` | E5 frozen, preprocess variants |
| `experiments/manifests/preprocess_alignment.yaml` | เอกสาร alignment baseline |
| `experiments/manifests/augment_sweep.yaml` | none / minority / error-driven augment |

แต่ละ variant มี `config:` overrides ที่ patch `rris.config` ชั่วคราว แล้ว train + eval holdout

---

## สคริปต์อื่นใน `experiments/`

```powershell
python experiments/xlmr_preprocess_ablation.py --epochs 1
python scripts/tune_baseline.py --append-try-log
python scripts/eda_baseline_data.py
```

---

## โฟลเดอร์ผลลัพธ์

```
experiments/
├── manifests/          # YAML configs
├── results/{run_id}/   # JSON ต่อ variant
└── baseline/           # try_log จาก tune_baseline (legacy)
```

Production artifacts ยังอยู่ที่ `artifacts/` — การรัน experiment จะเขียนทับ artifacts ของโมเดลที่ train

---

## Smoke / CI

```powershell
$env:RRIS_SMOKE="1"
pytest tests/ -q
pytest tests/test_predict_roundtrip.py -q   # รัน train smoke จริง
```

ดูเมตริกที่บันทึก: [evaluation.md](evaluation.md)
