# Experiments

วิธีรันชุดทดลองจาก manifest YAML แยกจาก pipeline production

---

## Orchestrator

```powershell
pip install pyyaml   # หรือ pip install -e ".[dev]"

# รัน manifest sweep — train + eval ทุก variant
python scripts/run_experiments.py experiments/manifests/baseline_sweep.yaml
python scripts/run_experiments.py experiments/manifests/xlmr_sweep.yaml
python scripts/run_experiments.py experiments/manifests/embedding_sweep.yaml
python scripts/run_experiments.py experiments/manifests/augment_sweep.yaml --run-id my_run

# ตัวเลือกเพิ่มเติม
python scripts/run_experiments.py experiments/manifests/xlmr_sweep.yaml --skip-train
python scripts/run_experiments.py experiments/manifests/baseline_sweep.yaml --variants v0_current_production a1_oversample3_no_weight
```

ผลลัพธ์: `experiments/results/{run_id}/{model}/{variant}.json` — MAE, F1, anomaly metrics บน holdout

---

## Manifests

| ไฟล์ | ทดลอง | Variants |
|------|--------|----------|
| `baseline_sweep.yaml` | balance, features, XGB params, truncation, augment, combos | 20+ |
| `xlmr_sweep.yaml` | preprocess, loss (focal/MSE/CE), LR, batch/seq, 3-class, epochs | 23 |
| `embedding_sweep.yaml` | models (E5/BGE/MiniLM/MPNet), classifiers, finetune, preprocess | 19 |
| `augment_sweep.yaml` | augment intensity, target count, synonym/shuffle prob | 7 |
| `preprocess_alignment.yaml` | เอกสาร alignment baseline × 6 strategies | 6 |

แต่ละ variant มี `config:` overrides ที่ patch `rris.config` ชั่วคราว แล้ว train + eval holdout

---

## Standalone Ablation Scripts

```powershell
# XLM-R preprocessing ablation (6 strategies, 1 epoch each)
python experiments/xlmr_preprocess_ablation.py --epochs 1

# XLM-R full preprocessing comparison
python experiments/run_all_preprocess.py --epochs 1 --max-samples 800

# Embedding model comparison (E5/BGE-M3/MiniLM/MPNet × LR/XGB/SVM)
python experiments/embedding_model_ablation.py
python experiments/embedding_model_ablation.py --models e5-base bge-m3 --classifiers lr xgb

# Baseline feature pipeline ablation (word/char/extra/LSA × K-Fold CV)
python experiments/baseline_feature_ablation.py --cv 3

# Baseline hyperparameter tuning
python scripts/tune_baseline.py --append-try-log

# Baseline EDA
python scripts/eda_baseline_data.py
```

---

## เปรียบเทียบผลทดลอง

```powershell
# ดูผลจากทุก run เรียง rank ตาม MAE
python experiments/compare_results.py

# กรองเฉพาะโมเดล / run
python experiments/compare_results.py --model baseline
python experiments/compare_results.py --run-id 20260604T0100Z

# ส่งออก CSV
python experiments/compare_results.py --export experiments/summary.csv
```

---

## โฟลเดอร์ผลลัพธ์

```
experiments/
├── manifests/                  # YAML configs
├── results/{run_id}/           # JSON ต่อ variant (จาก run_experiments.py)
├── baseline/                   # feature ablation, try_log, tune_log
├── xlmr/                       # preprocessing ablation logs
├── embedding/                  # model ablation logs
├── compare_results.py          # เปรียบเทียบผลทดลองทั้งหมด
├── embedding_model_ablation.py # standalone embedding model comparison
├── baseline_feature_ablation.py # standalone feature pipeline ablation
├── xlmr_preprocess_ablation.py # standalone XLM-R preprocess comparison
└── test_preprocess.py          # ตรวจผลลัพธ์ preprocessing
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
