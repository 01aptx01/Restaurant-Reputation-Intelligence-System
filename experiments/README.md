# Experiments (แยกจาก production `outputs/`)

โฟลเดอร์นี้เก็บ **ผลการทดลอง / จูน / EDA** — ไม่ปนกับ pipeline จริง (`outputs/eval`, `artifacts/`)

## โครงสร้าง

```
experiments/
├── manifests/                          # YAML sweep configs
│   ├── baseline_sweep.yaml             # Baseline: balance, features, XGB params (20+ variants)
│   ├── xlmr_sweep.yaml                 # XLM-R: preprocess, loss, LR, batch (23 variants)
│   ├── embedding_sweep.yaml            # Embedding: models, classifiers, finetune (19 variants)
│   ├── augment_sweep.yaml              # Augmentation: intensity, target, probability (7 variants)
│   └── preprocess_alignment.yaml       # Preprocessing strategy alignment (6 variants)
├── results/{run_id}/                   # JSON จาก run_experiments.py
├── baseline/                           # try_log, feature ablation, tune results
├── xlmr/                               # preprocessing ablation logs
├── embedding/                          # model ablation logs
├── run_all_preprocess.py               # รันทดลอง XLM-R preprocessing ทุก strategy รวดเดียว
├── xlmr_preprocess_ablation.py         # XLM-R preprocessing ablation (6 strategies)
├── embedding_model_ablation.py         # เปรียบเทียบ embedding models (E5/BGE-M3/MiniLM/MPNet)
├── baseline_feature_ablation.py        # Baseline feature pipeline ablation (CV)
├── compare_results.py                  # รวบรวมผลทดลองทั้งหมดเปรียบเทียบ rank
└── test_preprocess.py                  # ตรวจผลลัพธ์ preprocessing ตัวอย่าง
```

## รันทดลอง

### 1. Manifest Sweeps (ครบวงจร: train + eval ทุก variant)

```powershell
# Baseline sweep (20+ variants)
python scripts/run_experiments.py experiments/manifests/baseline_sweep.yaml

# XLM-R sweep (23 variants)
python scripts/run_experiments.py experiments/manifests/xlmr_sweep.yaml

# Embedding sweep (19 variants)
python scripts/run_experiments.py experiments/manifests/embedding_sweep.yaml

# Augmentation sweep
python scripts/run_experiments.py experiments/manifests/augment_sweep.yaml

# รันเฉพาะบาง variant
python scripts/run_experiments.py experiments/manifests/baseline_sweep.yaml --variants v0_current_production a1_oversample3_no_weight

# ข้ามการเทรน (ใช้ artifacts เดิม)
python scripts/run_experiments.py experiments/manifests/xlmr_sweep.yaml --skip-train
```

### 2. Standalone Ablation Scripts (เร็ว, เจาะลึก)

```powershell
# XLM-R: เปรียบเทียบ preprocessing 6 strategies
python experiments/xlmr_preprocess_ablation.py --epochs 1

# XLM-R: รันทุก strategy รวดเดียว (full comparison)
python experiments/run_all_preprocess.py --epochs 1 --max-samples 800

# Embedding: เปรียบเทียบ 5 embedding models
python experiments/embedding_model_ablation.py

# Embedding: เลือกเฉพาะบาง model
python experiments/embedding_model_ablation.py --models e5-base bge-m3 --classifiers lr xgb

# Baseline: Feature ablation (word/char/extra/LSA combinations)
python experiments/baseline_feature_ablation.py --cv 3

# Baseline: Optuna hyperparameter optimization
python scripts/tune_baseline.py --append-try-log
```

### 3. เปรียบเทียบผลทดลอง

```powershell
# ดูผลทดลองทั้งหมดเรียง rank
python experiments/compare_results.py

# กรองเฉพาะโมเดล
python experiments/compare_results.py --model baseline

# ส่งออก CSV
python experiments/compare_results.py --export experiments/summary.csv
```

รายละเอียด: [docs/experiments.md](../docs/experiments.md)

## Production (คนละโฟลเดอร์)

| งาน | Path |
|-----|------|
| โมเดลที่ใช้จริง | `artifacts/baseline/`, `artifacts/xlmr/`, `artifacts/embedding/` |
| Eval หลัง pipeline | `outputs/eval/eval_report.json` |
| Dashboard HTML | `outputs/reports/eval_report_viz.html` |

Config: [src/rris/config.py](../src/rris/config.py) · เริ่มต้น: [docs/getting-started.md](../docs/getting-started.md)
