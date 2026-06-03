# Directory Guide

คู่มือว่าแต่ละโฟลเดอร์/ไฟล์ทำอะไร — สำหรับคนเปิด repo ครั้งแรก

## โครงสร้างระดับบน

```text
Restaurant-Reputation-Intelligence-System/
├── README.md              ← เริ่มที่นี่
├── docs/                  ← คู่มือทั้งหมด
├── src/rris/              ← โค้ด Python หลัก (package)
├── scripts/               ← เตรียมข้อมูล / EDA / tune (รันครั้งเดียว)
├── data/                  ← ชุดข้อมูล (gitignore ไฟล์ใหญ่)
├── artifacts/             ← น้ำหนักโมเดลหลังเทรน
├── outputs/               ← ผล production
├── experiments/           ← ผลทดลอง / preprocess ablation
├── notebooks/             ← Jupyter pipeline (optional)
├── web_app/               ← Bun + Elysia + แผนที่
├── pyproject.toml
└── requirements.txt
```

## `docs/`

| ไฟล์ | เนื้อหา |
|------|---------|
| [getting-started.md](getting-started.md) | ติดตั้งและรัน pipeline |
| [directory-guide.md](directory-guide.md) | คู่มือนี้ |
| [preprocessing/README.md](preprocessing/README.md) | สรุป preprocessing ทุกโมเดล |
| [preprocessing/baseline.md](preprocessing/baseline.md) | TF-IDF, extra features |
| [preprocessing/xlmr.md](preprocessing/xlmr.md) | XLM-R tokenization |
| [preprocessing/embedding.md](preprocessing/embedding.md) | Sentence embedding |
| [preprocessing/text-normalization.md](preprocessing/text-normalization.md) | ฟังก์ชัน normalize ข้อความ |
| [training/README.md](training/README.md) | สรุปเทคนิคการเทรนทุกโมเดล |
| [training/baseline.md](training/baseline.md) | TF-IDF + XGBoost, 4 classifiers |
| [training/baseline_optuna.md](training/baseline_optuna.md) | Optuna + 5-fold CV |
| [training/xlmr.md](training/xlmr.md) | XLM-R fine-tune, AMP, Focal Loss |
| [training/embedding.md](training/embedding.md) | E5 embedding + classifier |

## Scope โมเดล

| กลุ่ม | โมเดล |
|------|--------|
| Train | `baseline`, `baseline_optuna`, `embedding`, `xlmr` |
| Inference / Eval / Score | `baseline`, `embedding`, `xlmr` |
| Web default | `auto` → best MAE จาก eval report |

## `src/rris/` — Library

| Path | กลุ่ม | ทำอะไร |
|------|--------|--------|
| `config.py` | Config | พาธ, hyperparameters, device, WEB_MODEL_* |
| `data/text.py` | Data | ล้างข้อความ, XLM-R preprocess |
| `data/loading.py` | Data | โหลด CSV, clean, distribution |
| `data/features.py` | Data | extra features, class weights, FocalLoss, augment |
| `utils.py` | Data | re-export ทั้ง `data/*` |
| `inference/common.py` | Inference | คลาสดาว, สี hex, head-tail truncate, ABSA keywords |
| `inference/prep.py` | Inference | prepare_scoring_dataframe |
| `inference/baseline.py` | Inference | predict_baseline |
| `inference/embedding.py` | Inference | predict_embedding |
| `inference/xlmr.py` | Inference | predict_xlmr |
| `inference/pipeline.py` | Inference | CLI main (ABSA, anomaly flags) |
| `evaluation/runner.py` | Eval | metrics, best_model, main eval |
| `visualization/eval_report.py` | Viz | Plotly HTML จาก eval JSON |
| `training/baseline.py` | Train | TF-IDF + XGBoost |
| `training/baseline_optuna.py` | Train | Optuna sweep |
| `training/xlmr.py` | Train | XLM-R fine-tune |
| `training/embedding.py` | Train | Sentence embedding + clf |
| `cli/*.py` | CLI | entry สำหรับ `python -m rris` |

## `scripts/` — One-off tools

| Script | เมื่อไหร่รัน | Input → Output |
|--------|--------------|----------------|
| `download_wongnai.py` | ดึงข้อมูล HF | → `data/wongnai/` |
| `generate_mock_data.py` | smoke test | → `data/mock/` |
| `merge.py` | รวม CSV ดิบ | → `data/merge/` |
| `reduce_merged.py` | ย่อขนาด train | → `data/merge/` |
| `augment_data.py` | เติมดาว 1–3 | → `*_augmented.csv` |
| `initialize_web_data.py` | ก่อนเปิดเว็บ | → `web_app/scored_reviews.json` |
| `eda_baseline_data.py` | วิเคราะห์ข้อมูล | → `experiments/baseline/eda_summary.json` |
| `tune_baseline.py` | จูน baseline | → `experiments/baseline/tune_log.json` |
| `summarize_tune_ceiling.py` | สรุปจูน | อ่าน tune log |
| `try_log_utils.py` | helper | ใช้โดย tune_baseline |

## `data/` (ตัวอย่างพาธใน config)

| โฟลเดอร์ | เนื้อหา |
|----------|---------|
| `wongnai/` | train_reduce.csv, test.csv |
| `mock/` | train/test จำลอง |
| `merge/` | merged CSV, holdout |
| `70k/` | ชุดใหญ่ / TSV สำหรับเว็บ |

## `artifacts/`

| โฟลเดอร์ | หลังเทรน |
|----------|----------|
| `baseline/` | tfidf, xgb, meta JSON |
| `xlmr/` | model weights, tokenizer |
| `embedding/` | classifier, meta JSON |

## `outputs/`

| โฟลเดอร์ | ไฟล์ตัวอย่าง |
|----------|--------------|
| `scores/` | scored_*.csv |
| `eval/` | eval_report.json (มี `best_model`) |
| `reports/` | eval_report_viz.html |

## `web_app/`

| File | บทบาท |
|------|--------|
| `index.ts` | Elysia server, API, spawn Python eval |
| `templates/index.html` | หน้า SPA |
| `static/js/dashboard.js` | Leaflet + UI |
| `initialize_data.py` | wrapper → `scripts/initialize_web_data.py` |
| `scored_reviews.json` | แคช (generated) |

## `experiments/`

สคริปต์ทดลอง: `run_all_preprocess.py`, `xlmr_preprocess_ablation.py`, `test_preprocess.py`  
ผลลัพธ์: `experiments/baseline/`, `experiments/xlmr/`
