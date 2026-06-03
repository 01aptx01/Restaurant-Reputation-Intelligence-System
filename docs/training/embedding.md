# Embedding Training

คำสั่ง: `python -m rris train embedding`

ไฟล์โค้ด: `src/rris/training/embedding.py`  
Inference: `src/rris/inference/embedding.py`

---

## สถาปัตยกรรม

```text
SentenceTransformer (multilingual-e5-base)
  → encode ข้อความเป็น vector (L2 normalized)
  → เทรน 4 classifiers บน embedding
  → เลือกตัวที่ Val MAE ต่ำสุด
```

Preprocessing: [docs/preprocessing/embedding.md](../preprocessing/embedding.md)

---

## ขั้นตอน 4 ขั้น

### Step 1 — โหลดข้อมูล

- Normalize ด้วย `XLMR_PREPROCESS_STRATEGY` (default: `aggressive`)
- Clean: min length 5, drop duplicates

### Step 2 — Extract embeddings

| Config | ค่า |
|--------|-----|
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-base` |
| `EMBEDDING_BATCH_SIZE` | 32 |
| `normalize_embeddings` | **True** |

- Cache vector ลง `data/embedding_cache_{hash}.joblib` (key จาก model + strategy + texts hash)
- Fine-tune opt-in: `python -m rris train embedding --finetune` → `artifacts/embedding/finetuned_model/`

### Step 3 — Train & compare classifiers

### Step 4 — Save artifacts

---

## การแบ่งข้อมูล

- **Stratified split** — `HOLDOUT_FRACTION = 0.2` เป็น validation
- Single split (ไม่มี nested holdout แยกไฟล์)

---

## Classifiers ที่เทียบกัน

| # | โมเดล | พารามิเตอร์หลัก |
|---|-------|----------------|
| 1 | Logistic Regression | `class_weight='balanced'`, max_iter=1000 |
| 2 | XGBoost | max_depth=4, lr=0.05, 300 trees, hist, GPU |
| 3 | Linear SVC + CalibratedClassifierCV | `class_weight='balanced'` |
| 4 | Random Forest | 100 trees, `class_weight='balanced'` |

เกณฑ์เลือก winner: **Val MAE** ต่ำสุด  
แสดงตารางเปรียบเทียบ Acc, MAE, F1 ทุกตัวก่อนเลือก

---

## Class imbalance

```python
class_weights = compute_class_weights(y_train, low_star_boost=XGB_LOW_STAR_BOOST)  # 3.0
sample_w = compute_sample_weights_from_ratings(y_train, class_weights)
```

- ใช้ `sample_weight` กับทุก classifier ที่รองรับ
- XGBoost: label ปรับเป็น 0–4 (`y - 1`)

---

## XGBoost (embedding) vs Baseline XGB

| หัวข้อ | Embedding clf | Baseline XGB |
|--------|---------------|--------------|
| Input | dense embedding vector | sparse TF-IDF |
| API | `XGBClassifier` (sklearn API) | Native `xgb.train` |
| Trees | 300 fixed | 800 + early stopping |
| Learning rate | 0.05 | 0.03 |

---

## Smoke mode

เมื่อ `RRIS_SMOKE=1`:

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

---

## Artifacts

| ไฟล์ | เนื้อหา |
|------|---------|
| `artifacts/embedding/clf_model.joblib` | classifier ที่ชนะ |
| `artifacts/embedding/embedding_meta.json` | model name, classifier, preprocess, val_mae, cache_key, finetune path |
| `data/embedding_cache_*.joblib` | hashed cached train/val embeddings |

**Fine-tune:** opt-in (`EMBEDDING_FINETUNE=False` by default) — supervised หรือ contrastive บน BGE-M3
