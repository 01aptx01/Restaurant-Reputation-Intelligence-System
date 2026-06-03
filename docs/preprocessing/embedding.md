# Embedding Preprocessing

โมเดล **Sentence Embedding** (`intfloat/multilingual-e5-base`) + classifier (XGBoost / Logistic Regression)

ไฟล์โค้ด: `src/rris/training/embedding.py`, `src/rris/inference/embedding.py`

---

## Pipeline ภาพรวม

```text
CSV/TSV
  → normalize ตาม XLMR_PREPROCESS_STRATEGY   (default: aggressive)
  → clean_review_dataframe
  → SentenceTransformer.encode (normalize_embeddings=True)
  → Classifier (predict_proba)
  → ค่าดาวเฉลี่ยจาก probability
```

ไม่มี feature engineering แยก (TF-IDF / extra features) — ใช้ vector จาก embedding model โดยตรง

---

## 1. Text normalization

### Train

ใช้ strategy เดียวกับ XLM-R train:

```python
strategy = XLMR_PREPROCESS_STRATEGY  # default: "aggressive"
normalize_func = PREPROCESS_REGISTRY.get(strategy, xlmr_normalize_text)
df = load_and_standardize_data(RAW_DATA_PATH, normalize_func=normalize_func)
```

รายละเอียด: [text-normalization.md](text-normalization.md)

### Inference — ขึ้นกับ entry point

| Path | normalize |
|------|-----------|
| `python -m rris evaluate --model embedding` | `aggressive` (จาก config) |
| `scripts/initialize_web_data.py --model embedding` | `aggressive` (จาก config) |
| `python -m rris score --model embedding` | `extended_normalize_text` ⚠️ |

⚠️ Pipeline CLI (`score`) ใช้ `extended_normalize_text` สำหรับโมเดลที่ไม่ใช่ xlmr — ไม่ตรงกับตอน train/eval

---

## 2. Data cleaning

| พารามิเตอร์ | ค่า |
|------------|-----|
| `min_text_length` | 5 |
| `drop_duplicates` | True |

---

## 3. Embedding extraction

| Config | ค่า |
|--------|-----|
| `EMBEDDING_MODEL_NAME` | `"intfloat/multilingual-e5-base"` |
| `EMBEDDING_BATCH_SIZE` | 32 |
| `EMBEDDING_MAX_LENGTH` | 128 (config มีไว้; encode ใช้ model default) |
| `normalize_embeddings` | **True** (L2 normalize vector) |

```python
X = embed_model.encode(
    texts,
    batch_size=EMBEDDING_BATCH_SIZE,
    normalize_embeddings=True,
)
```

Embedding cache: `data/embedding_cache.joblib` (ถ้าขนาด train/val ตรงกับ cache)

---

## 4. Classifier

เทรนเปรียบเทียบหลายตัวแล้วเลือกที่ดีที่สุดบน validation:

- Logistic Regression (`class_weight='balanced'`)
- XGBoost
- LinearSVC + calibration
- Random Forest

Sample weights: `compute_sample_weights_from_ratings` + `XGB_LOW_STAR_BOOST = 3.0`

---

## 5. Inference

`predict_embedding()`:

1. โหลด `SentenceTransformer` จาก `embedding_meta.json`
2. `encode` ข้อความที่ normalize แล้ว
3. `clf.predict_proba` → `expected_rating_from_probs`

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
| `artifacts/embedding/embedding_meta.json` | ชื่อ embedding model, preprocess strategy, classifier type |
| `artifacts/embedding/clf_model.joblib` | classifier ที่เลือก |
| `data/embedding_cache.joblib` | cache vector (optional) |
