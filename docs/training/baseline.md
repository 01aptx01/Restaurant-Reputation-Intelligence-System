# Baseline Training

คำสั่ง: `python -m rris train baseline`

ไฟล์โค้ด: `src/rris/training/baseline.py`  
Inference: `src/rris/inference/baseline.py`

---

## สถาปัตยกรรม

```text
TF-IDF (Word + Char) + Extra features
  → เทรน 4 classifiers พร้อมกัน
  → เลือกตัวที่ F1-macro บน val สูงสุด
  → บันทึก artifact
```

Preprocessing: [docs/preprocessing/baseline.md](../preprocessing/baseline.md)

---

## Classifiers ที่เทียบกัน

| # | โมเดล | หมายเหตุ |
|---|-------|----------|
| 1 | **XGBoost** (Native API) | `xgb.train`, early stopping |
| 2 | Logistic Regression | `class_weight='balanced'` |
| 3 | Linear SVC + CalibratedClassifierCV | ปรับ probability ให้ predict_proba ได้ |
| 4 | Random Forest | 100 trees, `class_weight='balanced'` |

เกณฑ์เลือก winner: **F1-macro** บน validation set

---

## การแบ่งข้อมูล

```text
ข้อมูลทั้งหมด (หลัง clean)
  ├─ 80% pool ──→ 80% train / 20% val   (stratified)
  └─ 20% holdout ──→ บันทึก holdout.csv (ไม่ใช้เทรน)
```

- `train_test_split(..., stratify=user_rating)`
- เปรียบเทียบ rating distribution train vs official test set (EDA drift check)

---

## จัดการ class imbalance

| เทคนิค | Config | รายละเอียด |
|--------|--------|------------|
| Undersample 4★ | `BASELINE_UNDERSAMPLE_STAR4_FRACTION = 0.65` | สุ่มเก็บ 4 ดาวเหลือ 65% |
| Oversample 1–2★ | `BASELINE_OVERSAMPLE_LOW_STARS = True`, `FACTOR = 5` | ทำซ้ำแถวดาวต่ำ 5 เท่า |
| Sample weights (XGB) | `XGB_USE_SAMPLE_WEIGHT = True` | balanced weights + `XGB_LOW_STAR_BOOST = 3.0` |
| class_weight | sklearn models | `'balanced'` บน LR, SVM, RF |

ฟังก์ชัน: `compute_class_weights()`, `compute_sample_weights_from_ratings()`, `oversample_low_star_reviews()`, `undersample_star_ratings()`

---

## เทคนิคเสริม (train only)

| เทคนิค | Config |
|--------|--------|
| Mock data mix | `BASELINE_MOCK_MIX_FRACTION = 0.2` |
| Text truncation | `MAX_REVIEW_CHARS = 500` |

---

## XGBoost hyperparameters

จาก `config.XGB_PARAMS`:

| พารามิเตอร์ | ค่า |
|------------|-----|
| `objective` | `multi:softprob` |
| `num_class` | 5 |
| `max_depth` | 4 |
| `eta` (learning rate) | 0.03 |
| `tree_method` | `hist` |
| `device` | cuda / cpu (auto) |
| `subsample` / `colsample_bytree` | 0.8 / 0.8 |
| `reg_lambda` / `reg_alpha` | 3.0 / 1.0 |
| `min_child_weight` | 5 |

Training loop:

- `XGB_ROUNDS = 800`
- `XGB_EARLY_STOPPING_ROUNDS = 50` — หยุดถ้า val mlogloss ไม่ดีขึ้น
- **GPU fallback** — ถ้า CUDA error สลับ train บน CPU อัตโนมัติ

---

## โหมดที่ปิดอยู่ (config มีไว้ แต่ default ปิด)

| Config | ค่า | ผลถ้าเปิด |
|--------|-----|-----------|
| `BASELINE_USE_LSA` | False | ลดมิติ TF-IDF ด้วย TruncatedSVD 400 มิติ |
| `BASELINE_USE_3CLASS` | False | ยุบ 1–2★ / 3★ / 4–5★ เป็น 3 คลาส |
| `BASELINE_USE_REGRESSION` | False | XGB objective เป็น `reg:squarederror` |
| `BASELINE_KFOLD` | 5 | ใช้ใน Optuna เท่านั้น ไม่ใช้ train ปกติ |

---

## Artifacts

| ไฟล์ | เนื้อหา |
|------|---------|
| `artifacts/baseline/tfidf_vectorizer.joblib` | Word TF-IDF |
| `artifacts/baseline/char_tfidf_vectorizer.joblib` | Char TF-IDF |
| `artifacts/baseline/xgb_model.json` หรือ `sklearn_model.joblib` | โมเดลที่ชนะ |
| `artifacts/baseline/baseline_meta.json` | `best_model_type`, hyperparams, flags |
| `data/merge/holdout.csv` | holdout 20% |
