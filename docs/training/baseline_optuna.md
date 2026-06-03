# Baseline Optuna Training

คำสั่ง: `python -m rris train baseline_optuna`

ไฟล์โค้ด: `src/rris/training/baseline_optuna.py`

Variant ของ Baseline ที่ใช้ **Optuna** ค้นหา hyperparameter และ classifier ที่ดีที่สุด  
Artifact บันทึกทับ path เดียวกับ baseline → inference ใช้ `predict_baseline` เหมือนกัน

---

## สถาปัตยกรรม

```text
TF-IDF features
  → Optuna TPE (30 trials × 5-fold CV)
  → เลือก classifier + hyperparams ที่ F1-macro สูงสุด
  → เทรนบน train ทั้งหมด → ทดสอบ holdout
  → บันทึก artifact (path เดียวกับ baseline)
```

---

## ความต่างจาก train baseline ปกติ

| หัวข้อ | Baseline ปกติ | Baseline Optuna |
|--------|---------------|-----------------|
| Hyperparameter | fixed ใน config | Optuna search |
| Validation | single val split | **5-fold Stratified CV** |
| Classifier | เทียบ 4 ตัว fixed params | Optuna เลือก + จูน params |
| Resampling | undersample/oversample/mock | ไม่มี |
| Word tokenizer | PyThaiNLP `newmm` | ไม่ใช้ custom tokenizer |
| Char TF-IDF analyzer | `char_wb` | `char` |

---

## Optuna setup

| รายการ | ค่า |
|--------|-----|
| Sampler | **TPE** (Tree-structured Parzen Estimator) |
| Direction | maximize F1-macro |
| Trials | 30 |
| CV | **StratifiedKFold** 5 folds (`BASELINE_KFOLD = 5`) |
| Seed | `RANDOM_STATE = 42` |

---

## Search space

Optuna เลือก classifier ก่อน แล้วจูน params ตามประเภท:

### XGBoost

| พารามิเตอร์ | ช่วงค้นหา |
|------------|-----------|
| `max_depth` | 3–9 |
| `learning_rate` | 1e-3 – 0.2 (log) |
| `n_estimators` | 100–500 (step 50) |
| `subsample` | 0.6–1.0 |
| `colsample_bytree` | 0.6–1.0 |
| `reg_lambda` | 1e-3 – 10.0 (log) |

### Logistic Regression

| พารามิเตอร์ | ช่วงค้นหา |
|------------|-----------|
| `C` | 1e-3 – 10.0 (log) |

### Linear SVM

| พารามิเตอร์ | ช่วงค้นหา |
|------------|-----------|
| `C` | 1e-3 – 10.0 (log) |

ไม่ใช้ CalibratedClassifierCV ระหว่าง CV (ความเร็ว)

### Random Forest

| พารามิเตอร์ | ช่วงค้นหา |
|------------|-----------|
| `n_estimators` | 50–300 (step 50) |
| `max_depth` | 10–50 |
| `min_samples_split` | 2–10 |

---

## การแบ่งข้อมูล

```text
ข้อมูลทั้งหมด
  ├─ 80% train/CV pool ──→ 5-fold CV ใน Optuna
  └─ 20% holdout ──→ ทดสอบหลังจูนเสร็จ (ไม่เคยเห็นใน CV)
```

- Truncate 500 ตัวอักษรก่อน split
- TF-IDF fit บน train pool เท่านั้น (ป้องกัน data leakage ไป holdout)

---

## Class imbalance

- `XGB_USE_SAMPLE_WEIGHT = True`
- `compute_class_weights` + `XGB_LOW_STAR_BOOST = 3.0` ในแต่ละ fold
- sklearn models: `class_weight='balanced'`

---

## ขั้นตอนหลัง Optuna

1. ดึง `best_params` และ `classifier` จาก study
2. เทรน final model บน train pool ทั้งหมด
3. ประเมินบน holdout
4. บันทึก vectorizer + model + meta ไป `artifacts/baseline/`

---

## Dependencies

```powershell
pip install optuna
```

ถ้าไม่มี optuna สคริปต์จะ exit พร้อมข้อความแจ้ง
