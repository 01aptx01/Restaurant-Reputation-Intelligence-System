# Training Guide

คู่มืออธิบายเทคนิคการเทรนแต่ละโมเดล — อัลกอริทึม, การจัดการ imbalance, hyperparameters และเกณฑ์เลือกโมเดล

## โมเดลที่รองรับ

| Train command | เอกสาร | ไฟล์โค้ด |
|---------------|--------|----------|
| `python -m rris train baseline` | [baseline.md](baseline.md) | `src/rris/training/baseline.py` |
| `python -m rris train baseline_optuna` | [baseline_optuna.md](baseline_optuna.md) | `src/rris/training/baseline_optuna.py` |
| `python -m rris train xlmr` | [xlmr.md](xlmr.md) | `src/rris/training/xlmr.py` |
| `python -m rris train embedding` | [embedding.md](embedding.md) | `src/rris/training/embedding.py` |

Preprocessing ก่อนเทรน: [docs/preprocessing/README.md](../preprocessing/README.md)

---

## ตารางเปรียบเทียบเทคนิคหลัก

| เทคนิค | Baseline | Optuna | XLM-R | Embedding |
|--------|:--------:|:------:|:-----:|:---------:|
| TF-IDF features | ✓ | ✓ | — | — |
| Sentence embedding | — | — | — | ✓ |
| Transformer fine-tune | — | — | ✓ | — |
| Undersample majority (4★) | ✓ | — | — | — |
| Oversample minority (1–2★) | ✓ | — | — | — |
| NLP data augmentation | — | — | ✓ | — |
| Sample / class weights | ✓ | ✓ | ✓ | ✓ |
| เทียบ 4 classifiers | ✓ | via Optuna | — | ✓ |
| Stratified K-Fold CV | — | ✓ (5-fold) | — | — |
| Hyperparameter tuning | fixed | Optuna TPE | fixed | fixed |
| Early stopping | XGB only | — | ✓ | — |
| AMP + gradient accumulation | — | — | ✓ | — |
| Focal Loss | — | — | ✓* | — |
| Ordinal regression | —** | — | ✓ | — |
| GPU | XGB | XGB | ✓ | encode |

\* ใช้เมื่อปิด regression mode และเปิด `XLMR_USE_FOCAL_LOSS`  
\*\* มีใน config แต่ `BASELINE_USE_REGRESSION = False`

---

## เทคนิคร่วมทุกโมเดล

1. **Stratified split** — รักษาสัดส่วนดาว 1–5 ใน train/val
2. **Text cleaning** — ลบข้อความว่าง, สั้นกว่า 5 ตัว, ซ้ำ
3. **Class imbalance** — sample weights, resampling หรือ class_weight
4. **Reproducibility** — `RANDOM_STATE = 42`
5. **5-class star rating** — ทำนายดาว 1–5 (ยกเว้น XLM-R ที่ใช้ ordinal regression เป็นหลัก)

---

## เกณฑ์เลือกโมเดล / classifier

| โมเดล | เกณฑ์ |
|-------|-------|
| Baseline | F1-macro บน validation |
| Baseline Optuna | F1-macro mean จาก 5-fold CV |
| XLM-R | val_acc สูงสุด (early stopping checkpoint) |
| Embedding | F1-macro บน validation |

---

## Config หลัก

```python
# src/rris/config.py
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.2

# Baseline imbalance
BASELINE_UNDERSAMPLE_STAR4_FRACTION = 0.65
BASELINE_OVERSAMPLE_FACTOR = 5
XGB_LOW_STAR_BOOST = 3.0

# XLM-R
EPOCHS = 3
LEARNING_RATE = 2e-5
XLMR_USE_REGRESSION = True
XLMR_USE_FOCAL_LOSS = True

# Augmentation (XLM-R)
AUGMENT_ENABLED = True
AUGMENT_TARGET_COUNT = 800
```

---

## โมเดลที่ยังไม่ implement

WangchanBERTa และ Hybrid Ensemble มี config path แต่ยังไม่มีสคริปต์ train ใน repo
