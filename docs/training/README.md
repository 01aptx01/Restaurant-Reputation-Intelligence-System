# Training Guide

คู่มืออธิบายเทคนิคการเทรนแต่ละโมเดล — อัลกอริทึม, การจัดการ imbalance, hyperparameters และเกณฑ์เลือกโมเดล

## โมเดลที่รองรับ

| Train command | เอกสาร | ไฟล์โค้ด |
|---------------|--------|----------|
| `python -m rris train baseline` | [baseline.md](baseline.md) | `src/rris/training/baseline.py` |
| `python -m rris train baseline_optuna` | [baseline_optuna.md](baseline_optuna.md) | `src/rris/training/baseline_optuna.py` |
| `python -m rris train xlmr` | [xlmr.md](xlmr.md) | `src/rris/training/xlmr.py` |
| `python -m rris train embedding` | [embedding.md](embedding.md) | `src/rris/training/embedding.py` |
| `python -m rris train embedding --finetune` | [embedding.md](embedding.md) | + `embedding_finetune.py` (opt-in) |

Preprocessing: [docs/preprocessing/README.md](../preprocessing/README.md) · เมตริก: [docs/evaluation.md](../evaluation.md)

---

## ตารางเปรียบเทียบเทคนิคหลัก

| เทคนิค | Baseline | Optuna | XLM-R | Embedding |
|--------|:--------:|:------:|:-----:|:---------:|
| TF-IDF + extra features | ✓ | ✓ | — | — |
| Sentence embedding | — | — | — | ✓ |
| Transformer fine-tune | — | — | ✓ | opt-in BGE-M3 |
| Undersample / oversample | ✓ | ✓ | — | — |
| NLP data augmentation | ✓ | ✓ | ✓ | — |
| Error-driven augment | opt-in | opt-in | opt-in | — |
| Sample / class weights | ✓ | ✓ | ✓ | ✓ |
| เทียบ 4 classifiers | ✓ | via Optuna | — | ✓ |
| Stratified K-Fold CV | — | ✓ (5-fold) | — | — |
| Hyperparameter tuning | fixed | Optuna TPE | fixed | fixed |
| Early stopping | XGB only | — | ✓ (val MAE) | — |
| เกณฑ์เลือก winner | **Val MAE** | **CV MAE** | **Val MAE** | **Val MAE** |

Augmentation รวมศูนย์ที่ `src/rris/data/augmentation.py` — เรียกผ่าน `apply_train_augmentation()`

---

## Config หลัก

```python
# Augmentation
AUGMENT_ENABLED = True
AUGMENT_FROM_ERRORS = False          # opt-in error CSV path

# Embedding fine-tune (opt-in)
EMBEDDING_FINETUNE = False
EMBEDDING_FINETUNE_MODEL = "BAAI/bge-m3"

# XLM-R — focal loss ไม่มีผลเมื่อ XLMR_USE_REGRESSION=True
XLMR_USE_REGRESSION = True
```
