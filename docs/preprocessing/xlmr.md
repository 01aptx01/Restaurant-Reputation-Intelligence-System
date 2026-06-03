# XLM-R Preprocessing

โมเดล **xlm-roberta-base** fine-tune ทำนายดาว 1–5 (Ordinal Regression เป็นหลัก)

ไฟล์โค้ด: `src/rris/training/xlmr.py`, `src/rris/inference/xlmr.py`, `src/rris/data/text.py`

---

## Pipeline ภาพรวม

```text
CSV/TSV
  → normalize ตาม XLMR_PREPROCESS_STRATEGY (บันทึกใน xlmr_meta.json)
  → clean_review_dataframe
  → [train only] apply_train_augmentation
  → head+tail token truncate (128 tokens)
  → XLM-R tokenizer → model
```

---

## 1. Text normalization

### Train

ใช้ strategy จาก `XLMR_PREPROCESS_STRATEGY` (default: **`aggressive`**) แล้วบันทึกลง `artifacts/xlmr/xlmr_meta.json`

### Inference / Eval / Score / Web

ใช้ **`prepare_scoring_for_model(path, "xlmr")`** → `resolve_normalize_func("xlmr")` อ่าน `preprocess_strategy` จาก meta (หรือ config ถ้าไม่มี meta)

Train และ inference ใช้ strategy เดียวกันหลัง retrain ที่มี `xlmr_meta.json`

รายละเอียดแต่ละ strategy: [text-normalization.md](text-normalization.md)

---

## 2. Data cleaning

| พารามิเตอร์ | ค่า |
|------------|-----|
| `min_text_length` | 5 (hardcode ใน train script) |
| `drop_duplicates` | True |

---

## 3. Data augmentation (train only)

ผ่าน `apply_train_augmentation()` จาก `src/rris/data/augmentation.py` เมื่อ `AUGMENT_ENABLED=True`

| Config | ค่า |
|--------|-----|
| `AUGMENT_TARGET_STARS` | (1, 2, 3) |
| `AUGMENT_TARGET_COUNT` | 800 |
| `AUGMENT_FROM_ERRORS` | opt-in + path จาก error export |

---

## 4. Token truncation

`head_tail_truncate_text()` — เก็บหัว+ท้ายประโยคให้พอดี `MAX_LENGTH` (128) tokens

---

## 5. Artifacts meta

`artifacts/xlmr/xlmr_meta.json`:

| Field | ความหมาย |
|-------|----------|
| `preprocess_strategy` | strategy ที่ใช้ตอน train |
| `best_val_mae` | MAE ต่ำสุดบน validation (เกณฑ์ checkpoint) |
| `best_epoch` | epoch ที่ save checkpoint |

---

## Config ที่เกี่ยวข้อง

```python
XLMR_PREPROCESS_STRATEGY = "aggressive"
MAX_LENGTH = 128
XLMR_USE_REGRESSION = True   # เมื่อ True, XLMR_USE_FOCAL_LOSS ไม่มีผล
```
