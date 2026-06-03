# XLM-R Preprocessing

โมเดล **xlm-roberta-base** fine-tune ทำนายดาว 1–5 (Ordinal Regression + Focal Loss)

ไฟล์โค้ด: `src/rris/training/xlmr.py`, `src/rris/inference/xlmr.py`, `src/rris/data/text.py`

---

## Pipeline ภาพรวม

```text
CSV/TSV
  → normalize ตาม XLMR_PREPROCESS_STRATEGY   (train: aggressive)
  → clean_review_dataframe
  → [train only] data augmentation (ดาว 1–2–3)
  → head+tail token truncate (128 tokens)
  → XLM-R tokenizer → model
```

---

## 1. Text normalization

### Train

ใช้ strategy จาก `XLMR_PREPROCESS_STRATEGY` ใน `config.py` (default: **`"aggressive"`**)

```python
normalize_func = PREPROCESS_REGISTRY.get(strategy, xlmr_normalize_text)
df = load_and_standardize_data(RAW_DATA_PATH, normalize_func=normalize_func)
```

รายละเอียดแต่ละ strategy: [text-normalization.md](text-normalization.md)

### Inference / Eval

ใช้ **`xlmr_normalize_text`** โดยตรง — **ไม่** ใช้ strategy จาก config

```python
# src/rris/evaluation/runner.py, scripts/initialize_web_data.py (model=xlmr)
prepare_scoring_dataframe(input, normalize_func=xlmr_normalize_text)
```

### ⚠️ Train vs inference ไม่ตรงกัน

| ช่วง | ฟังก์ชัน |
|------|----------|
| Train | `aggressive` (emoji, slang, URL, negation, …) |
| Inference | `xlmr_normalize_text` (เบากว่า — ไม่แปล emoji/slang) |

โมเดลที่เทรนด้วย `aggressive` แต่ inference ใช้ `xlmr_normalize_text` อาจได้ distribution ข้อความต่างจากตอน train

---

## 2. Data cleaning

| พารามิเตอร์ | ค่า |
|------------|-----|
| `min_text_length` | 5 (hardcode ใน train script) |
| `drop_duplicates` | True |

---

## 3. ขั้นตอนเฉพาะ train

### Data augmentation

เปิดด้วย `AUGMENT_ENABLED = True`:

| Config | ค่า |
|--------|-----|
| `AUGMENT_TARGET_STARS` | (1, 2, 3) |
| `AUGMENT_TARGET_COUNT` | 800 ต่อคลาส |
| `AUGMENT_SYNONYM_PROB` | 0.3 |
| `AUGMENT_SHUFFLE_PROB` | 0.2 |

ฟังก์ชัน: `augment_minority_classes()` ใน `src/rris/data/features.py`

### Class weights

`XLMR_USE_CLASS_WEIGHT = True`, `XLMR_LOW_STAR_BOOST = 1.5`

---

## 4. Tokenization (หลัง normalize)

| พารามิเตอร์ | ค่า (`config.py`) |
|------------|-------------------|
| `MAX_LENGTH` | 128 |
| Truncation | **Head + Tail** — เก็บครึ่งแรก + ครึ่งท้ายของ token sequence |
| `padding` | True (batch) |
| `add_special_tokens` | True |

ฟังก์ชัน: `head_tail_truncate_text()` ใน `src/rris/inference/common.py`

```text
ถ้า tokens > 126 (128 - 2 special tokens):
  เก็บ head_len tokens แรก + tail_len tokens ท้าย
  รวมแล้ว = max_length - 2
```

---

## 5. โหมดการทำนาย

จาก `XLMR_USE_REGRESSION = True`:

- Train: MSE loss (ordinal regression) + Focal Loss
- Inference: logits → ค่าดาวต่อเนื่อง → แปลงเป็น Gaussian probs → ค่าดาวเฉลี่ย

---

## 6. Inference entry points

| Path | normalize |
|------|-----------|
| `python -m rris evaluate --model xlmr` | `xlmr_normalize_text` |
| `python -m rris score --model xlmr` | `xlmr_normalize_text` |
| `scripts/initialize_web_data.py --model xlmr` | `xlmr_normalize_text` |

---

## Preprocessing ablation

ทดลองเปรียบเทียบ strategy ได้ด้วย:

```powershell
python experiments/xlmr_preprocess_ablation.py
python experiments/run_all_preprocess.py
```

บันทึกผล: `experiments/xlmr/preprocess_ablation_log.json`

---

## Artifacts

| Path | เนื้อหา |
|------|---------|
| `artifacts/xlmr/` | fine-tuned model + tokenizer (Hugging Face format) |
