# Preprocessing Guide

คู่มืออธิบายขั้นตอนล้างข้อความและแปลงฟีเจอร์ก่อนเข้าโมเดลแต่ละตัว

## โมเดลที่รองรับ

| โมเดล | เอกสาร | ไฟล์โค้ดหลัก |
|-------|--------|--------------|
| Baseline (TF-IDF + XGBoost) | [baseline.md](baseline.md) | `src/rris/training/baseline.py`, `src/rris/data/features.py` |
| XLM-RoBERTa | [xlmr.md](xlmr.md) | `src/rris/training/xlmr.py`, `src/rris/data/text.py` |
| Embedding (E5 + Classifier) | [embedding.md](embedding.md) | `src/rris/training/embedding.py` |

ฟังก์ชัน normalize ข้อความทั้งหมดอยู่ใน [text-normalization.md](text-normalization.md)

---

## ขั้นตอนร่วม (ทุกโมเดล)

### 1. โหลดและจัดคอลัมน์

`load_and_standardize_data()` ใน `src/rris/data/loading.py`:

- อ่าน CSV/TSV
- หาคอลัมน์ข้อความจาก alias: `review_body`, `text`, `review`
- หาคอลัมน์คะแนนจาก alias: `stars`, `user_rating`, `rating`, `star`
- แปลงคะแนนเป็นจำนวนเต็ม 1–5
- ใช้ฟังก์ชัน `normalize_func` กับทุกแถวข้อความ (ค่า default = `extended_normalize_text`)

### 2. ล้างข้อมูล

`clean_review_dataframe()` — ค่า config ใน `config.py`:

| พารามิเตอร์ | ค่า default | ความหมาย |
|------------|-------------|----------|
| `MIN_TEXT_LENGTH` | 5 | ลบรีวิวสั้นกว่า 5 ตัวอักษร |
| `DROP_DUPLICATE_TEXT` | True | ลบข้อความซ้ำ (เก็บแถวแรก) |
| `DUPLICATE_KEEP` | `"first"` | เก็บแถวไหนเมื่อเจอซ้ำ |

XLM-R และ Embedding ใช้ `min_text_length=5` แบบ hardcode ในสคริปต์เทรน (เทียบเท่า config)

---

## สรุปเปรียบเทียบ normalize ต่อโมเดล

| โมเดล | Train | Inference / Eval |
|-------|-------|---------------------|
| **Baseline** | `extended_normalize_text` | `extended_normalize_text` |
| **XLM-R** | `aggressive` (จาก `XLMR_PREPROCESS_STRATEGY`) | `xlmr_normalize_text` ⚠️ |
| **Embedding** | `aggressive` (จาก `XLMR_PREPROCESS_STRATEGY`) | `aggressive` (eval / web) · `extended_normalize_text` (pipeline CLI ถ้าไม่ใช่ xlmr) ⚠️ |

⚠️ **Train vs inference ไม่ตรงกัน** ในบาง path — ดูรายละเอียดใน [xlmr.md](xlmr.md) และ [embedding.md](embedding.md)

---

## จุดเข้า inference

| คำสั่ง / สคริปต์ | ฟังก์ชันเตรียมข้อมูล |
|-----------------|---------------------|
| `python -m rris evaluate --model baseline` | `prepare_scoring_dataframe(..., extended_normalize_text)` |
| `python -m rris evaluate --model xlmr` | `prepare_scoring_dataframe(..., xlmr_normalize_text)` |
| `python -m rris evaluate --model embedding` | `prepare_scoring_dataframe(..., aggressive)` |
| `python -m rris score --model xlmr` | `xlmr_normalize_text` |
| `python -m rris score --model baseline/embedding` | `extended_normalize_text` |
| `scripts/initialize_web_data.py` | ตามโมเดลที่เลือก (ดู [embedding.md](embedding.md)) |

---

## Config ที่เกี่ยวข้อง

```python
# src/rris/config.py
MAX_REVIEW_CHARS = 500              # Baseline เท่านั้น
XLMR_PREPROCESS_STRATEGY = "aggressive"
MAX_LENGTH = 128                    # XLM-R token limit
EMBEDDING_MAX_LENGTH = 128          # Embedding (encode ใช้ batch ไม่ truncate แยก)
```

---

## โมเดลที่ยังไม่ implement

WangchanBERTa, Hybrid Ensemble มี config path ใน `config.py` แต่ยังไม่มี train/inference pipeline — ไม่มีเอกสาร preprocessing แยก
