# Preprocessing Guide

คู่มืออธิบายขั้นตอนล้างข้อความและแปลงฟีเจอร์ก่อนเข้าโมเดลแต่ละตัว

## โมเดลที่รองรับ

| โมเดล | เอกสาร | ไฟล์โค้ดหลัก |
|-------|--------|--------------|
| Baseline (TF-IDF + XGBoost) | [baseline.md](baseline.md) | `src/rris/training/baseline.py`, `src/rris/data/features.py` |
| XLM-RoBERTa | [xlmr.md](xlmr.md) | `src/rris/training/xlmr.py`, `src/rris/data/text.py` |
| Embedding (E5 + Classifier) | [embedding.md](embedding.md) | `src/rris/training/embedding.py` |

ฟังก์ชัน normalize ข้อความทั้งหมดอยู่ใน [text-normalization.md](text-normalization.md)

Resolver กลาง: `src/rris/data/normalize.py` — `resolve_normalize_func(model)` อ่าน `preprocess_strategy` จาก artifact meta

---

## ขั้นตอนร่วม (ทุกโมเดล)

### 1. โหลดและจัดคอลัมน์

`load_and_standardize_data()` ใน `src/rris/data/loading.py`:

- อ่าน CSV/TSV
- หาคอลัมน์ข้อความจาก alias: `review_body`, `text`, `review`
- หาคอลัมน์คะแนนจาก alias: `stars`, `user_rating`, `rating`, `star`
- แปลงคะแนนเป็นจำนวนเต็ม 1–5
- ใช้ฟังก์ชัน `normalize_func` กับทุกแถวข้อความ

### 2. ล้างข้อมูล

`clean_review_dataframe()` — ค่า config ใน `config.py`:

| พารามิเตอร์ | ค่า default | ความหมาย |
|------------|-------------|----------|
| `MIN_TEXT_LENGTH` | 5 | ลบรีวิวสั้นกว่า 5 ตัวอักษร |
| `DROP_DUPLICATE_TEXT` | True | ลบข้อความซ้ำ (เก็บแถวแรก) |
| `DUPLICATE_KEEP` | `"first"` | เก็บแถวไหนเมื่อเจอซ้ำ |

---

## สรุปเปรียบเทียบ normalize ต่อโมเดล

| โมเดล | Train | Inference / Eval |
|-------|-------|---------------------|
| **Baseline** | `extended` (`preprocess_strategy` ใน meta) | `resolve_normalize_func("baseline")` |
| **XLM-R** | `XLMR_PREPROCESS_STRATEGY` → บันทึกใน `xlmr_meta.json` | อ่านจาก meta หรือ config default |
| **Embedding** | เหมือน XLM-R | อ่านจาก `embedding_meta.json` |

Train และ inference ใช้ strategy เดียวกันผ่าน `prepare_scoring_for_model(path, model)` — retrain หลัง overhaul เพื่อ populate meta ถ้า artifacts เก่าไม่มี field

---

## จุดเข้า inference

| คำสั่ง / สคริปต์ | ฟังก์ชันเตรียมข้อมูล |
|-----------------|---------------------|
| `python -m rris evaluate` | `prepare_scoring_for_model(..., model)` |
| `python -m rris score` | `prepare_scoring_for_model` |
| `scripts/initialize_web_data.py` | `prepare_scoring_for_model` |

---

## Config ที่เกี่ยวข้อง

```python
# src/rris/config.py
MAX_REVIEW_CHARS = 500              # Baseline truncation
XLMR_PREPROCESS_STRATEGY = "aggressive"
MAX_LENGTH = 128                    # XLM-R token limit
```
