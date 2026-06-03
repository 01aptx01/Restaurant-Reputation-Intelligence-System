# Baseline Preprocessing

โมเดล **TF-IDF + XGBoost** — แปลงข้อความเป็น sparse vector แล้ว classify ดาว 1–5

ไฟล์โค้ด: `src/rris/training/baseline.py`, `src/rris/inference/baseline.py`, `src/rris/data/features.py`

---

## Pipeline ภาพรวม

```text
CSV/TSV
  → extended_normalize_text          (โหลดข้อมูล)
  → clean_review_dataframe           (ลบว่าง / สั้น / ซ้ำ)
  → [train only] undersample 4★, oversample 1–2★, mock mix
  → truncate 500 ตัวอักษร
  → Word TF-IDF + Char TF-IDF + Extra features
  → XGBoost
```

Train และ inference ใช้ **`extended_normalize_text` เหมือนกัน**

---

## 1. Text normalization

ใช้ `extended_normalize_text` ตอน `load_and_standardize_data()` — ดูรายละเอียดใน [text-normalization.md](text-normalization.md)

สรุป: แปล emoji/สแลง → ลบ emoji → ตัวเลข→0 → lowercase

---

## 2. Data cleaning

จาก `config.py`:

| พารามิเตอร์ | ค่า |
|------------|-----|
| `MIN_TEXT_LENGTH` | 5 |
| `DROP_DUPLICATE_TEXT` | True |
| `DUPLICATE_KEEP` | `"first"` |

---

## 3. ขั้นตอนเฉพาะ train

| ขั้นตอน | Config | คำอธิบาย |
|---------|--------|----------|
| Undersample 4★ | `BASELINE_UNDERSAMPLE_STAR4_FRACTION = 0.65` | สุ่มเก็บ 4 ดาวแค่ 65% |
| Oversample 1–2★ | `BASELINE_OVERSAMPLE_LOW_STARS = True`, `FACTOR = 5` | ทำซ้ำรีวิวดาวต่ำ 5 เท่า |
| Mock mix | `BASELINE_MOCK_MIX_FRACTION = 0.2` | ปะปน mock data 20% |
| Truncate | `MAX_REVIEW_CHARS = 500` | ตัดข้อความสูงสุด 500 ตัวอักษร |

---

## 4. Word TF-IDF

สร้างด้วย `_build_word_tfidf()` — `sklearn.feature_extraction.text.TfidfVectorizer`

| พารามิเตอร์ | ค่า (`config.py`) |
|------------|-------------------|
| `tokenizer` | `thai_tokenizer` — normalize อีกรอบด้วย `extended_normalize_text` แล้ว `word_tokenize(..., engine="newmm")` |
| `token_pattern` | `None` (ปิด regex ภาษาอังกฤษ) |
| `max_features` | `TFIDF_MAX_FEATURES = 8000` |
| `ngram_range` | `TFIDF_NGRAM_RANGE = (1, 2)` — unigram + bigram |
| `min_df` | `TFIDF_MIN_DF = 2` |
| `max_df` | `TFIDF_MAX_DF = 0.9` |

---

## 5. Char TF-IDF

เปิดด้วย `BASELINE_USE_CHAR_TFIDF = True`

| พารามิเตอร์ | ค่า |
|------------|-----|
| `analyzer` | `"char_wb"` — char n-gram ภายในขอบเขตคำ |
| `ngram_range` | `BASELINE_CHAR_NGRAM_RANGE = (3, 5)` |
| `max_features` | `BASELINE_CHAR_MAX_FEATURES = 4000` |
| `min_df` / `max_df` | 2 / 0.9 (ใช้ค่าเดียวกับ Word TF-IDF) |

Char TF-IDF ใช้ข้อความที่ normalize แล้วโดยตรง **ไม่** ผ่าน `thai_tokenizer`

---

## 6. Extra features

เปิดด้วย `BASELINE_USE_EXTRA_FEATURES = True`  
ฟังก์ชัน: `compute_extra_features()` ใน `src/rris/data/features.py`

| มิติ | ชื่อ | การคำนวณ |
|------|------|----------|
| 1 | `char_len` | จำนวนตัวอักษรในข้อความ |
| 2 | `word_counts` | จำนวนคำหลัง PyThaiNLP `newmm` |
| 3 | `neg_counts` | จำนวนวลีปฏิเสธ: `"ไม่แนะนำ"`, `"ไม่ค่อย"`, `"ไม่"` |

---

## 7. การรวมฟีเจอร์

```text
Word TF-IDF (≤ 8,000)
  + Char TF-IDF (≤ 4,000)     [ถ้า BASELINE_USE_CHAR_TFIDF]
  + Extra (3)                  [ถ้า BASELINE_USE_EXTRA_FEATURES]
  ─────────────────────────
  ≈ 12,003 มิติ → XGBoost
```

LSA ปิดอยู่ (`BASELINE_USE_LSA = False`)

`fit` บน train set เท่านั้น · val/inference ใช้ `transform` อย่างเดียว

---

## 8. Inference

`src/rris/inference/baseline.py`:

1. `prepare_scoring_dataframe(..., extended_normalize_text)`
2. Truncate 500 ตัวอักษร (จาก `baseline_meta.json` หรือ config)
3. `vectorizer.transform` + `char_vectorizer.transform` (ถ้ามี)
4. ประกบ extra features
5. XGBoost `predict` → ค่าดาวเฉลี่ยจาก probability

---

## Artifacts

| ไฟล์ | เนื้อหา |
|------|---------|
| `artifacts/baseline/tfidf_vectorizer.joblib` | Word TF-IDF |
| `artifacts/baseline/char_tfidf_vectorizer.joblib` | Char TF-IDF |
| `artifacts/baseline/xgb_model.json` | โมเดล XGBoost |
| `artifacts/baseline/baseline_meta.json` | metadata (use_char_tfidf, max_review_chars, …) |
