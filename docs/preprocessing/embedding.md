# Embedding Preprocessing

โมเดล **Sentence Embedding** (default: `intfloat/multilingual-e5-base`) + classifier

ไฟล์โค้ด: `src/rris/training/embedding.py`, `src/rris/inference/embedding.py`

---

## Pipeline ภาพรวม

```text
CSV/TSV
  → normalize (strategy จาก train config / embedding_meta.json)
  → clean_review_dataframe
  → [opt-in] fine-tune SentenceTransformer
  → encode → classifier → expected rating จาก proba
```

---

## 1. Text normalization

### Train

ใช้ `XLMR_PREPROCESS_STRATEGY` (default: `aggressive`) — บันทึกใน `embedding_meta.json` เป็น `preprocess_strategy`

### Inference / Eval / Score / Web

`prepare_scoring_for_model(path, "embedding")` → `resolve_normalize_func("embedding")` อ่าน strategy จาก meta

Train และ inference สอดคล้องกันหลังมี `embedding_meta.json`

---

## 2. Data cleaning

| พารามิเตอร์ | ค่า |
|------------|-----|
| `min_text_length` | 5 |
| `drop_duplicates` | True |

---

## 3. Embedding cache

Hashed path: `data/embedding_cache_{key}.joblib` — key จาก model name, strategy, ขนาด train/val, fingerprint ข้อความ

Meta บันทึก `cache_key` — cache เก่า (`embedding_cache.joblib`) ไม่ใช้แล้ว

---

## 4. Fine-tune (opt-in)

```powershell
python -m rris train embedding --finetune
```

| Config | ค่า default |
|--------|-------------|
| `EMBEDDING_FINETUNE` | False |
| `EMBEDDING_FINETUNE_MODEL` | BAAI/bge-m3 |
| `EMBEDDING_FINETUNE_MODE` | supervised \| contrastive |

Inference โหลด encoder จาก `finetuned_model_path` ใน meta ผ่าน `resolve_embedding_model_path()`

---

## 5. Artifacts meta

`artifacts/embedding/embedding_meta.json` — `embedding_model`, `preprocess_strategy`, `classifier`, `val_mae`, `cache_key`, `finetune*`
