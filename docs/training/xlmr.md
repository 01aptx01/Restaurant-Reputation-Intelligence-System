# XLM-R Training

คำสั่ง: `python -m rris train xlmr`

ไฟล์โค้ด: `src/rris/training/xlmr.py`  
Inference: `src/rris/inference/xlmr.py`

---

## สถาปัตยกรรม

```text
xlm-roberta-base (pretrained)
  → fine-tune ด้วย PyTorch manual loop
  → Ordinal Regression (MSE) + early stopping
  → บันทึก best checkpoint
```

Preprocessing: [docs/preprocessing/xlmr.md](../preprocessing/xlmr.md)

---

## โหมดการทำนาย (config ปัจจุบัน)

| Config | ค่า | ผล |
|--------|-----|-----|
| `XLMR_USE_REGRESSION` | **True** | label = float 1.0–5.0, loss = MSE |
| `XLMR_USE_3CLASS` | False | ไม่ยุบเป็น Negative/Neutral/Positive |
| default 5-class | — | label index 0–4 (ถ้าปิด regression) |

---

## การแบ่งข้อมูล

- **Stratified 80% train / 20% val** (single split)
- ไม่แยก holdout แยกไฟล์เหมือน baseline

---

## Data augmentation (train only)

เปิดด้วย `AUGMENT_ENABLED = True` — เฉพาะ train split

| Config | ค่า |
|--------|-----|
| `AUGMENT_TARGET_STARS` | (1, 2, 3) |
| `AUGMENT_TARGET_COUNT` | 800 ต่อคลาส |
| `AUGMENT_SYNONYM_PROB` | 0.3 |
| `AUGMENT_SHUFFLE_PROB` | 0.2 |

เทคนิค:

1. **Synonym replacement** — WordNet Thai, แทนคำด้วยคำพ้องความหมาย
2. **Random word shuffle** — สลับลำดับคำในประโยค

ฟังก์ชัน: `augment_minority_classes()` ใน `src/rris/data/features.py`

---

## Loss function

ลำดับการเลือก (ใน `train_xlmr.py`):

| เงื่อนไข | Loss |
|----------|------|
| `XLMR_USE_REGRESSION = True` | **MSELoss** |
| `XLMR_USE_FOCAL_LOSS = True` (classification) | **FocalLoss** (γ=2.0, α=class weights) |
| มี class weights | **Weighted CrossEntropyLoss** |
| default | HuggingFace built-in loss |

Focal Loss: `src/rris/data/features.py` — โฟกัส hard examples ด้วย `(1 - p_t)^γ`

---

## Optimizer & learning rate

| รายการ | ค่า |
|--------|-----|
| Optimizer | **AdamW** |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Scheduler | **Linear warmup** — warmup 10% ของ total steps |
| Epochs | 3 max |

---

## เทคนิคประหยัด VRAM / เร่งความเร็ว

| เทคนิค | Config | รายละเอียด |
|--------|--------|------------|
| Gradient checkpointing | `XLMR_GRADIENT_CHECKPOINTING = True` | ลด VRAM ~50% |
| Mixed Precision (AMP) | `XLMR_USE_AMP = True` | FP16 บน CUDA |
| Gradient accumulation | `BATCH_SIZE=4`, `GRAD_ACCUM_STEPS=2` | effective batch = **8** |
| Head+Tail truncation | `MAX_LENGTH=128` | ใน `ReviewDataset` |
| CUDA alloc config | `expandable_segments:True` | ลด fragmentation OOM |

---

## Class imbalance

| Config | ค่า |
|--------|-----|
| `XLMR_USE_CLASS_WEIGHT` | True |
| `XLMR_LOW_STAR_BOOST` | 1.5× สำหรับดาว 1–2 |

ใช้กับ Weighted CE / Focal Loss (ไม่ใช้ใน pure MSE regression mode)

---

## Early stopping & checkpoint

| รายการ | ค่า |
|--------|-----|
| Metric | **val_acc** |
| Patience | `XLMR_EARLY_STOPPING_PATIENCE = 3` epochs |
| บันทึก | best checkpoint (ไม่ใช่ epoch สุดท้าย) |

```text
ถ้า val_acc ไม่ดีขึ้นติด 3 epoch → หยุด
restore best_state → save_pretrained
```

---

## Training loop

- Manual loop ใน `run_epoch()` — ไม่ใช้ HuggingFace Trainer
- Train loader: `shuffle=True`
- Val loader: `shuffle=False`
- ล้าง CUDA cache หลังแต่ละ epoch

---

## Hyperparameters สรุป

```python
XLMR_MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 4
XLMR_GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3
XLMR_FOCAL_GAMMA = 2.0
```

---

## Artifacts

| Path | เนื้อหา |
|------|---------|
| `artifacts/xlmr/` | model weights + tokenizer (Hugging Face format) |
