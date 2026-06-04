# -*- coding: utf-8 -*-
# ไฟล์ xlmr_large.py: ฝึกฝนโมเดล NLP ภาษาขั้นสูง (XLM-RoBERTa Large Fine-Tuning) ด้วย PyTorch Manual Training Loop

import os # เรียกใช้งานระบบจัดเตรียมไฟล์ของเครื่อง
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # บรรเทาการแบ่งส่วนแรมของการ์ดจอ (VRAM fragmentation) ป้องกัน CUDA OOM
import json
import numpy as np    # ไลบรารีการคำนวณเวกเตอร์และตัวเลข

import torch          # ไลบรารีหลักประมวลผล Tensor และโมเดลของ PyTorch
import torch.nn as nn # คอนเทนเนอร์โมดูลประสาท (Neural Network Modules) ของ PyTorch
from sklearn.model_selection import train_test_split # ฟังก์ชันสำหรับแบ่งข้อมูลออกเป็นชุด Train และ Validation
from torch.utils.data import DataLoader, Dataset    # โครงสร้างสำหรับห่อหุ้มและทะยอยส่งชุดข้อมูล (Dataloader)
# โหลดส่วนประกอบโมเดลและระบบ Linear Schedule จาก Hugging Face Transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from rris import config # นำเข้าค่าคงที่และการตั้งค่าทางคณิตศาสตร์หลัก
from rris import utils  # นำเข้าเครื่องมือทำความสะอาดและตรวจสอบข้อมูลหลัก
from rris.data.augmentation import apply_train_augmentation
from rris.inference.common import expected_rating_from_probs
from pathlib import Path

# --- Overrides for Large Model ---
config.XLMR_MODEL_NAME = "xlm-roberta-large"
config.XLMR_ARTIFACTS_DIR = str(Path(config.ARTIFACTS_DIR) / "xlmr_large")
config.XLMR_META_PATH = str(Path(config.ARTIFACTS_DIR) / "xlmr_large" / "xlmr_meta.json")
config.BATCH_SIZE = max(1, config.BATCH_SIZE // 4) # ลดขนาดแบทช์เพื่อป้องกัน OOM สำหรับ Large Model
config.XLMR_GRAD_ACCUM_STEPS = config.XLMR_GRAD_ACCUM_STEPS * 4 # เพิ่มสะสมเกรเดียนต์ชดเชย


class ReviewDataset(Dataset):
    """ชุดคลาสสำหรับแปลงตารางข้อมูลรีวิวภาษาไทยให้กลายเป็น Tensors ที่โทเคนไนเซอร์ประมวลผลได้สำเร็จ"""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts         # รายการข้อความรีวิว
        self.labels = labels       # ป้ายกำกับดัชนีดาว (0 ถึง 4 แทนคะแนนดาว 1 ถึง 5★)
        self.tokenizer = tokenizer # ตัวตัดคำประมวลผลระดับสูง (XLM-R Tokenizer)
        self.max_len = max_len     # ความยาวโทเคนสูงสุดต่อหนึ่งช่องรีวิว (เช่น 128)

    def __len__(self):
        """ส่งคืนขนาดจำนวนตัวอย่างทั้งหมดใน Dataset"""
        return len(self.texts)

    def __getitem__(self, item):
        """ดึงแถวข้อมูลข้อความรีวิวที่ดัชนีระบุมาแปรรูปเป็นรูปเวกเตอร์ PyTorch Tensor"""
        text = str(self.texts[item]) # ดึงข้อความรีวิวมาแปลงเป็นสตริงปลอดภัย
        label = self.labels[item]     # ดึงระดับดาวที่แมปไว้ (0-4)

        # เพิ่มระบบตัดคำแบบใหม่ **Head + Tail Truncation**
        # เพื่อรักษารายละเอียดต้นประโยคและบทสรุปช่วงปลายของรีวิวภาษาไทยที่ยาวเกินเกณฑ์ max_len ไว้
        tokens = self.tokenizer.tokenize(text) # หั่นข้อความดิบออกเป็นชิ้นคำย่อย (Tokens)
        if len(tokens) > self.max_len - 2:
            # หากความยาว Tokens รวมเกินเกณฑ์สูงสุด (-2 สำหรับโทเคนพิเศษ <s> และ </s>)
            head_len = (self.max_len - 2) // 2 # จัดสรรความยาวครึ่งหนึ่งให้ส่วนหัวรีวิว
            tail_len = (self.max_len - 2) - head_len # จัดสรรความยาวที่เหลือให้ส่วนท้ายรีวิว
            # ประกบประกบโทเคนตัวแรกสุดและตัวท้ายสุดเข้าด้วยกันเป็นก้อนคำใหม่
            tokens = tokens[:head_len] + tokens[-tail_len:]
            text = self.tokenizer.convert_tokens_to_string(tokens) # แปลงโทเคนประกบกลับเป็นข้อความสตริงเดียว

        # ดำเนินการตัดวิเคราะห์คำและสร้าง Tensors ประจำตัวอย่าง
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,       # แปะ <s> และ </s> ปิดหัวปิดท้าย
            max_length=self.max_len,       # กำหนดความยาวสูงสุด
            padding="max_length",          # เติม 0 ชดเชยจนความยาวครบเท่ากันทุกแถวใน Dataset (Static padding)
            truncation=True,               # ตัดหั่นเพิ่มเติมเพื่อป้องกันหลุดเกณฑ์
            return_tensors="pt",           # คืนรูปแบบ PyTorch Tensors
        )
        # ส่งดิกชันนารี Tensor ไปใช้งานกับ Dataloader โดยทำการบีบมิติมุมขวาสุดออก (Flatten)
        label_dtype = torch.float if isinstance(label, (float, np.floating)) else torch.long
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=label_dtype), # ปรับประเภทข้อมูลให้รองรับทั้ง Regression (Float) และ Classification (Long)
        }


def run_epoch(
    model,
    loader,
    optimizer=None,
    device=None,
    class_weights: torch.Tensor | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    scheduler=None,
    loss_fn_override=None,
):
    """สคริปต์ควบคุมการทำงานใน 1 รอบ Epoch ครอบคลุมทั้งขั้นตอนสอน (Train) และขั้นตอนประเมิน (Evaluate)"""
    is_train = optimizer is not None # ตรวจเช็คว่ารอบนี้เป็นการเทรนหรือไม่ (หากส่ง Optimizer เข้ามาจะถือว่าเป็นโหมดเทรน)
    model.train() if is_train else model.eval() # ปรับสถานะโมเดลให้เหมาะกับโหมดการทำงาน

    total_loss = 0.0 # ตัวแปรสะสมค่าความสูญเสียรวม (Loss)
    correct = 0      # ตัวนับความถูกต้องในการทำนาย
    total = 0        # ยอดนับจำนวนประชากรที่ประมวลผล
    
    loss_fn = None # นิยามสูตรคำนวณ Loss
    if loss_fn_override is not None:
        # ใช้ Loss Function ที่กำหนดจากภายนอก (เช่น FocalLoss)
        loss_fn = loss_fn_override
    elif is_train and class_weights is not None:
        # ใช้สูตร Cross Entropy Loss ร่วมกับเวกเตอร์น้ำหนักคำนวณชดเชย Imbalance (Weighted Cross Entropy)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    batch_idx = 0                  # ดัชนีก้าวของแบทช์
    n_batches = len(loader)        # จำนวนแบทช์รวมทั้งหมดใน Dataloader
    log_every = max(1, n_batches // 10) # กำหนดจังหวะความถี่ในการพิมพ์พิมพ์แสดงผลลงจอ Terminal
    
    if is_train:
        # ล้างประวัติเกรเดียนต์ย้อนกลับที่ค้างอยู่ในตัวนำน้ำหนักให้อยู่ในสถานะว่างเปล่า (set_to_none=True ช่วยเซฟแรมการ์ดจอ)
        optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        # โหลดข้อมูล Tensor ประจำแบทช์นี้เข้าไปยังการ์ดจอประมวลผล
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # เช็คความพร้อมของฮาร์ดแวร์เพื่อเปิดใช้งานตัวเร่งคำนวณทศนิยม FP16 (AMP) บนการ์ดจอ Nvidia CUDA
        amp_enabled = use_amp and device is not None and device.type == "cuda"
        
        # ปรับการเปิด-ปิดการสะสมค่าประเมินเกรเดียนต์ (Gradient computation) ตามรอบโหมด Train
        with torch.set_grad_enabled(is_train):
            # เปิด-ปิด ระบบเร่งประมวลผลทศนิยมแบบผสม (Automatic Mixed Precision: autocast)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                if loss_fn is not None:
                    # เทรนในแบบกำหนดถ่วงน้ำหนักคลาสประมวลผล
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    if outputs.logits.size(-1) == 1:
                        preds = outputs.logits.view(-1)
                        if class_weights is not None:
                            idx = torch.clamp(labels.long() - 1, 0, class_weights.size(0) - 1)
                            weights = class_weights[idx]
                            # Use manual weighted MSE
                            loss = (weights * (preds - labels.float()) ** 2).mean()
                        else:
                            loss = loss_fn(preds, labels.float())
                    else:
                        loss = loss_fn(outputs.logits, labels) # คำนวณความสูญเสียโดยแนบเวตดาวน้อยถ่วงใจ
                else:
                    # เทรนในแบบค่าสูญเสียความแม่นยำมาตรฐาน (แกะจากโมเดลโดยตรง)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss # โมเดลดึง Loss อัตโนมัติจาก HuggingFace

        # ขั้นตอนปรับค่าน้ำหนักย้อนกลับ (Backward pass) สำหรับโหมดฝึกฝน
        if is_train:
            scaled_loss = loss / grad_accum_steps # หารเฉลี่ยค่า Loss ตามจำนวนก้าวสะสมเกรเดียนต์ (Gradient Accumulation)
            if scaler is not None:
                # รันแบบมีตัวสเกลเลอร์ประคองความแม่นยำทศนิยมต่ำ (AMP GradScaler) ป้องกันตัวเลขกลายเป็นศูนย์ (Underflow)
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward() # รันคำนวณเกรเดียนต์ย้อนกลับแบบธรรมดา

            # เมื่อก้าวมาครบจนถึงจังหวะอัปเดตน้ำหนัก
            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer) # อัปเดตค่าน้ำหนักโดยผ่านเกราะสเกลเลอร์ AMP
                    scaler.update()        # ปรับค่าขนาดสเกลของการคำนวณรอบถัดไป
                else:
                    optimizer.step()       # ปรับค่าน้ำหนักเกรเดียนต์แบบมาตรฐาน
                
                if scheduler is not None:
                    scheduler.step()       # ให้ตัวจัดการระดับความเร็วเรียนรู้อัตโนมัติก้าวทำงานสเตปย่อยตาม
                
                optimizer.zero_grad(set_to_none=True) # ล้างแรมเกรเดียนต์พร้อมเข้าแบทช์ถัดไป

        batch_idx += 1 # เพิ่มระดับดัชนีแบทช์
        if is_train and batch_idx % log_every == 0:
            # พิมพ์สรุปผลงานระดับรอบย่อยลงจอวิจัย
            print(f"  train batch {batch_idx}/{n_batches} loss={loss.item():.4f}")

        total_loss += loss.item() * labels.size(0) # สะสมผลต่างความสูญเสียคูณจำนวนตัวอย่าง
        if outputs.logits.size(-1) == 1:
            preds = torch.clamp(torch.round(outputs.logits.view(-1)), 1.0, 5.0)
        else:
            preds = outputs.logits.argmax(dim=-1)      # ดึงคลาสดาวที่คะแนนลอจิตต์สูงที่สุดเป็นผลทำนาย
        correct += (preds == labels).sum().item()   # สะสมผลความถูกต้องของการทาย
        total += labels.size(0)                    # สะสมจำนวนประชากรที่ประมวลผล

    # ป้องกันกรณีเก็บเศษตกค้าง (Remainder batches) ตอนจบ Epoch หากมิติแบทช์หารไม่ลงตัวกับตัวสะสมน้ำหนัก
    if is_train and batch_idx % grad_accum_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # ส่งคืนผลลัพธ์เฉลี่ย ความสูญเสียเฉลี่ย (Average Loss) และ ความถูกต้องเฉลี่ย (Average Accuracy) ประจำรอบ
    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_loader_mae(
    model,
    loader,
    device,
    *,
    use_amp: bool = False,
    use_regression: bool = False,
    use_3class: bool = False,
):
    """Compute mean absolute error (stars 1-5) on a DataLoader."""
    model.eval()
    expected_parts = []
    true_parts = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            amp_enabled = use_amp and device is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
            if use_regression or logits.size(-1) == 1:
                expected_parts.append(logits.squeeze(-1).cpu().numpy())
            elif use_3class:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                expected_parts.append(
                    utils.class3_to_expected_star(np.argmax(probs, axis=1))
                )
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                expected_parts.append(expected_rating_from_probs(probs))

            if use_regression:
                true_parts.append(labels.cpu().numpy().astype(np.float64))
            elif use_3class:
                true_parts.append(
                    utils.class3_to_expected_star(labels.cpu().numpy().astype(int))
                )
            else:
                true_parts.append(labels.cpu().numpy().astype(np.float64) + 1.0)

    expected = np.concatenate(expected_parts)
    true = np.concatenate(true_parts)
    return float(np.mean(np.abs(expected - true)))


def _log_training_device(device: torch.device) -> None:
    """Print resolved device and a clear hint when CUDA exists but training falls back to CPU."""
    print(f"PyTorch {torch.__version__} | TORCH_DEVICE={config.TORCH_DEVICE}")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(idx)} (cuda:{idx})")
        return
    if torch.cuda.is_available():
        print(f"WARNING: CUDA is visible but training uses CPU. {config.cuda_device_hint()}")
    else:
        print("Training on CPU (no CUDA device reported by PyTorch).")


def main() -> None:
    """แกนโปรแกรมควบคุมการจูนโมเดลประมวลผลข้อความ NLP ภาษาไทยระดับสูง"""
    device = torch.device(config.TORCH_DEVICE) # ดึงพิกัดอุปกรณ์ประมวลผลหลัก (ชิป CUDA)
    _log_training_device(device)
    
    # กำหนดจำนวนคลาสตามโหมดการทำงาน (Regression, 3-class, หรือ 5-class)
    use_regression = getattr(config, "XLMR_USE_REGRESSION", False)
    use_3class = getattr(config, "XLMR_USE_3CLASS", False)
    
    if use_regression:
        num_labels = 1
        mode_label = "Ordinal Regression (1.0 to 5.0)"
    elif use_3class:
        num_labels = 3
        mode_label = "3-class (Negative/Neutral/Positive)"
    else:
        num_labels = 5
        mode_label = "5-class (1-5 stars)"
    print(f"--- Step 1: Initialize Tokenizer and Model (device={device}, mode={mode_label}) ---")
    
    # 1. โหลดเวตโมเดลพรีเทรนระดับสูงเข้ามาติดตั้งคลาสดาวแยกแยะ
    model = AutoModelForSequenceClassification.from_pretrained(
        config.XLMR_MODEL_NAME,
        num_labels=num_labels,
    )
    model.to(device) # โหลดตัวโมเดล NLP เข้ารันแรมการ์ดจอหลัก
    
    # 2. ปรับแต่งโครงสร้างเพื่อลดขีดจำกัดแรมการ์ดจอระดับสูงสุด (หากมี VRAM ต่ำสุดถึง 6GB)
    if config.XLMR_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable() # เปิดระบบฝากผลเกรเดียนต์ย้อนกลับเซฟแรม
        model.config.use_cache = False         # ปิดระบบแคชคำตอบช่วงกลางที่ไม่ได้ใช้งานย้อนกลับ
        
    tokenizer = AutoTokenizer.from_pretrained(config.XLMR_MODEL_NAME) # โหลดโทเคนไนเซอร์คู่บารมีของ XLM-R

    # 3. โหลดและจัดโครงสร้างข้อมูลข้อความผ่านกลยุทธ์ตัวทำความสะอาดตามที่กำหนดใน Config
    strategy = getattr(config, "XLMR_PREPROCESS_STRATEGY", "aggressive")
    normalize_func = utils.PREPROCESS_REGISTRY.get(strategy, utils.xlmr_normalize_text)
    print(f"  > Using preprocessing strategy: {strategy}")
    df = utils.load_and_standardize_data(
        config.RAW_DATA_PATH,
        normalize_func=normalize_func
    )
    # ทำการกรองลบขยะ รีวิวสั้น และรีวิวสแกนซ้ำซ้อน
    df, stats = utils.clean_review_dataframe(
        df,
        min_text_length=5,
        drop_duplicates=True
    )
    utils.log_cleaning_stats(stats, label="xlmr_train_pool") # บันทึกยอดขจัด Noise
    
    # 4. แยกข้อมูลออกเป็นชุดฝึกสอน (Train Split) และชุดประเมินชั่วคราว (Validation Split) ในสัดส่วน 80:20
    # โดยมีนโยบายการกระจายชั้นดาวให้มีความสม่ำเสมอเท่ากันในสองชุดข้อมูล (Stratify based on user_rating)
    df_train, df_val = train_test_split(
        df,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=df["user_rating"],
    )

    df_train = apply_train_augmentation(df_train)

    # ปรับระดับกลุ่มเป้าหมายตามโหมด
    if use_3class:
        # ยุบรวมคลาส: 1-2★ -> 0 (Negative), 3★ -> 1 (Neutral), 4-5★ -> 2 (Positive)
        train_labels = utils.rating_to_3class(df_train["user_rating"].values)
        val_labels = utils.rating_to_3class(df_val["user_rating"].values)
        print(f"\n  Label mapping: 1-2★ -> Negative(0), 3★ -> Neutral(1), 4-5★ -> Positive(2)")
    else:
        if getattr(config, "XLMR_USE_REGRESSION", False):
            # โหมด Ordinal Regression ใช้ค่าดาวเต็มๆ 1.0 - 5.0 เป็นเป้าหมายแบบทศนิยม (Float)
            train_labels = df_train["user_rating"].values.astype(np.float32)
            val_labels = df_val["user_rating"].values.astype(np.float32)
            print(f"\n  Label mapping: Ordinal Regression (1.0 to 5.0)")
        else:
            # ปรับระดับกลุ่มเป้าหมายให้อยู่ในช่วงระดับดัชนี 0-4 (1-5★ ปรับลงมา 1 ดัชนี)
            train_labels = df_train["user_rating"].values - 1
            val_labels = df_val["user_rating"].values - 1
    train_ratings = df_train["user_rating"].values # คะแนนดาวจริงของชุดสอน

    # 5. วิเคราะห์ค่าชดเชยน้ำหนักอสมมาตร (Class Weights) เพื่อถ่วงความแม่นยำให้กลุ่มดาวคะแนนน้อย
    class_weight_np = None
    class_weights_tensor = None
    if config.XLMR_USE_CLASS_WEIGHT and not use_regression:
        # คำนวณเวกเตอร์ถ่วงน้ำหนักความถี่ผกผัน พร้อมมีตัวเร่งลงโทษดาวต่ำ 1-2 ดาว 1.5 เท่า
        class_weight_np = utils.compute_class_weights(
            train_ratings,
            num_classes=num_labels,
            low_star_boost=config.XLMR_LOW_STAR_BOOST,
        )
        # โหลดค่าคำนวณเก็บไว้ในแรมการ์ดจอพร้อมทำงานในรูปของ PyTorch Float Tensor
        class_weights_tensor = torch.tensor(
            class_weight_np, dtype=torch.float32, device=device
        )
    # พิมพ์กราฟแจกแจงความถี่และน้ำหนักชดเชยออกจอประวัติศาสตร์การรัน
    utils.print_rating_distribution(
        train_ratings, class_weight_np, label="train"
    )

    batch_size = config.BATCH_SIZE                       # ขนาดแบทช์ย่อย
    grad_accum_steps = max(1, config.XLMR_GRAD_ACCUM_STEPS) # รอบสะสมน้ำหนัก
    max_length = config.MAX_LENGTH                       # มิติความยาวประโยค

    # 6. สร้าง Dataset สำหรับเตรียมส่งข้อความ
    train_dataset = ReviewDataset(
        df_train["text"].values,
        train_labels,
        tokenizer,
        max_length,
    )
    val_dataset = ReviewDataset(
        df_val["text"].values,
        val_labels,
        tokenizer,
        max_length,
    )

    # 7. สร้าง DataLoader สำหรับลูปสับเปลี่ยนข้อมูลและสุ่มสับแถว (Shuffle=True สำหรับชุดสอนเท่านั้น)
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
    )

    # 8. นิยามตัวปรับค่าน้ำหนักสุดล้ำ AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,          # อัตราเร็วในการเรียนรู้ (เช่น 2e-5)
        weight_decay=config.WEIGHT_DECAY,  # ตัวลดทอนน้ำหนัก (Weight Decay) ช่วยควบคุม Overfitting
    )

    # 9. นิยามตัวควบคุมความเร็วระดับสูง **Linear Warmup Learning Rate Scheduler** ของ Hugging Face
    scheduler = None
    if config.XLMR_USE_LR_SCHEDULER:
        # คำนวณรอบสเตปรวมในการสอนทั้งหมดในระบบ
        total_steps = len(train_loader) * config.EPOCHS // grad_accum_steps
        warmup_steps = int(total_steps * 0.1)  # ปูระยะค่อยๆ ไต่ระดับความเร็วรอบแรก 10% ของงานทั้งหมด (Warmup)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    train_class_weights = class_weights_tensor if config.XLMR_USE_CLASS_WEIGHT else None
    use_amp = config.XLMR_USE_AMP and device.type == "cuda" # ความพร้อมทำงานแบบผสมความคมชัดทศนิยม
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) # ตัวสเกลเกรเดียนต์ป้องกัน Underflow คู่อุปกรณ์ประมวลผล FP16
    effective_batch = batch_size * grad_accum_steps # ขนาดความจุแบทช์แท้จริงเฉลี่ยสะสมน้ำหนัก

    # ==========================================
    # สร้าง Loss Function: MSE, Focal Loss หรือ Weighted CrossEntropy
    # ==========================================
    train_loss_fn = None  # None = ใช้ loss จาก HuggingFace model โดยตรง
    use_focal = getattr(config, "XLMR_USE_FOCAL_LOSS", False)
    
    if use_regression:
        train_loss_fn = nn.MSELoss()
        print(f"  > Using MSELoss for Ordinal Regression")
    elif use_focal:
        focal_gamma = getattr(config, "XLMR_FOCAL_GAMMA", 2.0)
        focal_alpha = getattr(config, "XLMR_FOCAL_ALPHA", None)
        # ถ้า focal_alpha เป็น None ให้ใช้ class_weights แทน
        if focal_alpha is None and train_class_weights is not None:
            focal_alpha = train_class_weights
        train_loss_fn = utils.FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        train_loss_fn = train_loss_fn.to(device)
        print(f"  > Using Focal Loss (gamma={focal_gamma}, alpha={'class_weights' if focal_alpha is not None else 'None'})")
    elif train_class_weights is not None:
        train_loss_fn = nn.CrossEntropyLoss(weight=train_class_weights)
        print(f"  > Using Weighted CrossEntropyLoss")
    else:
        print(f"  > Using default CrossEntropyLoss (from HuggingFace model)")
    
    print(
        f"--- Step 2: Training (batch={batch_size}, "
        f"accum={grad_accum_steps}, effective_batch={effective_batch}, "
        f"max_length={max_length}, amp={use_amp}) ---"
    )

    # กำหนดสถานะและสถิติเริ่มต้นของระบบ Early Stopping (ตรวจหาจุด MAE ต่ำสุด)
    best_val_mae = float("inf")
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    # 10. แกนลูปวนฝึกฝนโมเดลผ่านรอบ Epoch
    for epoch in range(config.EPOCHS):
        # 10.1 รันรอบฝึกฝนค่าน้ำหนัก
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            class_weights=train_class_weights,
            scaler=scaler if use_amp else None,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
            scheduler=scheduler,
            loss_fn_override=train_loss_fn,
        )
        # 10.2 รันรอบตรวจสอบความแม่นยำชุด Validation (ไม่มีตัวป้อนเกรเดียนต์และ Optimizer)
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            device=device,
            class_weights=None,
            use_amp=use_amp,
        )
        val_mae = evaluate_loader_mae(
            model,
            val_loader,
            device,
            use_amp=use_amp,
            use_regression=use_regression,
            use_3class=use_3class,
        )
        # เคลียร์ล้างหน่วยความจำแคชการ์ดจอที่ตกค้างเพื่อรักษาระดับการระบายความร้อนการ์ดจอ
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_mae={val_mae:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_acc = val_acc
            best_epoch = epoch + 1   # บันทึก Epoch แห่งชัยชนะ
            # บันทึกโคลนน้ำหนักพารามิเตอร์ทั้งหมดในโมเดลเก็บค้างในระบบแรมปกติ (CPU RAM) เพื่อป้องกันโมเดลถดถอยปลายทาง
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()
            patience_counter = 0 # รีเซ็ตตัวนับการรอเมื่อค้นพบสิ่งที่ดีขึ้น
        else:
            patience_counter += 1 # บวกจำนวนรอเพิ่มขึ้นหากผลลัพธ์รอบนี้ทรงตัวหรือดิ่งแย่ลง

        if scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"] # พิมพ์ระดับความเร็วการปรับตัวปัจจุบันลงรายงาน
            print(f"  lr={current_lr:.2e}")

        # 10.4 ระบบ Early Stopping สั่งยุติขั้นตอนสอนล่วงหน้าทันทีเพื่อป้องกันโมเดล Overfit หรือฝึกวนจนหลุดกรอบ
        if patience_counter >= config.XLMR_EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no val_mae improvement for "
                f"{config.XLMR_EARLY_STOPPING_PATIENCE} epochs)"
            )
            break # ทะลายออกนอกลูป Epoch

    print("--- Step 3: Saving Weights Manually ---")
    if best_state is not None:
        # ดึงน้ำหนักตัวแบบ NLP รุ่นที่แข็งแกร่งที่สุดในประวัติศาสตร์รอบ Epoch กลับมาติดตั้งใส่ร่างโมเดล
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint from epoch {best_epoch} (val_mae={best_val_mae:.4f})")
    else:
        print("Warning: no improvement seen; saving final epoch weights.")

    # 11. ดำเนินการสร้างไดเรกทอรีและเขียนเวตไฟล์และบอร์กโทเคนลงไฟล์เป้าหมาย (artifacts/xlmr/)
    out_dir = config.XLMR_ARTIFACTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)     # บันทึกพารามิเตอร์น้ำหนักเวตตัวแบบหลักของ HuggingFace
    tokenizer.save_pretrained(out_dir) # บันทึกคลังคำศัพท์และโทเคนไนเซอร์
    xlmr_meta = {
        "preprocess_strategy": strategy,
        "best_val_mae": best_val_mae if best_val_mae != float("inf") else None,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "use_regression": use_regression,
        "use_3class": use_3class,
    }
    with open(config.XLMR_META_PATH, "w", encoding="utf-8") as f:
        json.dump(xlmr_meta, f, indent=2)
    print(f"Done PyTorch loop training! Saved to {out_dir}")


if __name__ == "__main__":
    main() # วิ่งตัวฝึกฝนเมื่อสคริปต์ทำงานโดดๆ
