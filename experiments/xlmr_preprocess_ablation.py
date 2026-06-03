# -*- coding: utf-8 -*-
"""
experiments/xlmr_preprocess_ablation.py
=======================================
สคริปต์ทดลองเปรียบเทียบ Preprocessing Strategy 6 แบบ สำหรับ XLM-RoBERTa
- รันแต่ละ strategy บน train_reduce.csv
- เทรน XLM-R 1 epoch (เร็ว) แล้ววัด val_acc / val_loss
- บันทึกผลทุกรอบลง experiments/xlmr/preprocess_ablation_log.json
- พิมพ์ตารางสรุปเปรียบเทียบตอนจบ

Usage:
    python experiments/xlmr_preprocess_ablation.py
    python experiments/xlmr_preprocess_ablation.py --strategies default minimal aggressive
    python experiments/xlmr_preprocess_ablation.py --epochs 2 --max-samples 500
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# ต้อง pip install -e . เพื่อ import rris
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rris import config, utils
from rris.training.xlmr import ReviewDataset, run_epoch


def run_single_experiment(
    strategy_name: str,
    preprocess_fn: callable,
    epochs: int = 1,
    max_samples: int = 0,
    batch_size: int = 8,
    max_length: int = 128,
) -> dict:
    """รันการทดลอง 1 strategy: โหลดข้อมูล -> preprocess -> เทรน -> วัดผล"""
    device = torch.device(config.TORCH_DEVICE)
    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Device: {device} | Epochs: {epochs} | MaxLen: {max_length}")
    print(f"{'='*60}")

    # 1. โหลดและ preprocess ข้อมูลด้วย strategy ที่เลือก
    t0 = time.time()
    df = utils.load_and_standardize_data(
        config.RAW_DATA_PATH, normalize_func=preprocess_fn
    )
    df, clean_stats = utils.clean_review_dataframe(
        df, min_text_length=5, drop_duplicates=True
    )
    preprocess_time = time.time() - t0

    # จำกัดจำนวนตัวอย่างถ้าระบุ (สำหรับรันเร็ว)
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=config.RANDOM_STATE).reset_index(drop=True)

    print(f"  Rows after clean: {len(df)} (preprocess took {preprocess_time:.1f}s)")

    # 2. แสดงตัวอย่างข้อความ 3 แถวแรกเพื่อเห็นผลลัพธ์ preprocessing
    print(f"  Sample texts:")
    for i, row in df.head(3).iterrows():
        text_preview = row["text"][:80] + "..." if len(row["text"]) > 80 else row["text"]
        print(f"    [{row['user_rating']}★] {text_preview}")

    # 3. แบ่ง train/val
    df_train, df_val = train_test_split(
        df, test_size=0.2, random_state=config.RANDOM_STATE,
        stratify=df["user_rating"],
    )
    train_labels = df_train["user_rating"].values - 1
    val_labels = df_val["user_rating"].values - 1

    # 4. คำนวณ class weights
    class_weight_np = utils.compute_class_weights(
        df_train["user_rating"].values, low_star_boost=config.XLMR_LOW_STAR_BOOST
    )
    class_weights_tensor = torch.tensor(class_weight_np, dtype=torch.float32, device=device)

    # 5. โหลด tokenizer & model (ใหม่ทุกรอบเพื่อความยุติธรรม)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.XLMR_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.XLMR_MODEL_NAME, num_labels=5
    )
    model.to(device)
    if config.XLMR_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # 6. สร้าง Dataset & DataLoader
    train_ds = ReviewDataset(df_train["text"].values, train_labels, tokenizer, max_length)
    val_ds = ReviewDataset(df_val["text"].values, val_labels, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 7. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    use_amp = config.XLMR_USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 8. เทรนและวัดผล
    epoch_results = []
    best_val_acc = 0.0
    t_train_start = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer=optimizer, device=device,
            class_weights=class_weights_tensor, scaler=scaler if use_amp else None,
            grad_accum_steps=1, use_amp=use_amp,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, device=device, use_amp=use_amp,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

        best_val_acc = max(best_val_acc, val_acc)
        epoch_results.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })
        print(f"  Epoch {epoch+1}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    train_time = time.time() - t_train_start

    # 9. ล้างแรม
    del model, tokenizer, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "strategy": strategy_name,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "epochs": epochs,
        "max_length": max_length,
        "batch_size": batch_size,
        "preprocess_time_sec": round(preprocess_time, 2),
        "train_time_sec": round(train_time, 2),
        "best_val_acc": round(best_val_acc, 4),
        "final_val_loss": epoch_results[-1]["val_loss"],
        "epoch_results": epoch_results,
        "clean_stats": clean_stats,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="XLM-R Preprocessing Ablation Study")
    parser.add_argument("--strategies", nargs="+", default=list(utils.PREPROCESS_REGISTRY.keys()),
                        help="เลือก strategy ที่ต้องการทดสอบ (default: ทั้งหมด)")
    parser.add_argument("--epochs", type=int, default=1, help="จำนวน epoch ต่อ strategy (default: 1)")
    parser.add_argument("--max-samples", type=int, default=0, help="จำกัดจำนวนตัวอย่าง (0=ไม่จำกัด)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    # ตรวจสอบ strategy ที่เลือก
    for s in args.strategies:
        if s not in utils.PREPROCESS_REGISTRY:
            print(f"ERROR: Unknown strategy '{s}'. Available: {list(utils.PREPROCESS_REGISTRY.keys())}")
            sys.exit(1)

    # รันทดลองทุก strategy
    all_results = []
    for strategy_name in args.strategies:
        preprocess_fn = utils.PREPROCESS_REGISTRY[strategy_name]
        result = run_single_experiment(
            strategy_name=strategy_name,
            preprocess_fn=preprocess_fn,
            epochs=args.epochs,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        all_results.append(result)

    # บันทึกผลลง JSON
    log_dir = os.path.join(config.EXPERIMENTS_DIR, "xlmr")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "preprocess_ablation_log.json")

    # อ่านผลเดิมถ้ามี แล้ว append เข้าไป
    existing = []
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.extend(all_results)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\nResults appended to: {log_path}")

    # พิมพ์ตารางสรุป
    print(f"\n{'='*70}")
    print(f"  PREPROCESSING ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Strategy':<15} {'Val Acc':>10} {'Val Loss':>10} {'Prep(s)':>8} {'Train(s)':>9}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*8} {'-'*9}")

    # เรียงตาม val_acc จากมากไปน้อย
    sorted_results = sorted(all_results, key=lambda x: x["best_val_acc"], reverse=True)
    for r in sorted_results:
        marker = " ★" if r == sorted_results[0] else ""
        print(f"  {r['strategy']:<15} {r['best_val_acc']:>10.4f} {r['final_val_loss']:>10.4f} "
              f"{r['preprocess_time_sec']:>8.1f} {r['train_time_sec']:>9.1f}{marker}")

    print(f"\n  Best strategy: {sorted_results[0]['strategy']} "
          f"(val_acc={sorted_results[0]['best_val_acc']:.4f})")


if __name__ == "__main__":
    main()
