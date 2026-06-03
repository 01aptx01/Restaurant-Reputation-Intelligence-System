# -*- coding: utf-8 -*-
"""
run_all_preprocess.py — รันทดลอง Preprocessing ทุกตัวรวดเดียว แล้วเลือกตัวที่ดีที่สุด
====================================================================================
สคริปต์นี้จะ:
  1. วนลูปรัน XLM-R ทั้ง 6 strategies (default, minimal, aggressive, keep_digits, emoji_tag, segment)
  2. เทรนแต่ละตัว 1 epoch บนข้อมูลจริง (หรือจำกัด --max-samples เพื่อความเร็ว)
  3. เก็บผลทุกรอบลง experiments/xlmr/full_comparison.json
  4. พิมพ์ตารางสรุป + ประกาศตัวที่ชนะ (Best Strategy)
  5. เขียน best_strategy.txt ไว้ให้ train_xlmr.py อ่านใช้ได้เลย

Usage:
  python experiments/run_all_preprocess.py                    # รันทั้งหมด (ข้อมูลเต็ม 1 epoch)
  python experiments/run_all_preprocess.py --max-samples 800  # จำกัด 800 แถว (รันเร็ว ~5 นาที)
  python experiments/run_all_preprocess.py --epochs 2         # เทรน 2 epoch ต่อ strategy
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# เพิ่ม project root เข้า path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from rris import config, utils
from train_xlmr import ReviewDataset, run_epoch

# ===== ตั้งค่าผลลัพธ์ =====
OUTPUT_DIR = os.path.join(config.EXPERIMENTS_DIR, "xlmr")
RESULT_PATH = os.path.join(OUTPUT_DIR, "full_comparison.json")
BEST_PATH = os.path.join(OUTPUT_DIR, "best_strategy.txt")


def train_one_strategy(name, preprocess_fn, args):
    """เทรน XLM-R 1 strategy แล้วส่งคืนผลลัพธ์"""
    device = torch.device(config.TORCH_DEVICE)

    # — โหลด + preprocess —
    t0 = time.time()
    df = utils.load_and_standardize_data(config.RAW_DATA_PATH, normalize_func=preprocess_fn)
    df, stats = utils.clean_review_dataframe(df, min_text_length=5, drop_duplicates=True)
    prep_sec = time.time() - t0

    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    # — แบ่ง train/val —
    df_train, df_val = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["user_rating"]
    )
    train_labels = df_train["user_rating"].values - 1
    val_labels = df_val["user_rating"].values - 1

    # — class weights —
    cw = utils.compute_class_weights(df_train["user_rating"].values, low_star_boost=config.XLMR_LOW_STAR_BOOST)
    cw_tensor = torch.tensor(cw, dtype=torch.float32, device=device)

    # — โหลด model (ใหม่ทุกรอบ) —
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.XLMR_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(config.XLMR_MODEL_NAME, num_labels=5)
    model.to(device)
    if config.XLMR_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # — DataLoader —
    bs = args.batch_size
    ml = args.max_length
    train_loader = DataLoader(ReviewDataset(df_train["text"].values, train_labels, tokenizer, ml), batch_size=bs, shuffle=True)
    val_loader = DataLoader(ReviewDataset(df_val["text"].values, val_labels, tokenizer, ml), batch_size=bs, shuffle=False)

    # — Optimizer —
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    use_amp = config.XLMR_USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # — Training loop —
    best_val_acc = 0.0
    epochs_log = []
    t1 = time.time()

    for ep in range(args.epochs):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, optimizer=optimizer, device=device,
            class_weights=cw_tensor, scaler=scaler if use_amp else None,
            grad_accum_steps=1, use_amp=use_amp,
        )
        va_loss, va_acc = run_epoch(model, val_loader, device=device, use_amp=use_amp)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        best_val_acc = max(best_val_acc, va_acc)
        epochs_log.append({"ep": ep+1, "tr_loss": round(tr_loss,4), "tr_acc": round(tr_acc,4),
                           "va_loss": round(va_loss,4), "va_acc": round(va_acc,4)})
        print(f"    Epoch {ep+1}/{args.epochs}  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  va_acc={va_acc:.4f}")

    train_sec = time.time() - t1

    # — cleanup —
    del model, tokenizer, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "strategy": name,
        "n_train": len(df_train), "n_val": len(df_val),
        "epochs": args.epochs, "max_length": ml, "batch_size": bs,
        "prep_sec": round(prep_sec, 1), "train_sec": round(train_sec, 1),
        "best_val_acc": round(best_val_acc, 4),
        "final_val_loss": epochs_log[-1]["va_loss"],
        "epochs_log": epochs_log,
        "clean_stats": stats,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    strategies = list(utils.PREPROCESS_REGISTRY.items())  # ทุกตัว
    total = len(strategies)

    print("=" * 70)
    print(f"  XLM-R PREPROCESSING FULL COMPARISON ({total} strategies)")
    print(f"  epochs={args.epochs}  max_samples={args.max_samples or 'ALL'}  device={config.TORCH_DEVICE}")
    print("=" * 70)

    results = []
    t_all = time.time()

    for i, (name, fn) in enumerate(strategies, 1):
        print(f"\n[{i}/{total}] === {name.upper()} ===")
        r = train_one_strategy(name, fn, args)
        results.append(r)
        elapsed = time.time() - t_all
        remaining = elapsed / i * (total - i)
        print(f"    Done ({r['train_sec']:.0f}s). ETA remaining: {remaining/60:.1f} min")

    total_time = time.time() - t_all

    # ===== บันทึกผลลัพธ์ =====
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_record = {
        "run_timestamp": datetime.now().isoformat(),
        "total_time_sec": round(total_time, 1),
        "args": vars(args),
        "results": results,
    }

    # append ต่อจากรอบเก่า (ถ้ามี)
    history = []
    if os.path.isfile(RESULT_PATH):
        with open(RESULT_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(run_record)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # ===== หาตัวที่ดีที่สุด =====
    ranked = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)
    winner = ranked[0]

    with open(BEST_PATH, "w", encoding="utf-8") as f:
        f.write(winner["strategy"])

    # ===== ตารางสรุป =====
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY (sorted by val_acc)")
    print("=" * 70)
    print(f"  {'#':>2}  {'Strategy':<15}  {'Val Acc':>8}  {'Val Loss':>9}  {'Prep':>6}  {'Train':>7}")
    print(f"  {'--':>2}  {'-'*15}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*7}")
    for rank, r in enumerate(ranked, 1):
        tag = " << BEST" if rank == 1 else ""
        print(f"  {rank:>2}  {r['strategy']:<15}  {r['best_val_acc']:>8.4f}  {r['final_val_loss']:>9.4f}  "
              f"{r['prep_sec']:>5.1f}s  {r['train_sec']:>6.1f}s{tag}")

    print(f"\n  Winner: {winner['strategy']}  (val_acc = {winner['best_val_acc']:.4f})")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Results saved: {RESULT_PATH}")
    print(f"  Best strategy: {BEST_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
