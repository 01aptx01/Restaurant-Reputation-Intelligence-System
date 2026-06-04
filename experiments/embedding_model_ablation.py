# -*- coding: utf-8 -*-
"""
experiments/embedding_model_ablation.py
=======================================
สคริปต์ทดลองเปรียบเทียบ Embedding Models หลายตัว สำหรับ Thai Restaurant Review Classification
- ทดลองแต่ละ embedding model บน train data
- เทรน classifier (LR/XGB/SVM) แล้ววัด val_mae / val_acc / f1
- บันทึกผลทุกรอบลง experiments/embedding/model_ablation_log.json
- พิมพ์ตารางสรุปเปรียบเทียบตอนจบ

Usage:
    python experiments/embedding_model_ablation.py
    python experiments/embedding_model_ablation.py --models e5-base bge-m3
    python experiments/embedding_model_ablation.py --max-samples 500 --classifiers lr xgb
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from rris import config, utils
from rris.evaluation.selection import classifier_val_mae, pick_lowest_mae

# Registry ของ embedding models ที่ต้องการทดสอบ
EMBEDDING_MODELS = {
    "e5-base": "intfloat/multilingual-e5-base",
    "e5-large": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}

CLASSIFIERS = {
    "lr": lambda: LogisticRegression(
        class_weight="balanced", max_iter=1000, n_jobs=-1,
        random_state=config.RANDOM_STATE,
    ),
    "xgb": lambda: __import__("xgboost").XGBClassifier(
        objective="multi:softprob", num_class=5, max_depth=4,
        learning_rate=0.05, n_estimators=300, tree_method="hist",
        device=config.XGB_DEVICE, random_state=config.RANDOM_STATE,
    ),
    "svm": lambda: LinearSVC(
        max_iter=1000, class_weight="balanced", dual=False,
        random_state=config.RANDOM_STATE,
    ),
    "rf": lambda: RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1,
        random_state=config.RANDOM_STATE,
    ),
}


def run_single_experiment(
    model_key: str,
    model_name: str,
    df_train, df_val,
    y_train, y_val,
    classifiers: list[str],
    batch_size: int = 32,
) -> dict:
    """รันการทดลอง 1 embedding model: encode -> train classifiers -> วัดผล"""
    device = config.TORCH_DEVICE if config.TORCH_DEVICE != "cpu" else "cpu"

    print(f"\n{'=' * 60}")
    print(f"  Embedding Model: {model_key} ({model_name})")
    print(f"  Device: {device}")
    print(f"{'=' * 60}")

    # 1. Encode
    t0 = time.time()
    try:
        embed_model = SentenceTransformer(model_name, device=device)
        X_train = embed_model.encode(
            df_train["text"].tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        X_val = embed_model.encode(
            df_val["text"].tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embed_dim = X_train.shape[1]
        encode_time = time.time() - t0
        print(f"  Encoded in {encode_time:.1f}s | dim={embed_dim}")
    except Exception as exc:
        print(f"  ERROR loading model: {exc}")
        return {
            "model_key": model_key,
            "model_name": model_name,
            "status": "error",
            "error": str(exc),
        }
    finally:
        del embed_model
        if device != "cpu":
            import torch
            torch.cuda.empty_cache()

    # 2. Train & evaluate classifiers
    class_w = utils.compute_class_weights(y_train, low_star_boost=config.XGB_LOW_STAR_BOOST)
    sample_w = utils.compute_sample_weights_from_ratings(y_train, class_w)

    clf_results = {}
    best_clf_name = None
    best_mae = float("inf")

    for clf_key in classifiers:
        if clf_key not in CLASSIFIERS:
            print(f"  Skipping unknown classifier: {clf_key}")
            continue

        t1 = time.time()
        try:
            clf = CLASSIFIERS[clf_key]()

            if clf_key == "xgb":
                clf.fit(X_train, y_train - 1, sample_weight=sample_w, verbose=False)
                preds = clf.predict(X_val) + 1
            elif clf_key == "svm":
                clf.fit(X_train, y_train, sample_weight=sample_w)
                try:
                    from sklearn.frozen import FrozenEstimator
                    clf = CalibratedClassifierCV(FrozenEstimator(clf))
                    clf.fit(X_val, y_val)
                except ImportError:
                    clf = CalibratedClassifierCV(clf, cv=3)
                    clf.fit(X_train, y_train, sample_weight=sample_w)
                preds = clf.predict(X_val)
            else:
                clf.fit(X_train, y_train, sample_weight=sample_w)
                preds = clf.predict(X_val)

            acc = accuracy_score(y_val, preds)
            mae = float(mean_absolute_error(y_val, preds))
            f1m = f1_score(y_val, preds, average="macro")
            train_time = time.time() - t1

            clf_results[clf_key] = {
                "accuracy": round(acc, 4),
                "mae": round(mae, 4),
                "f1_macro": round(f1m, 4),
                "train_time_sec": round(train_time, 1),
            }
            print(f"    {clf_key:>5}: MAE={mae:.4f} Acc={acc:.4f} F1={f1m:.4f} ({train_time:.1f}s)")

            if mae < best_mae:
                best_mae = mae
                best_clf_name = clf_key

        except Exception as exc:
            clf_results[clf_key] = {"error": str(exc)}
            print(f"    {clf_key:>5}: ERROR — {exc}")

    return {
        "model_key": model_key,
        "model_name": model_name,
        "embed_dim": embed_dim,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "encode_time_sec": round(encode_time, 1),
        "classifiers": clf_results,
        "best_classifier": best_clf_name,
        "best_mae": round(best_mae, 4) if best_mae < float("inf") else None,
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Embedding Model Ablation Study")
    parser.add_argument(
        "--models", nargs="+", default=list(EMBEDDING_MODELS.keys()),
        help=f"Embedding models to test (default: all). Available: {list(EMBEDDING_MODELS.keys())}",
    )
    parser.add_argument(
        "--classifiers", nargs="+", default=["lr", "xgb", "svm"],
        help="Classifiers to test (default: lr xgb svm)",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0=no limit)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--preprocess", default="aggressive",
        choices=list(utils.PREPROCESS_REGISTRY.keys()),
        help="Preprocessing strategy (default: aggressive)",
    )
    args = parser.parse_args()

    # Validate models
    for m in args.models:
        if m not in EMBEDDING_MODELS:
            print(f"ERROR: Unknown model '{m}'. Available: {list(EMBEDDING_MODELS.keys())}")
            sys.exit(1)

    # Load data
    normalize_fn = utils.PREPROCESS_REGISTRY.get(args.preprocess, utils.xlmr_normalize_text)
    df = utils.load_and_standardize_data(config.RAW_DATA_PATH, normalize_func=normalize_fn)
    df, stats = utils.clean_review_dataframe(df, min_text_length=5, drop_duplicates=True)

    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=config.RANDOM_STATE).reset_index(drop=True)

    df_train, df_val = train_test_split(
        df, test_size=0.2, random_state=config.RANDOM_STATE,
        stratify=df["user_rating"],
    )
    y_train = df_train["user_rating"].values
    y_val = df_val["user_rating"].values

    print(f"{'=' * 70}")
    print(f"  EMBEDDING MODEL ABLATION STUDY")
    print(f"  Models: {args.models}")
    print(f"  Classifiers: {args.classifiers}")
    print(f"  Preprocess: {args.preprocess}")
    print(f"  Train: {len(df_train)} | Val: {len(df_val)}")
    print(f"{'=' * 70}")

    # Run experiments
    all_results = []
    for model_key in args.models:
        model_name = EMBEDDING_MODELS[model_key]
        result = run_single_experiment(
            model_key, model_name,
            df_train, df_val, y_train, y_val,
            classifiers=args.classifiers,
            batch_size=args.batch_size,
        )
        all_results.append(result)

    # Save results
    log_dir = os.path.join(config.EXPERIMENTS_DIR, "embedding")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "model_ablation_log.json")

    existing = []
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.extend(all_results)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\nResults appended to: {log_path}")

    # Summary table
    ok_results = [r for r in all_results if r.get("status") == "ok"]
    if ok_results:
        ok_results.sort(key=lambda r: r.get("best_mae") or float("inf"))

        print(f"\n{'=' * 80}")
        print(f"  EMBEDDING MODEL ABLATION SUMMARY")
        print(f"{'=' * 80}")
        print(f"  {'Model':<12} {'Dim':>5} {'Best CLF':<8} {'MAE':>7} "
              f"{'Encode':>8} {'LR MAE':>8} {'XGB MAE':>8} {'SVM MAE':>8}")
        print(f"  {'-' * 12} {'-' * 5} {'-' * 8} {'-' * 7} "
              f"{'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

        for r in ok_results:
            clfs = r.get("classifiers", {})
            tag = " ★" if r == ok_results[0] else ""
            lr_mae = clfs.get("lr", {}).get("mae", "—")
            xgb_mae = clfs.get("xgb", {}).get("mae", "—")
            svm_mae = clfs.get("svm", {}).get("mae", "—")

            lr_str = f"{lr_mae:>8.4f}" if isinstance(lr_mae, (int, float)) else f"{lr_mae:>8}"
            xgb_str = f"{xgb_mae:>8.4f}" if isinstance(xgb_mae, (int, float)) else f"{xgb_mae:>8}"
            svm_str = f"{svm_mae:>8.4f}" if isinstance(svm_mae, (int, float)) else f"{svm_mae:>8}"

            print(
                f"  {r['model_key']:<12} {r.get('embed_dim', 0):>5} "
                f"{r.get('best_classifier', '?'):<8} {r.get('best_mae', 0):>7.4f} "
                f"{r.get('encode_time_sec', 0):>7.1f}s "
                f"{lr_str} {xgb_str} {svm_str}{tag}"
            )

        best = ok_results[0]
        print(f"\n  🏆 Best: {best['model_key']} + {best.get('best_classifier', '?')} "
              f"(MAE={best.get('best_mae', 0):.4f})")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
