# -*- coding: utf-8 -*-
"""
experiments/baseline_feature_ablation.py
========================================
Ablation study สำหรับ Baseline: ทดสอบ feature combinations ทุกแบบ
- Word TF-IDF ± Char TF-IDF ± Extra Features ± LSA
- ใช้ Cross-validation เพื่อให้ผลมีความน่าเชื่อถือ
- บันทึกผลลง experiments/baseline/feature_ablation_log.json

Usage:
    python experiments/baseline_feature_ablation.py
    python experiments/baseline_feature_ablation.py --cv 3 --max-samples 2000
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rris import config, utils
from rris.data.augmentation import apply_train_augmentation
from rris.training.baseline import fit_vectorizer_and_features


def _train_and_eval(
    X_train, X_val,
    y_train: np.ndarray, y_val: np.ndarray,
    use_xgb: bool = True,
) -> dict:
    """Train XGBoost or Logistic Regression and return metrics."""
    import xgboost as xgb

    if use_xgb:
        # Sample weights
        class_w = utils.compute_class_weights(y_train, low_star_boost=config.XGB_LOW_STAR_BOOST)
        sample_w = utils.compute_sample_weights_from_ratings(y_train, class_w)

        dtrain = xgb.DMatrix(X_train, label=y_train - 1, weight=sample_w)
        dval = xgb.DMatrix(X_val, label=y_val - 1)

        params = dict(config.XGB_PARAMS)
        params["device"] = config.XGB_DEVICE

        bst = xgb.train(
            params=params, dtrain=dtrain,
            num_boost_round=200,
            evals=[(dval, "val")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )

        y_prob = bst.predict(dval)
        y_pred = np.argmax(y_prob, axis=1) + 1
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000,
            n_jobs=-1, random_state=config.RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

    mae = float(mean_absolute_error(y_val, y_pred))
    acc = float(accuracy_score(y_val, y_pred))
    f1m = float(f1_score(y_val, y_pred, average="macro"))

    return {"mae": mae, "accuracy": acc, "f1_macro": f1m}


def run_feature_combination(
    combo_name: str,
    feature_config: dict,
    df_pool: pd.DataFrame,
    n_folds: int = 3,
) -> dict:
    """ทดลอง 1 feature combination ด้วย K-Fold CV."""
    print(f"\n{'=' * 50}")
    print(f"  Feature: {combo_name}")
    print(f"  Config: {feature_config}")
    print(f"{'=' * 50}")

    # Apply config overrides temporarily
    snapshot = {}
    for key, value in feature_config.items():
        snapshot[key] = getattr(config, key)
        setattr(config, key, value)

    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
        fold_results = []
        t0 = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_pool, df_pool["user_rating"])):
            fold_train = df_pool.iloc[train_idx].reset_index(drop=True)
            fold_val = df_pool.iloc[val_idx].reset_index(drop=True)

            # Apply text truncation
            fold_train = utils.apply_text_truncation(fold_train, config.MAX_REVIEW_CHARS)
            fold_val = utils.apply_text_truncation(fold_val, config.MAX_REVIEW_CHARS)

            # Build features
            try:
                _, _, _, X_train, X_val, explained = fit_vectorizer_and_features(fold_train, fold_val)
            except Exception as exc:
                print(f"    Fold {fold_idx + 1}: Feature error — {exc}")
                continue

            y_train = fold_train["user_rating"].values.astype(int)
            y_val = fold_val["user_rating"].values.astype(int)

            metrics = _train_and_eval(X_train, X_val, y_train, y_val)
            fold_results.append(metrics)
            print(f"    Fold {fold_idx + 1}: MAE={metrics['mae']:.4f} Acc={metrics['accuracy']:.4f} F1={metrics['f1_macro']:.4f}")

        total_time = time.time() - t0

        if not fold_results:
            return {
                "combo": combo_name,
                "config": feature_config,
                "status": "error",
                "error": "All folds failed",
            }

        avg_mae = np.mean([r["mae"] for r in fold_results])
        avg_acc = np.mean([r["accuracy"] for r in fold_results])
        avg_f1 = np.mean([r["f1_macro"] for r in fold_results])
        std_mae = np.std([r["mae"] for r in fold_results])

        print(f"    Mean: MAE={avg_mae:.4f}±{std_mae:.4f} Acc={avg_acc:.4f} F1={avg_f1:.4f}")

        return {
            "combo": combo_name,
            "config": feature_config,
            "n_folds": n_folds,
            "n_samples": len(df_pool),
            "avg_mae": round(avg_mae, 4),
            "std_mae": round(std_mae, 4),
            "avg_accuracy": round(avg_acc, 4),
            "avg_f1_macro": round(avg_f1, 4),
            "fold_results": fold_results,
            "total_time_sec": round(total_time, 1),
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        # Restore config
        for key, value in snapshot.items():
            setattr(config, key, value)


def main():
    parser = argparse.ArgumentParser(description="Baseline Feature Ablation Study")
    parser.add_argument("--cv", type=int, default=3, help="Number of CV folds (default: 3)")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0=no limit)")
    args = parser.parse_args()

    # Load data
    df = utils.load_and_standardize_data(config.RAW_DATA_PATH)
    df, _ = utils.clean_review_dataframe(df, min_text_length=config.MIN_TEXT_LENGTH, drop_duplicates=True)

    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=config.RANDOM_STATE).reset_index(drop=True)

    print(f"{'=' * 70}")
    print(f"  BASELINE FEATURE ABLATION STUDY ({args.cv}-Fold CV)")
    print(f"  Samples: {len(df)}")
    print(f"{'=' * 70}")

    # Define feature combinations to test
    COMBOS = [
        # Format: (name, {config_key: value, ...})
        ("word_only", {
            "BASELINE_USE_CHAR_TFIDF": False,
            "BASELINE_USE_EXTRA_FEATURES": False,
            "BASELINE_USE_LSA": False,
        }),
        ("word+extra", {
            "BASELINE_USE_CHAR_TFIDF": False,
            "BASELINE_USE_EXTRA_FEATURES": True,
            "BASELINE_USE_LSA": False,
        }),
        ("word+char", {
            "BASELINE_USE_CHAR_TFIDF": True,
            "BASELINE_USE_EXTRA_FEATURES": False,
            "BASELINE_USE_LSA": False,
        }),
        ("word+char+extra", {
            "BASELINE_USE_CHAR_TFIDF": True,
            "BASELINE_USE_EXTRA_FEATURES": True,
            "BASELINE_USE_LSA": False,
        }),
        ("word+lsa300", {
            "BASELINE_USE_CHAR_TFIDF": False,
            "BASELINE_USE_EXTRA_FEATURES": False,
            "BASELINE_USE_LSA": True,
            "LSA_N_COMPONENTS": 300,
            "TFIDF_MAX_FEATURES": 10000,
        }),
        ("word+lsa300+extra", {
            "BASELINE_USE_CHAR_TFIDF": False,
            "BASELINE_USE_EXTRA_FEATURES": True,
            "BASELINE_USE_LSA": True,
            "LSA_N_COMPONENTS": 300,
            "TFIDF_MAX_FEATURES": 10000,
        }),
        ("word+char+lsa400+extra", {
            "BASELINE_USE_CHAR_TFIDF": True,
            "BASELINE_USE_EXTRA_FEATURES": True,
            "BASELINE_USE_LSA": True,
            "LSA_N_COMPONENTS": 400,
            "TFIDF_MAX_FEATURES": 10000,
        }),
        ("tfidf_5k", {
            "TFIDF_MAX_FEATURES": 5000,
            "BASELINE_USE_CHAR_TFIDF": True,
            "BASELINE_USE_EXTRA_FEATURES": True,
            "BASELINE_USE_LSA": False,
        }),
        ("tfidf_12k", {
            "TFIDF_MAX_FEATURES": 12000,
            "BASELINE_USE_CHAR_TFIDF": True,
            "BASELINE_USE_EXTRA_FEATURES": True,
            "BASELINE_USE_LSA": False,
        }),
        ("char_6k_ngram26", {
            "BASELINE_USE_CHAR_TFIDF": True,
            "BASELINE_CHAR_MAX_FEATURES": 6000,
            "BASELINE_CHAR_NGRAM_RANGE": (2, 6),
            "BASELINE_USE_EXTRA_FEATURES": True,
            "BASELINE_USE_LSA": False,
        }),
    ]

    all_results = []
    for name, feature_config in COMBOS:
        result = run_feature_combination(name, feature_config, df, n_folds=args.cv)
        all_results.append(result)

    # Save results
    log_dir = os.path.join(config.EXPERIMENTS_DIR, "baseline")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "feature_ablation_log.json")

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
    ok_results.sort(key=lambda r: r.get("avg_mae", float("inf")))

    print(f"\n{'=' * 80}")
    print(f"  FEATURE ABLATION SUMMARY ({args.cv}-Fold CV)")
    print(f"{'=' * 80}")
    print(f"  {'#':>2}  {'Features':<25} {'MAE':>7} {'±':>1} {'Std':>6} {'Acc':>7} {'F1-M':>7} {'Time':>7}")
    print(f"  {'--':>2}  {'-' * 25} {'-' * 7} {' ':>1} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 7}")

    for rank, r in enumerate(ok_results, 1):
        tag = " ★" if rank == 1 else ""
        time_str = f"{r.get('total_time_sec', 0):.0f}s"
        print(
            f"  {rank:>2}  {r['combo']:<25} {r['avg_mae']:>7.4f} ± {r['std_mae']:>6.4f} "
            f"{r['avg_accuracy']:>7.4f} {r['avg_f1_macro']:>7.4f} {time_str:>7}{tag}"
        )

    if ok_results:
        best = ok_results[0]
        print(f"\n  🏆 Best features: {best['combo']} (MAE={best['avg_mae']:.4f} ± {best['std_mae']:.4f})")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
