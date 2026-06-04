# -*- coding: utf-8 -*-
# ไฟล์ train_embedding.py: สคริปต์ฝึกสอนโมเดลโดยใช้ Sentence Embedding (เช่น BGE-M3/E5) เป็นฟีเจอร์
# พร้อมระบบฝึกสอนแบบคู่ขนานเพื่อเปรียบเทียบและคัดเลือก Classifier ที่ดีที่สุด (Logistic Regression หรือ XGBoost) อัตโนมัติ

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from rris import config
from rris import utils
from rris.evaluation.selection import classifier_val_mae, pick_lowest_mae
from rris.training.embedding_cache import embedding_cache_path, texts_fingerprint
from rris.training.embedding_finetune import finetune_embedding_model, resolve_embedding_model_path


def main(finetune: bool = False) -> None:
    print("--- Step 1: Loading data ---")
    # ใช้ preprocess แบบเดียวกับ XLM-R ดีที่สุด (aggressive) หรือที่เซ็ตไว้ใน config
    strategy = getattr(config, "XLMR_PREPROCESS_STRATEGY", "aggressive")
    normalize_func = utils.PREPROCESS_REGISTRY.get(strategy, utils.xlmr_normalize_text)
    
    df = utils.load_and_standardize_data(config.RAW_DATA_PATH, normalize_func=normalize_func)
    df, stats = utils.clean_review_dataframe(df, min_text_length=5, drop_duplicates=True)
    utils.log_cleaning_stats(stats, label="embedding_train_pool")

    # Split train/val
    df_train, df_val = train_test_split(
        df,
        test_size=config.HOLDOUT_FRACTION,
        random_state=config.RANDOM_STATE,
        stratify=df["user_rating"],
    )

    y_train = df_train["user_rating"].values
    y_val = df_val["user_rating"].values

    do_finetune = finetune or getattr(config, "EMBEDDING_FINETUNE", False)
    embed_model_name = config.EMBEDDING_MODEL_NAME
    finetuned_path = None
    if do_finetune:
        print("\n--- Step 1.5: Fine-tuning embedding model (opt-in) ---")
        finetuned_path = finetune_embedding_model(
            df_train["text"].tolist(),
            y_train,
        )
        embed_model_name = finetuned_path

    print("\n--- Step 2: Extracting Sentence Embeddings ---")
    texts_hash = texts_fingerprint(
        df_train["text"].tolist() + df_val["text"].tolist()
    )
    cache_path = embedding_cache_path(
        embed_model_name,
        strategy,
        len(df_train),
        len(df_val),
        texts_hash,
    )
    
    X_train = X_val = None
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        cache_data = joblib.load(cache_path)
        if len(cache_data["X_train"]) == len(y_train) and len(cache_data["X_val"]) == len(y_val):
            X_train = cache_data["X_train"]
            X_val = cache_data["X_val"]

    if X_train is None:
        print(f"Loading embedding model: {embed_model_name}")
        device = config.TORCH_DEVICE if config.TORCH_DEVICE != "cpu" else "cpu"
        embed_model = SentenceTransformer(embed_model_name, device=device)
        
        print(f"Encoding {len(df_train)} training texts...")
        t0 = time.time()
        X_train = embed_model.encode(
            df_train["text"].tolist(), 
            batch_size=config.EMBEDDING_BATCH_SIZE, 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        print(f"Done encoding train in {time.time()-t0:.2f}s")
        
        print(f"Encoding {len(df_val)} validation texts...")
        X_val = embed_model.encode(
            df_val["text"].tolist(), 
            batch_size=config.EMBEDDING_BATCH_SIZE, 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        print("Caching embeddings...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump({"X_train": X_train, "X_val": X_val}, cache_path)

    print("\n--- Step 3: Training and Comparing Classifiers ---")
    
    # 3.0 Compute sample weights for models that support it
    class_weights = utils.compute_class_weights(y_train, low_star_boost=config.XGB_LOW_STAR_BOOST)
    sample_w = utils.compute_sample_weights_from_ratings(y_train, class_weights)

    clf_type_config = getattr(config, "EMBEDDING_CLF_TYPE", "auto").lower()
    models_mae = {}

    # --- 1. Logistic Regression ---
    if clf_type_config in ["auto", "lr", "logistic_regression"]:
        print("\n[1/4] Training Logistic Regression...")
        lr_clf = LogisticRegression(
            class_weight='balanced', 
            max_iter=config.EMBEDDING_LR_MAX_ITER,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        lr_clf.fit(X_train, y_train, sample_weight=sample_w)
        lr_val_preds = lr_clf.predict(X_val)
        lr_acc = accuracy_score(y_val, lr_val_preds)
        lr_mae = classifier_val_mae(lr_clf, X_val, y_val)
        lr_f1 = f1_score(y_val, lr_val_preds, average='macro')
        models_mae["logistic_regression"] = (lr_clf, lr_mae, lr_acc, lr_f1)

    # --- 2. XGBoost ---
    if clf_type_config in ["auto", "xgb", "xgboost"]:
        print("\n[2/4] Training XGBoost...")
        import xgboost as xgb
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 5,
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "tree_method": "hist",
            "device": config.XGB_DEVICE,
            "random_state": config.RANDOM_STATE
        }
        if hasattr(config, "XGB_PARAMS") and isinstance(config.XGB_PARAMS, dict):
            xgb_params.update(config.XGB_PARAMS)
            
        xgb_clf = xgb.XGBClassifier(**xgb_params)
        y_train_xgb = y_train - 1
        y_val_xgb = y_val - 1
        
        xgb_clf.fit(
            X_train, y_train_xgb, 
            sample_weight=sample_w,
            eval_set=[(X_train, y_train_xgb), (X_val, y_val_xgb)],
            verbose=False
        )
        xgb_val_preds_raw = xgb_clf.predict(X_val)
        xgb_val_preds = xgb_val_preds_raw + 1
        xgb_acc = accuracy_score(y_val, xgb_val_preds)
        xgb_mae = classifier_val_mae(xgb_clf, X_val, y_val)
        xgb_f1 = f1_score(y_val, xgb_val_preds, average='macro')
        models_mae["xgboost"] = (xgb_clf, xgb_mae, xgb_acc, xgb_f1)
    
    # --- 3. Linear SVM ---
    if clf_type_config in ["auto", "svm", "linear_svm"]:
        print("\n[3/4] Training Linear SVM...")
        svm = LinearSVC(max_iter=1000, class_weight='balanced', dual=False, random_state=config.RANDOM_STATE)
        svm.fit(X_train, y_train, sample_weight=sample_w)
        try:
            from sklearn.frozen import FrozenEstimator
            svm_clf = CalibratedClassifierCV(FrozenEstimator(svm))
            svm_clf.fit(X_val, y_val)
        except ImportError:
            svm_clf = CalibratedClassifierCV(svm, cv=3)
            svm_clf.fit(X_train, y_train, sample_weight=sample_w)
        svm_val_preds = svm_clf.predict(X_val)
        svm_acc = accuracy_score(y_val, svm_val_preds)
        svm_mae = classifier_val_mae(svm_clf, X_val, y_val)
        svm_f1 = f1_score(y_val, svm_val_preds, average='macro')
        models_mae["linear_svm"] = (svm_clf, svm_mae, svm_acc, svm_f1)
    
    # --- 4. Random Forest ---
    if clf_type_config in ["auto", "rf", "random_forest"]:
        print("\n[4/4] Training Random Forest...")
        rf_clf = RandomForestClassifier(
            n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=config.RANDOM_STATE
        )
        rf_clf.fit(X_train, y_train, sample_weight=sample_w)
        rf_val_preds = rf_clf.predict(X_val)
        rf_acc = accuracy_score(y_val, rf_val_preds)
        rf_mae = classifier_val_mae(rf_clf, X_val, y_val)
        rf_f1 = f1_score(y_val, rf_val_preds, average='macro')
        models_mae["random_forest"] = (rf_clf, rf_mae, rf_acc, rf_f1)

    # --- 5. Compare and Select ---
    print("\n" + "="*65)
    print("--- Classifier Comparative Analysis ---")
    print(f"{'Classifier':<25} | {'Val Acc':<10} | {'Val MAE':<10} | {'Val F1-Macro':<12}")
    print("-" * 65)
    for name, (_, mae, acc, f1) in models_mae.items():
        print(f"{name:<25} | {acc:<10.4f} | {mae:<10.4f} | {f1:<12.4f}")
    print("-" * 65)
    best_clf_type = pick_lowest_mae({k: v[1] for k, v in models_mae.items()})
    best_clf, best_mae, best_val_acc, best_f1 = models_mae[best_clf_type]

    print(f"\nWINNER: {best_clf_type.upper()} (Val MAE: {best_mae:.4f}, F1-Macro: {best_f1:.4f})")
    print("="*65 + "\n")

    print("--- Step 4: Saving artifacts ---")
    os.makedirs(config.EMBEDDING_ARTIFACTS_DIR, exist_ok=True)
    model_path = os.path.join(config.EMBEDDING_ARTIFACTS_DIR, "clf_model.joblib")
    joblib.dump(best_clf, model_path)
    
    meta = {
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "finetuned_model_path": finetuned_path,
        "classifier": best_clf_type,
        "preprocess_strategy": strategy,
        "cache_key": os.path.basename(cache_path),
        "val_mae": best_mae,
        "val_accuracy": best_val_acc,
        "val_f1_macro": best_f1,
        "finetune": do_finetune,
        "finetune_mode": config.EMBEDDING_FINETUNE_MODE if do_finetune else None,
    }
    meta_path = os.path.join(config.EMBEDDING_ARTIFACTS_DIR, "embedding_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Saved {best_clf_type.upper()} classifier to {model_path}")
    print("Done embedding training!")

if __name__ == "__main__":
    import sys
    main(finetune="--finetune" in sys.argv)