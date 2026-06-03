# -*- coding: utf-8 -*-
# ไฟล์ train_baseline_optuna.py: สคริปต์ขั้นสูงสำหรับการค้นหาพารามิเตอร์ที่ดีที่สุด (Hyperparameter Optimization)
# ทำงานด้วย Optuna แบบ Bayesian Search ผสานกับ Stratified K-Fold Cross Validation (5 พับ)
# ครอบคลุมการแข่งขันระหว่าง XGBoost, Logistic Regression, Linear SVM และ Random Forest

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.sparse import hstack
import xgboost as xgb

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None  # type: ignore[assignment,misc]
    TPESampler = None  # type: ignore[assignment,misc]

from rris import config
from rris import utils

def main():
    if optuna is None:
        print("Optuna is not installed. Please run: pip install optuna", file=sys.stderr)
        sys.exit(1)

    print("--- 🌟 Advanced Optuna K-Fold Optimization for Baseline ---")

    # 1. โหลดข้อมูล
    print("Loading data...")
    df = utils.load_and_standardize_data(config.WONGNAI_TRAIN_PATH)
    df, _ = utils.clean_review_dataframe(df, min_text_length=config.MIN_TEXT_LENGTH, drop_duplicates=True)
    
    # ตัดความยาวตัวอักษรให้ตรงกับตอน Inference เพื่อป้องกัน Train-Test Mismatch
    df = utils.apply_text_truncation(df, config.MAX_REVIEW_CHARS)
    
    # 2. แบ่ง Holdout Set ออก 20% สำหรับทับสอบรอบสุดท้ายแบบไม่เคยเห็นหน้ากันมาก่อน
    train_df, holdout_df = train_test_split(
        df,
        test_size=config.HOLDOUT_FRACTION,
        random_state=config.RANDOM_STATE,
        stratify=df["user_rating"],
    )
    
    print(f"Train/CV Size: {len(train_df)}, Holdout Size: {len(holdout_df)}")
    
    # 3. เตรียม TF-IDF (ฟิตเฉพาะบน Train/CV เพื่อป้องกัน Data Leakage ไป Holdout)
    print("Extracting TF-IDF Features...")
    word_vec = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
    )
    X_train_word = word_vec.fit_transform(train_df["text"])
    
    char_vec = None
    X_train_char = None
    if config.BASELINE_USE_CHAR_TFIDF:
        char_vec = TfidfVectorizer(
            analyzer="char",
            max_features=config.BASELINE_CHAR_MAX_FEATURES,
            ngram_range=config.BASELINE_CHAR_NGRAM_RANGE,
        )
        X_train_char = char_vec.fit_transform(train_df["text"])
        X_train_features = hstack([X_train_word, X_train_char], format="csr")
    else:
        X_train_features = X_train_word

    y_train = train_df["user_rating"].values
    
    # 4. ฟังก์ชันเป้าหมายของ Optuna (Objective Function)
    def objective(trial):
        # ให้ Optuna เลือกอัลกอริทึม
        classifier_name = trial.suggest_categorical("classifier", ["xgboost", "logistic", "svm", "random_forest"])
        
        # ค้นหาพารามิเตอร์ที่เหมาะสมกับแต่ละอัลกอริทึม
        if classifier_name == "xgboost":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "objective": "multi:softprob",
                "num_class": 5,
                "tree_method": "hist",
                "device": config.XGB_DEVICE,
                "random_state": config.RANDOM_STATE
            }
            clf = xgb.XGBClassifier(**params)
            
        elif classifier_name == "logistic":
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            clf = LogisticRegression(C=C, class_weight='balanced', max_iter=1000, n_jobs=-1, random_state=config.RANDOM_STATE)
            
        elif classifier_name == "svm":
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            svm_base = LinearSVC(C=C, class_weight='balanced', dual=False, max_iter=1000, random_state=config.RANDOM_STATE)
            # ไม่ใช้ CalibratedClassifierCV ในระหว่าง CV เพื่อความรวดเร็ว
            clf = svm_base
            
        elif classifier_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 10, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
            }
            clf = RandomForestClassifier(**params, class_weight='balanced', n_jobs=-1, random_state=config.RANDOM_STATE)

        # 5. ประเมินผลด้วย 5-Fold Stratified K-Fold
        skf = StratifiedKFold(n_splits=config.BASELINE_KFOLD, shuffle=True, random_state=config.RANDOM_STATE)
        fold_f1_scores = []
        
        for train_idx, val_idx in skf.split(X_train_features, y_train):
            X_fold_train, X_fold_val = X_train_features[train_idx], X_train_features[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # การถ่วงน้ำหนัก (ถ้าตั้งไว้)
            sample_weights = None
            if config.XGB_USE_SAMPLE_WEIGHT:
                class_weights = utils.compute_class_weights(y_fold_train, low_star_boost=config.XGB_LOW_STAR_BOOST)
                sample_weights = utils.compute_sample_weights_from_ratings(y_fold_train, class_weights)
            
            # ถ้าเป็น XGBoost จะต้องปรับ y ให้เริ่มจาก 0-4
            if classifier_name == "xgboost":
                clf.fit(X_fold_train, y_fold_train - 1, sample_weight=sample_weights, verbose=False)
                preds = clf.predict(X_fold_val) + 1
            else:
                clf.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
                preds = clf.predict(X_fold_val)
                
            fold_f1 = f1_score(y_fold_val, preds, average='macro')
            fold_f1_scores.append(fold_f1)
            
        return np.mean(fold_f1_scores)

    # 6. เริ่มกระบวนการ Optuna
    print(f"\n--- Starting Optuna Optimization ({config.BASELINE_KFOLD}-Fold CV) ---")
    sampler = TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="baseline_optimization")
    
    # รัน 30 Trials (สามารถเพิ่มได้ถ้ามีเวลา)
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
    print("\n" + "="*50)
    print("🏆 OPTIMIZATION FINISHED")
    print(f"Best Trial: F1-Macro = {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    print("="*50)
    
    # 7. เทรนโมเดลที่ดีที่สุดด้วยข้อมูล Train ทั้งหมดเพื่อทดสอบกับ Holdout Set
    print("\n--- Training Best Model on Full CV Set ---")
    best_params = study.best_params
    best_clf_name = best_params.pop("classifier")
    
    sample_weights = None
    if config.XGB_USE_SAMPLE_WEIGHT:
        class_weights = utils.compute_class_weights(y_train, low_star_boost=config.XGB_LOW_STAR_BOOST)
        sample_weights = utils.compute_sample_weights_from_ratings(y_train, class_weights)
        
    final_clf = None
    if best_clf_name == "xgboost":
        best_params.update({"objective": "multi:softprob", "num_class": 5, "tree_method": "hist", "device": config.XGB_DEVICE, "random_state": config.RANDOM_STATE})
        final_clf = xgb.XGBClassifier(**best_params)
        final_clf.fit(X_train_features, y_train - 1, sample_weight=sample_weights)
    elif best_clf_name == "logistic":
        final_clf = LogisticRegression(C=best_params["C"], class_weight='balanced', max_iter=1000, n_jobs=-1, random_state=config.RANDOM_STATE)
        final_clf.fit(X_train_features, y_train, sample_weight=sample_weights)
    elif best_clf_name == "svm":
        svm_base = LinearSVC(C=best_params["C"], class_weight='balanced', dual=False, max_iter=1000, random_state=config.RANDOM_STATE)
        svm_base.fit(X_train_features, y_train, sample_weight=sample_weights)
        final_clf = CalibratedClassifierCV(svm_base, cv='prefit')
        final_clf.fit(X_train_features, y_train)
    elif best_clf_name == "random_forest":
        final_clf = RandomForestClassifier(**best_params, class_weight='balanced', n_jobs=-1, random_state=config.RANDOM_STATE)
        final_clf.fit(X_train_features, y_train, sample_weight=sample_weights)
        
    # 8. ประเมินผลบน Holdout Set สุดท้าย (เพื่อความโปร่งใส)
    X_hold_word = word_vec.transform(holdout_df["text"])
    if X_train_char is not None:
        X_hold_char = char_vec.transform(holdout_df["text"])
        X_hold_features = hstack([X_hold_word, X_hold_char], format="csr")
    else:
        X_hold_features = X_hold_word
        
    y_holdout = holdout_df["user_rating"].values
    
    if best_clf_name == "xgboost":
        final_preds = final_clf.predict(X_hold_features) + 1
    else:
        final_preds = final_clf.predict(X_hold_features)
        
    final_f1 = f1_score(y_holdout, final_preds, average='macro')
    print(f"\n🎯 Holdout Set F1-Macro (Unseen Data): {final_f1:.4f}")
    
    # 9. บันทึก Artifacts แทนที่ Baseline เดิม
    print("\n--- Saving Artifacts ---")
    os.makedirs(config.BASELINE_ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(word_vec, config.TFIDF_VECTORIZER_PATH)
    if char_vec:
        joblib.dump(char_vec, config.CHAR_TFIDF_VECTORIZER_PATH)
        
    if best_clf_name == "xgboost":
        # ดึง Booster ดิบออกมาเซฟ
        final_clf.get_booster().save_model(config.XGB_MODEL_PATH)
    else:
        sklearn_model_path = os.path.join(config.BASELINE_ARTIFACTS_DIR, "sklearn_model.joblib")
        joblib.dump(final_clf, sklearn_model_path)
        
    # บันทึก Metadata
    meta = {
        "best_model_type": "xgboost" if best_clf_name == "xgboost" else best_clf_name.replace("logistic", "logistic_regression").replace("svm", "linear_svm"),
        "best_f1_macro": final_f1,
        "optuna_cv_f1": study.best_value,
        "best_params": study.best_params,
        "use_char_tfidf": config.BASELINE_USE_CHAR_TFIDF,
        "use_extra_features": False,
        "use_lsa": False,
        "use_regression": False,
        "use_3class": False,
        "max_review_chars": config.MAX_REVIEW_CHARS
    }
    with open(config.BASELINE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Deployment artifacts updated with the Optuna Champion: {best_clf_name.upper()}!")

if __name__ == "__main__":
    main()