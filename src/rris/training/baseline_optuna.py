# -*- coding: utf-8 -*-
"""Optuna hyperparameter search for baseline with full feature pipeline and MAE objective."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC
import xgboost as xgb

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None  # type: ignore[assignment,misc]
    TPESampler = None  # type: ignore[assignment,misc]

from rris import config
from rris import utils
from rris.data.augmentation import apply_train_augmentation
from rris.data.normalize import BASELINE_PREPROCESS_STRATEGY
from rris.evaluation.selection import classifier_val_mae
from rris.inference.common import expected_rating_from_probs
from rris.training.baseline import fit_vectorizer_and_features, save_artifacts


def _fold_mae(
    clf,
    X_val,
    y_val_stars: np.ndarray,
    classifier_name: str,
) -> float:
    if classifier_name == "xgboost" and hasattr(clf, "predict_proba"):
        return classifier_val_mae(clf, X_val, y_val_stars)
    if classifier_name == "svm":
        preds = clf.predict(X_val)
        return float(mean_absolute_error(y_val_stars, preds.astype(np.float64)))
    return classifier_val_mae(clf, X_val, y_val_stars)


def main() -> None:
    if optuna is None:
        print("Optuna is not installed. Please run: pip install optuna", file=sys.stderr)
        sys.exit(1)

    print("--- Advanced Optuna K-Fold Optimization for Baseline (MAE objective) ---")

    df = utils.load_and_standardize_data(config.WONGNAI_TRAIN_PATH)
    df, _ = utils.clean_review_dataframe(
        df,
        min_text_length=config.MIN_TEXT_LENGTH,
        drop_duplicates=True,
    )
    df = utils.apply_text_truncation(df, config.MAX_REVIEW_CHARS)

    train_df, holdout_df = train_test_split(
        df,
        test_size=config.HOLDOUT_FRACTION,
        random_state=config.RANDOM_STATE,
        stratify=df["user_rating"],
    )
    train_df = utils.prepare_baseline_train_df(train_df)
    holdout_df = utils.apply_text_truncation(holdout_df, config.MAX_REVIEW_CHARS)
    y_train_stars = train_df["user_rating"].values.astype(int)

    print(f"Train/CV Size: {len(train_df)}, Holdout Size: {len(holdout_df)}")

    def objective(trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical(
            "classifier", ["xgboost", "logistic", "svm", "random_forest"]
        )

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
                "random_state": config.RANDOM_STATE,
            }
        elif classifier_name == "logistic":
            params = {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
        elif classifier_name == "svm":
            params = {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 10, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }

        skf = StratifiedKFold(
            n_splits=config.BASELINE_KFOLD,
            shuffle=True,
            random_state=config.RANDOM_STATE,
        )
        fold_maes: list[float] = []

        for train_idx, val_idx in skf.split(train_df, y_train_stars):
            fold_train = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val = train_df.iloc[val_idx].reset_index(drop=True)
            _, _, _, X_fold_train, X_fold_val, _ = fit_vectorizer_and_features(
                fold_train, fold_val
            )
            y_fold_train = fold_train["user_rating"].values.astype(int)
            y_fold_val = fold_val["user_rating"].values.astype(int)

            sample_weights = None
            if config.XGB_USE_SAMPLE_WEIGHT:
                class_weights = utils.compute_class_weights(
                    y_fold_train, low_star_boost=config.XGB_LOW_STAR_BOOST
                )
                sample_weights = utils.compute_sample_weights_from_ratings(
                    y_fold_train, class_weights
                )

            if classifier_name == "xgboost":
                clf = xgb.XGBClassifier(**params)
                clf.fit(
                    X_fold_train,
                    y_fold_train - 1,
                    sample_weight=sample_weights,
                    verbose=False,
                )
            elif classifier_name == "logistic":
                clf = LogisticRegression(
                    C=params["C"],
                    class_weight="balanced",
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=config.RANDOM_STATE,
                )
                clf.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            elif classifier_name == "svm":
                clf = LinearSVC(
                    C=params["C"],
                    class_weight="balanced",
                    dual=False,
                    max_iter=1000,
                    random_state=config.RANDOM_STATE,
                )
                clf.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            else:
                clf = RandomForestClassifier(
                    **params,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=config.RANDOM_STATE,
                )
                clf.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)

            fold_maes.append(_fold_mae(clf, X_fold_val, y_fold_val, classifier_name))

        return float(np.mean(fold_maes))

    print(f"\n--- Starting Optuna Optimization ({config.BASELINE_KFOLD}-Fold CV, minimize MAE) ---")
    sampler = TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="baseline_mae_optimization",
    )
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("\n" + "=" * 50)
    print("OPTIMIZATION FINISHED")
    print(f"Best Trial: Val MAE = {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    print("=" * 50)

    print("\n--- Training Best Model on Full CV Set ---")
    best_params = dict(study.best_params)
    best_clf_name = best_params.pop("classifier")

    vectorizer, char_vectorizer, svd, X_train, X_hold, explained = fit_vectorizer_and_features(
        train_df, holdout_df
    )
    y_train = train_df["user_rating"].values.astype(int)
    y_holdout = holdout_df["user_rating"].values.astype(int)

    sample_weights = None
    if config.XGB_USE_SAMPLE_WEIGHT:
        class_weights = utils.compute_class_weights(
            y_train, low_star_boost=config.XGB_LOW_STAR_BOOST
        )
        sample_weights = utils.compute_sample_weights_from_ratings(y_train, class_weights)

    final_clf = None
    if best_clf_name == "xgboost":
        best_params.update(
            {
                "objective": "multi:softprob",
                "num_class": 5,
                "tree_method": "hist",
                "device": config.XGB_DEVICE,
                "random_state": config.RANDOM_STATE,
            }
        )
        final_clf = xgb.XGBClassifier(**best_params)
        final_clf.fit(X_train, y_train - 1, sample_weight=sample_weights)
        hold_preds = expected_rating_from_probs(final_clf.predict_proba(X_hold))
    elif best_clf_name == "logistic":
        final_clf = LogisticRegression(
            C=best_params["C"],
            class_weight="balanced",
            max_iter=1000,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )
        final_clf.fit(X_train, y_train, sample_weight=sample_weights)
        hold_preds = expected_rating_from_probs(final_clf.predict_proba(X_hold))
    elif best_clf_name == "svm":
        svm_base = LinearSVC(
            C=best_params["C"],
            class_weight="balanced",
            dual=False,
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        )
        svm_base.fit(X_train, y_train, sample_weight=sample_weights)
        final_clf = CalibratedClassifierCV(svm_base, cv="prefit")
        final_clf.fit(X_train, y_train)
        hold_preds = expected_rating_from_probs(final_clf.predict_proba(X_hold))
    else:
        final_clf = RandomForestClassifier(
            **best_params,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )
        final_clf.fit(X_train, y_train, sample_weight=sample_weights)
        hold_preds = expected_rating_from_probs(final_clf.predict_proba(X_hold))

    hold_mae = float(mean_absolute_error(y_holdout, hold_preds))
    hold_f1 = float(
        f1_score(
            y_holdout,
            np.clip(np.round(hold_preds), 1, 5).astype(int),
            average="macro",
        )
    )
    print(f"\nHoldout MAE (Unseen Data): {hold_mae:.4f} | F1-Macro: {hold_f1:.4f}")

    model_type_map = {
        "xgboost": "xgboost",
        "logistic": "logistic_regression",
        "svm": "linear_svm",
        "random_forest": "random_forest",
    }
    best_model_type = model_type_map[best_clf_name]

    meta = {
        "best_model_type": best_model_type,
        "best_val_mae": hold_mae,
        "best_f1_macro": hold_f1,
        "preprocess_strategy": BASELINE_PREPROCESS_STRATEGY,
        "optuna_cv_mae": study.best_value,
        "best_params": study.best_params,
        "use_char_tfidf": config.BASELINE_USE_CHAR_TFIDF,
        "use_extra_features": config.BASELINE_USE_EXTRA_FEATURES,
        "use_lsa": config.BASELINE_USE_LSA,
        "use_regression": False,
        "use_3class": False,
        "max_review_chars": config.MAX_REVIEW_CHARS,
        "explained_variance": explained,
    }

    save_model = final_clf
    if best_clf_name == "xgboost":
        save_model = final_clf.get_booster()

    print("\n--- Saving Artifacts ---")
    save_artifacts(vectorizer, char_vectorizer, svd, save_model, meta)
    print(f"Deployment artifacts updated with Optuna champion: {best_clf_name.upper()}!")


if __name__ == "__main__":
    main()
