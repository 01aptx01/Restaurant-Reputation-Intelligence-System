# -*- coding: utf-8 -*-
"""Baseline (TF-IDF + XGBoost / sklearn) inference."""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack

from rris import config, utils
from rris.inference.common import expected_rating_from_probs, exit_missing_artifacts


def load_baseline_meta() -> dict:
    if os.path.isfile(config.BASELINE_META_PATH):
        with open(config.BASELINE_META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "use_lsa": config.BASELINE_USE_LSA,
        "use_extra_features": config.BASELINE_USE_EXTRA_FEATURES,
    }


def _truncate_for_inference(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    max_chars = meta.get("max_review_chars", config.MAX_REVIEW_CHARS)
    return utils.apply_text_truncation(df, max_chars)


def transform_baseline_features(
    df: pd.DataFrame,
    vectorizer,
    svd,
    meta: dict,
    char_vectorizer=None,
) -> np.ndarray:
    df_model = _truncate_for_inference(df, meta)
    X_vec = vectorizer.transform(df_model["text"])

    if meta.get("use_char_tfidf", config.BASELINE_USE_CHAR_TFIDF) and char_vectorizer:
        X_char = char_vectorizer.transform(df_model["text"])
        X_vec = hstack([X_vec, X_char], format="csr")

    use_lsa = meta.get("use_lsa", config.BASELINE_USE_LSA)
    use_extra = meta.get("use_extra_features", config.BASELINE_USE_EXTRA_FEATURES)

    if use_lsa and svd is not None:
        main = svd.transform(X_vec)
        if use_extra:
            extra = utils.compute_extra_features(df_model["text"])
            return np.hstack([main, extra])
        return main

    if use_extra:
        extra = utils.compute_extra_features(df_model["text"])
        return hstack([X_vec, csr_matrix(extra)], format="csr")
    return X_vec


def _predict_baseline_raw(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    vectorizer_path = config.TFIDF_VECTORIZER_PATH
    model_path = config.XGB_MODEL_PATH
    lsa_path = config.LSA_TRANSFORMER_PATH

    if not os.path.isfile(vectorizer_path):
        exit_missing_artifacts(
            "Baseline artifacts not found. Run: python -m rris train baseline"
        )

    meta = load_baseline_meta()
    use_lsa = meta.get("use_lsa", config.BASELINE_USE_LSA)
    best_model_type = meta.get("best_model_type", "xgboost")

    if use_lsa and not os.path.isfile(lsa_path):
        exit_missing_artifacts(
            f"LSA transformer not found at {lsa_path}\n"
            "Re-run: python -m rris train baseline"
        )

    vectorizer = joblib.load(vectorizer_path)
    char_vectorizer = None
    if meta.get("use_char_tfidf", config.BASELINE_USE_CHAR_TFIDF):
        char_path = config.CHAR_TFIDF_VECTORIZER_PATH
        if os.path.isfile(char_path):
            char_vectorizer = joblib.load(char_path)
    svd = joblib.load(lsa_path) if use_lsa else None

    X_features = transform_baseline_features(
        df, vectorizer, svd, meta, char_vectorizer=char_vectorizer
    )

    if best_model_type == "xgboost":
        bst = xgb.Booster()
        bst.load_model(model_path)
        raw = bst.predict(xgb.DMatrix(X_features))
    else:
        sklearn_model_path = os.path.join(config.BASELINE_ARTIFACTS_DIR, "sklearn_model.joblib")
        raw = joblib.load(sklearn_model_path).predict_proba(X_features)

    if meta.get("use_regression", config.BASELINE_USE_REGRESSION):
        pred = np.clip(raw + 1.0, 1.0, 5.0)
        return pred.astype(np.float64), None

    n_class = 3 if meta.get("use_3class", config.BASELINE_USE_3CLASS) else 5
    probs = raw.reshape(-1, n_class) if raw.ndim == 1 else raw
    if n_class == 3:
        class_idx = np.argmax(probs, axis=1)
        return utils.class3_to_expected_star(class_idx), probs
    return expected_rating_from_probs(probs), probs


def predict_baseline(df: pd.DataFrame) -> np.ndarray:
    expected, _ = _predict_baseline_raw(df)
    return expected


def predict_baseline_with_probs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    return _predict_baseline_raw(df)
