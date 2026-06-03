# -*- coding: utf-8 -*-
"""Helpers for picking models/classifiers by validation MAE."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error

from rris.inference.common import expected_rating_from_probs


def expected_ratings_from_classifier(clf: Any, X_val, y_val: np.ndarray) -> np.ndarray:
    """Predict star ratings (1-5 float) from a sklearn-like classifier."""
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_val)
        return expected_rating_from_probs(probs)
    raw = clf.predict(X_val)
    if raw.ndim == 1 and np.issubdtype(raw.dtype, np.floating):
        return np.clip(raw.astype(np.float64), 1.0, 5.0)
    return raw.astype(np.float64)


def classifier_val_mae(clf: Any, X_val, y_val: np.ndarray) -> float:
    """Validation MAE for a classifier on star labels 1-5."""
    expected = expected_ratings_from_classifier(clf, X_val, y_val)
    return float(mean_absolute_error(y_val, expected))


def xgb_booster_val_mae(bst, dval, y_val: np.ndarray, use_regression: bool = False) -> float:
    """Validation MAE for an XGBoost Booster on a DMatrix."""
    raw = bst.predict(dval)
    if use_regression:
        expected = np.clip(raw.astype(np.float64) + 1.0, 1.0, 5.0)
    else:
        probs = raw.reshape(-1, 5) if raw.ndim == 1 else raw
        expected = expected_rating_from_probs(probs)
    return float(mean_absolute_error(y_val, expected))


def pick_lowest_mae(scores: dict[str, float]) -> str:
    """Return the key with the lowest MAE."""
    if not scores:
        raise ValueError("No candidates to compare")
    return min(scores, key=scores.get)
