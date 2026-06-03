# -*- coding: utf-8 -*-
"""Extended evaluation metrics for star rating and anomaly detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from rris import config


def rounded_stars(expected: np.ndarray) -> np.ndarray:
    return np.clip(np.round(expected), 1, 5).astype(int)


def compute_extended_metrics(
    y_true: np.ndarray,
    expected: np.ndarray,
    *,
    anomaly_threshold: float | None = None,
) -> dict:
    """Compute standard and extended metrics for star-rating evaluation."""
    threshold = (
        config.ANOMALY_THRESHOLD if anomaly_threshold is None else anomaly_threshold
    )
    y_true = y_true.astype(int)
    y_pred = rounded_stars(expected)
    delta = np.abs(y_true.astype(np.float64) - expected.astype(np.float64))
    abs_rounded_error = np.abs(y_true - y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=[1, 2, 3, 4, 5],
        output_dict=True,
        zero_division=0,
    )

    y_flag = delta >= threshold
    y_severe = abs_rounded_error >= 2
    if y_severe.any() or y_flag.any():
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_severe.astype(int),
            y_flag.astype(int),
            average="binary",
            zero_division=0,
        )
        anomaly_precision = float(prec)
        anomaly_recall = float(rec)
        anomaly_f1 = float(f1)
    else:
        anomaly_precision = anomaly_recall = anomaly_f1 = 0.0

    return {
        "n_samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, expected)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, expected))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "off_by_one_accuracy": float(np.mean(abs_rounded_error <= 1)),
        "recall_star_1": float(report.get("1", {}).get("recall", 0.0)),
        "recall_star_2": float(report.get("2", {}).get("recall", 0.0)),
        "anomaly_rate": float(np.mean(y_flag)),
        "anomaly_precision": anomaly_precision,
        "anomaly_recall": anomaly_recall,
        "anomaly_f1": anomaly_f1,
        "severe_error_rate": float(np.mean(y_severe)),
        "per_class_recall": {
            str(k): float(v.get("recall", 0.0))
            for k, v in report.items()
            if k.isdigit()
        },
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[1, 2, 3, 4, 5]
        ).tolist(),
    }
