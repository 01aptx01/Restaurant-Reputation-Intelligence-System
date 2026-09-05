# -*- coding: utf-8 -*-
"""Tests for extended evaluation metrics."""

from __future__ import annotations

import numpy as np

from rris.evaluation.metrics import compute_extended_metrics


def test_perfect_predictions() -> None:
    y = np.array([1, 2, 3, 4, 5])
    expected = y.astype(float)
    m = compute_extended_metrics(y, expected)
    assert m["mae"] == 0.0
    assert m["accuracy"] == 1.0
    assert m["off_by_one_accuracy"] == 1.0
    assert m["anomaly_rate"] == 0.0


def test_off_by_one_accuracy() -> None:
    y = np.array([3, 3, 3])
    expected = np.array([4.0, 2.0, 3.0])
    m = compute_extended_metrics(y, expected)
    assert m["off_by_one_accuracy"] == 1.0
    assert m["mae"] > 0.0


def test_anomaly_metrics_present() -> None:
    y = np.array([1, 5])
    expected = np.array([5.0, 1.0])
    m = compute_extended_metrics(y, expected, anomaly_threshold=2.0)
    assert "anomaly_precision" in m
    assert "anomaly_recall" in m
    assert "recall_star_1" in m
