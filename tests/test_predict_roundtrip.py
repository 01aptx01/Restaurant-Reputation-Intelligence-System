# -*- coding: utf-8 -*-
"""Smoke round-trip: train baseline in smoke mode and predict."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RRIS_SMOKE") != "1",
    reason="Set RRIS_SMOKE=1 to run integration smoke tests",
)


def test_baseline_predict_shape_and_range(tmp_path, monkeypatch) -> None:
    from rris import config
    from rris.inference.baseline import predict_baseline_with_probs
    from rris.inference.prep import prepare_scoring_for_model
    from rris.training import baseline as train_baseline

    mock_csv = tmp_path / "mock_train.csv"
    rows = []
    for star in range(1, 6):
        for i in range(20):
            rows.append({"text": f"รีวิวดาว{star} ข้อความ{i}", "user_rating": star})
    pd.DataFrame(rows).to_csv(mock_csv, index=False)

    monkeypatch.setattr(config, "RAW_DATA_PATH", str(mock_csv))
    monkeypatch.setattr(config, "WONGNAI_TRAIN_PATH", str(mock_csv))
    monkeypatch.setattr(config, "BASELINE_OVERSAMPLE_LOW_STARS", False)
    monkeypatch.setattr(config, "AUGMENT_ENABLED", False)
    monkeypatch.setattr(config, "BASELINE_MOCK_MIX_FRACTION", 0.0)

    train_baseline.main()

    df = prepare_scoring_for_model(str(mock_csv), "baseline")
    expected, probs = predict_baseline_with_probs(df)
    assert len(expected) == len(df)
    assert probs.shape[0] == len(df)
    assert np.all(expected >= 1.0)
    assert np.all(expected <= 5.0)
