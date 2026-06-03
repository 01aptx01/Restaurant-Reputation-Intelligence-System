# -*- coding: utf-8 -*-
"""Tests for extra feature extraction."""

from __future__ import annotations

import numpy as np

from rris.data.features import compute_extra_features


def test_compute_extra_features_shape() -> None:
    texts = ["ร้านอาหารดี", "ไม่ชอบเลย แย่มาก"]
    feats = compute_extra_features(texts)
    assert feats.shape == (2, 3)
    assert feats.dtype == np.float64


def test_char_len_increases_with_text() -> None:
    short = compute_extra_features(["abc"])
    long = compute_extra_features(["abcdefgh"])
    assert long[0, 0] > short[0, 0]
