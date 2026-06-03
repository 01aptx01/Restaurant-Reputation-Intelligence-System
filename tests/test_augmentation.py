# -*- coding: utf-8 -*-
"""Tests for augmentation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rris.data.augmentation import (
    augment_minority_classes,
    augment_random_shuffle,
    augment_synonym_replace,
    augment_text,
    build_confusion_summary,
)


def test_augment_synonym_replace_deterministic() -> None:
    rng = np.random.RandomState(42)
    text = "ร้านอาหารอร่อยมาก"
    a = augment_synonym_replace(text, replace_prob=0.0, rng=rng)
    b = augment_synonym_replace(text, replace_prob=0.0, rng=rng)
    assert a == b == text


def test_augment_random_shuffle_keeps_tokens_when_prob_zero() -> None:
    rng = np.random.RandomState(0)
    text = "ร้านอาหารดีมาก"
    assert augment_random_shuffle(text, shuffle_prob=0.0, rng=rng) == text


def test_augment_text_deterministic_with_seed() -> None:
    rng1 = np.random.RandomState(123)
    rng2 = np.random.RandomState(123)
    text = "บริการดีอาหารอร่อย"
    assert augment_text(text, rng=rng1) == augment_text(text, rng=rng2)


def test_augment_minority_classes_noop_when_above_target() -> None:
    df = pd.DataFrame(
        {"text": ["a"] * 10, "user_rating": [1] * 10},
    )
    out = augment_minority_classes(df, target_stars=(1,), target_count=5, random_state=0)
    assert len(out) == len(df)


def test_build_confusion_summary() -> None:
    y_true = np.array([1, 5, 3])
    expected = np.array([5.0, 1.0, 3.2])
    summary = build_confusion_summary(y_true, expected, min_delta=2)
    assert summary["severe_count"] >= 2
    assert "confusion_pairs" in summary
