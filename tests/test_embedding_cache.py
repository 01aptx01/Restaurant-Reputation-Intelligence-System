# -*- coding: utf-8 -*-
"""Tests for hashed embedding cache paths."""

from __future__ import annotations

from rris.training.embedding_cache import embedding_cache_path, texts_fingerprint


def test_texts_fingerprint_changes_with_content() -> None:
    a = texts_fingerprint(["hello", "world"])
    b = texts_fingerprint(["hello", "world!"])
    assert a != b


def test_cache_path_changes_with_model_or_strategy() -> None:
    h = texts_fingerprint(["x"])
    p1 = embedding_cache_path("model-a", "aggressive", 100, 20, h)
    p2 = embedding_cache_path("model-b", "aggressive", 100, 20, h)
    p3 = embedding_cache_path("model-a", "minimal", 100, 20, h)
    assert p1 != p2
    assert p1 != p3
    assert "embedding_cache_" in p1


def test_cache_path_changes_with_counts() -> None:
    h = texts_fingerprint(["x"])
    p1 = embedding_cache_path("m", "s", 100, 20, h)
    p2 = embedding_cache_path("m", "s", 101, 20, h)
    assert p1 != p2
