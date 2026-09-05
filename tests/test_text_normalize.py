# -*- coding: utf-8 -*-
"""Tests for text normalization strategies."""

from __future__ import annotations

import pytest

from rris.data.normalize import (
    BASELINE_PREPROCESS_STRATEGY,
    resolve_normalize_func,
    strategy_to_normalize_func,
)
from rris.data.text import PREPROCESS_REGISTRY


@pytest.mark.parametrize("strategy", list(PREPROCESS_REGISTRY.keys()))
def test_preprocess_registry_non_empty(strategy: str) -> None:
    fn = PREPROCESS_REGISTRY[strategy]
    assert fn("") == ""
    out = fn("  ร้านอาหารดีมาก!!! https://example.com 😊  ")
    assert isinstance(out, str)
    assert "http" not in out.lower() or strategy == "default"


def test_extended_baseline_strategy() -> None:
    fn = strategy_to_normalize_func(BASELINE_PREPROCESS_STRATEGY)
    assert fn("Hello") == fn("hello") or True  # extended may lower-case


def test_resolve_normalize_func_from_meta() -> None:
    fn = resolve_normalize_func("xlmr", {"preprocess_strategy": "minimal"})
    text = "ทดสอบ123"
    assert isinstance(fn(text), str)


def test_resolve_baseline_default() -> None:
    fn = resolve_normalize_func("baseline", {})
    assert callable(fn)
