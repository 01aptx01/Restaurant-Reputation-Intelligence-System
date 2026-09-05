# -*- coding: utf-8 -*-
"""Tests for resolve_normalize_func."""

from __future__ import annotations

from rris.data.normalize import load_model_meta, resolve_normalize_func, resolve_preprocess_strategy


def test_resolve_preprocess_strategy_defaults() -> None:
    assert resolve_preprocess_strategy("baseline", {}) == "extended"
    assert resolve_preprocess_strategy("xlmr", {}) in (
        "aggressive",
        "default",
        "minimal",
        "keep_digits",
        "emoji_tag",
        "segment",
    )


def test_load_model_meta_missing_returns_empty() -> None:
    assert load_model_meta("nonexistent_model_xyz") == {}


def test_resolve_normalize_callable() -> None:
    fn = resolve_normalize_func("embedding", {"preprocess_strategy": "segment"})
    assert fn("ข้อความทดสอบ") == fn("ข้อความทดสอบ")
