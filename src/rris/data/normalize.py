# -*- coding: utf-8 -*-
"""Resolve text normalization functions for train and inference."""

from __future__ import annotations

import json
import os
from collections.abc import Callable

from rris import config
from rris.data.text import (
    PREPROCESS_REGISTRY,
    extended_normalize_text,
    xlmr_normalize_text,
)

BASELINE_PREPROCESS_STRATEGY = "extended"


def strategy_to_normalize_func(strategy: str) -> Callable[[str], str]:
    """Map a strategy name to a normalize callable."""
    if strategy == BASELINE_PREPROCESS_STRATEGY:
        return extended_normalize_text
    return PREPROCESS_REGISTRY.get(strategy, xlmr_normalize_text)


def load_model_meta(model: str) -> dict:
    """Load artifact metadata for a model key, or empty dict."""
    if model == "baseline":
        path = config.BASELINE_META_PATH
    elif model == "xlmr":
        path = config.XLMR_META_PATH
    elif model == "embedding":
        path = os.path.join(config.EMBEDDING_ARTIFACTS_DIR, "embedding_meta.json")
    else:
        return {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_preprocess_strategy(model: str, meta: dict | None = None) -> str:
    """Return preprocess strategy name for a model."""
    meta = meta if meta is not None else load_model_meta(model)
    if model == "baseline":
        return meta.get("preprocess_strategy", BASELINE_PREPROCESS_STRATEGY)
    if model in ("xlmr", "embedding"):
        return meta.get(
            "preprocess_strategy",
            getattr(config, "XLMR_PREPROCESS_STRATEGY", "aggressive"),
        )
    return BASELINE_PREPROCESS_STRATEGY


def resolve_normalize_func(
    model: str,
    meta: dict | None = None,
) -> Callable[[str], str]:
    """Return the normalize function that matches train-time preprocessing."""
    strategy = resolve_preprocess_strategy(model, meta)
    return strategy_to_normalize_func(strategy)
