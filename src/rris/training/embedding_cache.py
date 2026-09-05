# -*- coding: utf-8 -*-
"""Hashed embedding cache paths."""

from __future__ import annotations

import hashlib
import os

from rris import config


def texts_fingerprint(texts: list[str]) -> str:
    joined = "\n".join(texts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def embedding_cache_path(
    model_name: str,
    preprocess_strategy: str,
    n_train: int,
    n_val: int,
    texts_hash: str,
) -> str:
    key = hashlib.sha256(
        f"{model_name}|{preprocess_strategy}|{n_train}|{n_val}|{texts_hash}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return os.path.join(config.DATA_DIR, f"embedding_cache_{key}.joblib")
