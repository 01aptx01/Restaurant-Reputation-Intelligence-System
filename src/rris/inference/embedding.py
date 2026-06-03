# -*- coding: utf-8 -*-
"""Sentence embedding + classifier inference."""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from rris import config
from rris.inference.cache import get as cache_get, set as cache_set
from rris.inference.common import expected_rating_from_probs, exit_missing_artifacts

_CACHE_KEY = "embedding_session"


def _load_embedding_session():
    cached = cache_get(_CACHE_KEY)
    if cached is not None:
        return cached

    from sentence_transformers import SentenceTransformer

    meta_path = os.path.join(config.EMBEDDING_ARTIFACTS_DIR, "embedding_meta.json")
    model_path = os.path.join(config.EMBEDDING_ARTIFACTS_DIR, "clf_model.joblib")

    if not os.path.isfile(meta_path) or not os.path.isfile(model_path):
        exit_missing_artifacts(
            "Embedding artifacts not found. Run: python -m rris train embedding"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    clf = joblib.load(model_path)
    device = config.TORCH_DEVICE if config.TORCH_DEVICE != "cpu" else "cpu"
    embed_model = SentenceTransformer(meta["embedding_model"], device=device)
    return cache_set(_CACHE_KEY, (embed_model, clf))


def predict_embedding_with_probs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    embed_model, clf = _load_embedding_session()
    X = embed_model.encode(
        df["text"].tolist(),
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=len(df) > 64,
        normalize_embeddings=True,
    )
    probs = clf.predict_proba(X)
    return expected_rating_from_probs(probs), probs


def predict_embedding(df: pd.DataFrame) -> np.ndarray:
    expected, _ = predict_embedding_with_probs(df)
    return expected
