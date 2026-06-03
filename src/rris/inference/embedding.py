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
from rris.training.embedding_finetune import resolve_embedding_model_path

_CACHE_KEY = "embedding_session"


def _load_embedding_meta() -> dict:
    meta_path = os.path.join(config.EMBEDDING_ARTIFACTS_DIR, "embedding_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_embedding_session():
    cached = cache_get(_CACHE_KEY)
    if cached is not None:
        return cached

    from sentence_transformers import SentenceTransformer

    meta = _load_embedding_meta()
    model_path = os.path.join(config.EMBEDDING_ARTIFACTS_DIR, "clf_model.joblib")

    if not meta or not os.path.isfile(model_path):
        exit_missing_artifacts(
            "Embedding artifacts not found. Run: python -m rris train embedding"
        )

    clf = joblib.load(model_path)
    device = config.TORCH_DEVICE if config.TORCH_DEVICE != "cpu" else "cpu"
    embed_name = resolve_embedding_model_path(meta)
    embed_model = SentenceTransformer(embed_name, device=device)
    return cache_set(_CACHE_KEY, (embed_model, clf, meta))


def predict_embedding_with_probs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    embed_model, clf, _meta = _load_embedding_session()
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
