# -*- coding: utf-8 -*-
"""XLM-RoBERTa sequence classification inference."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rris import config
from rris.inference.cache import get as cache_get, set as cache_set
from rris.inference.common import (
    expected_rating_from_probs,
    exit_missing_artifacts,
    gaussian_probs_from_expected,
    head_tail_truncate_text,
)

_CACHE_KEY = "xlmr_session"


def _load_xlmr_session():
    cached = cache_get(_CACHE_KEY)
    if cached is not None:
        return cached

    model_dir = config.XLMR_ARTIFACTS_DIR
    config_json = os.path.join(model_dir, "config.json")
    if not os.path.isdir(model_dir) or not os.path.isfile(config_json):
        exit_missing_artifacts(
            f"XLM-R artifacts not found at {model_dir}\n"
            "Run: python -m rris train xlmr"
        )

    device = torch.device(config.TORCH_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    model.to(device)
    model.eval()
    session = (tokenizer, model, device)
    return cache_set(_CACHE_KEY, session)


def predict_xlmr_with_probs(
    df: pd.DataFrame, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer, model, device = _load_xlmr_session()
    texts = df["text"].astype(str).tolist()
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            truncated = [
                head_tail_truncate_text(t, tokenizer, config.MAX_LENGTH)
                for t in batch_texts
            ]
            encoding = tokenizer(
                truncated,
                add_special_tokens=True,
                max_length=config.MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            if getattr(config, "XLMR_USE_REGRESSION", False):
                expected_vals = logits.squeeze(-1).cpu().numpy()
                probs = gaussian_probs_from_expected(expected_vals)
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    stacked = np.vstack(all_probs)
    return expected_rating_from_probs(stacked), stacked


def predict_xlmr(df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
    expected, _ = predict_xlmr_with_probs(df, batch_size)
    return expected
