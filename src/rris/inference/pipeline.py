# -*- coding: utf-8 -*-
"""CLI scoring pipeline: load data, predict, ABSA, anomaly flags."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from rris import config, utils
from rris.inference.baseline import predict_baseline
from rris.inference.common import ABSA_KEYWORDS, get_hex_color
from rris.inference.embedding import predict_embedding
from rris.inference.prep import prepare_scoring_dataframe
from rris.inference.xlmr import predict_xlmr

try:
    from pythainlp.tokenize import sent_tokenize

    HAS_SENT_TOKENIZE = True
except ImportError:
    HAS_SENT_TOKENIZE = False

_TRANSFORMER_MODELS = frozenset(("xlmr",))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score reviews and flag anomalies.")
    parser.add_argument(
        "--model",
        choices=("baseline", "xlmr", "embedding"),
        default="baseline",
        help="Model: baseline, xlmr, or embedding",
    )
    parser.add_argument(
        "--input",
        default=config.RAW_DATA_PATH,
        help="Input CSV path (default: Wongnai train)",
    )
    parser.add_argument(
        "--output",
        default=config.DEFAULT_SCORED_OUTPUT,
        help="Output CSV path for scored reviews",
    )
    parser.add_argument(
        "--skip-absa",
        action="store_true",
        help="Skip aspect-based sentiment (faster; avoids repeated model loads)",
    )
    return parser.parse_args()


def _resolve_predict_fn(model: str):
    return {
        "baseline": predict_baseline,
        "embedding": predict_embedding,
        "xlmr": predict_xlmr,
    }[model]


def main() -> None:
    args = parse_args()
    print(f"--- Running Inference & Integrity Check (model={args.model}) ---")

    norm_fn = (
        utils.xlmr_normalize_text
        if args.model in _TRANSFORMER_MODELS
        else utils.extended_normalize_text
    )
    df = prepare_scoring_dataframe(args.input, normalize_func=norm_fn)
    predict_fn = _resolve_predict_fn(args.model)
    df["ai_expected_rating"] = predict_fn(df)

    if args.skip_absa:
        print("Skipping ABSA (--skip-absa).")
    elif HAS_SENT_TOKENIZE:
        print("--- Running Aspect-Based Sentiment Analysis (ABSA) ---")
        aspect_rows: list[tuple[int, str, str]] = []
        for row_idx, text in enumerate(df["text"]):
            for sent in sent_tokenize(text, engine="whitespace+newline"):
                for aspect, keywords in ABSA_KEYWORDS.items():
                    if any(kw in sent for kw in keywords):
                        aspect_rows.append((row_idx, aspect, sent))

        food_scores: list[float | None] = [None] * len(df)
        service_scores: list[float | None] = [None] * len(df)
        atmos_scores: list[float | None] = [None] * len(df)
        aspect_buckets = {
            "food": food_scores,
            "service": service_scores,
            "atmosphere": atmos_scores,
        }

        if aspect_rows:
            absa_df = pd.DataFrame({"text": [r[2] for r in aspect_rows]})
            absa_ratings = predict_fn(absa_df)
            sums: dict[tuple[int, str], float] = {}
            counts: dict[tuple[int, str], int] = {}
            for (row_idx, aspect, _), rating in zip(aspect_rows, absa_ratings):
                key = (row_idx, aspect)
                sums[key] = sums.get(key, 0.0) + float(rating)
                counts[key] = counts.get(key, 0) + 1
            for (row_idx, aspect), total in sums.items():
                aspect_buckets[aspect][row_idx] = total / counts[(row_idx, aspect)]

        df["aspect_food"] = food_scores
        df["aspect_service"] = service_scores
        df["aspect_atmosphere"] = atmos_scores
        df["aspect_food"] = df["aspect_food"].fillna(df["ai_expected_rating"])
        df["aspect_service"] = df["aspect_service"].fillna(df["ai_expected_rating"])
        df["aspect_atmosphere"] = df["aspect_atmosphere"].fillna(
            df["ai_expected_rating"]
        )
    else:
        print("Warning: PyThaiNLP not found, skipping ABSA.")

    df["ai_hex_color"] = df["ai_expected_rating"].apply(get_hex_color)
    df["delta"] = np.abs(df["user_rating"] - df["ai_expected_rating"])
    df["is_anomaly"] = df["delta"] >= config.ANOMALY_THRESHOLD

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Finished scoring! Check result at '{args.output}'")
