"""Build web_app/scored_reviews.json cache for the dashboard (inference + anomaly flags)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rris import config, utils
from rris.evaluation.runner import resolve_best_model
from rris.inference.baseline import predict_baseline_with_probs
from rris.inference.common import get_hex_color
from rris.inference.embedding import predict_embedding_with_probs
from rris.inference.prep import prepare_scoring_dataframe
from rris.inference.xlmr import predict_xlmr_with_probs

ROOT = Path(__file__).resolve().parent.parent
WEB_APP_DIR = ROOT / "web_app"
DEFAULT_OUTPUT = WEB_APP_DIR / "scored_reviews.json"

_SCORING_MODELS = ("baseline", "embedding", "xlmr")

_PREDICT_WITH_PROBS = {
    "baseline": predict_baseline_with_probs,
    "embedding": predict_embedding_with_probs,
    "xlmr": predict_xlmr_with_probs,
}


def resolve_scoring_model(requested: str) -> str:
    if requested == "auto":
        return resolve_best_model()
    if requested in _PREDICT_WITH_PROBS:
        return requested
    raise ValueError(f"Unknown model: {requested}")


def main():
    print("Python data initializer running...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("auto", *_SCORING_MODELS),
        default=config.WEB_MODEL_DEFAULT,
        help="Model for dashboard prediction (auto = best from eval report)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to scored_reviews.json (default: web_app/scored_reviews.json)",
    )
    args = parser.parse_args()

    model = resolve_scoring_model(args.model)
    if args.model == "auto":
        print(f"Auto-selected model: {model}")

    input_file = ROOT / "data" / "70k" / "70k.tsv"
    if not input_file.exists():
        input_file = Path(config.TEST_PATH)

    if model == "xlmr":
        norm_fn = utils.xlmr_normalize_text
    elif model == "embedding":
        strategy = getattr(config, "XLMR_PREPROCESS_STRATEGY", "aggressive")
        norm_fn = utils.PREPROCESS_REGISTRY.get(strategy, utils.xlmr_normalize_text)
    else:
        norm_fn = utils.extended_normalize_text

    df = prepare_scoring_dataframe(str(input_file), normalize_func=norm_fn)
    if "place_name" not in df.columns:
        n_places = min(8, max(1, len(df) // 5))
        place_names = [f"ร้านที่ {i + 1}" for i in range(n_places)]
        df["place_name"] = [place_names[i % n_places] for i in range(len(df))]
    df = df.groupby("place_name").head(5)
    place_names = df["place_name"].dropna().unique()
    df = df[df["place_name"].isin(place_names)].copy()

    print(f"Predicting with model: {model}")
    predict_fn = _PREDICT_WITH_PROBS[model]
    expected, probs = predict_fn(df)

    df["ai_expected_rating"] = expected.astype(float)
    df["ai_hex_color"] = df["ai_expected_rating"].apply(get_hex_color)
    df["delta"] = np.abs(df["user_rating"] - df["ai_expected_rating"]).astype(float)
    df["is_anomaly"] = (df["delta"] >= config.ANOMALY_THRESHOLD).astype(bool)

    review_list = df.to_dict(orient="records")

    if not any(r["is_anomaly"] for r in review_list):
        review_list[0]["user_rating"] = 5
        review_list[0]["ai_expected_rating"] = 1.2
        review_list[0]["delta"] = 3.8
        review_list[0]["is_anomaly"] = True
        review_list[0]["text"] = (
            "(FLAGGED ANOMALY TEST) บริการแย่มาก อาหารเค็มและเย็นชืด "
            "สกปรก ไม่แนะนำให้ไปกินเลย เสียดายเงินที่สุด"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(review_list, f, indent=2, ensure_ascii=False)
    print(f"Scoring complete! Saved to {output_path}")


if __name__ == "__main__":
    main()
