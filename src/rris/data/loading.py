# -*- coding: utf-8 -*-
"""Load CSV/TSV reviews and clean review DataFrames."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from rris.data.text import RATING_ALIASES, TEXT_ALIASES, extended_normalize_text


def _pick_column(columns: list[str], aliases: tuple[str, ...], kind: str) -> str:
    alias_lower = [a.lower() for a in aliases]
    matches = [c for c in columns if c.lower() in alias_lower]
    if not matches:
        raise ValueError(
            f"Could not find {kind} column. Expected one of {aliases}, got {list(columns)}"
        )
    return matches[0]


def load_and_standardize_data(
    file_path: str, normalize_func=extended_normalize_text
) -> pd.DataFrame:
    if file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, sep="\t", encoding="utf-16", on_bad_lines="skip")
    else:
        df = pd.read_csv(file_path)
    text_col = _pick_column(list(df.columns), TEXT_ALIASES, "text")
    rating_col = _pick_column(list(df.columns), RATING_ALIASES, "rating")

    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=[rating_col])
    standard_df = pd.DataFrame(
        {
            "text": df[text_col].astype(str).map(normalize_func),
            "user_rating": df[rating_col].astype(float).round().astype(int),
        }
    )
    if "place_name" in df.columns:
        standard_df["place_name"] = df["place_name"]
    invalid = ~standard_df["user_rating"].between(1, 5)
    if invalid.any():
        raise ValueError(
            f"user_rating must be 1..5; found {int(invalid.sum())} invalid row(s)"
        )
    return standard_df


def clean_review_dataframe(
    df: pd.DataFrame,
    min_text_length: int,
    drop_duplicates: bool = True,
    duplicate_keep: str = "first",
) -> tuple[pd.DataFrame, dict]:
    stats: dict = {
        "initial_rows": int(len(df)),
        "removed_empty": 0,
        "removed_short": 0,
        "removed_duplicate": 0,
        "final_rows": 0,
    }
    out = df.copy()
    empty_mask = out["text"].str.len() == 0
    stats["removed_empty"] = int(empty_mask.sum())
    out = out.loc[~empty_mask]
    short_mask = out["text"].str.len() < min_text_length
    stats["removed_short"] = int(short_mask.sum())
    out = out.loc[~short_mask]
    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates(subset=["text"], keep=duplicate_keep)
        stats["removed_duplicate"] = before - len(out)
    stats["final_rows"] = int(len(out))
    return out.reset_index(drop=True), stats


def log_cleaning_stats(stats: dict, label: str = "dataset") -> None:
    print(f"Cleaning stats ({label}):")
    print(f"  initial rows:      {stats['initial_rows']}")
    print(f"  removed empty:     {stats['removed_empty']}")
    print(f"  removed short:     {stats['removed_short']}")
    print(f"  removed duplicate: {stats['removed_duplicate']}")
    print(f"  final rows:        {stats['final_rows']}")


def rating_distribution_dict(ratings_1_to_5: np.ndarray) -> dict[int, int]:
    counts = pd.Series(ratings_1_to_5).value_counts().sort_index()
    return {star: int(counts.get(star, 0)) for star in range(1, 6)}


def print_rating_distribution(
    ratings_1_to_5: np.ndarray,
    class_weights: np.ndarray | None = None,
    label: str = "train",
) -> None:
    dist = rating_distribution_dict(ratings_1_to_5)
    print(f"Rating distribution ({label}):")
    for star in range(1, 6):
        print(f"  star {star}: {dist[star]}")
    if class_weights is not None:
        print(f"Class weights (index 0..4): {np.round(class_weights, 4).tolist()}")


def compare_rating_distributions(
    train_ratings: np.ndarray,
    test_ratings: np.ndarray | None = None,
    label_train: str = "train split",
    label_test: str = "test",
) -> None:
    train_dist = rating_distribution_dict(train_ratings)
    n_train = max(sum(train_dist.values()), 1)
    print(f"\n--- Rating distribution: {label_train} (n={n_train}) ---")
    for star in range(1, 6):
        c = train_dist[star]
        print(f"  star {star}: {c} ({100.0 * c / n_train:.1f}%)")
    if test_ratings is None:
        return
    test_dist = rating_distribution_dict(test_ratings)
    n_test = max(sum(test_dist.values()), 1)
    print(f"\n--- Rating distribution: {label_test} (n={n_test}) ---")
    for star in range(1, 6):
        c = test_dist[star]
        print(f"  star {star}: {c} ({100.0 * c / n_test:.1f}%)")
    print("\n--- Train vs test share delta (test% - train%) ---")
    for star in range(1, 6):
        train_pct = 100.0 * train_dist[star] / n_train
        test_pct = 100.0 * test_dist[star] / n_test
        print(f"  star {star}: {test_pct - train_pct:+.1f} pp")


def truncate_text(text: str, max_chars: int) -> str:
    if not max_chars or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def apply_text_truncation(df: pd.DataFrame, max_chars: int) -> pd.DataFrame:
    if not max_chars or max_chars <= 0:
        return df
    out = df.copy()
    out["text"] = out["text"].map(lambda t: truncate_text(t, max_chars))
    return out


def mix_mock_training_data(
    wongnai_df: pd.DataFrame,
    mock_path: str,
    mock_fraction: float,
    *,
    min_text_length: int = 5,
    drop_duplicates: bool = True,
    duplicate_keep: str = "first",
    random_state: int = 42,
) -> pd.DataFrame:
    if mock_fraction <= 0 or mock_path is None or not os.path.isfile(mock_path):
        return wongnai_df.reset_index(drop=True)
    mock_df = load_and_standardize_data(mock_path)
    mock_df, _ = clean_review_dataframe(
        mock_df,
        min_text_length=min_text_length,
        drop_duplicates=drop_duplicates,
        duplicate_keep=duplicate_keep,
    )
    n_mock = max(1, int(len(wongnai_df) * mock_fraction))
    n_mock = min(n_mock, len(mock_df))
    mock_sample = mock_df.sample(n=n_mock, random_state=random_state)
    combined = pd.concat([wongnai_df, mock_sample], ignore_index=True)
    return combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
