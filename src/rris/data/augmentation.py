# -*- coding: utf-8 -*-
"""Unified NLP augmentation for training pipelines."""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize

from rris import config


def _get_thai_synonyms(word: str) -> list[str]:
    try:
        from pythainlp.corpus import wordnet

        synsets = wordnet.synsets(word, lang="tha")
        synonyms = set()
        for syn in synsets:
            for lemma in syn.lemma_names("tha"):
                if lemma != word and lemma.strip():
                    synonyms.add(lemma)
        return list(synonyms)
    except Exception:
        return []


def augment_synonym_replace(
    text: str,
    replace_prob: float = 0.3,
    rng: np.random.RandomState | None = None,
) -> str:
    if rng is None:
        rng = np.random.RandomState()
    tokens = word_tokenize(text, engine="newmm")
    if len(tokens) < 2:
        return text
    new_tokens = []
    for token in tokens:
        if rng.random() < replace_prob and len(token) > 1:
            synonyms = _get_thai_synonyms(token)
            new_tokens.append(rng.choice(synonyms) if synonyms else token)
        else:
            new_tokens.append(token)
    return "".join(new_tokens)


def augment_random_shuffle(
    text: str,
    shuffle_prob: float = 0.2,
    rng: np.random.RandomState | None = None,
) -> str:
    if rng is None:
        rng = np.random.RandomState()
    if rng.random() > shuffle_prob:
        return text
    tokens = word_tokenize(text, engine="newmm")
    if len(tokens) < 3:
        return text
    rng.shuffle(tokens)
    return "".join(tokens)


def augment_text(
    text: str,
    synonym_prob: float = 0.3,
    shuffle_prob: float = 0.2,
    rng: np.random.RandomState | None = None,
) -> str:
    if rng is None:
        rng = np.random.RandomState()
    technique = rng.choice(["synonym", "shuffle", "both"])
    if technique == "synonym":
        return augment_synonym_replace(text, replace_prob=synonym_prob, rng=rng)
    if technique == "shuffle":
        return augment_random_shuffle(text, shuffle_prob=1.0, rng=rng)
    augmented = augment_synonym_replace(text, replace_prob=synonym_prob, rng=rng)
    return augment_random_shuffle(augmented, shuffle_prob=1.0, rng=rng)


def augment_minority_classes(
    df: pd.DataFrame,
    target_stars: tuple[int, ...] = (1, 2, 3),
    target_count: int = 800,
    synonym_prob: float = 0.3,
    shuffle_prob: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    augmented_rows = []
    for star in target_stars:
        star_df = df[df["user_rating"] == star]
        current_count = len(star_df)
        if current_count >= target_count:
            print(f"  {star} stars: {current_count} rows (>= {target_count}, skip)")
            continue
        needed = target_count - current_count
        print(f"  {star} stars: {current_count} -> augmenting {needed} more")
        source_texts = star_df["text"].values
        for i in range(needed):
            original = source_texts[i % len(source_texts)]
            augmented = augment_text(
                original,
                synonym_prob=synonym_prob,
                shuffle_prob=shuffle_prob,
                rng=rng,
            )
            augmented_rows.append({"text": augmented, "user_rating": star})
    if not augmented_rows:
        print("  No augmentation needed")
        return df.reset_index(drop=True)
    augmented_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, augmented_df], ignore_index=True)
    print(f"\n  Augmentation: {len(df)} -> {len(combined)} rows (+{len(augmented_rows)})")
    return combined


def augment_from_error_csv(
    df: pd.DataFrame,
    error_csv_path: str,
    *,
    factor: int = 2,
    synonym_prob: float = 0.3,
    shuffle_prob: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Augment texts from severe-error export to target frequent confusion patterns."""
    if not error_csv_path or not os.path.isfile(error_csv_path):
        print(f"  Error-driven augment skipped: file not found ({error_csv_path})")
        return df
    errors = pd.read_csv(error_csv_path, encoding="utf-8")
    if errors.empty or "text" not in errors.columns:
        return df
    rng = np.random.RandomState(random_state)
    rows = []
    for _, row in errors.iterrows():
        star = int(row.get("user_rating", row.get("true_star", 3)))
        text = str(row["text"])
        for _ in range(factor):
            rows.append(
                {
                    "text": augment_text(
                        text,
                        synonym_prob=synonym_prob,
                        shuffle_prob=shuffle_prob,
                        rng=rng,
                    ),
                    "user_rating": star,
                }
            )
    if not rows:
        return df
    extra = pd.DataFrame(rows)
    combined = pd.concat([df, extra], ignore_index=True)
    print(f"  Error-driven augment: +{len(extra)} rows from {error_csv_path}")
    return combined


def apply_train_augmentation(df_train: pd.DataFrame) -> pd.DataFrame:
    """Apply configured augmentation steps to a training dataframe."""
    out = df_train
    if getattr(config, "AUGMENT_ENABLED", False):
        print("\n--- Data Augmentation (minority classes) ---")
        out = augment_minority_classes(
            out,
            target_stars=config.AUGMENT_TARGET_STARS,
            target_count=config.AUGMENT_TARGET_COUNT,
            synonym_prob=config.AUGMENT_SYNONYM_PROB,
            shuffle_prob=config.AUGMENT_SHUFFLE_PROB,
            random_state=config.AUGMENT_RANDOM_STATE,
        )
    error_path = getattr(config, "AUGMENT_FROM_ERRORS_PATH", "")
    if getattr(config, "AUGMENT_FROM_ERRORS", False) and error_path:
        print("\n--- Data Augmentation (error-driven) ---")
        out = augment_from_error_csv(
            out,
            error_path,
            factor=getattr(config, "AUGMENT_FROM_ERRORS_FACTOR", 2),
            synonym_prob=config.AUGMENT_SYNONYM_PROB,
            shuffle_prob=config.AUGMENT_SHUFFLE_PROB,
            random_state=config.AUGMENT_RANDOM_STATE,
        )
    return out


def build_confusion_summary(
    y_true: np.ndarray,
    expected: np.ndarray,
    *,
    min_delta: int = 2,
) -> dict:
    """Summarize severe (true, pred) star pairs for error-driven augmentation."""
    y_true = y_true.astype(int)
    y_pred = np.clip(np.round(expected), 1, 5).astype(int)
    delta = np.abs(y_true - y_pred)
    mask = delta >= min_delta
    pairs = Counter(
        (int(t), int(p)) for t, p in zip(y_true[mask], y_pred[mask])
    )
    return {
        "min_delta": min_delta,
        "severe_count": int(mask.sum()),
        "confusion_pairs": {f"{t}->{p}": c for (t, p), c in pairs.most_common()},
    }


def write_confusion_summary(path: str, summary: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
