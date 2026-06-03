# -*- coding: utf-8 -*-
"""Extra features, sampling, class weights, focal loss, and augmentation."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pythainlp.tokenize import word_tokenize
from sklearn.utils.class_weight import compute_class_weight

from rris.data.loading import clean_review_dataframe, load_and_standardize_data
from rris.data.text import NEGATION_PHRASES, extended_normalize_text


def count_negations(text: str, phrases: tuple[str, ...] = NEGATION_PHRASES) -> int:
    if not text:
        return 0
    remaining = text
    total = 0
    for phrase in phrases:
        if not phrase:
            continue
        n = remaining.count(phrase)
        total += n
        remaining = remaining.replace(phrase, " ")
    return total


def compute_extra_features(texts: pd.Series | list[str]) -> np.ndarray:
    series = texts if isinstance(texts, pd.Series) else pd.Series(list(texts))
    char_len = series.str.len().astype(np.float64).values.reshape(-1, 1)
    word_counts = np.array(
        [len(word_tokenize(t, engine="newmm")) for t in series],
        dtype=np.float64,
    ).reshape(-1, 1)
    neg_counts = np.array(
        [count_negations(t) for t in series],
        dtype=np.float64,
    ).reshape(-1, 1)
    return np.hstack([char_len, word_counts, neg_counts])


def rating_to_3class(ratings_1_to_5: np.ndarray) -> np.ndarray:
    r = ratings_1_to_5.astype(int)
    out = np.ones(len(r), dtype=int)
    out[(r == 1) | (r == 2)] = 0
    out[r == 3] = 1
    out[(r == 4) | (r == 5)] = 2
    return out


def class3_to_expected_star(class_idx: np.ndarray) -> np.ndarray:
    centers = np.array([1.5, 3.0, 4.5], dtype=np.float64)
    idx = np.clip(class_idx.astype(int), 0, 2)
    return centers[idx]


def undersample_star_ratings(
    df: pd.DataFrame,
    star: int = 4,
    keep_fraction: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    if keep_fraction >= 1.0:
        return df.reset_index(drop=True)
    mask = df["user_rating"] == star
    keep = df.loc[mask].sample(frac=keep_fraction, random_state=random_state)
    rest = df.loc[~mask]
    return pd.concat([rest, keep], ignore_index=True)


def oversample_low_star_reviews(
    df: pd.DataFrame,
    factor: int = 3,
    low_stars: tuple[int, ...] = (1, 2),
) -> pd.DataFrame:
    if factor <= 1:
        return df.reset_index(drop=True)
    low = df[df["user_rating"].isin(low_stars)]
    if low.empty:
        return df.reset_index(drop=True)
    extra_copies = max(0, factor - 1)
    parts = [df]
    for _ in range(extra_copies):
        parts.append(low.copy())
    return pd.concat(parts, ignore_index=True)


def per_class_recall(classification_report: dict) -> dict[int, float]:
    recalls: dict[int, float] = {}
    for star in range(1, 6):
        key = str(star)
        if key in classification_report:
            recalls[star] = float(classification_report[key].get("recall", 0.0))
    return recalls


def export_error_analysis(
    df: pd.DataFrame,
    expected: np.ndarray,
    probs: np.ndarray | None,
    out_dir: str,
    *,
    min_delta: int = 2,
    low_conf_threshold: float = 0.4,
    prefix: str = "errors",
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    pred_rounded = np.clip(np.round(expected), 1, 5).astype(int)
    y_true = df["user_rating"].values.astype(int)
    delta = np.abs(pred_rounded - y_true)
    severe_mask = delta >= min_delta
    severe_path = os.path.join(out_dir, f"{prefix}_severe_delta_ge_{min_delta}.csv")
    severe_df = pd.DataFrame(
        {
            "text": df.loc[severe_mask, "text"].values,
            "user_rating": y_true[severe_mask],
            "pred_star_rounded": pred_rounded[severe_mask],
            "ai_expected_rating": expected[severe_mask],
            "abs_error": delta[severe_mask],
        }
    )
    severe_df.to_csv(severe_path, index=False, encoding="utf-8")
    paths = {"severe_errors": severe_path}
    if probs is not None:
        max_prob = probs.max(axis=1)
        correct = pred_rounded == y_true
        low_conf_mask = correct & (max_prob < low_conf_threshold)
        low_path = os.path.join(
            out_dir, f"{prefix}_low_conf_correct_lt_{low_conf_threshold}.csv"
        )
        pd.DataFrame(
            {
                "text": df.loc[low_conf_mask, "text"].values,
                "user_rating": y_true[low_conf_mask],
                "pred_star_rounded": pred_rounded[low_conf_mask],
                "ai_expected_rating": expected[low_conf_mask],
                "max_prob": max_prob[low_conf_mask],
            }
        ).to_csv(low_path, index=False, encoding="utf-8")
        paths["low_confidence_correct"] = low_path
    return paths


def thai_tokenizer(text: str) -> list[str]:
    return word_tokenize(extended_normalize_text(text), engine="newmm")


def compute_class_weights(
    ratings_1_to_5: np.ndarray,
    num_classes: int = 5,
    low_star_boost: float = 1.0,
) -> np.ndarray:
    classes = np.arange(num_classes)
    if num_classes == 3:
        labels = rating_to_3class(ratings_1_to_5)
    else:
        labels = ratings_1_to_5.astype(int) - 1
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    ).astype(np.float64)
    if low_star_boost != 1.0:
        if num_classes == 3:
            weights[0] *= low_star_boost
        else:
            weights[0] *= low_star_boost
            weights[1] *= low_star_boost
    return weights


def compute_sample_weights_from_ratings(
    ratings_1_to_5: np.ndarray,
    class_weights: np.ndarray,
) -> np.ndarray:
    indices = ratings_1_to_5.astype(int) - 1
    return class_weights[indices]


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance in transformer training."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        targets_one_hot = F.one_hot(targets, num_classes).float()
        p_t = (p * targets_one_hot).sum(dim=-1)
        log_p_t = (log_p * targets_one_hot).sum(dim=-1)
        focal_weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        loss = -focal_weight * log_p_t
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


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
