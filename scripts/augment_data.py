# -*- coding: utf-8 -*-
# สคริปต์ Data Augmentation สำหรับแก้ Class Imbalance (รีวิว 1–3 ดาว)

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rris import config

try:
    from pythainlp.tokenize import word_tokenize
    HAS_PYTHAINLP = True
except ImportError:
    HAS_PYTHAINLP = False
    print("Warning: PyThaiNLP is not installed. Augmentation will rely only on Random Deletion/Swap.")


def get_synonym(word):
    """หาคำพ้องความหมายภาษาไทย (Synonym)"""
    if not HAS_PYTHAINLP:
        return word

    try:
        from pythainlp.corpus import wordnet

        syns = wordnet.synsets(word)
        if not syns:
            return word

        lemmas = syns[0].lemma_names("tha")
        if not lemmas:
            return word

        choices = [lemma for lemma in lemmas if lemma != word]
        if choices:
            return random.choice(choices)
    except Exception:
        pass

    return word


def synonym_replacement(text, p=0.3):
    """สุ่มเปลี่ยนคำศัพท์ด้วยคำพ้องความหมาย (Synonym Replacement)"""
    if not HAS_PYTHAINLP:
        return text

    words = word_tokenize(text, engine="newmm")
    new_words = []

    for w in words:
        if w.strip() and random.random() < p:
            new_words.append(get_synonym(w))
        else:
            new_words.append(w)

    return "".join(new_words)


def random_deletion(text, p=0.1):
    """สุ่มลบคำศัพท์ในประโยคออก (Random Deletion)"""
    if not HAS_PYTHAINLP:
        words = text.split()
    else:
        words = word_tokenize(text, engine="newmm")

    if len(words) <= 3:
        return text

    new_words = [w for w in words if random.random() > p]
    if not new_words:
        return text

    return "".join(new_words) if HAS_PYTHAINLP else " ".join(new_words)


def augment_review(text):
    """สุ่มใช้งานเทคนิค Augmentation"""
    choice = random.random()
    if choice < 0.5:
        return synonym_replacement(text, p=config.AUGMENT_SYNONYM_PROB)
    return random_deletion(text, p=0.1)


def main():
    print("--- Starting Thai Data Augmentation Pipeline ---")
    if not config.AUGMENT_ENABLED:
        print("Data Augmentation is disabled in config.py")
        return

    random.seed(config.AUGMENT_RANDOM_STATE)

    df = pd.read_csv(config.WONGNAI_TRAIN_PATH)
    print(f"Original Dataset Size: {len(df)} rows")

    class_counts = df["user_rating"].value_counts().to_dict()
    print("Current Class Distribution:", class_counts)

    augmented_rows = []

    for star in config.AUGMENT_TARGET_STARS:
        current_count = class_counts.get(star, 0)
        target = config.AUGMENT_TARGET_COUNT

        if current_count >= target:
            print(f"Star {star}: Already reached target ({current_count} >= {target})")
            continue

        needed = target - current_count
        print(f"Star {star}: Augmenting {needed} new samples...")

        subset = df[df["user_rating"] == star].copy()
        if len(subset) == 0:
            continue

        texts = subset["text"].tolist()

        for _ in tqdm(range(needed), desc=f"Augmenting {star}★"):
            base_text = random.choice(texts)
            aug_text = augment_review(base_text)
            augmented_rows.append({"text": aug_text, "user_rating": star})

    if not augmented_rows:
        print("\nNo augmentation was necessary.")
        return

    df_aug = pd.DataFrame(augmented_rows)
    df_combined = pd.concat([df, df_aug], ignore_index=True)
    df_combined = df_combined.sample(
        frac=1.0, random_state=config.AUGMENT_RANDOM_STATE
    ).reset_index(drop=True)

    out_path = config.WONGNAI_TRAIN_PATH.replace(".csv", "_augmented.csv")
    df_combined.to_csv(out_path, index=False)
    print(f"\nAugmented dataset saved to: {out_path}")
    print(f"New Dataset Size: {len(df_combined)} rows")
    print("New Class Distribution:")
    print(df_combined["user_rating"].value_counts())


if __name__ == "__main__":
    main()
