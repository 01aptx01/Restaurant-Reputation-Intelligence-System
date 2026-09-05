# -*- coding: utf-8 -*-
"""Standalone data augmentation script (uses unified rris.data.augmentation)."""

from __future__ import annotations

import pandas as pd

from rris import config
from rris.data.augmentation import augment_minority_classes


def main() -> None:
    print("--- Starting Thai Data Augmentation Pipeline ---")
    if not config.AUGMENT_ENABLED:
        print("Data Augmentation is disabled in config.py")
        return

    df = pd.read_csv(config.WONGNAI_TRAIN_PATH)
    print(f"Original Dataset Size: {len(df)} rows")
    print("Current Class Distribution:", df["user_rating"].value_counts().to_dict())

    df_combined = augment_minority_classes(
        df,
        target_stars=config.AUGMENT_TARGET_STARS,
        target_count=config.AUGMENT_TARGET_COUNT,
        synonym_prob=config.AUGMENT_SYNONYM_PROB,
        shuffle_prob=config.AUGMENT_SHUFFLE_PROB,
        random_state=config.AUGMENT_RANDOM_STATE,
    )

    if len(df_combined) == len(df):
        print("\nNo augmentation was necessary.")
        return

    out_path = config.WONGNAI_TRAIN_PATH.replace(".csv", "_augmented.csv")
    df_combined.to_csv(out_path, index=False)
    print(f"\nAugmented dataset saved to: {out_path}")
    print(f"New Dataset Size: {len(df_combined)} rows")
    print("New Class Distribution:")
    print(df_combined["user_rating"].value_counts())


if __name__ == "__main__":
    main()
