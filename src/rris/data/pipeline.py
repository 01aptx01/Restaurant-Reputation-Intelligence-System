# -*- coding: utf-8 -*-
"""Data preparation pipeline for model training."""

import pandas as pd
from rris import config
from rris.data.loading import mix_mock_training_data, apply_text_truncation
from rris.data.features import undersample_star_ratings, oversample_low_star_reviews
from rris.data.augmentation import apply_train_augmentation

def prepare_baseline_train_df(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Apply standard resampling and augmentation steps for baseline training."""
    train_df = df
    
    if getattr(config, "BASELINE_MOCK_MIX_FRACTION", 0) > 0:
        before = len(train_df)
        train_df = mix_mock_training_data(
            train_df,
            getattr(config, "MOCK_TRAIN_PATH", None),
            config.BASELINE_MOCK_MIX_FRACTION,
            min_text_length=config.MIN_TEXT_LENGTH,
            drop_duplicates=config.DROP_DUPLICATE_TEXT,
            duplicate_keep=config.DUPLICATE_KEEP,
            random_state=config.RANDOM_STATE,
        )
        if verbose: 
            print(f"Mock mix (fraction={config.BASELINE_MOCK_MIX_FRACTION}): {before} -> {len(train_df)} rows")
            
    if getattr(config, "BASELINE_UNDERSAMPLE_STAR4_FRACTION", 1.0) < 1.0:
        before = len(train_df)
        train_df = undersample_star_ratings(
            train_df,
            star=4,
            keep_fraction=config.BASELINE_UNDERSAMPLE_STAR4_FRACTION,
            random_state=config.RANDOM_STATE,
        )
        if verbose: 
            print(f"Undersample 4-star (keep={config.BASELINE_UNDERSAMPLE_STAR4_FRACTION}): {before} -> {len(train_df)} rows")
            
    if getattr(config, "BASELINE_OVERSAMPLE_LOW_STARS", False):
        before = len(train_df)
        train_df = oversample_low_star_reviews(
            train_df,
            factor=config.BASELINE_OVERSAMPLE_FACTOR,
        )
        if verbose: 
            print(f"Oversampled low stars (factor={config.BASELINE_OVERSAMPLE_FACTOR}): {before} -> {len(train_df)} rows")
            
    train_df = apply_train_augmentation(train_df)
    train_df = apply_text_truncation(train_df, config.MAX_REVIEW_CHARS)
    
    return train_df
