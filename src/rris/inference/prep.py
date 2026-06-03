# -*- coding: utf-8 -*-
"""Load and clean dataframes before scoring."""

from __future__ import annotations

import os

import pandas as pd

from rris import config, utils


def prepare_scoring_dataframe(file_path: str, normalize_func=None) -> pd.DataFrame:
    if normalize_func is None:
        normalize_func = utils.extended_normalize_text

    df = utils.load_and_standardize_data(file_path, normalize_func=normalize_func)
    df, stats = utils.clean_review_dataframe(
        df,
        min_text_length=config.MIN_TEXT_LENGTH,
        drop_duplicates=config.DROP_DUPLICATE_TEXT,
        duplicate_keep=config.DUPLICATE_KEEP,
    )
    utils.log_cleaning_stats(stats, label=os.path.basename(file_path))
    return df
