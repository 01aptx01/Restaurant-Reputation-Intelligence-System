"""Merge Wongnai + Joe Beach Capital review CSVs into data/merge/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rris import config

MERGE_DIR = Path(config.DATA_DIR) / "merge"
WONGNAI_CSV = MERGE_DIR / "wongnai-restaurant-review_train.csv"
JOEBEACH_CSV = MERGE_DIR / "joebeachcapital_restaurant_review.csv"
OUTPUT_CSV = MERGE_DIR / "merged_restaurant_reviews.csv"


def main():
    df_wongnai = pd.read_csv(WONGNAI_CSV)
    df_joebeach = pd.read_csv(JOEBEACH_CSV)

    df_wongnai_clean = df_wongnai[["review_body", "stars"]].rename(
        columns={"review_body": "Review", "stars": "Rating"}
    )
    df_joebeach_clean = df_joebeach[["Review", "Rating"]]

    df_merged = pd.concat([df_wongnai_clean, df_joebeach_clean], ignore_index=True)

    print(f"Wongnai Rows: {len(df_wongnai)}")
    print(f"Joe Beach Capital Rows: {len(df_joebeach)}")
    print(f"Total Rows After Merge: {len(df_merged)}")

    MERGE_DIR.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Merged file saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
