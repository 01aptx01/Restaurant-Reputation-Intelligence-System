# Scripts

สคริปต์ **เตรียมข้อมูลและวิจัย** — ไม่ใช่เทรนโมเดลหลัก (ใช้ `python -m rris train <model>`)

| Script | Purpose |
|--------|---------|
| `download_wongnai.py` | Download Wongnai → `data/wongnai/` |
| `generate_mock_data.py` | Mock train/test → `data/mock/` |
| `merge.py` | Merge raw CSVs → `data/merge/` |
| `reduce_merged.py` | Shrink merged training set |
| `augment_data.py` | Augment low-star classes |
| `initialize_web_data.py` | Build `web_app/scored_reviews.json` |
| `eda_baseline_data.py` | EDA → `experiments/baseline/` |
| `tune_baseline.py` | Hyperparameter sweep |
| `summarize_tune_ceiling.py` | Summarize tune results |

Production CLI: `python -m rris score|evaluate|visualize|train`

Library: `src/rris/` — see [docs/directory-guide.md](../docs/directory-guide.md)
