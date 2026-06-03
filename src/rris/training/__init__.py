"""Model training scripts."""

from . import baseline
from . import baseline_optuna
from . import embedding
from . import xlmr

train_baseline = baseline
train_baseline_optuna = baseline_optuna
train_embedding = embedding
train_xlmr = xlmr

__all__ = [
    "baseline",
    "baseline_optuna",
    "embedding",
    "xlmr",
    "train_baseline",
    "train_baseline_optuna",
    "train_embedding",
    "train_xlmr",
]
