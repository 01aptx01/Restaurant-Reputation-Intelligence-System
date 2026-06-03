"""Model training scripts (lazy submodules to avoid importing heavy deps at package load)."""

from __future__ import annotations

import importlib

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

_ALIASES = {
    "train_baseline": "baseline",
    "train_baseline_optuna": "baseline_optuna",
    "train_embedding": "embedding",
    "train_xlmr": "xlmr",
}


def __getattr__(name: str):
    target = _ALIASES.get(name, name)
    if target not in ("baseline", "baseline_optuna", "embedding", "xlmr"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return importlib.import_module(f".{target}", __name__)
