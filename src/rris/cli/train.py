"""CLI: train models (baseline, baseline_optuna, xlmr, embedding)."""

from __future__ import annotations

import argparse
import sys

TRAINERS = {
    "baseline": "rris.training.baseline",
    "baseline_optuna": "rris.training.baseline_optuna",
    "xlmr": "rris.training.xlmr",
    "embedding": "rris.training.embedding",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a review rating model.")
    parser.add_argument(
        "model",
        choices=list(TRAINERS.keys()),
        help="Which training script to run",
    )
    args, remainder = parser.parse_known_args(argv)
    module_name = TRAINERS[args.model]
    import importlib

    mod = importlib.import_module(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [f"{args.model}.py", *remainder]
        if hasattr(mod, "main"):
            mod.main()
        elif hasattr(mod, "__main__"):
            pass
        else:
            raise SystemExit(f"No main() in {module_name}")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
