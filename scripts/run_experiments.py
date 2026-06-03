#!/usr/bin/env python3
"""Run experiment manifests: train variants and record evaluation metrics."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rris import config
from rris.evaluation.runner import evaluate_model
from rris.inference.baseline import predict_baseline_with_probs
from rris.inference.embedding import predict_embedding_with_probs
from rris.inference.prep import prepare_scoring_for_model
from rris.inference.xlmr import predict_xlmr

TRAINERS = {
    "baseline": "rris.training.baseline",
    "baseline_optuna": "rris.training.baseline_optuna",
    "xlmr": "rris.training.xlmr",
    "embedding": "rris.training.embedding",
}


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required for experiment manifests. "
            "Install with: pip install pyyaml"
        ) from exc
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(overrides: dict) -> dict:
    snapshot: dict = {}
    for key, value in overrides.items():
        if key in ("name", "hypothesis", "skip_train"):
            continue
        if key == "XGB_PARAMS" and isinstance(value, dict):
            snapshot["XGB_PARAMS"] = copy.deepcopy(config.XGB_PARAMS)
            config.XGB_PARAMS = {**config.XGB_PARAMS, **value}
        elif hasattr(config, key):
            snapshot[key] = getattr(config, key)
            setattr(config, key, value)
        else:
            raise KeyError(f"Unknown config key: {key}")
    return snapshot


def restore_config(snapshot: dict) -> None:
    for key, value in snapshot.items():
        setattr(config, key, value)


def eval_model_on_holdout(model: str, input_path: str | None = None) -> dict:
    path = input_path or config.HOLDOUT_PATH
    df = prepare_scoring_for_model(path, model)
    if model == "baseline":
        expected, _ = predict_baseline_with_probs(df)
    elif model == "embedding":
        expected, _ = predict_embedding_with_probs(df)
    elif model == "xlmr":
        expected = predict_xlmr(df)
    else:
        raise ValueError(f"Unsupported eval model: {model}")
    return evaluate_model(model, df, expected)


def run_train(model: str, *, finetune: bool = False) -> None:
    module_name = TRAINERS[model]
    mod = importlib.import_module(module_name)
    if model == "embedding":
        mod.main(finetune=finetune)
    else:
        mod.main()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run RRIS experiment manifests.")
    parser.add_argument(
        "manifest",
        nargs="?",
        default=str(ROOT / "experiments" / "manifests" / "baseline_sweep.yaml"),
        help="Path to manifest YAML",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output run id (default: UTC timestamp)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Evaluate only (artifacts must exist)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Pass --finetune to embedding training",
    )
    parser.add_argument(
        "--eval-model",
        default=None,
        help="Override model key for evaluation (default: manifest model)",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = _load_yaml(manifest_path)
    model = manifest.get("model", "baseline")
    eval_model = args.eval_model or model
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "experiments" / "results" / run_id / model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {manifest_path.name} | model={model} | run_id={run_id}")
    summary: list[dict] = []

    for variant in manifest.get("variants", []):
        name = variant.get("name", "unnamed")
        hypothesis = variant.get("hypothesis", "")
        skip_train = args.skip_train or variant.get("skip_train", False)
        overrides = variant.get("config") or {}

        print(f"\n{'=' * 60}\nVariant: {name}\n{hypothesis}\n{'=' * 60}")
        snapshot = apply_overrides(overrides)
        result: dict = {
            "manifest": manifest_path.name,
            "variant": name,
            "hypothesis": hypothesis,
            "model": model,
            "config_overrides": overrides,
        }

        try:
            if not skip_train:
                run_train(model, finetune=args.finetune)
            metrics = eval_model_on_holdout(eval_model)
            result["metrics"] = {
                k: metrics.get(k)
                for k in (
                    "mae",
                    "accuracy",
                    "f1_macro",
                    "off_by_one_accuracy",
                    "recall_star_1",
                    "recall_star_2",
                    "anomaly_rate",
                    "anomaly_precision",
                    "anomaly_recall",
                    "anomaly_f1",
                )
                if k in metrics
            }
            result["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 — experiment runner logs all failures
            result["status"] = "error"
            result["error"] = str(exc)
            print(f"ERROR in variant {name}: {exc}")
        finally:
            restore_config(snapshot)

        out_path = out_dir / f"{name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        summary.append({"variant": name, "status": result["status"], "path": str(out_path)})
        print(f"Wrote {out_path}")

    index_path = out_dir / "_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "manifest": manifest_path.name,
                "model": model,
                "variants": summary,
            },
            f,
            indent=2,
        )
    print(f"\nDone. Index: {index_path}")


if __name__ == "__main__":
    main()
