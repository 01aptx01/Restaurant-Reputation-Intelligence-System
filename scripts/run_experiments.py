#!/usr/bin/env python3
"""Run experiment manifests: train variants and record evaluation metrics.

Supports all model types: baseline, baseline_optuna, xlmr, embedding.
Prints a ranked summary table at the end to compare all variants.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import time
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

# Metrics to extract from evaluation results
REPORT_METRICS = (
    "mae",
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "rmse",
    "off_by_one_accuracy",
    "recall_star_1",
    "recall_star_2",
    "anomaly_rate",
    "anomaly_precision",
    "anomaly_recall",
    "anomaly_f1",
)


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


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_summary_table(summary: list[dict]) -> None:
    """Print a ranked comparison table of all completed variants."""
    completed = [s for s in summary if s.get("status") == "ok" and s.get("metrics")]
    if not completed:
        print("\nNo successful variants to compare.")
        return

    # Sort by MAE (lower is better)
    completed.sort(key=lambda s: s["metrics"].get("mae", float("inf")))

    print(f"\n{'=' * 100}")
    print("  EXPERIMENT RESULTS — RANKED BY MAE (lower is better)")
    print(f"{'=' * 100}")

    header = (
        f"  {'#':>2}  {'Variant':<28} {'MAE':>7} {'Acc':>7} {'F1-M':>7} "
        f"{'Off1':>7} {'R★1':>6} {'R★2':>6} {'Time':>7}"
    )
    print(header)
    print(f"  {'--':>2}  {'-' * 28} {'-' * 7} {'-' * 7} {'-' * 7} "
          f"{'-' * 7} {'-' * 6} {'-' * 6} {'-' * 7}")

    for rank, s in enumerate(completed, 1):
        m = s["metrics"]
        tag = " ★" if rank == 1 else ""
        duration = _format_duration(s.get("duration", 0))

        mae = m.get("mae", 0)
        acc = m.get("accuracy", 0)
        f1m = m.get("f1_macro", 0)
        off1 = m.get("off_by_one_accuracy", 0)
        r1 = m.get("recall_star_1", m.get("per_class_recall", {}).get("1", 0))
        r2 = m.get("recall_star_2", m.get("per_class_recall", {}).get("2", 0))

        print(
            f"  {rank:>2}  {s['variant']:<28} {mae:>7.4f} {acc:>7.4f} {f1m:>7.4f} "
            f"{off1:>7.4f} {r1:>6.3f} {r2:>6.3f} {duration:>7}{tag}"
        )

    best = completed[0]
    worst = completed[-1]
    print(f"\n  🏆 Best:  {best['variant']} (MAE={best['metrics']['mae']:.4f})")
    if len(completed) > 1:
        improvement = worst["metrics"]["mae"] - best["metrics"]["mae"]
        print(f"  📊 Spread: {improvement:.4f} MAE ({best['variant']} → {worst['variant']})")

    print(f"{'=' * 100}")


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
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Run only these variant names (space-separated). Default: all.",
    )
    parser.add_argument(
        "--holdout-path",
        default=None,
        help="Override holdout CSV path for evaluation.",
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

    # Filter variants if --variants is specified
    all_variants = manifest.get("variants", [])
    if args.variants:
        wanted = set(args.variants)
        all_variants = [v for v in all_variants if v.get("name") in wanted]
        if not all_variants:
            raise SystemExit(f"No matching variants for: {args.variants}")

    print(f"{'=' * 70}")
    print(f"  RRIS Experiment Runner")
    print(f"  Manifest: {manifest_path.name} | Model: {model} | Run: {run_id}")
    print(f"  Variants: {len(all_variants)} | Skip-train: {args.skip_train}")
    print(f"{'=' * 70}")

    summary: list[dict] = []
    total_start = time.time()

    for idx, variant in enumerate(all_variants, 1):
        name = variant.get("name", "unnamed")
        hypothesis = variant.get("hypothesis", "")
        skip_train = args.skip_train or variant.get("skip_train", False)
        overrides = variant.get("config") or {}

        print(f"\n{'=' * 60}")
        print(f"  [{idx}/{len(all_variants)}] Variant: {name}")
        print(f"  Hypothesis: {hypothesis}")
        print(f"{'=' * 60}")

        snapshot = apply_overrides(overrides)
        variant_start = time.time()
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

            holdout = args.holdout_path or None
            metrics = eval_model_on_holdout(eval_model, holdout)

            # Extract metrics
            result["metrics"] = {
                k: metrics.get(k)
                for k in REPORT_METRICS
                if k in metrics
            }
            # Also store per-class recall if available
            if "per_class_recall" in metrics:
                result["metrics"]["per_class_recall"] = metrics["per_class_recall"]

            result["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 — experiment runner logs all failures
            result["status"] = "error"
            result["error"] = str(exc)
            print(f"ERROR in variant {name}: {exc}")
        finally:
            restore_config(snapshot)

        duration = time.time() - variant_start
        result["duration"] = round(duration, 1)

        out_path = out_dir / f"{name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        summary.append(result)

        elapsed = time.time() - total_start
        remaining = elapsed / idx * (len(all_variants) - idx)
        print(f"  ✓ {name} done in {_format_duration(duration)} "
              f"(ETA: {_format_duration(remaining)} remaining)")
        print(f"  Wrote {out_path}")

    # Save index
    index_path = out_dir / "_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "manifest": manifest_path.name,
                "model": model,
                "total_time_sec": round(time.time() - total_start, 1),
                "variants": [
                    {
                        "variant": s["variant"],
                        "status": s["status"],
                        "mae": s.get("metrics", {}).get("mae"),
                        "duration": s.get("duration"),
                        "path": str(out_dir / f"{s['variant']}.json"),
                    }
                    for s in summary
                ],
            },
            f,
            indent=2,
        )

    # Print summary table
    print_summary_table(summary)

    total_time = time.time() - total_start
    print(f"\nTotal time: {_format_duration(total_time)}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
