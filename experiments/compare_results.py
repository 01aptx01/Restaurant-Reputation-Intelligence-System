#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/compare_results.py
==============================
เปรียบเทียบผลทดลองจากหลาย run / manifest พร้อมพิมพ์ตาราง rank
และส่งออก CSV สรุปเพื่อวิเคราะห์ต่อ

Usage:
    python experiments/compare_results.py                          # ค้นหาผล run ล่าสุดทั้งหมด
    python experiments/compare_results.py --run-id 20260604T0100Z  # เจาะจง run
    python experiments/compare_results.py --model baseline         # กรองเฉพาะโมเดล
    python experiments/compare_results.py --export results.csv     # ส่งออก CSV
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"


def collect_results(
    run_id: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """รวบรวมผลทดลองจากโฟลเดอร์ results/"""
    results = []

    if not RESULTS_DIR.is_dir():
        print(f"No results directory found at {RESULTS_DIR}", file=sys.stderr)
        return results

    # วนลูป run directories
    run_dirs = sorted(RESULTS_DIR.iterdir())
    if run_id:
        run_dirs = [d for d in run_dirs if d.name == run_id]

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        for model_dir in sorted(run_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model and model_dir.name != model:
                continue

            for json_file in sorted(model_dir.glob("*.json")):
                if json_file.name.startswith("_"):
                    continue  # skip _index.json

                with json_file.open(encoding="utf-8") as f:
                    data = json.load(f)

                data["_run_id"] = run_dir.name
                data["_file"] = str(json_file)
                results.append(data)

    return results


def print_comparison_table(results: list[dict]) -> None:
    """พิมพ์ตารางเปรียบเทียบเรียงลำดับ MAE"""
    ok_results = [r for r in results if r.get("status") == "ok" and r.get("metrics")]
    failed = [r for r in results if r.get("status") != "ok"]

    if not ok_results:
        print("No successful experiment results found.")
        return

    # จัดเรียงตาม MAE
    ok_results.sort(key=lambda r: r["metrics"].get("mae", float("inf")))

    print(f"\n{'=' * 110}")
    print("  EXPERIMENT RESULTS COMPARISON — Ranked by MAE")
    print(f"{'=' * 110}")

    header = (
        f"  {'#':>2}  {'Run':>16} {'Model':<10} {'Variant':<28} "
        f"{'MAE':>7} {'Acc':>7} {'F1-M':>7} {'Off1':>7} {'Time':>6}"
    )
    print(header)
    print(f"  {'--':>2}  {'-' * 16} {'-' * 10} {'-' * 28} "
          f"{'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 6}")

    for rank, r in enumerate(ok_results, 1):
        m = r["metrics"]
        run_id = r.get("_run_id", "?")[:16]
        model = r.get("model", "?")[:10]
        variant = r.get("variant", "?")[:28]
        mae = m.get("mae", 0)
        acc = m.get("accuracy", 0)
        f1m = m.get("f1_macro", 0)
        off1 = m.get("off_by_one_accuracy", 0)
        dur = r.get("duration", 0)
        tag = " ★" if rank == 1 else ""

        if dur >= 3600:
            time_str = f"{dur / 3600:.1f}h"
        elif dur >= 60:
            time_str = f"{dur / 60:.1f}m"
        else:
            time_str = f"{dur:.0f}s"

        print(
            f"  {rank:>2}  {run_id:>16} {model:<10} {variant:<28} "
            f"{mae:>7.4f} {acc:>7.4f} {f1m:>7.4f} {off1:>7.4f} {time_str:>6}{tag}"
        )

    if len(ok_results) > 1:
        best = ok_results[0]
        worst = ok_results[-1]
        spread = worst["metrics"]["mae"] - best["metrics"]["mae"]
        print(f"\n  🏆 Best:  {best['variant']} (MAE={best['metrics']['mae']:.4f})")
        print(f"  📊 Spread: {spread:.4f} MAE over {len(ok_results)} variants")

    if failed:
        print(f"\n  ⚠️  Failed variants: {len(failed)}")
        for r in failed:
            print(f"     - {r.get('variant', '?')}: {r.get('error', 'unknown')}")

    print(f"{'=' * 110}")


def export_csv(results: list[dict], output_path: str) -> None:
    """ส่งออกผลเปรียบเทียบเป็น CSV"""
    import csv

    ok_results = [r for r in results if r.get("status") == "ok" and r.get("metrics")]
    ok_results.sort(key=lambda r: r["metrics"].get("mae", float("inf")))

    fieldnames = [
        "rank", "run_id", "model", "variant", "hypothesis",
        "mae", "accuracy", "f1_macro", "f1_weighted", "rmse",
        "off_by_one_accuracy", "duration_sec", "status",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(ok_results, 1):
            m = r["metrics"]
            writer.writerow({
                "rank": rank,
                "run_id": r.get("_run_id", ""),
                "model": r.get("model", ""),
                "variant": r.get("variant", ""),
                "hypothesis": r.get("hypothesis", ""),
                "mae": m.get("mae"),
                "accuracy": m.get("accuracy"),
                "f1_macro": m.get("f1_macro"),
                "f1_weighted": m.get("f1_weighted"),
                "rmse": m.get("rmse"),
                "off_by_one_accuracy": m.get("off_by_one_accuracy"),
                "duration_sec": r.get("duration"),
                "status": r.get("status"),
            })
    print(f"\nExported {len(ok_results)} results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--run-id", default=None, help="Filter by run ID")
    parser.add_argument("--model", default=None, help="Filter by model (baseline/xlmr/embedding)")
    parser.add_argument("--export", default=None, help="Export results to CSV path")
    args = parser.parse_args()

    results = collect_results(run_id=args.run_id, model=args.model)

    if not results:
        print("No experiment results found.")
        print(f"Run experiments first: python scripts/run_experiments.py experiments/manifests/baseline_sweep.yaml")
        sys.exit(1)

    print_comparison_table(results)

    if args.export:
        export_csv(results, args.export)


if __name__ == "__main__":
    main()
