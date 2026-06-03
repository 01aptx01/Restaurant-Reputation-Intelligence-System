"""python -m rris [score|evaluate|visualize|train ...]"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "Usage: python -m rris <command> [args...]\n\n"
            "Commands:\n"
            "  score      Score reviews and flag anomalies\n"
            "  evaluate   Evaluate model metrics\n"
            "  visualize  Build Plotly eval HTML report\n"
            "  train      Train a model (baseline, xlmr, ...)\n"
        )
        raise SystemExit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]
    old = sys.argv
    try:
        if cmd == "score":
            sys.argv = ["rris-score", *rest]
            from rris.cli.score import main as run

            run()
        elif cmd == "evaluate":
            sys.argv = ["rris-evaluate", *rest]
            from rris.cli.evaluate import main as run

            run()
        elif cmd == "visualize":
            sys.argv = ["rris-visualize", *rest]
            from rris.cli.visualize import main as run

            run()
        elif cmd == "train":
            from rris.cli.train import main as run

            run(rest)
        else:
            print(f"Unknown command: {cmd}", file=sys.stderr)
            raise SystemExit(1)
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
