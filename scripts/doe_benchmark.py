"""Follow recommend_next_step on the benchmark's known processes and print how each study ended.

Usage::

    uv run python scripts/doe_benchmark.py                 # every case, 20 seeds
    uv run python scripts/doe_benchmark.py --seeds 5 --case adhesion

Each row is one case: the mean and worst shortfall of the final settings from the true best, in
noise standard deviations, the mean runs used, how the studies ended, and the commonest path.
"""

from __future__ import annotations

import argparse

import pandas as pd

from process_improve.experiments.benchmark import CASES, run_benchmark


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print its table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20, help="studies per case (default 20)")
    parser.add_argument("--case", default="", help="run only the cases whose name contains this text")
    args = parser.parse_args(argv)
    cases = [case for case in CASES if args.case.lower() in case.name.lower()]
    if not cases:
        parser.error(f"no case matches {args.case!r}; the cases are: {', '.join(c.name for c in CASES)}")
    with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.precision", 3):
        print(run_benchmark(cases, n_seeds=args.seeds).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
