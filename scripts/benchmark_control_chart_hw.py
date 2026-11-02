"""Time the Holt-Winters ``ControlChart`` fit with the default lambda grid search.

Reports best-of-``--repeat`` wall-clock time for ``ControlChart(variant="hw").calculate_limits(y)``
with no lambdas supplied, so the 5x5 grid search runs (26 Holt-Winters fits per call). The
series is seeded N(50, 2) noise. This is a measurement tool, not a test: nothing asserts on
the timings, which depend on the machine.

Usage
-----
    uv run python scripts/benchmark_control_chart_hw.py [--sizes 200 1000 5000] [--repeat 5]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from process_improve.monitoring.control_charts import ControlChart


def time_fit(n: int, repeat: int) -> float:
    """Return the best of ``repeat`` wall-clock times (seconds) for one grid-search fit."""
    y = 50.0 + 2.0 * np.random.default_rng(n).standard_normal(n)
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        ControlChart(variant="hw").calculate_limits(y)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    """Parse the arguments and print a Markdown table of timings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 1000, 5000], help="series lengths N")
    parser.add_argument("--repeat", type=int, default=5, help="runs per size; the minimum is reported")
    args = parser.parse_args()

    print("| N | seconds per fit | microseconds per row per HW fit |")
    print("|---:|---:|---:|")
    for n in args.sizes:
        seconds = time_fit(n, args.repeat)
        print(f"| {n:,} | {seconds:.4g} | {seconds / (26 * n) * 1e6:.3g} |", flush=True)


if __name__ == "__main__":
    main()
