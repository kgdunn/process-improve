"""Time the PCA and PLS calls a multivariate browser app would make, natively or under Pyodide.

Usage::

    uv run python scripts/multivariate_timings.py                    # natively
    uv run python scripts/multivariate_timings.py --only tablet      # the workloads whose name has "tablet"
    node scripts/pyodide_run.mjs dist/<wheel> scripts/multivariate_timings.py   # under Pyodide

Prints one row per workload: the first (cold) run and the median of the warm repeats, in seconds.
Pyodide runs Python on one thread, so compare it with a native run on one thread too:
``OPENBLAS_NUM_THREADS=1``. The cross-validation settings are ScorePilot's: 7 folds, the curve to 20
components, PLS with 10 repeats and the 1-SE rule, PCA with element-wise k-fold.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from process_improve.multivariate import PCA, PLS, MCUVScaler, vip

if TYPE_CHECKING:
    from collections.abc import Callable

DATA = Path(__file__).resolve().parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
SEED = 13
#: A call that takes longer than this is timed once: it is its own average.
LONG = 5.0


def workloads() -> dict[str, Callable[[], object]]:
    """Return the workloads by name, with their data loaded and scaled."""
    tablet_raw = pd.read_csv(DATA / "tablet-spectra.csv", header=None, index_col=0)
    tablet = MCUVScaler().fit_transform(tablet_raw)
    # The tablet spectra carry no Y: a synthetic one from two spectral bands gives PLS a real signal.
    rng = np.random.default_rng(SEED)
    band = tablet.iloc[:, 100:120].mean(axis=1) - tablet.iloc[:, 400:430].mean(axis=1)
    tablet_y = pd.DataFrame({"y": band + rng.normal(0, 0.1, len(tablet))})
    ldpe = pd.read_csv(DATA / "LDPE" / "LDPE.csv", index_col=0)
    ldpe_x, ldpe_y = MCUVScaler().fit_transform(ldpe.iloc[:, :14]), MCUVScaler().fit_transform(ldpe.iloc[:, 14:])
    big = MCUVScaler().fit_transform(pd.DataFrame(rng.normal(size=(5000, 200)) @ rng.normal(size=(200, 200))))
    cv = {"max_components": 20, "cv": 7, "random_state": SEED}

    def pca_contributions() -> None:
        model = PCA(n_components=3).fit(tablet)
        model.t2_contributions(tablet)
        model.spe_contributions(tablet)

    return {
        "Load the tablet spectra CSV (460 x 650)": lambda: pd.read_csv(DATA / "tablet-spectra.csv", header=None),
        "PCA fit, LDPE, A=3": lambda: PCA(n_components=3).fit(ldpe_x),
        "PCA fit, tablet, A=3": lambda: PCA(n_components=3).fit(tablet),
        "PCA fit and T2 and SPE contributions, tablet, A=3": pca_contributions,
        "PCA fit, 5000 x 200, A=5": lambda: PCA(n_components=5).fit(big),
        "PLS fit and VIP, LDPE, A=3": lambda: vip(PLS(n_components=3, scale=False).fit(ldpe_x, ldpe_y)),
        "PLS fit, tablet, A=5": lambda: PLS(n_components=5, scale=False).fit(tablet, tablet_y),
        "PCA cross-validation, LDPE": lambda: PCA.select_n_components(ldpe_x, cv_scheme="ekf", **cv),
        "PCA cross-validation, tablet": lambda: PCA.select_n_components(tablet, cv_scheme="ekf", **cv),
        "PLS cross-validation, LDPE": lambda: PLS.select_n_components(ldpe_x, ldpe_y, n_repeats=10, **cv),
        "PLS cross-validation, tablet": lambda: PLS.select_n_components(tablet, tablet_y, n_repeats=10, **cv),
    }


def timed(fn: Callable[[], object], repeats: int) -> tuple[float, float]:
    """Return the cold run's seconds and the median of the warm repeats."""
    times = []
    for _ in range(1 + repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
        if times[0] > LONG:
            break
    return times[0], statistics.median(times[1:] or times)


def main(argv: list[str] | None = None) -> int:
    """Time the workloads and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3, help="warm repeats of a short call (default 3)")
    parser.add_argument("--only", default="", help="run only the workloads whose name contains this text")
    args = parser.parse_args(argv)
    chosen = {name: fn for name, fn in workloads().items() if args.only.lower() in name.lower()}
    if not chosen:
        parser.error(f"no workload matches {args.only!r}")
    warnings.simplefilter("ignore")  # convergence notes from the fits are not what is measured here
    print(f"{'Workload':52s} {'cold [s]':>9s} {'warm [s]':>9s}")
    for name, fn in chosen.items():
        cold, warm = timed(fn, args.repeats)
        print(f"{name:52s} {cold:9.3f} {warm:9.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
