#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["process-improve[expt]"]
# ///
"""Verify a two-level design before anyone runs it in a lab.

Why this exists
---------------
A language model asked to write out a fractional factorial design will often
produce one that looks right and is not. Vazquez et al. (2026) evaluated
GPT-5.1 and Gemini 2.5 Flash on 36 construction tasks and found reliable
results only up to about eight factors; past that the models returned designs
of resolution 1 or 2, non-regular arrays presented as regular fractions, and
tables with missing cells. Every one of those failures is invisible by
inspection and expensive in the lab.

So: never trust a design matrix that was typed rather than generated. Run it
through this script. It reports the resolution actually implied by the matrix,
independent of whatever the matrix was labelled, and fails loudly when the
design is worse than asked for.

Usage
-----
Check a design and print a report::

    python verify_design.py design.csv

Fail (non-zero exit) unless the design meets a bar::

    python verify_design.py design.csv --require-resolution 4
    python verify_design.py design.csv --require-resolution 4 --expect-runs 16 --expect-factors 7

Compare candidate designs and rank them by minimum moment aberration::

    python verify_design.py a.csv b.csv c.csv --compare

Input format
------------
CSV with one column per factor and one row per run, two levels per column in
any coding (-1/1, 0/1, low/high, False/True). Columns named ``Run``,
``RunOrder`` or ``Block`` are ignored. Response columns must not be present;
pass ``--factors A,B,C`` to select explicitly if the file has extras.

Exit codes
----------
0   the design passed every requested check
1   bad usage (unreadable file, no factor columns, mixed-level column)
3   the design failed a check (resolution, shape, or balance)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

EXIT_USAGE = 1
EXIT_FAILED_CHECK = 3

#: How many moments to show before eliding the tail.
_PATTERN_PREVIEW = 8

_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}

_MEANING = {
    1: "Not level-balanced. At least one column has unequal numbers of low and high settings. "
    "This is not a usable factorial design.",
    2: "A main effect is aliased with another main effect. Two columns carry the same "
    "information, so their effects can never be separated. Do not run this design.",
    3: "Main effects are clear of each other but aliased with two-factor interactions. "
    "Usable for screening only, and only if you accept that a large apparent main effect "
    "may really be an interaction.",
    4: "Main effects are clear of each other and of two-factor interactions, but some pairs "
    "of two-factor interactions are aliased with each other. The standard screening choice.",
    5: "Main effects and two-factor interactions are all clear of each other. Interactions are "
    "aliased only with three-factor interactions. Suitable for modelling, not just screening.",
}


def _load(path: Path, factors: list[str] | None) -> Any:
    """Read a design CSV into a DataFrame of factor columns only."""
    import pandas as pd

    if not path.is_file():
        sys.exit(f"No such file: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        sys.exit(f"Could not read {path}: {exc}")

    if factors:
        missing = [name for name in factors if name not in frame.columns]
        if missing:
            sys.exit(f"{path}: requested factor(s) not in file: {', '.join(missing)}")
        return frame[factors]
    return frame


def _describe(result: Any, label: str) -> list[str]:
    """Render one design's verdict as report lines."""
    roman = _ROMAN.get(result.resolution, str(result.resolution))
    lines = [
        f"{label}",
        f"  {result.n_runs} runs x {result.n_factors} factors",
        f"  Resolution {roman} (strength {result.strength})",
        f"  {_MEANING.get(result.resolution, 'No aliasing detectable within this many factors.')}",
        "  Moment aberration pattern: " + ", ".join(f"{k:.4g}" for k in result.pattern[:_PATTERN_PREVIEW]),
    ]
    if len(result.pattern) > _PATTERN_PREVIEW:
        lines[-1] += ", ..."
    attained = len(result.lower_bounds) - 1
    if attained < len(result.pattern):
        lines.append(f"  First moment short of its lower bound: K_{attained + 1}")
    return lines


def _check(result: Any, args: argparse.Namespace, label: str) -> list[str]:
    """Return a list of failure messages; empty means the design passed."""
    failures = []
    if args.require_resolution is not None and result.resolution < args.require_resolution:
        failures.append(
            f"{label}: resolution {result.resolution} is below the required "
            f"{args.require_resolution}. {_MEANING.get(result.resolution, '')}".rstrip()
        )
    if args.expect_runs is not None and result.n_runs != args.expect_runs:
        failures.append(f"{label}: expected {args.expect_runs} runs, found {result.n_runs}.")
    if args.expect_factors is not None and result.n_factors != args.expect_factors:
        failures.append(f"{label}: expected {args.expect_factors} factors, found {result.n_factors}.")
    return failures


def _rank(results: list[tuple[Path, Any]]) -> None:
    """Print the designs ordered by minimum moment aberration, best first."""
    if len({(r.n_runs, r.n_factors) for _, r in results}) > 1:
        print("Cannot rank designs of different sizes; the moment pattern scales with n and m.")
        return
    ranked = sorted(results, key=lambda item: item[1].exact_pattern)
    print("Ranked by minimum moment aberration, best first:")
    for position, (path, _) in enumerate(ranked, start=1):
        print(f"  {position}. {path}")
    print()


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="verify_design.py",
        description="Check what a two-level design matrix actually is, not what it claims to be.",
    )
    parser.add_argument("design", nargs="+", type=Path, help="one or more design CSV files")
    parser.add_argument("--factors", help="comma-separated factor columns, if the file has extras")
    parser.add_argument(
        "--require-resolution",
        type=int,
        help="fail unless every design reaches at least this resolution",
    )
    parser.add_argument("--expect-runs", type=int, help="fail unless the run count matches")
    parser.add_argument("--expect-factors", type=int, help="fail unless the factor count matches")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="rank the designs by minimum moment aberration (same size only)",
    )
    args = parser.parse_args(argv)

    try:
        from process_improve.experiments import moment_aberration
        from process_improve.experiments._moment_aberration import NotTwoLevelError
    except ImportError as exc:  # pragma: no cover - environment guard
        sys.exit(
            f"Could not import process_improve ({exc}).\n"
            "Install it with:  pip install 'process-improve[expt]'\n"
            "or run this script with:  uv run --script verify_design.py ..."
        )

    selected = args.factors.split(",") if args.factors else None
    results = []
    for path in args.design:
        frame = _load(path, selected)
        try:
            results.append((path, moment_aberration(frame)))
        except NotTwoLevelError as exc:
            sys.exit(f"{path}: {exc}")
        except ValueError as exc:
            sys.exit(f"{path}: {exc}")

    failures: list[str] = []
    for path, result in results:
        print("\n".join(_describe(result, str(path))))
        print()
        failures.extend(_check(result, args, str(path)))

    if args.compare and len(results) > 1:
        _rank(results)

    if failures:
        print("FAILED")
        for message in failures:
            print(f"  {message}")
        return EXIT_FAILED_CHECK

    if args.require_resolution or args.expect_runs or args.expect_factors:
        print("PASSED: every requested check is satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
