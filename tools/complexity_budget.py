"""Measure, rank, and ratchet down the complexity suppressions (ENG-25, #307).

The four complexity rules (`C901`, `PLR0912`, `PLR0913`, `PLR0915`) are silenced
in-line at many sites in ``src/process_improve``. Counting the ``# noqa``
comments is the obvious way to track that, and it is the wrong way: a directive
that no longer suppresses anything still counts, and a function exempted by a
`per-file-ignores` entry does not. This module asks ruff instead, with
``--ignore-noqa``, so the number is the count of functions that genuinely breach
a threshold today.

Two uses:

* ``python tools/complexity_budget.py`` prints the current counts against
  :data:`BUDGET` and ranks the worst offenders, which answers "what do I split
  next?". Exit status is non-zero when the count has grown.
* ``tests/test_complexity_ratchet.py`` calls :func:`check` so CI enforces the
  same thing, in both directions: the count may not rise, and when it falls,
  :data:`BUDGET` has to be lowered to match. That is what keeps "each release
  lowers the count" true rather than aspirational.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Rules tracked by the ratchet, in the order they are reported.
RULES = ("C901", "PLR0912", "PLR0913", "PLR0915")

#: Highest count of each rule that ``src/process_improve`` may contain.
#:
#: These are a ratchet, not a target: lower them whenever a refactor earns it,
#: never raise them. A pull request that pushes a count over its budget has made
#: the code worse in the specific way #307 is about, and CI says so.
BUDGET: dict[str, int] = {
    "C901": 47,
    "PLR0912": 26,
    "PLR0913": 78,
    "PLR0915": 31,
}

#: Where the ratchet is headed: half of the 2026-06 baseline of 185, by v2.0.
TARGET_TOTAL = 91
TARGET_RELEASE = "2.0"

#: Package directory the budget applies to. Tests, examples and notebooks are
#: excluded: complexity there is a different (and largely accepted) trade-off.
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "process_improve"

#: ruff writes the measured value and the threshold into the message, e.g.
#: "`fit` is too complex (40 > 10)" or "Too many branches (62 > 12)".
_MEASUREMENT = re.compile(r"\((\d+) > (\d+)\)")
_DEF = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)")


@dataclass(frozen=True)
class Offender:
    """One function that breaches one complexity rule."""

    code: str
    path: Path
    row: int
    function: str
    value: int
    threshold: int

    @property
    def excess(self) -> float:
        """How far past the threshold, as a multiple, so rules rank comparably."""
        return self.value / self.threshold

    def __str__(self) -> str:
        """Render the rule, the measurement and the threshold on one line."""
        return f"{self.code} {self.value:>4} > {self.threshold:<3} ({self.excess:.1f}x)"


def measure(package: Path = PACKAGE) -> list[Offender]:
    """Run ruff over `package` and return every tracked complexity breach.

    Parameters
    ----------
    package : Path
        Directory to lint. Defaults to :data:`PACKAGE`.

    Returns
    -------
    list[Offender]
        One entry per (function, rule) pair that exceeds its threshold, with
        in-line ``# noqa`` directives ignored.

    Raises
    ------
    RuntimeError
        If ruff is not installed, or fails for a reason other than reporting
        violations (exit status 1 means "found something", which is expected).
    """
    argv = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        str(package),
        f"--select={','.join(RULES)}",
        "--ignore-noqa",
        "--output-format=json",
        "--no-cache",
    ]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603 - fixed argv
    except OSError as exc:  # pragma: no cover - only when the interpreter itself is broken
        msg = f"could not run ruff: {exc}"
        raise RuntimeError(msg) from exc
    if completed.returncode not in (0, 1):
        msg = f"ruff exited {completed.returncode}: {completed.stderr.strip()}"
        raise RuntimeError(msg)

    return sorted(
        (_offender(entry) for entry in json.loads(completed.stdout)),
        key=lambda off: (-off.excess, str(off.path), off.row),
    )


def _offender(entry: dict) -> Offender:
    """Build an :class:`Offender` from one ruff JSON diagnostic."""
    match = _MEASUREMENT.search(entry["message"])
    if match is None:  # pragma: no cover - would mean ruff changed its message format
        msg = f"cannot parse a measurement out of ruff's message: {entry['message']!r}"
        raise RuntimeError(msg)
    path = Path(entry["filename"])
    row = entry["location"]["row"]
    return Offender(
        code=entry["code"],
        path=path,
        row=row,
        function=_function_name(path, row),
        value=int(match.group(1)),
        threshold=int(match.group(2)),
    )


def _function_name(path: Path, row: int) -> str:
    """Read the `def` line the diagnostic points at, for a readable report."""
    try:
        line = path.read_text(encoding="utf-8").splitlines()[row - 1]
    except (OSError, IndexError):  # pragma: no cover - the file ruff just read
        return "?"
    match = _DEF.match(line)
    return match.group(1) if match else "?"


def counts(offenders: list[Offender]) -> dict[str, int]:
    """Tally `offenders` per rule, with an explicit zero for unbreached rules."""
    tally = dict.fromkeys(RULES, 0)
    for offender in offenders:
        tally[offender.code] += 1
    return tally


def check(offenders: list[Offender] | None = None) -> list[str]:
    """Compare the measured counts against :data:`BUDGET`.

    Parameters
    ----------
    offenders : list[Offender] | None
        Result of :func:`measure`; measured afresh when omitted.

    Returns
    -------
    list[str]
        One message per rule that is off budget, empty when every rule matches
        exactly. A count above budget is a regression; a count below it means
        :data:`BUDGET` is stale and must be lowered to lock the win in.
    """
    tally = counts(measure() if offenders is None else offenders)
    problems = []
    for rule, budget in BUDGET.items():
        found = tally[rule]
        if found > budget:
            problems.append(
                f"{rule}: {found} breaches, budget is {budget}. Refactor, or justify raising the budget in #307."
            )
        elif found < budget:
            problems.append(
                f"{rule}: {found} breaches, budget is still {budget}. "
                f"Lower BUDGET['{rule}'] to {found} in tools/complexity_budget.py to keep the win."
            )
    return problems


def report(offenders: list[Offender], worst: int) -> str:
    """Render the budget table and the `worst` offenders as text."""
    tally = counts(offenders)
    total = sum(tally.values())
    lines = [f"{'rule':<9} {'found':>6} {'budget':>7}", "-" * 24]
    lines += [f"{rule:<9} {tally[rule]:>6} {BUDGET[rule]:>7}" for rule in RULES]
    lines += [
        "-" * 24,
        f"{'total':<9} {total:>6} {sum(BUDGET.values()):>7}",
        "",
        f"Target: {TARGET_TOTAL} by v{TARGET_RELEASE} ({total - TARGET_TOTAL} to go).",
        "",
        f"Worst {worst} offenders (ranked by how far past the threshold):",
    ]
    for offender in offenders[:worst]:
        where = offender.path.relative_to(PACKAGE.parents[1])
        lines.append(f"  {offender!s:<28} {offender.function}()  {where}:{offender.row}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the budget report; exit non-zero when a count is off budget."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worst", type=int, default=15, help="how many offenders to list (default: 15)")
    args = parser.parse_args(argv)

    offenders = measure()
    print(report(offenders, args.worst))
    problems = check(offenders)
    if problems:
        print("\n" + "\n".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
