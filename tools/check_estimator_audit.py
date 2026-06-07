"""One-off audit: run sklearn.utils.estimator_checks.check_estimator on every
public estimator in process_improve.multivariate and tally pass/fail per check.

Uses the sklearn 1.6+ callback API to collect every check's outcome rather
than aborting on the first failure.
"""
from __future__ import annotations

import warnings
from collections import defaultdict

from sklearn.utils.estimator_checks import check_estimator

from process_improve.multivariate.methods import (
    PCA,
    PLS,
    MCUVScaler,
)

ESTIMATORS = [
    ("MCUVScaler", MCUVScaler()),
    ("PCA(n_components=2)", PCA(n_components=2)),
    ("PLS(n_components=2)", PLS(n_components=2)),
]


def audit(name: str, est):
    """Run check_estimator with a callback that records each check's outcome."""
    results: list[tuple[str, str, str]] = []  # (check_name, status, reason)

    def callback(*, estimator, check_name, exception=None, status=None, **_) -> None:
        if exception is None:
            results.append((check_name, "PASS", ""))
        else:
            short = str(exception).splitlines()[0][:240]
            results.append(
                (check_name, "FAIL", f"{type(exception).__name__}: {short}")
            )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_estimator(est, on_fail="warn", on_skip="warn", callback=callback)
    except Exception as exc:  # noqa: BLE001
        results.append(("<setup>", "ERROR", f"{type(exc).__name__}: {exc}"))

    return results


def main() -> None:
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0})
    failure_details: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for name, est in ESTIMATORS:
        print(f"\n===== {name} =====")
        for check_name, status, reason in audit(name, est):
            if status == "PASS":
                summary[name]["pass"] += 1
            else:
                summary[name]["fail"] += 1
                failure_details[name].append((check_name, reason))

    print("\n\n===== SUMMARY =====")
    for name, s in summary.items():
        total = s["pass"] + s["fail"]
        pct = (s["pass"] / total * 100) if total else 0.0
        print(f"{name}: {s['pass']}/{total} passed ({pct:.1f}%)")

    print("\n===== FAILURES (grouped by reason) =====")
    for name, fails in failure_details.items():
        print(f"\n--- {name} ({len(fails)} failures) ---")
        bucketed: dict[str, list[str]] = defaultdict(list)
        for check_name, reason in fails:
            bucketed[reason].append(check_name)
        for reason, checks in sorted(bucketed.items(), key=lambda kv: -len(kv[1])):
            print(f"\n  [{len(checks)}x] {reason}")
            for c in sorted(checks)[:6]:
                print(f"     - {c}")
            if len(checks) > 6:
                print(f"     ... and {len(checks) - 6} more")


if __name__ == "__main__":
    main()
