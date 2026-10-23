"""Combine the per-file test durations the coverage shards measure, and say when they are stale.

Each ``coverage`` shard in CI runs ``pytest --shard=K/N --store-durations``, which rewrites
``tests/.test_durations.json``: the committed durations, with the files that shard ran measured
again. A file's entry differs from the committed one only in the copy of the shard that ran it,
so the combined file takes, for each test file, the value that differs. A file no shard measured
again keeps its committed value.

The shards are balanced on the committed durations. When the seconds each shard measured differ
by more than ``SPREAD`` of the largest, the committed file no longer predicts the suite: this
prints a GitHub Actions warning that says so. Refreshing it is then a download of the combined
file, not a run of the whole suite. Standard library only.

    python tools/shard_durations.py tests/.test_durations.json shard-1.json shard-2.json -o combined.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: The largest gap between the most and the least loaded shard, as a fraction of the most, before a warning.
SPREAD = 0.25


def combine(committed: dict[str, float], shards: list[dict[str, float]]) -> tuple[dict[str, float], list[float]]:
    """Return the combined durations, and the seconds each shard measured again."""
    combined = dict(committed)
    loads = []
    for shard in shards:
        measured = {name: seconds for name, seconds in shard.items() if committed.get(name) != seconds}
        combined |= measured
        loads.append(round(sum(measured.values()), 2))
    return dict(sorted(combined.items())), loads


def stale(loads: list[float], spread: float = SPREAD) -> bool:
    """Whether the shards' measured loads differ by more than ``spread`` of the largest."""
    return bool(loads) and max(loads) > 0 and (max(loads) - min(loads)) > spread * max(loads)


def main(argv: list[str] | None = None) -> None:
    """Write the combined durations and print each shard's load, with a warning when they are stale."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("committed", type=Path, help="the committed tests/.test_durations.json")
    parser.add_argument("shards", type=Path, nargs="+", help="each shard's tests/.test_durations.json, in order")
    parser.add_argument("-o", "--output", type=Path, required=True, help="where to write the combined durations")
    args = parser.parse_args(argv)

    def read(path: Path) -> dict[str, float]:
        return json.loads(path.read_text(encoding="utf-8"))

    combined, loads = combine(read(args.committed), [read(path) for path in args.shards])
    args.output.write_text(json.dumps(combined, indent=1) + "\n", encoding="utf-8")
    for k, load in enumerate(loads, start=1):
        print(f"shard {k}: {load:.0f} s of tests")
    if stale(loads):
        print(
            f"::warning title=Stale test durations::the shards measured {min(loads):.0f} s to {max(loads):.0f} s "
            "of tests. Replace tests/.test_durations.json with this run's test-durations artifact."
        )


if __name__ == "__main__":
    main()
