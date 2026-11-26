"""Write the minimum-aberration fractions of 12 to 50 factors in 64 and 128 runs, which no exhaustive search reaches.

A two-level regular fraction of ``k`` factors in ``2**r`` runs is ``k`` distinct non-zero columns of ``r`` bits. Its
word-length pattern ``(A3, A4, ...)`` follows from the code the columns span (the MacWilliams identities, as in
``trade_off._word_counts``), so a set of columns is scored in microseconds. A tabu search swaps one column for another,
always taking the swap with the smallest pattern that is not tabu, and keeps the smallest pattern found; it starts
again from the best design after a few random swaps (an iterated local search, :func:`search`). A search cannot prove
that its design has minimum aberration: the tests do, by comparing each pattern with the one the published,
exhaustive catalogs give (``tests/test_experiments_trade_off.py``). The two run counts take about two hours each.

Run from the repository root::

    uv run python scripts/minimum_aberration_table.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from process_improve.experiments.trade_off import _factor_names, _krawtchouk

OUT = Path("src/process_improve/datasets/experiments/minimum_aberration.json")
SIZES = {64: range(12, 51), 128: range(12, 51)}
STEPS, KICKS, SWAPS = 1600, 100, 7  # the first tabu search's steps, then the kicks and their random swaps
ABOUT = (
    "Minimum-aberration 2^(k-p) fractions of 12 to 50 factors in 64 and 128 runs, written by "
    "scripts/minimum_aberration_table.py. Each size is a string of generator words over the base factors A, B, ..., "
    "one for each added factor in turn (G, H, J, ... after Z come a, b, ...). A search found them; each has the "
    "word-length pattern of the minimum-aberration design of the published, exhaustive catalogs, which the tests check."
)
SOURCES = [
    "Chen, Sun and Wu (1993), International Statistical Review 61, 131-145: 64 runs, resolution IV (12 to 32 factors).",
    "Mee (2009), A Comprehensive Guide to Factorial Two-Level Experimentation, Appendix G: 64 runs, 33 to 37 factors.",
    "Xu (2009), Technometrics 51, 262-277: 128 runs, up to 24 factors.",
    (
        "Block and Mee (2005), Journal of Quality Technology 37, 282-293, with corrigendum 38 (2006) 196: 128 runs, "
        "25 to 64 factors."
    ),
    (
        "64 runs, 38 to 50 factors: FrF2's catalog, from a duplicated 32-run minimum-aberration design and "
        "complementary design theory (Mee, 2009, Section 6.2.2)."
    ),
]


def pattern(weights: np.ndarray, kraw: np.ndarray, r: int) -> tuple[int, ...]:
    """Word-length pattern ``(A3, ..., Ak)`` from the weights of the non-zero codewords."""
    k = kraw.shape[0] - 1
    return tuple(int(math.comb(k, j) + kraw[weights, j].sum()) >> r for j in range(3, k + 1))


def tabu_search(chosen: np.ndarray, odd: np.ndarray, kraw: np.ndarray, rng: np.random.Generator, steps: int) -> tuple:
    """Swap one column for another ``steps`` times; return the smallest pattern met, and its columns as a mask.

    Each step takes the swap with the smallest pattern whose two columns are not tabu, and makes them tabu for a few
    steps, so the search walks on past a local minimum instead of going back to it.
    """
    r, k = odd.shape[0].bit_length(), int(chosen.sum())
    chosen, weights, tabu = chosen.copy(), odd[:, chosen].sum(axis=1), np.zeros(len(chosen), dtype=int)
    best = (pattern(weights, kraw, r), chosen.copy())
    for step in range(steps):
        out, into = np.meshgrid(np.flatnonzero(chosen), np.flatnonzero(~chosen), indexing="ij")
        out, into = out.ravel(), into.ravel()
        moved = weights[:, None] - odd[:, out] + odd[:, into]  # the weights after each swap
        kept = np.flatnonzero((tabu[out] <= step) & (tabu[into] <= step))
        kept = kept if len(kept) else np.arange(len(out))  # with few columns left, all can be tabu
        for j in range(3, k + 1):  # lexicographic: the fewest words of each length in turn
            counts = kraw[moved[:, kept], j].sum(axis=0)
            kept = kept[counts == counts.min()]
            if len(kept) == 1:
                break
        pick = kept[rng.integers(len(kept))]
        chosen[out[pick]], chosen[into[pick]] = False, True
        weights = moved[:, pick]
        tabu[[out[pick], into[pick]]] = step + max(5, k // 3) + rng.integers(3)
        if (found := pattern(weights, kraw, r)) < best[0]:
            best = (found, chosen.copy())
    return best


def search(r: int, k: int, seed: int, fewer: list[int]) -> list[int]:
    """Return the ``k`` columns (integers of ``r`` bits) with the smallest word-length pattern found.

    An iterated local search: a tabu search from the design of ``k - 1`` factors (``fewer``) with its best column
    added, since minimum-aberration designs are often nested; then, ``KICKS`` times, a shorter tabu search from the
    best design after ``SWAPS`` random swaps, which leaves the valley the search is in without losing all of it.
    """
    rng = np.random.default_rng(seed)
    points = np.arange(1, 2**r)  # every possible column, and every non-zero codeword u
    odd = np.array([[(u & p).bit_count() & 1 for p in points] for u in points])  # odd[u, p]: column p counts in u
    kraw = _krawtchouk(k)
    chosen = np.zeros(len(points), dtype=bool)
    if len(fewer) == k - 1:
        chosen[np.array(fewer) - 1] = True
        free = np.flatnonzero(~chosen)
        chosen[min(free, key=lambda p: pattern(odd[:, chosen].sum(axis=1) + odd[:, p], kraw, r))] = True
    else:
        chosen[rng.choice(len(points), k, replace=False)] = True
    best = tabu_search(chosen, odd, kraw, rng, STEPS)
    for _ in range(KICKS):
        kicked = best[1].copy()
        for _ in range(SWAPS):
            kicked[rng.choice(np.flatnonzero(kicked))] = False
            kicked[rng.choice(np.flatnonzero(~kicked))] = True
        if (found := tabu_search(kicked, odd, kraw, rng, STEPS // 2))[0] < best[0]:
            best = found
    return points[best[1]].tolist()


def generators(columns: list[int], r: int) -> list[str]:
    """Write the columns as generator words over ``r`` independent columns taken as the base factors."""
    base, span = [], {0: 0}  # span: each combination of the base columns, and which base columns make it
    for column in sorted(columns):
        if column not in span:
            span |= {value ^ column: mask | 1 << len(base) for value, mask in span.items()}
            base.append(column)
    letters = _factor_names(r)
    words = ["".join(letters[i] for i in range(r) if span[c] >> i & 1) for c in columns if c not in base]
    return sorted(words, key=lambda word: (len(word), word))


def columns_of(words: str, r: int) -> list[int]:
    """Return the columns of a design written as generator words: the base factors' own, then one per word."""
    letters = _factor_names(r)
    return [1 << i for i in range(r)] + [sum(1 << letters.index(c) for c in word) for word in words.split()]


def main() -> None:
    """Search every size not in the table yet, and write the table after each one.

    A size starts from the design the table holds for one factor fewer, so a run that stops part way and is
    started again writes the same table as one that does not stop. Delete the file to search every size again.
    """
    designs = json.loads(OUT.read_text(encoding="utf-8"))["designs"] if OUT.exists() else {}
    for runs, factors in SIZES.items():
        r, found = runs.bit_length() - 1, designs.setdefault(str(runs), {})
        for k in factors:
            if str(k) not in found:
                fewer = columns_of(found[str(k - 1)], r) if str(k - 1) in found else []
                found[str(k)] = " ".join(generators(search(r, k, seed=runs * 100 + k, fewer=fewer), r))
                table = {"about": ABOUT, "sources": SOURCES, "designs": designs}
                OUT.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
                print(runs, k, found[str(k)], flush=True)


if __name__ == "__main__":
    main()
