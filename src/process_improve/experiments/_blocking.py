# (c) Kevin Dunn, 2010-2026. MIT License.

"""Assign the runs of a design to blocks so that block differences disturb the factor effects least.

Two methods, chosen by the design:

- **Confounding** (regular two-level designs: full factorials and fractional
  factorials). ``2^b`` blocks are defined by ``b`` interaction words; every run's
  block is read off the signs of those words. All ``2^b - 1`` products of the words
  (the block contrasts) are then confounded with blocks, together with their aliases
  in a fraction. The words are chosen so that no block contrast is a main effect or
  aliased with one, then so that as few as possible are two-factor interactions, then
  for the longest words (minimum aberration blocking; Sun, Wu and Chen 1997).
- **Exchange** (every other design). Runs are assigned to blocks of equal size (or
  sizes differing by one) by swapping the blocks of two runs while that raises
  ``det(Xc'Xc)``, with ``Xc`` the model columns centred within each block: the
  information left for the factor effects once each block has its own mean. A
  design whose runs split into orthogonal blocks reaches the unblocked value.

Centre runs carry no sign for any word, so the confounding method shares them out
across the blocks in turn.

References
----------
Sun, D. X., Wu, C. F. J. and Chen, Y. (1997). Optimal blocking schemes for 2^n and
2^(n-p) designs. *Technometrics*, 39(3), 298-307.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

#: Largest number of block-generator sets the confounding search scores exhaustively.
_MAX_GENERATOR_SETS = 200_000
#: Random starts and passes of the block exchange.
_EXCHANGE_STARTS = 4
_EXCHANGE_PASSES = 30


@dataclass
class Blocking:
    """Block labels (1-based, one per run) and how they were chosen."""

    labels: np.ndarray
    method: str
    generators: list[str] = field(default_factory=list)
    confounded_with: list[str] = field(default_factory=list)
    model: str | None = None


def _word(mask: int, names: list[str]) -> str:
    return "".join(name for i, name in enumerate(names) if mask >> i & 1)


def _span(masks: list[int]) -> list[int]:
    """Every product (XOR) of a subset of ``masks``, the empty product (0) included."""
    group = {0}
    for m in masks:
        group |= {g ^ m for g in group}
    return sorted(group)


def defining_subgroup(signs: np.ndarray) -> list[int]:
    """Words (bit masks over the columns) whose product is constant over every run of a two-level design."""
    n, k = signs.shape
    bits = signs > 0
    return [
        mask
        for mask in range(1, 2**k)
        if abs(int(np.prod(np.where(bits[:, [i for i in range(k) if mask >> i & 1]], 1, -1), axis=1).sum())) == n
    ]


def is_regular_two_level(design: np.ndarray) -> bool:
    """Whether the non-centre runs are +/-1 and form a regular fraction (all words either constant or balanced)."""
    corners = design[~np.all(design == 0, axis=1)]
    if len(corners) == 0 or not np.all(np.isin(corners, (-1.0, 1.0))) or corners.shape[1] > 15:
        return False
    k = corners.shape[1]
    for mask in range(1, 2**k):
        total = abs(int(np.prod(corners[:, [i for i in range(k) if mask >> i & 1]], axis=1).sum()))
        if total not in (0, len(corners)):
            return False
    return True


def confounding_blocks(design: np.ndarray, n_blocks: int, names: list[str]) -> Blocking:
    """Block a regular two-level design by confounding ``log2(n_blocks)`` interaction words.

    Raises
    ------
    ValueError
        If ``n_blocks`` is not a power of 2, or every choice of words confounds a main
        effect with blocks.
    """
    b = n_blocks.bit_length() - 1
    if n_blocks < 2 or 2**b != n_blocks:
        raise ValueError(f"A two-level factorial splits into 2, 4, 8, ... blocks; got n_blocks={n_blocks}.")
    k = design.shape[1]
    centre = np.all(design == 0, axis=1)
    corners = design[~centre]
    subgroup = [0, *defining_subgroup(corners)]

    def shortest_alias(mask: int) -> int:
        return min((mask ^ d).bit_count() for d in subgroup)

    pool = [m for m in range(1, 2**k) if m not in subgroup and shortest_alias(m) >= 2]
    pool.sort(key=lambda m: (-shortest_alias(m), -m.bit_count(), m))
    while len(pool) > b and _n_choose(len(pool), b) > _MAX_GENERATOR_SETS:
        pool = pool[: max(b, int(len(pool) * 0.8))]

    best, best_key = None, None
    for generators in itertools.combinations(pool, b):
        contrasts = _span(list(generators))[1:]
        if len(contrasts) != n_blocks - 1:
            continue  # the words were not independent
        lengths = sorted(shortest_alias(c) for c in contrasts)
        if lengths[0] < 2:
            continue  # a main effect (or the mean) would be confounded with blocks
        key = (lengths.count(2), [-length for length in lengths])
        if best_key is None or key < best_key:
            best, best_key = generators, key
    if best is None:
        raise ValueError(
            f"Every way of splitting this {len(corners)}-run design into {n_blocks} blocks confounds a main effect "
            "with blocks; use fewer blocks or a larger design."
        )

    labels = np.zeros(len(design), dtype=int)
    for i, mask in enumerate(best):
        columns = [j for j in range(k) if mask >> j & 1]
        labels[~centre] += (np.prod(corners[:, columns], axis=1) > 0).astype(int) << i
    labels[centre] = np.arange(int(centre.sum())) % n_blocks
    contrasts = _span(list(best))[1:]
    return Blocking(
        labels=labels + 1,
        method="confounding",
        generators=[_word(m, names) for m in best],
        confounded_with=[_word(m, names) for m in contrasts],
    )


def _n_choose(n: int, r: int) -> int:
    from math import comb  # noqa: PLC0415

    return comb(n, r)


def _model_columns(design: np.ndarray, model: str) -> np.ndarray:
    """Return main effects, plus two-factor interactions, plus squares of columns with more than two levels."""
    columns = [design]
    k = design.shape[1]
    if model in ("interactions", "quadratic"):
        columns += [design[:, [i]] * design[:, [j]] for i, j in itertools.combinations(range(k), 2)]
    if model == "quadratic":
        columns += [design[:, [j]] ** 2 for j in range(k) if len(np.unique(design[:, j])) > 2]
    return np.hstack(columns)


def _within_block_information(x: np.ndarray, labels: np.ndarray, n_blocks: int) -> float:
    """``log det(Xc'Xc)``, with each column of ``x`` centred within each block."""
    centred = x.copy()
    for block in range(n_blocks):
        rows = labels == block
        centred[rows] -= centred[rows].mean(axis=0)
    sign, logdet = np.linalg.slogdet(centred.T @ centred)
    return float(logdet) if sign > 0 else -np.inf


def _climb(x: np.ndarray, labels: np.ndarray, n_blocks: int) -> tuple[np.ndarray, float]:
    """Swap the blocks of two runs while that raises the within-block information; return the labels and its value."""
    value = _within_block_information(x, labels, n_blocks)
    for _ in range(_EXCHANGE_PASSES):
        improved = False
        for i, j in itertools.combinations(range(len(labels)), 2):
            if labels[i] == labels[j]:
                continue
            labels[i], labels[j] = labels[j], labels[i]
            trial = _within_block_information(x, labels, n_blocks)
            if trial > value + 1e-10:
                value, improved = trial, True
            else:
                labels[i], labels[j] = labels[j], labels[i]
        if not improved:
            break
    return labels, value


def exchange_blocks(design: np.ndarray, n_blocks: int, rng: np.random.Generator) -> Blocking:
    """Block any design into ``n_blocks`` near-equal blocks by pairwise label swaps.

    The model is the largest of quadratic, interactions and main effects that the
    blocked design can still estimate.

    Raises
    ------
    ValueError
        If ``n_blocks`` is below 2 or above half the runs, or the main effects cannot
        be estimated with a separate mean in every block.
    """
    n = len(design)
    if not 2 <= n_blocks <= n // 2:
        raise ValueError(f"n_blocks must be between 2 and half the {n} runs; got {n_blocks}.")
    balanced = np.arange(n) % n_blocks
    for model in ("quadratic", "interactions", "main_effects"):
        x = _model_columns(design, model)
        if x.shape[1] + n_blocks > n:
            continue
        best_labels, best_value = None, -np.inf
        for _ in range(_EXCHANGE_STARTS):
            labels, value = _climb(x, rng.permutation(balanced), n_blocks)
            if value > best_value:
                best_labels, best_value = labels, value
        if best_labels is not None and np.isfinite(best_value):
            return Blocking(labels=best_labels + 1, method="exchange", model=model)
    raise ValueError(
        f"The main effects cannot be estimated once each of {n_blocks} blocks has its own mean; use fewer blocks."
    )
