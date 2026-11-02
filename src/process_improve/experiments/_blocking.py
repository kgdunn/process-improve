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
- **Portions** (central composite designs). The axial runs are a block of their own
  and the cube runs fill the others, split by confounding as above, each portion with
  its own centre runs (Box and Hunter 1957). The blocks are then orthogonal to every
  linear and interaction term; they are orthogonal to the squares too when each block
  holds the share of each factor's sum of squares that it holds of the runs, which
  :func:`process_improve.experiments.designs_response_surface.blocking_alpha` arranges.
- **Exchange** (every other design). Runs are assigned to blocks of equal size (or
  sizes differing by one) by swapping the blocks of two runs while that raises
  ``det(Xc'Xc)``, with ``Xc`` the model columns centred within each block: the
  information left for the factor effects once each block has its own mean. A
  design whose runs split into orthogonal blocks reaches the unblocked value.

Centre runs carry no sign for any word, so the confounding method shares them out
across the blocks in turn.

References
----------
Box, G. E. P. and Hunter, J. S. (1957). Multi-factor experimental designs for exploring
response surfaces. *Annals of Mathematical Statistics*, 28(1), 195-241.

Sun, D. X., Wu, C. F. J. and Chen, Y. (1997). Optimal blocking schemes for 2^n and
2^(n-p) designs. *Technometrics*, 39(3), 298-307.
"""

from __future__ import annotations

import itertools
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

#: Largest number of block-generator sets the confounding search scores exhaustively.
_MAX_GENERATOR_SETS = 200_000
#: Random starts and passes of the block exchange.
_EXCHANGE_STARTS = 4
_EXCHANGE_PASSES = 30
#: Gain in ``log det`` an exchange must make to count: less is rounding, which differs between platforms.
_EXCHANGE_GAIN = 1e-10


@dataclass
class Blocking:
    """Block labels (1-based, one per run) and how they were chosen."""

    labels: np.ndarray
    method: str
    generators: list[str] = field(default_factory=list)
    confounded_with: list[str] = field(default_factory=list)
    model: str | None = None
    #: For a central composite design: whether the blocks are orthogonal to the squares too.
    orthogonal: bool | None = None
    #: For a fraction: each word confounded with blocks, and the words aliased with it in the fraction.
    aliases: dict[str, list[str]] = field(default_factory=dict)


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


def confounding_blocks(
    design: np.ndarray, n_blocks: int, names: list[str], generators: list[str] | None = None
) -> Blocking:
    """Block a regular two-level design by confounding ``log2(n_blocks)`` interaction words.

    The words are chosen as the module describes, or given as ``generators``: each a word in the
    factors' names (``"ADE"``, or ``"A*D*E"`` when a name is longer than one letter). In a fraction
    each confounded word carries its aliases, which ``Blocking.aliases`` lists.

    Raises
    ------
    ValueError
        If ``n_blocks`` is not a power of 2, every choice of words confounds a main effect with
        blocks, or the given generators are not ``log2(n_blocks)`` independent words of the
        factors, outside the fraction's defining relation, that leave every main effect clear of
        the blocks.
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

    if generators is not None:
        return _blocking_from_words(design, _given_generators(generators, names, b, subgroup), names, subgroup)

    pool = [m for m in range(1, 2**k) if m not in subgroup and shortest_alias(m) >= 2]
    pool.sort(key=lambda m: (-shortest_alias(m), -m.bit_count(), m))
    while len(pool) > b and _n_choose(len(pool), b) > _MAX_GENERATOR_SETS:
        pool = pool[: max(b, int(len(pool) * 0.8))]

    best, best_key = None, None
    for candidate in itertools.combinations(pool, b):
        contrasts = _span(list(candidate))[1:]
        if len(contrasts) != n_blocks - 1:
            continue  # the words were not independent
        lengths = sorted(shortest_alias(c) for c in contrasts)
        if lengths[0] < 2:
            continue  # a main effect (or the mean) would be confounded with blocks
        key = (lengths.count(2), [-length for length in lengths])
        if best_key is None or key < best_key:
            best, best_key = candidate, key
    if best is None:
        raise ValueError(
            f"Every way of splitting this {len(corners)}-run design into {n_blocks} blocks confounds a main effect "
            "with blocks; use fewer blocks or a larger design."
        )

    return _blocking_from_words(design, list(best), names, subgroup)


def _blocking_from_words(design: np.ndarray, words: list[int], names: list[str], subgroup: list[int]) -> Blocking:
    """Return the blocks the signs of ``words`` define, the centre runs shared out among them in turn.

    Every product of the words is confounded with blocks, and in a fraction each carries the words
    aliased with it: its products with the defining relation's words.
    """
    k = design.shape[1]
    centre = np.all(design == 0, axis=1)
    corners = design[~centre]
    labels = np.zeros(len(design), dtype=int)
    for i, mask in enumerate(words):
        columns = [j for j in range(k) if mask >> j & 1]
        labels[~centre] += (np.prod(corners[:, columns], axis=1) > 0).astype(int) << i
    labels[centre] = np.arange(int(centre.sum())) % 2 ** len(words)
    contrasts = _span(words)[1:]
    return Blocking(
        labels=labels + 1,
        method="confounding",
        generators=[_word(m, names) for m in words],
        confounded_with=[_word(m, names) for m in contrasts],
        aliases=_alias_chains(contrasts, subgroup, names),
    )


def _alias_chains(contrasts: list[int], subgroup: list[int], names: list[str]) -> dict[str, list[str]]:
    """Return each block contrast's aliases in a fraction, shortest first; nothing for a full factorial."""
    relation = subgroup[1:]
    return {
        _word(c, names): [_word(c ^ d, names) for d in sorted(relation, key=lambda d: ((c ^ d).bit_count(), c ^ d))]
        for c in contrasts
        if relation
    }


def _word_mask(word: str, names: list[str]) -> int:
    """Return a word of factor names as a bit mask over them: ``"ADE"``, or ``"A*D*E"`` (also ``:``) for long names.

    A name that appears twice cancels, as the product of a column with itself is 1.
    """
    index = {name: i for i, name in enumerate(names)}
    text = word.strip()
    parts = re.split(r"[*:]", text) if re.search(r"[*:]", text) else list(text)
    if not text or any(part not in index for part in parts):
        raise ValueError(
            f"Block generator {word!r} must be a product of the factors {names}: write it as letters "
            '("ADE") when every name is one letter, or join the names with \'*\' ("Temp*Feed").'
        )
    mask = 0
    for part in parts:
        mask ^= 1 << index[part]
    return mask


def _given_generators(generators: list[str], names: list[str], b: int, subgroup: list[int]) -> list[int]:
    """Return the given block generators as masks, refusing a set that cannot block the design."""
    if len(generators) != b:
        raise ValueError(f"{2**b} blocks need {b} block generators, one per halving; got {len(generators)}.")
    words = [_word_mask(word, names) for word in generators]
    contrasts = _span(words)[1:]
    if len(contrasts) != 2**b - 1:
        raise ValueError(f"The block generators {generators} are not independent: one is a product of the others.")
    for word, mask in zip(generators, words, strict=True):
        if mask in subgroup:
            raise ValueError(
                f"Block generator {word!r} is constant over the runs (it is in the defining relation), so it "
                "cannot split them into blocks."
            )
    for contrast in contrasts:
        main = [d for d in subgroup if (contrast ^ d).bit_count() == 1]
        if main:
            raise ValueError(
                f"The block generators {generators} confound the main effect {_word(contrast ^ main[0], names)} "
                f"with blocks (through {_word(contrast, names)}); choose words that leave every main effect clear."
            )
    return words


def ccd_blocks(design: np.ndarray, axial: np.ndarray, n_blocks: int, names: list[str]) -> Blocking:
    """Block a central composite design: its axial runs in the last block, its cube runs in the others.

    ``axial`` marks the axial runs and the axial portion's centre runs. The cube portion
    and its centre runs form ``n_blocks - 1`` blocks, split by confounding interaction
    words when there is more than one (:func:`confounding_blocks`).

    Raises
    ------
    ValueError
        If ``n_blocks - 1`` is not a power of 2.
    """
    n_cube_blocks = n_blocks - 1
    if n_cube_blocks < 1 or n_cube_blocks & (n_cube_blocks - 1):
        raise ValueError(
            "A central composite design is blocked as its axial runs plus 1, 2, 4, ... blocks of cube runs, "
            f"so n_blocks must be 2, 3, 5, 9, ...; got n_blocks={n_blocks}."
        )
    labels = np.full(len(design), n_blocks)
    generators: list[str] = []
    confounded: list[str] = []
    if n_cube_blocks > 1:
        cube = confounding_blocks(design[~axial], n_cube_blocks, names)
        labels[~axial] = cube.labels
        generators, confounded = cube.generators, cube.confounded_with
        _warn_on_lost_interactions(design[~axial], cube.labels, names)
    else:
        labels[~axial] = 1
    return Blocking(
        labels=labels,
        method="ccd_portions",
        generators=generators,
        confounded_with=confounded,
        orthogonal=squares_orthogonal_to_blocks(design, labels),
    )


def _warn_on_lost_interactions(cube: np.ndarray, labels: np.ndarray, names: list[str]) -> None:
    """Warn when splitting a CCD's cube confounds a two-factor interaction with blocks.

    An interaction is lost when its column is constant within every cube block; a CCD
    is run for the full quadratic model, which then cannot estimate it.
    """
    corners = ~np.all(cube == 0, axis=1)
    lost = []
    for i, j in itertools.combinations(range(cube.shape[1]), 2):
        product = np.sign(cube[corners, i] * cube[corners, j])
        if all(len(np.unique(product[labels[corners] == b])) == 1 for b in np.unique(labels[corners])):
            lost.append(f"{names[i]}:{names[j]}")
    if lost:
        warnings.warn(
            f"Splitting this central composite design's cube into {len(np.unique(labels))} blocks confounds "
            f"{', '.join(lost)} with blocks, so the full quadratic model cannot estimate "
            f"{'it' if len(lost) == 1 else 'them'}. Use fewer blocks, or a larger cube.",
            UserWarning,
            stacklevel=4,
        )


def squares_orthogonal_to_blocks(design: np.ndarray, labels: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether the blocks are orthogonal to every factor's square.

    They are when each block holds the same share of each squared column's sum as of
    the runs, so the centred squares sum to zero within every block.
    """
    squares = design.astype(float) ** 2
    totals = squares.sum(axis=0)
    for block in np.unique(labels):
        inside = labels == block
        share = inside.sum() / len(labels)
        if not np.allclose(squares[inside].sum(axis=0), share * totals, rtol=tol, atol=tol):
            return False
    return True


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


def _centred_within_blocks(x: np.ndarray, labels: np.ndarray, n_blocks: int) -> np.ndarray:
    """Return ``x`` with each column centred within each block: what is left once each block has its own mean."""
    centred = x.copy()
    for block in range(n_blocks):
        rows = labels == block
        centred[rows] -= centred[rows].mean(axis=0)
    return centred


def _within_block_information(x: np.ndarray, labels: np.ndarray, n_blocks: int) -> float:
    """``log det(Xc'Xc)``, with each column of ``x`` centred within each block."""
    centred = _centred_within_blocks(x, labels, n_blocks)
    sign, logdet = np.linalg.slogdet(centred.T @ centred)
    return float(logdet) if sign > 0 else -np.inf


def _estimable(x: np.ndarray, labels: np.ndarray, n_blocks: int) -> bool:
    """Whether every column of ``x`` can be estimated once each block has its own mean."""
    return int(np.linalg.matrix_rank(_centred_within_blocks(x, labels, n_blocks))) == x.shape[1]


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
            if trial > value + _EXCHANGE_GAIN:
                value, improved = trial, True
            else:
                labels[i], labels[j] = labels[j], labels[i]
        if not improved:
            break
    return labels, value


def exchange_blocks(design: np.ndarray, n_blocks: int, rng: np.random.Generator) -> Blocking:
    """Block any design into ``n_blocks`` near-equal blocks by pairwise label swaps.

    The model is the largest of quadratic, interactions and main effects that the
    blocked design can still estimate. A model the design cannot estimate even in one
    block is passed over: every split would leave it singular, and its determinant
    would be rounding error, so the split it picked would differ between platforms.
    For the same reason a start replaces an earlier one only when it is better by more
    than rounding.

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
    one_block = np.zeros(n, dtype=int)
    for model in ("quadratic", "interactions", "main_effects"):
        x = _model_columns(design, model)
        if x.shape[1] + n_blocks > n or not _estimable(x, one_block, 1):
            continue
        best_labels, best_value = None, -np.inf
        for _ in range(_EXCHANGE_STARTS):
            labels, value = _climb(x, rng.permutation(balanced), n_blocks)
            if value > best_value + _EXCHANGE_GAIN:
                best_labels, best_value = labels, value
        if best_labels is not None and _estimable(x, best_labels, n_blocks):
            return Blocking(labels=best_labels + 1, method="exchange", model=model)
    raise ValueError(
        f"The main effects cannot be estimated once each of {n_blocks} blocks has its own mean; use fewer blocks."
    )
