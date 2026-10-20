# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Three-level fractional factorials, :math:`3^{k-p}`, made from generators in arithmetic modulo 3.

Each factor takes the digits 0, 1 and 2, its low, middle and high levels. A :math:`3^{k-p}` fraction
runs the full :math:`3^{k-p}` design in its base factors and sets each of the :math:`p` derived
factors by a *generator*, a sum of base digits modulo 3 (Montgomery, *Design and Analysis of
Experiments*, section 9.3):

.. code-block:: text

    "D=A2BC2"   means   x_D = 2 x_A + x_B + 2 x_C  (mod 3)

A generator gives a word of the defining relation: here :math:`A^2BC^2D^2`, the derived factor moved
across with the exponent :math:`-1 = 2`. A word and its square are the same component, written with
its first exponent 1, so this one is :math:`AB^2CD`, Montgomery's ``I = AB2CD`` (all exponents times
2, modulo 3). The relation holds every product of powers of the generator words, :math:`(3^p - 1)/2`
components in all, and the fewest letters in any of them is the design's resolution. An effect is
aliased with its products with each component and with that component's square.

Exponents are written as a digit after a one-letter factor name (``AB2C``), or as ``^2`` or ``²``;
names longer than one letter are joined with ``*`` (``Temp*Feed^2``).
"""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor

#: The arithmetic of a three-level fraction: its generators and defining relation work modulo 3.
MODULUS = 3

#: Most components of a defining relation listed in the metadata, as for a two-level fraction: six
#: generators give 364 components, seven give 1093.
_MAX_LISTED_COMPONENTS = 1023

#: Most generators whose relation is worked out, for its resolution: ten give 3^10 products of powers
#: to examine, enough for the 13 factors of a 27-run fraction.
_MAX_EXAMINED_GENERATORS = 10

#: A letter of a word of one-letter names, with its exponent: ``B``, ``B2``, ``B^2`` or ``B²``.
_LETTER = re.compile(r"([^\d^²])(?:\^?([12])|(²))?")

#: A factor of a word of longer names, joined by ``*``, with its exponent: ``Feed``, ``Feed^2``, ``Feed²``.
_NAMED = re.compile(r"(.+?)(?:\^([12])|(²))?")


def dispatch_three_level_fraction(
    factors: list[Factor], generators: list[str], codes: list[np.ndarray], resolution: int | None = None
) -> tuple[np.ndarray, dict]:
    """Build a :math:`3^{k-p}` fraction of three-level factors from its generators.

    Parameters
    ----------
    factors : list[Factor]
        Factors with three levels each: continuous factors with three ``levels``, or categorical
        factors with three levels.
    generators : list[str]
        One generator for each derived factor, such as ``"D=A2BC2"``, its right-hand side a word in
        the base factors.
    codes : list[numpy.ndarray]
        The three coded levels of each factor, in the order of the digits 0, 1 and 2.
    resolution : int or None
        The least resolution the design must reach; generators that fall short raise ``ValueError``.

    Returns
    -------
    tuple[np.ndarray, dict]
        The coded design matrix, the first base factor changing fastest, and metadata with
        ``generators_used``, ``modulus`` (3), ``levels_per_factor``, ``resolution``,
        ``defining_relation`` (each component as ``"I=AB2CD"``, fewest letters first) and
        ``aliases`` (each main effect's alias chain). A relation of more than 1023 components
        (more than six generators) is not listed: ``defining_relation_note`` says so, and more
        than ten generators leave the resolution unknown as well.

    Raises
    ------
    ValueError
        If a generator is not of the form ``"D=A2BC2"`` (one derived factor on the left, a word in
        the base factors on the right), a factor is derived twice, or the design falls short of
        *resolution*.
    """
    names = [f.name for f in factors]
    derived, weights = _parse_generators(generators, names)
    base = [j for j in range(len(names)) if j not in derived]
    digits = np.zeros((MODULUS ** len(base), len(names)), dtype=int)
    digits[:, base] = np.array(list(itertools.product(range(MODULUS), repeat=len(base))))[:, ::-1]
    for column, weight in zip(derived, weights, strict=True):
        digits[:, column] = digits @ weight % MODULUS
    meta: dict = {"generators_used": list(generators), "modulus": MODULUS, "levels_per_factor": [3] * len(names)}
    meta.update(_relation_metadata([_generator_word(c, w) for c, w in zip(derived, weights, strict=True)], names))
    achieved = meta.get("resolution")
    if resolution is not None and (achieved is None or achieved < resolution):
        found = "a design whose resolution is not worked out" if achieved is None else f"a resolution-{achieved} design"
        msg = (
            f"The generators {list(generators)} give {found}, short of the resolution={resolution} asked for. "
            "Pass generators that reach the resolution."
        )
        raise ValueError(msg)
    coded = np.column_stack([codes[j][digits[:, j]] for j in range(len(names))])
    return coded.astype(float), meta


def _relation_metadata(generator_words: list[np.ndarray], names: list[str]) -> dict:
    """Return the resolution, the defining relation and the main effects' alias chains, as far as listed."""
    n_components = (MODULUS ** len(generator_words) - 1) // 2
    if len(generator_words) > _MAX_EXAMINED_GENERATORS:
        return {"defining_relation_note": f"The defining relation has {n_components} components, too many to examine."}
    words = _relation(np.array(generator_words))
    meta: dict = {"resolution": int(np.count_nonzero(words, axis=1).min())}
    if n_components > _MAX_LISTED_COMPONENTS:
        meta["defining_relation_note"] = f"The defining relation has {n_components} components, too many to list."
        return meta
    words = np.array(sorted(words.tolist(), key=_order))
    identity = np.eye(len(names), dtype=int)
    meta["defining_relation"] = ["I=" + _component(word, names) for word in words]
    meta["aliases"] = {name: _aliases(identity[j], words, names) for j, name in enumerate(names)}
    return meta


def _parse_generators(generators: list[str], names: list[str]) -> tuple[list[int], list[np.ndarray]]:
    """Return each generator's derived factor and the weights it puts on the base factors."""
    derived, weights = [], []
    for generator in generators:
        lhs, equals, rhs = generator.partition("=")
        left = _exponents(lhs, names, generator)
        if not equals or np.count_nonzero(left) != 1 or left.max() != 1:
            msg = f"Generator {generator!r} must have the form 'D=A2BC2': one derived factor, then a word in others."
            raise ValueError(msg)
        derived.append(int(np.argmax(left)))
        weights.append(_exponents(rhs, names, generator))
    repeated = sorted({names[column] for column in derived if derived.count(column) > 1})
    if repeated:
        msg = f"Each derived factor takes one generator; {repeated} have more than one."
        raise ValueError(msg)
    for generator, weight in zip(generators, weights, strict=True):
        named = [names[j] for j in np.flatnonzero(weight) if j in derived]
        if named or not weight.any():
            reason = f"{named} are derived factors themselves" if named else "its right-hand side names no factor"
            msg = f"Generator {generator!r} must set its factor from the base factors; {reason}."
            raise ValueError(msg)
    return derived, weights


def _exponents(word: str, names: list[str], generator: str) -> np.ndarray:
    """Return the exponent of each factor in a word such as ``"A2BC2"`` or ``"Temp*Feed^2"``, modulo 3."""
    text = word.replace(" ", "")
    matches: list[re.Match[str] | None]
    if any(len(name) > 1 for name in names):
        matches = [_NAMED.fullmatch(token) for token in text.split("*")] if text else []
    else:
        text = text.replace("*", "")
        letters = list(_LETTER.finditer(text))
        # None stands for a character no letter took, such as the 3 of "A3B".
        matches = [*letters] if "".join(m.group(0) for m in letters) == text else [*letters, None]
    exponents = np.zeros(len(names), dtype=int)
    for match in matches:
        if match is None or match.group(1) not in names:
            what = "cannot be read" if match is None else f"names {match.group(1)!r}, which is not a factor"
            msg = (
                f"Generator {generator!r} {what}. Write each side as a word in the factors {names}, such as "
                "'A2BC2', or 'Temp*Feed^2' when a name is longer than one letter."
            )
            raise ValueError(msg)
        exponents[names.index(match.group(1))] += 2 if match.group(2) == "2" or match.group(3) else 1
    return exponents % MODULUS


def _generator_word(column: int, weights: np.ndarray) -> np.ndarray:
    """Return the defining word of ``x_column = weights . x``: the weights, with 2 (-1) on the derived factor."""
    word = weights.copy()
    word[column] = MODULUS - 1
    return _normalised(word)[0]


def _normalised(words: np.ndarray) -> np.ndarray:
    """Return each word (one per row) written with its first exponent 1: the word itself or its square."""
    words = np.atleast_2d(words) % MODULUS
    first = words[np.arange(len(words)), np.argmax(words != 0, axis=1)]
    return words * np.where(first == 0, 1, first)[:, None] % MODULUS  # 2 * 2 = 4 = 1 modulo 3


def _relation(generator_words: np.ndarray) -> np.ndarray:
    """Return every component of the defining relation, one per row: each product of powers of the words."""
    powers = np.array(list(itertools.product(range(MODULUS), repeat=len(generator_words))))[1:]
    return np.unique(_normalised(powers @ generator_words), axis=0)


def _aliases(effect: np.ndarray, words: np.ndarray, names: list[str]) -> list[str]:
    """Return the components an effect is aliased with: its products with each component and its square."""
    chain = np.unique(_normalised(np.vstack([effect + words, effect + 2 * words])), axis=0)
    return [_component(np.array(word), names) for word in sorted(chain.tolist(), key=_order)]


def _order(word: list[int]) -> tuple[int, list[int], list[int]]:
    """Sort components by their number of letters, then by the factors in them, then by the exponents."""
    return int(np.count_nonzero(word)), [0 if e else 1 for e in word], list(word)


def _component(word: np.ndarray, names: list[str]) -> str:
    """Write a component as Montgomery does, ``AB2CD``, or ``Temp*Feed^2`` for longer names."""
    long_names = any(len(name) > 1 for name in names)
    parts = [
        name + ("" if power == 1 else ("^2" if long_names else "2"))
        for name, power in zip(names, word, strict=True)
        if power
    ]
    return ("*" if long_names else "").join(parts)
