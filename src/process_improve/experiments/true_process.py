# (c) Kevin Dunn, 2010-2026. MIT License.

"""A known process, for simulating the results of a designed experiment.

Before running a design on the plant, it can be run on a process whose true model is
known: simulate the responses, analyse them as if they were real, and compare what
the analysis finds with the truth. That shows what a design can and cannot recover,
such as an interaction aliased with a main effect, or a quadratic too small to detect
at the noise level, before any experiment is spent.

Also see
--------
process_improve.experiments.compare.compare_designs : the same question asked of the design alone.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from process_improve._random import check_random_state
from process_improve.experiments.designs_utils import RESERVED_COLUMN_NAMES
from process_improve.experiments.factor import DesignResult
from process_improve.experiments.optimization import _extract_b_and_B

if TYPE_CHECKING:
    from collections.abc import Mapping

#: A square, written ``I(A ** 2)`` as the analysis names it, or ``A**2`` or ``A^2``.
_SQUARE = re.compile(r"I\(\s*(\w+)\s*\*\*\s*2\s*\)|(\w+)\s*(?:\*\*|\^)\s*2")

_TERM_HELP = (
    "a term is 'Intercept', a factor such as 'A', a product of two factors such as 'A:B' or 'A*B', "
    "or a square such as 'I(A ** 2)', 'A**2' or 'A^2'"
)


def _factor_name(name: str, term: str) -> str:
    """Return a factor name used in a term, refusing anything that is not an identifier."""
    if not name.isidentifier() or name == "Intercept":
        raise ValueError(f"Cannot read the term {term!r}: {name!r} is not a factor name; {_TERM_HELP}.")
    return name


def _canonical(term: str) -> tuple[str, tuple[str, ...]]:
    """Write a term as ``analyze_experiment`` names it, with the factors it multiplies.

    Raises
    ------
    ValueError
        If the term is not the intercept, a factor, a product of two factors or a square.
    """
    text = str(term).strip()
    if text == "Intercept":
        return text, ()
    square = _SQUARE.fullmatch(text)
    if square:
        name = _factor_name(square.group(1) or square.group(2), text)
        return f"I({name} ** 2)", (name, name)
    parts = [part.strip() for part in re.split(r"[:*]", text)]
    if len(parts) == 1:
        return _factor_name(parts[0], text), (parts[0],)
    if len(parts) == 2:
        first, second = (_factor_name(part, text) for part in parts)
        if first == second:
            return f"I({first} ** 2)", (first, first)
        return f"{first}:{second}", (first, second)
    raise ValueError(f"Cannot use the term {text!r}: only second-order models are simulated; {_TERM_HELP}.")


def _canonical_coefficients(coefficients: Mapping[str, float]) -> dict[str, float]:
    """Normalise the terms' spellings, refusing a term given twice or a value that is not a number."""
    canonical: dict[str, float] = {}
    seen: dict[tuple[str, ...], str] = {}
    for term, value in coefficients.items():
        name, factors = _canonical(term)
        key = tuple(sorted(factors))
        if key in seen:
            raise ValueError(f"The terms {seen[key]!r} and {term!r} are the same term; give it once.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"The coefficient of {term!r} must be a finite number, not {value!r}.")
        seen[key] = str(term)
        canonical[name] = number
    return canonical


def _coded_frame(design: DesignResult | pd.DataFrame) -> pd.DataFrame:
    """Return a design's factor columns in coded units."""
    if isinstance(design, DesignResult):
        return design.design[list(design.factor_names)]
    if isinstance(design, pd.DataFrame):
        return design[[c for c in design.columns if c not in RESERVED_COLUMN_NAMES]]
    raise TypeError(
        "Pass the design as a DesignResult or a DataFrame of coded factor columns, so the terms can find "
        "their factors by name."
    )


@dataclass(frozen=True, eq=False)
class TrueProcess:
    r"""A process with a known second-order model, in coded units, and normal noise.

    The response at coded settings :math:`\mathbf{x}` is

    .. math::

        y = b_0 + \sum_i b_i x_i + \sum_{i<j} b_{ij} x_i x_j + \sum_i b_{ii} x_i^2 + \varepsilon,
        \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2),

    with the coefficients named as :func:`analyze_experiment` reports them, so the
    analysis of simulated runs can be set beside the truth term by term. A factor
    of the design that no term names is inert: it has no effect on the response.

    Parameters
    ----------
    coefficients : mapping of str to float
        The terms and their coefficients. A term is ``"Intercept"``, a factor
        (``"A"``), a product of two factors (``"A:B"`` or ``"A*B"``) or a square
        (``"I(A ** 2)"``, ``"A**2"`` or ``"A^2"``); each is stored under the name
        ``analyze_experiment`` uses. Terms of third order and above are refused, as
        is a term given twice under two spellings.
    sigma : float
        The standard deviation of the run-to-run noise, in the response's units.

    Attributes
    ----------
    coefficients : Mapping[str, float]
        The terms under their canonical names, read-only.
    sigma : float
        The noise standard deviation.

    Raises
    ------
    ValueError
        If a term cannot be read, a coefficient is not a finite number, or ``sigma``
        is not a positive finite number.

    Examples
    --------
    >>> import pandas as pd
    >>> process = TrueProcess({"Intercept": 62, "A": 4, "B*A": 1.5, "C^2": -2.25}, sigma=0.5)
    >>> dict(process.coefficients)
    {'Intercept': 62.0, 'A': 4.0, 'B:A': 1.5, 'I(C ** 2)': -2.25}
    >>> runs = pd.DataFrame({"A": [-1, 1, 0], "B": [1, 1, 0], "C": [0, 1, -1]})
    >>> process.mean(runs).tolist()
    [56.5, 65.25, 59.75]
    """

    coefficients: Mapping[str, float]
    sigma: float

    def __post_init__(self) -> None:
        """Normalise the terms and freeze them; check the noise level."""
        object.__setattr__(self, "coefficients", MappingProxyType(_canonical_coefficients(self.coefficients)))
        sigma = float(self.sigma)
        if not (math.isfinite(sigma) and sigma > 0):
            raise ValueError(f"sigma must be a positive number, not {self.sigma!r}.")
        object.__setattr__(self, "sigma", sigma)

    @property
    def factors(self) -> tuple[str, ...]:
        """Return the factors the terms use, in the order they first appear."""
        names: dict[str, None] = {}
        for term in self.coefficients:
            for name in _canonical(term)[1]:
                names[name] = None
        return tuple(names)

    def mean(self, design: DesignResult | pd.DataFrame) -> np.ndarray:
        """Return the noise-free response at each run of a design.

        Parameters
        ----------
        design : DesignResult or pandas.DataFrame
            The runs in coded units: a :class:`DesignResult` (its coded ``design``) or
            a DataFrame with a column per factor; ``RunOrder`` and ``Block`` are
            ignored.

        Returns
        -------
        numpy.ndarray
            One value per run.

        Raises
        ------
        ValueError
            If a term uses a factor the design does not have.
        TypeError
            If the design is not a DesignResult or a DataFrame.
        """
        frame = _coded_frame(design)
        names = [str(c) for c in frame.columns]
        missing = [f for f in self.factors if f not in names]
        if missing:
            raise ValueError(f"The process uses the factor(s) {missing}, which the design does not have: {names}.")
        terms = [{"term": term, "coefficient": value} for term, value in self.coefficients.items()]
        intercept, linear, quadratic = _extract_b_and_B(terms, names)
        x = frame.to_numpy(dtype=float)
        return intercept + x @ linear + np.einsum("ij,jk,ik->i", x, quadratic, x)

    def simulate(
        self,
        design: DesignResult | pd.DataFrame,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> pd.Series:
        """Return simulated responses: the true mean at each run plus normal noise.

        Parameters
        ----------
        design : DesignResult or pandas.DataFrame
            The runs in coded units, as for :meth:`mean`.
        random_state : int, numpy.random.Generator or None
            Seed or generator for the noise; the same integer gives the same responses.
            ``None`` draws fresh noise on every call.

        Returns
        -------
        pandas.Series
            One response per run, named ``"y"``, on the design's index.
        """
        frame = _coded_frame(design)
        rng = check_random_state(random_state)
        values = self.mean(design) + rng.normal(0.0, self.sigma, len(frame))
        return pd.Series(values, index=frame.index, name="y")
