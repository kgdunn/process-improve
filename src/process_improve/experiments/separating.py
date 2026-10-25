# (c) Kevin Dunn, 2010-2026. MIT License.

"""A straight limit proposed from settings: the cut that rules out the settings that cannot be run.

Someone who knows which settings their process cannot run (the corner where it is both
hottest and fastest, say) often knows it better than they can write the inequality. Two
functions turn that knowledge into a linear :class:`~process_improve.experiments.Constraint`
in actual units, written for a person to read (``"T + 25 * P <= 250"``):

- :func:`separating_constraint` takes the settings that cannot be run and those that can,
  and proposes the one straight cut that rules out the first and keeps the second, as far
  from both as it can be;
- :func:`constraint_through_cuts` takes one corner of the factor box and, for each factor,
  how far it can go from that corner with the others held there, and writes the limit
  through those cut points.

When no straight cut separates the two sets of settings, :func:`separating_constraint`
raises :class:`NotSeparableError`, which names the settings that lie inside the other set.

Only numpy, ``scipy.optimize`` (``linprog`` with HiGHS, ``nnls`` and SLSQP), pandas (to
accept the settings as a DataFrame) and the standard library are used, and every step is
deterministic, so a proposal is the same on every platform, Pyodide included.
"""

from __future__ import annotations

import keyword
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize, nnls

from process_improve.experiments._lm import suggest_identifier
from process_improve.experiments._region_geometry import round_significant
from process_improve.experiments.designs_constrained import rename_factors
from process_improve.experiments.factor import Constraint, Factor, FactorType

logger = logging.getLogger(__name__)

#: Margin of the separability linear program at or below which no straight cut separates the two sets.
_SEPARABLE = 1e-9
#: Weight of the offset ``c`` in the least-distance problem: small, so the plane found is (to within this
#: fraction) the one of maximum margin, which leaves ``c`` free.
_OFFSET_WEIGHT = 1e-4
#: Relative duality gap and constraint violation (in units of the margin) that certify a least-distance solution.
_CERTIFY_GAP = 1e-8
_CERTIFY_SLACK = 1e-3
#: Coefficients smaller than this fraction of the sum of their sizes are set to zero (coded units).
_SNAP = 1e-6
#: Most the rounded coefficients (coded units, scaled to a sum of sizes of 1) may move from the plane found.
_MOVE = 0.01
#: Most significant digits tried for the coefficients: at 15 digits a float is written as it is.
_MAX_DIGITS = 15
#: Significant digits the scaled coefficients are rounded to before fewer are tried. The solver's last-bit noise
#: differs between platforms, so a coefficient exactly halfway between two candidates (22.5, rounded to two digits)
#: would otherwise round up on one and down on another, and two equal coefficients could round apart.
_CLEAN_DIGITS = 10
#: Most significant digits a number of the limit through cut points is written with.
_CUT_DIGITS = 12
#: Distance (coded units) within which a point counts as a corner of the box or as equal to another point.
_TOL = 1e-12
#: ``scipy.optimize.linprog`` status of a problem with a solution.
_SOLVED = 0


class NotSeparableError(ValueError):
    """No straight cut rules out the settings that cannot be run and keeps all those that can.

    Raised by :func:`separating_constraint`. It names the settings to blame, when some are:
    a setting that cannot be run but lies between settings that can (inside their convex
    hull), so any straight cut that rules it out rules out one of them too; and a setting
    that can be run but lies between settings that cannot, so any straight cut that rules
    them out rules it out too. When the two sets overlap without any one setting lying
    inside the other (the four corners of a square marked alternately, say), both lists are
    empty.

    Parameters
    ----------
    message : str
        The explanation shown to the reader.
    inside_can : Sequence[int]
        Indices into ``cannot`` of the settings that lie inside the hull of ``can``.
    inside_cannot : Sequence[int]
        Indices into ``can`` of the settings that lie inside the hull of ``cannot``.

    Attributes
    ----------
    inside_can : list[int]
        As the parameter, in increasing order.
    inside_cannot : list[int]
        As the parameter, in increasing order.
    """

    def __init__(self, message: str, inside_can: Sequence[int] = (), inside_cannot: Sequence[int] = ()) -> None:
        super().__init__(message)
        self.inside_can = sorted(int(i) for i in inside_can)
        self.inside_cannot = sorted(int(i) for i in inside_cannot)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle with the indices, which ``ValueError`` alone would drop."""
        return type(self), (str(self), self.inside_can, self.inside_cannot)


@dataclass(frozen=True)
class _Box:
    """The factor box: the names, and the ranges that set the coded units."""

    names: list[str]
    low: np.ndarray
    high: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        return (self.low + self.high) / 2.0

    @property
    def half(self) -> np.ndarray:
        return (self.high - self.low) / 2.0

    def coded(self, actual: np.ndarray) -> np.ndarray:
        """Return actual settings in coded units, ``-1`` at each factor's low end and ``+1`` at its high end."""
        return (actual - self.centre) / self.half


def _box(factors: Sequence[Factor]) -> _Box:
    """Return the box of ``factors``, which must be continuous factors whose names can be written in a limit."""
    if not factors:
        raise ValueError("A limit needs at least one factor.")
    names = []
    for factor in factors:
        if factor.type != FactorType.continuous:
            raise ValueError(
                f"Factor {factor.name!r} is {factor.type.value}: a limit is proposed over continuous factors only."
            )
        if not factor.name.isidentifier() or keyword.iskeyword(factor.name):
            raise ValueError(
                f"Factor {factor.name!r} cannot be named in a limit: its name is not a valid identifier. "
                f"Try {suggest_identifier(factor.name)!r}."
            )
        names.append(factor.name)
    if len(set(names)) < len(names):
        raise ValueError(f"The factors must have different names; got {names}.")
    low = np.array([f.low for f in factors], dtype=float)
    high = np.array([f.high for f in factors], dtype=float)
    return _Box(names, low, high)


def _as_points(
    points: np.ndarray | pd.DataFrame | Sequence[Sequence[float]] | Sequence[Mapping[str, float]], box: _Box, label: str
) -> np.ndarray:
    """Return ``points`` as an ``(n, k)`` array in actual units, checked to be finite and inside the box.

    ``points`` is a 2-D array-like with one column per factor, in factor order, a pandas
    DataFrame with a column per factor name, or a sequence of mappings from factor name
    to value.
    """
    k = len(box.names)
    rows: Any = points
    if isinstance(points, pd.DataFrame):
        missing = [n for n in box.names if n not in points.columns]
        if missing:
            raise ValueError(f"The settings in {label!r} have no column for the factors {missing}.")
        rows = points[box.names].to_numpy(dtype=float)
    elif isinstance(points, Sequence) and points and all(isinstance(p, Mapping) for p in points):
        settings = cast("Sequence[Mapping[str, float]]", points)
        for p in settings:
            missing = [n for n in box.names if n not in p]
            if missing:
                raise ValueError(f"A setting in {label!r} gives no value for the factors {missing}.")
        rows = [[p[n] for n in box.names] for p in settings]
    array = np.asarray(rows, dtype=float)
    if array.size == 0:
        return np.empty((0, k))
    if array.ndim != 2 or array.shape[1] != k:
        raise ValueError(
            f"{label!r} must hold one setting per row, each with a value for the {k} factors {box.names}; "
            f"got an array of shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Every value in {label!r} must be a finite number.")
    margin = 1e-9 * (box.high - box.low)
    outside = np.flatnonzero(np.any((array < box.low - margin) | (array > box.high + margin), axis=1))
    if len(outside):
        where = _setting(array[outside[0]], box)
        raise ValueError(f"Setting {int(outside[0])} of {label!r}, {where}, is outside the factors' ranges.")
    return np.clip(array, box.low, box.high)


def _setting(point: np.ndarray, box: _Box) -> str:
    """Return a setting as ``"T = 200, P = 3"`` for a message."""
    return ", ".join(f"{name} = {float(value):g}" for name, value in zip(box.names, point, strict=True))


# ---------------------------------------------------------------------------
# Step 1: is there a straight cut at all?
# ---------------------------------------------------------------------------


def _separation(cannot: np.ndarray, can: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return ``(w, c, t)`` maximising ``t`` with ``w @ p - c >= t`` on ``cannot`` and ``<= -t`` on ``can``.

    Points are in coded units and ``-1 <= w_j <= 1``, so ``t`` is bounded; the two sets
    can be separated by a straight cut exactly when the largest ``t`` is positive.
    """
    k = cannot.shape[1]
    ones_p, ones_q = np.ones((len(cannot), 1)), np.ones((len(can), 1))
    a_ub = np.vstack([np.hstack([-cannot, ones_p, ones_p]), np.hstack([can, -ones_q, ones_q])])
    cost = np.zeros(k + 2)
    cost[-1] = -1.0
    bounds = [(-1.0, 1.0)] * k + [(None, None), (None, None)]
    result = linprog(cost, A_ub=a_ub, b_ub=np.zeros(len(a_ub)), bounds=bounds, method="highs")
    if result.status != _SOLVED:  # pragma: no cover - bounded and feasible (w = 0, c = 0, t = 0) by construction
        raise RuntimeError(f"The separability linear program did not solve: {result.message}")
    x = np.asarray(result.x)
    return x[:k], float(x[k]), float(x[k + 1])


def _in_hull(point: np.ndarray, others: np.ndarray) -> bool:
    """Return whether ``point`` lies in the convex hull of ``others`` (coded units, all inside the box; some given)."""
    if np.any(np.all(np.abs(others - point) <= _TOL, axis=1)):
        return True
    if np.any(point < others.min(axis=0) - _TOL) or np.any(point > others.max(axis=0) + _TOL):
        return False
    if np.all(np.abs(np.abs(point) - 1.0) <= _TOL):  # a corner of the box is in no hull of other points of the box
        return False
    a_eq = np.vstack([others.T, np.ones((1, len(others)))])
    b_eq = np.append(point, 1.0)
    result = linprog(np.zeros(len(others)), A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return bool(result.status == _SOLVED)


def _not_separable(cannot: np.ndarray, can: np.ndarray, box: _Box) -> NotSeparableError:
    """Return the error naming the settings of each set that lie inside the other's hull."""
    inside_can = [i for i, p in enumerate(cannot) if _in_hull(p, can)]
    inside_cannot = [i for i, q in enumerate(can) if _in_hull(q, cannot)]
    reasons = []
    if inside_can:
        reasons.append(
            f"cannot[{inside_can[0]}] ({_setting(box.centre + cannot[inside_can[0]] * box.half, box)}) lies between "
            "settings that can be run, so a cut that rules it out rules some of them out too"
        )
    if inside_cannot:
        reasons.append(
            f"can[{inside_cannot[0]}] ({_setting(box.centre + can[inside_cannot[0]] * box.half, box)}) lies between "
            "settings that cannot be run, so a cut that rules them out rules it out too"
        )
    if not reasons:
        reasons.append("the two sets are interleaved, so every straight cut leaves a setting on the wrong side")
    message = "No straight cut rules out the settings that cannot be run and keeps those that can: " + "; ".join(
        reasons
    )
    return NotSeparableError(message + ".", inside_can, inside_cannot)


# ---------------------------------------------------------------------------
# Step 2: the cut of maximum margin, which is unique
# ---------------------------------------------------------------------------


def _least_distance(g: np.ndarray, h: np.ndarray) -> np.ndarray | None:
    """Return ``z`` minimising ``||z||`` subject to ``g @ z >= h``, or None when the solution cannot be certified.

    The least-distance problem is solved through the non-negative least-squares problem
    ``min ||E u - f||, u >= 0`` with ``E = [g.T; h]`` and ``f = (0, ..., 0, 1)`` (Lawson and
    Hanson, 1974, chapter 23): ``z = g.T u / (1 - h @ u)``. The answer is kept only when it
    meets every inequality and its duality gap is zero, which certifies it optimal: on
    degenerate problems, such as the corners of a box, the solver's answer can miss the
    optimum.

    References
    ----------
    Lawson, C. L. and Hanson, R. J. (1974) *Solving Least Squares Problems*. Prentice-Hall.
    """
    e = np.vstack([g.T, h[None, :]])
    f = np.zeros(g.shape[1] + 1)
    f[-1] = 1.0
    try:
        u, _ = nnls(e, f)
    except RuntimeError:  # pragma: no cover - an iteration limit, in versions of scipy that raise for it
        return None
    denominator = 1.0 - float(h @ u)
    if denominator <= 1e-12:  # the inequalities have no common point
        return None
    z = g.T @ u / denominator
    size = float(z @ z)
    gap = size - float(u @ h) / denominator
    if not (size > 0.0 and np.min(g @ z - h) >= -_CERTIFY_SLACK and abs(gap) <= _CERTIFY_GAP * size):
        return None
    return z


def _max_margin_plane(cannot: np.ndarray, can: np.ndarray, start: tuple[np.ndarray, float] | None = None) -> np.ndarray:
    """Return ``w`` of the plane ``w @ x = c`` of maximum margin between two separable sets (coded units).

    It minimises ``||w||`` subject to ``w @ p - c >= 1`` on ``cannot`` and ``w @ q - c <= -1``
    on ``can``: the support vector machine of hard margin, whose ``w`` is unique. The
    problem is a least-distance problem once ``c`` is carried as a variable with the small
    weight ``_OFFSET_WEIGHT`` (it is free in the original), and is solved through
    :func:`_least_distance`; SLSQP from the separating plane ``start`` takes over when that
    answer cannot be certified. Coefficients below ``_SNAP`` of the sum of their sizes are
    set to zero, smallest first and each only while the plane still separates the sets, so
    a factor the sets do not depend on gets exactly zero.

    Parameters
    ----------
    cannot, can : np.ndarray
        ``(n, k)`` and ``(m, k)`` points in coded units, separable by a straight cut.
    start : tuple[np.ndarray, float] or None
        A separating plane ``(w, c)`` with ``w @ p - c > 0 > w @ q - c``, from the
        separability linear program; found here when not given.

    Returns
    -------
    np.ndarray
        ``w``, scaled so the margin is 1 on each side.
    """
    k = cannot.shape[1]
    weight = _OFFSET_WEIGHT
    g = np.vstack(
        [
            np.hstack([cannot, -np.ones((len(cannot), 1)) / weight]),
            np.hstack([-can, np.ones((len(can), 1)) / weight]),
        ]
    )
    z = _least_distance(g, np.ones(len(g)))
    w = None if z is None else z[:k]
    if w is None or not _separates(w, cannot, can):
        if start is None:
            w_lp, c_lp, t = _separation(cannot, can)
            start = (w_lp / t, c_lp / t)
        w = _slsqp_plane(cannot, can, start)
    small = np.flatnonzero(np.abs(w) < _SNAP * np.abs(w).sum())
    for j in small[np.argsort(np.abs(w[small]), kind="stable")]:  # smallest first, each only if the cut still holds
        snapped = w.copy()
        snapped[j] = 0.0
        if _separates(snapped, cannot, can):
            w = snapped
    gap = float(np.min(cannot @ w) - np.max(can @ w))
    return w * 2.0 / gap


def _separates(w: np.ndarray, cannot: np.ndarray, can: np.ndarray) -> bool:
    """Return whether the direction ``w`` puts every point of ``cannot`` beyond every point of ``can``."""
    return bool(np.all(np.isfinite(w)) and np.min(cannot @ w) > np.max(can @ w))


def _slsqp_plane(cannot: np.ndarray, can: np.ndarray, start: tuple[np.ndarray, float]) -> np.ndarray:
    """Return ``w`` of the plane of maximum margin by SLSQP from a separating plane, or that plane's ``w``."""
    k = cannot.shape[1]
    signed = np.vstack([np.hstack([cannot, -np.ones((len(cannot), 1))]), np.hstack([-can, np.ones((len(can), 1))])])
    result = minimize(
        lambda z: float(z[:k] @ z[:k]),
        np.append(start[0], start[1]),
        jac=lambda z: np.append(2.0 * z[:k], 0.0),
        constraints=[{"type": "ineq", "fun": lambda z: signed @ z - 1.0, "jac": lambda _z: signed}],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    w = np.asarray(result.x[:k], dtype=float)
    if _separates(w, cannot, can):
        return w
    logger.warning("The plane of maximum margin was not found; the separating plane of the linear program is used.")
    return np.asarray(start[0], dtype=float)


# ---------------------------------------------------------------------------
# Writing the cut for a person: rounding, the right-hand side, the text
# ---------------------------------------------------------------------------


def _exact(value: float) -> Fraction:
    """Return a float as the decimal its shortest representation shows, exactly."""
    return Fraction(repr(float(value)))


def _decimal_text(value: Fraction) -> str:
    """Return a terminating decimal in positional notation, a whole number without ``".0"``."""
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value.denominator == 1:
        return f"{sign}{value.numerator}"
    places = 0
    while (value * 10**places).denominator != 1:
        places += 1
    digits = str((value * 10**places).numerator).rjust(places + 1, "0")
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def _terminates(value: Fraction) -> bool:
    """Return whether a fraction has a finite decimal expansion (its denominator has no prime factor but 2 and 5)."""
    denominator = value.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    return denominator == 1


def _digits_written(coefficients: Sequence[Fraction]) -> int:
    """Return the digits it takes to write the coefficients: none for a 1, which is left out, or for a 0."""
    count = 0
    for c in coefficients:
        if c != 0 and abs(c) != 1:
            count += sum(ch.isdigit() for ch in _decimal_text(abs(c)))
    return count


def _significant(value: Fraction) -> int:
    """Return the significant digits of a terminating decimal."""
    return len(_decimal_text(abs(value)).replace(".", "").strip("0")) or 1


def _roundest(low: Fraction, high: Fraction) -> Fraction:
    """Return the roundest number in the middle fifth of ``(low, high)``, the one nearest the middle among equals.

    The roundest number is a multiple of the largest step ``1``, ``2`` or ``5`` times a
    power of ten that has a multiple there: zero when the interval holds it, then multiples
    of 1000 before 500, 200, 100, 50 and so on. Equally near multiples go to the smaller
    in size. The arithmetic is exact.
    """
    middle = (low + high) / 2
    half_width = (high - low) / 10
    left, right = middle - half_width, middle + half_width
    if left <= 0 <= right:
        return Fraction(0)
    top = max(abs(left), abs(right))
    # The power of ten at or below ``top``, exactly: the digits of its numerator and denominator place it within one.
    exponent = len(str(top.numerator)) - len(str(top.denominator))
    if Fraction(10) ** exponent > top:
        exponent -= 1
    while True:  # a step no wider than the interval always has a multiple in it, so this ends
        for factor in (5, 2, 1):
            step = factor * Fraction(10) ** exponent
            first, last = math.ceil(left / step), math.floor(right / step)
            if first <= last:
                near = {min(max(m, first), last) for m in (math.floor(middle / step), math.ceil(middle / step))}
                return min((m * step for m in near), key=lambda v: (abs(v - middle), abs(v), v))
        exponent -= 1


def _gap(
    coefficients: np.ndarray, cannot: np.ndarray, can: np.ndarray, exact: Sequence[Fraction]
) -> tuple[Fraction, Fraction] | None:
    """Return the exact ``(largest over can, smallest over cannot)`` of ``coefficients @ x``, or None if they overlap.

    The values are found in floating point, and only the points within rounding error of
    each extreme are recomputed exactly, from the decimals their floats show.
    """
    over_cannot, over_can = cannot @ coefficients, can @ coefficients
    reach = np.abs(coefficients) @ np.max(np.abs(np.vstack([cannot, can])), axis=0)
    tol = 1e-9 * max(float(reach), 1e-300)
    if over_cannot.min() < over_can.max() - tol:
        return None

    def dot(point: np.ndarray) -> Fraction:
        return sum((c * _exact(x) for c, x in zip(exact, point, strict=True) if c), Fraction(0))

    low = max(dot(can[i]) for i in np.flatnonzero(over_can >= over_can.max() - tol))
    high = min(dot(cannot[i]) for i in np.flatnonzero(over_cannot <= over_cannot.min() + tol))
    return (low, high) if low < high else None


def _rounded(a: np.ndarray, half: np.ndarray, cannot: np.ndarray, can: np.ndarray) -> tuple[list[Fraction], Fraction]:
    """Round the plane ``a @ x = .`` (actual units) for a reader, keeping every point on its side.

    Two scalings are tried: the smallest coefficient made 1, and the largest; each is
    rounded by :func:`_round_scaling`. The scaling whose coefficients take fewer digits to
    write is kept, the smallest made 1 on a tie.

    Returns
    -------
    tuple[list[Fraction], Fraction]
        The coefficients and right-hand side, exactly as they are written; the points of
        ``cannot`` lie above it, those of ``can`` below.
    """
    target = a * half / np.abs(a * half).sum()
    sizes = np.abs(a[a != 0])
    written = [_round_scaling(a / scale, half, target, cannot, can) for scale in (sizes.min(), sizes.max())]
    return min(written, key=lambda form: _digits_written(form[0]))


def _round_scaling(
    scaled: np.ndarray, half: np.ndarray, target: np.ndarray, cannot: np.ndarray, can: np.ndarray
) -> tuple[list[Fraction], Fraction]:
    """Round the coefficients ``scaled`` to the fewest significant digits that keep the plane where it was.

    The digits are the fewest at which the coefficients, in coded units and scaled to a
    sum of sizes of 1, move by at most ``_MOVE`` in total from ``target`` and still put
    every point of ``cannot`` above every point of ``can``, which is checked exactly (see
    :func:`_gap`). The right-hand side is then the roundest number in the middle fifth of
    the gap between the two (see :func:`_roundest`), so every point lies strictly on its
    side.

    Up to ``_CLEAN_DIGITS`` digits, the coefficients rounded are ``scaled`` first rounded to
    ``_CLEAN_DIGITS`` digits, which removes the solver's last-bit noise: an exact tie then
    rounds the same way on every platform (half to even), and equal coefficients round
    alike. Only a cut that needs more digits than that is rounded from ``scaled`` itself.
    """
    clean = round_significant(scaled, _CLEAN_DIGITS)
    for digits in range(1, _MAX_DIGITS + 1):
        coefficients = round_significant(clean if digits <= _CLEAN_DIGITS else scaled, digits)
        coded = coefficients * half
        if np.abs(coded / np.abs(coded).sum() - target).sum() > _MOVE + 1e-12:
            continue
        exact = [_exact(c) for c in coefficients]
        gap = _gap(coefficients, cannot, can, exact)
        if gap is not None:
            return exact, _roundest(*gap)
    raise RuntimeError(  # pragma: no cover - at 15 digits the plane is where it was, and separates by its margin
        "The proposed limit could not be written without moving a setting to the wrong side."
    )


def _expression(names: Sequence[str], coefficients: Sequence[Fraction], rhs: Fraction) -> str:
    """Write ``coefficients @ x <= rhs`` (the excluded side above) as a limit's canonical text.

    The factors keep their order; the first term is made positive, turning ``<=`` into
    ``>=`` when its coefficient was negative; a coefficient of 1 is left out, and whole
    numbers are written without ``".0"``. The text is then canonicalised by parsing it and
    writing it back (``ast.unparse``), which also checks it is a limit the library reads.
    """
    first = next(c for c in coefficients if c != 0)
    operator = "<="
    if first < 0:
        coefficients, rhs, operator = [-c for c in coefficients], -rhs, ">="
    terms: list[str] = []
    for name, c in zip(names, coefficients, strict=True):
        if c == 0:
            continue
        size = _decimal_text(abs(c))
        term = name if size == "1" else f"{size} * {name}"
        terms.append(term if not terms else f"{'-' if c < 0 else '+'} {term}")
    return rename_factors(f"{' '.join(terms)} {operator} {_decimal_text(rhs)}", {})


# ---------------------------------------------------------------------------
# The two public functions
# ---------------------------------------------------------------------------


def separating_constraint(
    factors: Sequence[Factor],
    cannot: np.ndarray | pd.DataFrame | Sequence[Sequence[float]] | Sequence[Mapping[str, float]],
    can: np.ndarray | pd.DataFrame | Sequence[Sequence[float]] | Sequence[Mapping[str, float]],
) -> Constraint:
    """Propose the straight limit that rules out the settings that cannot be run and keeps those that can.

    The limit is found in two steps, in coded units (each factor's range mapped to
    ``[-1, 1]``, so a factor counts by its range, not by its units):

    1. Whether any straight cut separates the two sets: a linear program maximises the
       margin ``t`` of a plane ``w @ x = c`` with every ``|w_j| <= 1``. When the best margin
       is not positive, :class:`NotSeparableError` is raised; one small linear program per
       setting finds the settings that lie inside the other set's convex hull, to name them.
    2. The cut itself: the plane of maximum margin, which lies as far from the nearest
       settings of both sets as it can (the hard-margin support vector machine). It is
       unique, so a symmetric pattern gives a symmetric cut, and a factor the pattern does
       not depend on gets exactly zero and drops out of the limit.

    The plane is then written for a reader, in actual units. Its coefficients are scaled so
    the smallest is 1, and also so the largest is 1, and each scaling is rounded to the
    fewest significant digits at which the coefficients (coded, scaled to a sum of sizes of
    1) move by at most 1% in total and every setting stays on its side; the scaling whose
    coefficients take fewer digits to write is kept. The rounding starts from the
    coefficients rounded to 10 significant digits, so the solver's last-bit noise, which
    differs between platforms, cannot decide which way a coefficient exactly halfway
    between two candidates goes: it goes to the even one, and equal coefficients stay
    equal. The right-hand side is the roundest number (a multiple of 1, 2 or 5 times a
    power of ten, as large as fits) in the middle fifth of the gap between the two sets,
    nearest its middle. Every setting is then checked, in exact decimal arithmetic, to be
    strictly on its side.

    Parameters
    ----------
    factors : Sequence[Factor]
        The continuous factors the settings are given for; their ranges set the coded
        units, and their names are written into the limit.
    cannot : np.ndarray, pandas.DataFrame, or a sequence of sequences or of mappings
        The settings that cannot be run, in actual units: one row per setting with a value
        per factor in factor order (a 2-D array or a list of lists), a pandas DataFrame
        with a column per factor, or a list of ``{name: value}`` mappings. Each must lie
        within the factors' ranges.
    can : np.ndarray, pandas.DataFrame, or a sequence of sequences or of mappings
        The settings that can be run, in the same forms. Settings in neither list (another
        group of marked corners, say) do not count either way.

    Returns
    -------
    Constraint
        A linear constraint in actual units, such as ``"T + 25 * P <= 250"``, under which
        every setting of ``cannot`` is ruled out and every setting of ``can`` is allowed,
        none of them on the boundary. Its first term is positive.

    Raises
    ------
    NotSeparableError
        When no straight cut separates the two sets.
    ValueError
        When either set is empty, a setting is outside the factors' ranges or has the
        wrong number of values, or a factor is not continuous or has a name that cannot be
        written in a limit.

    See Also
    --------
    constraint_through_cuts : the limit through a cut point on each factor from one corner.

    Examples
    --------
    >>> import itertools
    >>> factors = [
    ...     Factor(name="T", low=150, high=200),
    ...     Factor(name="P", low=1, high=3),
    ...     Factor(name="F", low=10, high=20),
    ... ]
    >>> corners = [list(c) for c in itertools.product((150, 200), (1, 3), (10, 20))]
    >>> cannot = [c for c in corners if c[0] == 200 and c[1] == 3]  # T and P both high
    >>> can = [c for c in corners if c not in cannot] + [[175, 2, 15]]  # the rest, and the centre
    >>> separating_constraint(factors, cannot, can).expression
    'T + 25 * P <= 250'
    """
    box = _box(factors)
    cannot_actual, can_actual = _as_points(cannot, box, "cannot"), _as_points(can, box, "can")
    if not len(cannot_actual):
        raise ValueError("Give at least one setting that cannot be run: with none, there is nothing to rule out.")
    if not len(can_actual):
        raise ValueError(
            "Give at least one setting that can be run: with none, any cut would do, including one that leaves "
            "nothing to run."
        )
    cannot_coded, can_coded = box.coded(cannot_actual), box.coded(can_actual)
    w_lp, c_lp, t = _separation(cannot_coded, can_coded)
    if t <= _SEPARABLE:
        raise _not_separable(cannot_coded, can_coded, box)
    w = _max_margin_plane(cannot_coded, can_coded, start=(w_lp / t, c_lp / t))
    coefficients, rhs = _rounded(w / box.half, box.half, cannot_actual, can_actual)
    return Constraint(expression=_expression(box.names, coefficients, rhs), type="linear")


def constraint_through_cuts(
    factors: Sequence[Factor],
    corner: Mapping[str, float] | Sequence[float],
    cuts: Mapping[str, float | None],
) -> Constraint:
    """Write the straight limit that cuts a corner off the box at given values of the factors.

    From the ``corner`` (a setting that cannot be run), each limited factor can go up to
    (or down to) its cut value with the other factors held at the corner. The limit is the
    plane through those cut points, ``sum((X_j - v_j) / (cut_j - v_j)) >= 1`` over the
    limited factors, where ``v`` is the corner: it rules out the corner and the settings
    near it, keeps the cut points on its boundary, and does not depend on a factor whose
    cut is ``None``. The arithmetic is exact (each value is read as the decimal its float
    shows), so round inputs give whole-number coefficients. The limit is scaled so its
    smallest coefficient is 1, so its largest is 1, or to whole numbers, keeping whichever
    of these writes the coefficients exactly with the fewest digits; it is written as
    :func:`separating_constraint` writes it.

    Parameters
    ----------
    factors : Sequence[Factor]
        The continuous factors.
    corner : Mapping[str, float] or Sequence[float]
        A corner of the box, in actual units: each factor at its low or its high value, as
        ``{name: value}`` or a sequence in factor order.
    cuts : Mapping[str, float or None]
        For each factor, the value it can reach from the corner with the others held there,
        or ``None`` (or no entry) when the factor does not limit this. A cut must lie
        inside the factor's range and differ from the corner's value.

    Returns
    -------
    Constraint
        A linear constraint in actual units, such as ``"T + 20 * P <= 250"``.

    Raises
    ------
    ValueError
        When a factor is not continuous or its name cannot be written in a limit, the
        corner is not a corner of the box, ``cuts`` names a factor that is not given, no
        factor is cut, or a cut lies outside its factor's range or at or beyond the corner.

    See Also
    --------
    separating_constraint : the limit proposed from settings that cannot and can be run.

    Examples
    --------
    >>> factors = [Factor(name="T", low=150, high=200), Factor(name="P", low=1, high=3)]
    >>> constraint_through_cuts(factors, {"T": 200, "P": 3}, {"T": 190, "P": 2.5}).expression
    'T + 20 * P <= 250'
    """
    box = _box(factors)
    vertex = _corner(corner, box)
    unknown = sorted(set(cuts) - set(box.names))
    if unknown:
        raise ValueError(f"The cuts name {unknown}, which are not among the factors {box.names}.")
    alpha: list[Fraction] = []
    for name, v, low, high in zip(box.names, vertex, box.low, box.high, strict=True):
        cut = cuts.get(name)
        if cut is None:
            alpha.append(Fraction(0))
            continue
        cut = float(cut)
        if not math.isfinite(cut) or not low <= cut <= high:
            raise ValueError(f"The cut of {name!r} at {cut:g} is outside its range, {low:g} to {high:g}.")
        if (cut >= v) if v == high else (cut <= v):
            raise ValueError(
                f"The cut of {name!r} at {cut:g} is at or beyond the corner's {name} = {v:g}: it must lie between "
                "the corner and the other end of the range."
            )
        alpha.append(1 / (_exact(cut) - _exact(v)))
    if not any(alpha):
        raise ValueError("Give a cut for at least one factor: with none, there is no limit to write.")
    # sum(alpha_j (X_j - v_j)) >= 1 keeps the settings away from the corner; written as a @ X <= b, with a = -alpha.
    a = [-x for x in alpha]
    b = -(1 + sum((x * _exact(v) for x, v in zip(alpha, vertex, strict=True)), Fraction(0)))
    coefficients, rhs = _exact_scaling(a, b)
    return Constraint(expression=_expression(box.names, coefficients, rhs), type="linear")


def _corner(corner: Mapping[str, float] | Sequence[float], box: _Box) -> list[float]:
    """Return the corner's values in factor order, each checked to be its factor's low or high value."""
    if isinstance(corner, Mapping):
        missing = [n for n in box.names if n not in corner]
        if missing:
            raise ValueError(f"The corner gives no value for the factors {missing}.")
        values = [corner[n] for n in box.names]
    else:
        values = list(corner)
        if len(values) != len(box.names):
            raise ValueError(f"The corner must give {len(box.names)} values, one per factor {box.names}.")
    out = []
    for name, value, low, high in zip(box.names, values, box.low, box.high, strict=True):
        tol = 1e-9 * (high - low)
        if abs(float(value) - low) <= tol:
            out.append(float(low))
        elif abs(float(value) - high) <= tol:
            out.append(float(high))
        else:
            raise ValueError(
                f"The corner's {name} = {float(value):g} is not a corner of the box: it must be {low:g} or {high:g}."
            )
    return out


def _exact_scaling(a: list[Fraction], b: Fraction) -> tuple[list[Fraction], Fraction]:
    """Scale an exact limit ``a @ x <= b`` to the form that writes its coefficients exactly with the fewest digits.

    The candidates are the smallest coefficient made 1, the largest made 1, and whole
    numbers with no common factor; one counts only when each of its numbers is a
    terminating decimal of at most ``_CUT_DIGITS`` significant digits. Ties go to the
    candidates in that order. When none counts, the first is rounded to ``_CUT_DIGITS``
    significant digits.
    """
    sizes = [abs(x) for x in a if x]
    candidates = [(1 / min(sizes)), (1 / max(sizes))]
    denominators = math.lcm(*(x.denominator for x in [*a, b]))
    numerators = math.gcd(*(int(x * denominators) for x in [*a, b] if x))
    candidates.append(Fraction(denominators, numerators))
    best: tuple[int, list[Fraction], Fraction] | None = None
    for scale in candidates:
        numbers = [x * scale for x in a], b * scale
        every = [*numbers[0], numbers[1]]
        if all(_terminates(x) and _significant(x) <= _CUT_DIGITS for x in every if x):
            written = _digits_written(numbers[0])
            if best is None or written < best[0]:
                best = (written, *numbers)
    if best is not None:
        return best[1], best[2]
    scale = candidates[0]
    return [_round_fraction(x * scale) for x in a], _round_fraction(b * scale)


def _round_fraction(value: Fraction, digits: int = _CUT_DIGITS) -> Fraction:
    """Return ``value`` rounded to ``digits`` significant digits, as an exact terminating decimal."""
    with localcontext() as context:
        context.prec = digits
        return Fraction(Decimal(value.numerator) / Decimal(value.denominator))
