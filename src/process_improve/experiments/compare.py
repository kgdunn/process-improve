# (c) Kevin Dunn, 2010-2026. MIT License.

r"""Compare designs for one fixed model on nine measures of design quality.

The question a comparison answers is narrow on purpose: the model is already chosen,
and the designs differ only in where they place their runs. Each design is scored on
the same model, in coded units over the cube :math:`[-1, 1]^k`, on the measures that
*Process Improvement using Data* defines in "Nine measures down one column"
(:data:`BOOK_URL`):

* five *alphabetic optimality criteria*, each one summary of the information matrix
  :math:`\mathbf{M} = \mathbf{X}^T\mathbf{X}` of the model matrix :math:`\mathbf{X}`:
  :math:`A/p` and :math:`I`, which average; :math:`E` and :math:`G`, which take the
  worst case of the same two quantities; and :math:`D`, which does neither;
* the largest absolute correlation between two second-order terms, read off the
  design's correlation map;
* the power to detect a main effect, a two-factor interaction and a quadratic of a
  stated size.

They come after three rows that decide whether the design belongs in the comparison
at all: its runs, whether it can fit the model, and the degrees of freedom it leaves
to estimate the error. No single design is best on every row, and the comparison
does not pick one.

When some settings cannot be run, a :class:`~process_improve.experiments.region.DesignRegion`
with the limits on them can be given: :math:`I` and :math:`G` are then taken over the
settings that can be run instead of over the whole cube.

Also see
--------
process_improve.experiments.evaluate.evaluate_design : many more measures, one design at a time.
process_improve.experiments.true_process.TrueProcess : simulate a design's results from a known model.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from process_improve.experiments.designs_utils import RESERVED_COLUMN_NAMES
from process_improve.experiments.evaluate import _coefficient_power
from process_improve.experiments.factor import DesignResult, Factor, FactorType
from process_improve.experiments.region import DesignRegion
from process_improve.tool_spec import clean

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The models a comparison can be made for. ``main_quadratic`` is the book's model for
#: the nine measures: the intercept, the main effects and the pure quadratics, without
#: the two-factor interactions.
MODELS = ("main_effects", "interactions", "main_quadratic", "quadratic")

#: Where the book defines the measures.
BOOK_URL = "https://learnche.org/pid/design-analysis-experiments/omars-designs#doe-omars-metric-choice"

#: The largest grid, in points, over which G is searched; 7 levels per factor up to
#: five factors, then 5 and 3 levels while the grid stays within this size.
_GRID_LIMIT = 100_000

#: Beyond the grid limit, G is searched over the cube's corners and this many points
#: drawn uniformly from the cube with a fixed seed.
_SAMPLED_POINTS = 20_000

#: Rows of the prediction grid expanded into model columns at a time.
_CHUNK = 4096

#: Over a region, I is averaged over this many points drawn uniformly from it, with a
#: generator seeded with :data:`_REGION_SEED`, so every comparison draws the same points.
_REGION_SAMPLES = 20_000
_REGION_SEED = 0

#: A column whose centred norm is below this, relative to the run count, is constant.
_CONSTANT = 1e-9


@dataclass(frozen=True)
class Measure:
    """One row of a :class:`DesignComparison`.

    Attributes
    ----------
    key : str
        The key the measure has in :attr:`DesignComparison.values`.
    label : str
        The row label.
    direction : str
        ``"higher"`` or ``"lower"``, whichever is preferred; empty for the feasibility
        rows, which are facts about the design rather than a scale.
    group : str
        ``"Feasibility"``, ``"Averages"``, ``"Worst cases"``, ``"Neither"`` or
        ``"Power"``: the book's grouping of the measures.
    description : str
        What the measure is, in one sentence.
    """

    key: str
    label: str
    direction: str
    group: str
    description: str


#: The rows of a comparison, in the order the table lists them.
#:
#: :meta hide-value:
MEASURES: tuple[Measure, ...] = (
    Measure("runs", "Runs", "", "Feasibility", "The number of runs in the design."),
    Measure(
        "fits",
        "Fits the model?",
        "",
        "Feasibility",
        "Whether every coefficient of the model can be estimated: the model matrix has full column rank.",
    ),
    Measure(
        "residual_df",
        "Residual degrees of freedom",
        "",
        "Feasibility",
        "The runs left after the fit to estimate the run-to-run noise; counted from the rank of the model "
        "matrix when the design cannot fit the model.",
    ),
    Measure(
        "a",
        "A/p, average coefficient variance",
        "lower",
        "Averages",
        "tr(M^-1)/p: the coefficients' variances averaged over the p coefficients, in units of sigma^2.",
    ),
    Measure(
        "i",
        "I, average prediction variance",
        "lower",
        "Averages",
        "tr(M^-1 B): the variance of a prediction averaged over the cube, in units of sigma^2; B holds the "
        "averages of the products of the model terms there.",
    ),
    Measure(
        "e",
        "E, smallest eigenvalue of M",
        "higher",
        "Worst cases",
        "The worst-estimated combination of the coefficients has variance sigma^2 / E.",
    ),
    Measure(
        "g",
        "G, largest prediction variance",
        "lower",
        "Worst cases",
        "The variance of a prediction at the worst point of the cube, in units of sigma^2.",
    ),
    Measure(
        "d",
        "D, |M|^(1/p)",
        "higher",
        "Neither",
        "The joint precision of all p coefficients: their joint confidence region has a volume "
        "proportional to |M|^(-1/2).",
    ),
    Measure(
        "max_r",
        "Max |r| between second-order terms",
        "lower",
        "Neither",
        "The largest absolute correlation between two second-order terms (squares and two-factor "
        "interactions), whether or not the model includes them.",
    ),
    Measure(
        "power_main",
        "Power, main effect",
        "higher",
        "Power",
        "The probability that the test of the least precisely estimated main effect is significant when "
        "its coefficient is effect_size noise standard deviations.",
    ),
    Measure(
        "power_interaction",
        "Power, interaction",
        "higher",
        "Power",
        "As for the main effect, for the least precisely estimated two-factor interaction.",
    ),
    Measure(
        "power_quadratic",
        "Power, quadratic",
        "higher",
        "Power",
        "As for the main effect, for the least precisely estimated pure quadratic.",
    ),
)

#: The definitions of I and G when they are taken over a region rather than over the cube.
_OVER_THE_REGION = {
    "i": "tr(M^-1 B): the variance of a prediction averaged over the region, in units of sigma^2; B holds the "
    "averages of the products of the model terms over a uniform sample of the region.",
    "g": "The variance of a prediction at the worst point of the region, in units of sigma^2.",
}

_ARROWS = {"higher": "↑", "lower": "↓", "": ""}

#: The term types power is reported for, and the measure each one fills.
_POWER = (("main", "power_main"), ("interaction", "power_interaction"), ("quadratic", "power_quadratic"))


# --------------------------------------------------------------------------- the result
@dataclass(frozen=True, eq=False)
class DesignComparison:
    """Designs scored for one model on the measures in :data:`MEASURES`.

    Attributes
    ----------
    model : str
        The model every design is scored on, one of :data:`MODELS`.
    terms : tuple[str, ...]
        The model's terms, named as :func:`analyze_experiment` names them
        (``"Intercept"``, ``"A"``, ``"A:B"``, ``"I(A ** 2)"``).
    effect_size : float
        The coefficient size, in noise standard deviations, at which power is computed.
    significance_level : float
        The level of the tests behind the power.
    designs : tuple[str, ...]
        The designs' labels, each with its size (``"DSD [n=9]"``), in the order given.
    values : Mapping[str, Mapping[str, Any]]
        For each design label, each measure's value by its :attr:`Measure.key`. A
        measure the design cannot provide is ``None``: every measure after the
        feasibility rows when the design cannot fit the model, power when no degrees
        of freedom are left, and power for a term type the model does not have.
    notes : Mapping[str, tuple[str, ...]]
        For each design label, sentences that qualify its numbers.
    correlations : Mapping[str, pandas.DataFrame]
        For each design label, the absolute Pearson correlation between its main
        effects, squares and two-factor interactions, in that order; ``NaN`` for a
        column that does not vary. The block of squares and interactions is the one
        ``"max_r"`` reads.
    g_points : str
        The points over which G was searched, shared by every design.
    region : DesignRegion or None
        The region I and G were taken over, as it was given; ``None`` for the cube
        :math:`[-1, 1]^k`.
    """

    model: str
    terms: tuple[str, ...]
    effect_size: float
    significance_level: float
    designs: tuple[str, ...]
    values: Mapping[str, Mapping[str, Any]]
    notes: Mapping[str, tuple[str, ...]]
    correlations: Mapping[str, pd.DataFrame]
    g_points: str
    region: DesignRegion | None = None

    @property
    def measures(self) -> tuple[Measure, ...]:
        """Return the rows of the comparison, with I and G defined over the region when there is one.

        Returns
        -------
        tuple[Measure, ...]
            :data:`MEASURES` without a region, or with one that has no constraints (the
            cube's exact moments are used then); otherwise the same rows with the
            descriptions of I and G saying they are taken over the region.
        """
        if self.region is None or not self.region.is_constrained:
            return MEASURES
        return tuple(dataclasses.replace(m, description=_OVER_THE_REGION.get(m.key, m.description)) for m in MEASURES)

    @property
    def table(self) -> pd.DataFrame:
        """Return the comparison in the book's layout, for reading.

        Returns
        -------
        pandas.DataFrame
            One row per measure, labelled by :attr:`Measure.label`; a ``"Better"``
            column with an arrow for the preferred direction; then one column per
            design. The entries are text: counts as integers, the measures to four
            significant figures as the book prints them, ``"Fits the model?"`` as
            ``"yes"`` or ``"no (rank r)"``, and an empty string where the design
            cannot provide a value. :attr:`values` and :attr:`long` hold the numbers.
        """
        rows = {}
        for measure in MEASURES:
            row: dict[str, Any] = {"Better": _ARROWS[measure.direction]}
            for label in self.designs:
                row[label] = _display(measure.key, self.values[label])
            rows[measure.label] = row
        return pd.DataFrame.from_dict(rows, orient="index")

    @property
    def long(self) -> pd.DataFrame:
        """Return one row per design and measure.

        Returns
        -------
        pandas.DataFrame
            Columns ``design``, ``key``, ``measure``, ``group``, ``direction`` and
            ``value``.
        """
        return pd.DataFrame(
            [
                {
                    "design": label,
                    "key": m.key,
                    "measure": m.label,
                    "group": m.group,
                    "direction": m.direction,
                    "value": self.values[label][m.key],
                }
                for label in self.designs
                for m in MEASURES
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison as JSON-safe Python values.

        Returns
        -------
        dict
            The attributes, with the measures' definitions (:attr:`measures`) under
            ``"measures"``, each correlation matrix as ``{"terms": [...], "matrix":
            [[...]]}`` and the region as :meth:`DesignRegion.to_dict` writes it (``None``
            without one); ``NaN`` and infinities become ``None``.
        """
        return clean(
            {
                "model": self.model,
                "terms": list(self.terms),
                "effect_size": self.effect_size,
                "significance_level": self.significance_level,
                "designs": list(self.designs),
                "measures": [m.__dict__ for m in self.measures],
                "values": {label: dict(v) for label, v in self.values.items()},
                "notes": {label: list(n) for label, n in self.notes.items()},
                "correlations": {
                    label: {"terms": list(frame.columns), "matrix": frame.to_numpy().tolist()}
                    for label, frame in self.correlations.items()
                },
                "g_points": self.g_points,
                "book": BOOK_URL,
                "region": None if self.region is None else self.region.to_dict(),
            }
        )


def _display(key: str, values: Mapping[str, Any]) -> str:
    """Return a value as the table shows it."""
    value = values[key]
    if key == "fits":
        return "yes" if value else f"no (rank {values['rank']})"
    if value is None:
        return ""
    return str(value) if isinstance(value, (int, np.integer)) else f"{value:#.4g}"


# --------------------------------------------------------------------------- the model
@dataclass(frozen=True)
class _Model:
    """A model's terms as exponent vectors, with what every design is scored against."""

    name: str
    exponents: np.ndarray  # (p, k) integer powers of each factor in each term
    terms: tuple[str, ...]
    kinds: tuple[str, ...]  # "intercept", "main", "interaction" or "quadratic"
    moments: np.ndarray  # (p, p) averages of the products of the terms over the cube, or over the region
    grid: np.ndarray  # (n_points, p) the model's columns at the points G is searched over
    g_points: str
    region: DesignRegion | None = None  # the region over the designs' factors, None for the cube


def _model_exponents(k: int, model: str) -> np.ndarray:
    """List a model's terms as exponent vectors over k factors, intercept first."""
    eye = np.eye(k, dtype=int)
    rows = [np.zeros(k, dtype=int), *eye]
    if model in {"interactions", "quadratic"}:
        rows += [eye[i] + eye[j] for i, j in itertools.combinations(range(k), 2)]
    if model in {"main_quadratic", "quadratic"}:
        rows += [2 * eye[i] for i in range(k)]
    return np.array(rows, dtype=int)


def _term_name(exponent: np.ndarray, names: Sequence[str]) -> tuple[str, str]:
    """Name a term as ``analyze_experiment`` does, and say what kind of term it is."""
    used = [i for i, power in enumerate(exponent) if power]
    if not used:
        return "Intercept", "intercept"
    if exponent[used[0]] == 2:
        return f"I({names[used[0]]} ** 2)", "quadratic"
    if len(used) == 1:
        return names[used[0]], "main"
    return ":".join(names[i] for i in used), "interaction"


def _expand(points: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """Return the model's columns at the points: each term is a product of powers of the factors."""
    return np.prod(points[:, None, :] ** exponents[None, :, :], axis=2)


def _moments(exponents: np.ndarray) -> np.ndarray:
    """Average every product of two model terms over the cube, exactly.

    Over :math:`[-1, 1]`, the average of :math:`x^a` is 0 for odd :math:`a` and
    :math:`1/(a + 1)` for even :math:`a`; the factors are independent and uniform
    over the cube, so the averages multiply.
    """
    powers = exponents[:, None, :] + exponents[None, :, :]
    averages = np.where(powers % 2 == 1, 0.0, 1.0 / (powers + 1.0))
    return np.prod(averages, axis=2)


def _g_points(k: int) -> tuple[np.ndarray, str]:
    """Return the points of the cube over which G is searched, and how they were chosen."""
    for levels in (7, 5, 3):
        if levels**k <= _GRID_LIMIT:
            grid = np.array(list(itertools.product(np.linspace(-1.0, 1.0, levels), repeat=k)))
            return grid, f"a {levels}-level grid of {len(grid)} points"
    corners = np.array(list(itertools.product([-1.0, 1.0], repeat=k)))
    sampled = np.random.default_rng(0).uniform(-1.0, 1.0, size=(_SAMPLED_POINTS, k))
    return np.vstack([corners, sampled]), (
        f"the {len(corners)} corners and {_SAMPLED_POINTS} points drawn uniformly from the cube"
    )


def _expand_in_chunks(points: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """Return the model's columns at many points, :data:`_CHUNK` rows at a time."""
    return np.vstack([_expand(points[i : i + _CHUNK], exponents) for i in range(0, len(points), _CHUNK)])


def _region_points(region: DesignRegion, k: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Return a uniform sample of the region, the points G is searched over, and how they were chosen.

    The sample is :data:`_REGION_SAMPLES` points drawn with a generator seeded with
    :data:`_REGION_SEED`, the same on every run. G is searched over the region's support
    points (its vertices, where its constraints cross the edges of a 3-level grid, and
    that grid's points inside it), the points of the cube's own grid (see
    :func:`_g_points`) that lie in the region, and the sample.
    """
    sample = region.sample(_REGION_SAMPLES, np.random.default_rng(_REGION_SEED))
    grid, described = _g_points(k)
    inside = grid[region.feasible(grid)]
    support = region.seed_points()  # none for a box too large for a boundary grid
    parts = [f"the {len(inside)} points of {described} that lie in the region"]
    if len(support):
        parts.append(f"its {len(support)} support points")
    parts.append(f"{len(sample)} points drawn uniformly from it")
    description = ", ".join(parts[:-1]) + " and " + parts[-1]
    return sample, np.vstack([support, inside, sample]), description


def _build_model(k: int, model: str, names: Sequence[str], region: DesignRegion | None = None) -> _Model:
    """Set up a model once, for scoring every design on it, over the cube or over a region."""
    exponents = _model_exponents(k, model)
    named = [_term_name(e, names) for e in exponents]
    if region is None:
        points, description = _g_points(k)
        moments = _moments(exponents)
    else:
        sample, points, description = _region_points(region, k)
        sampled = _expand_in_chunks(sample, exponents)
        moments = sampled.T @ sampled / len(sample)
    return _Model(
        name=model,
        exponents=exponents,
        terms=tuple(n for n, _ in named),
        kinds=tuple(kind for _, kind in named),
        moments=moments,
        grid=_expand_in_chunks(points, exponents),
        g_points=description,
        region=region,
    )


# --------------------------------------------------------------------------- the designs
def _coded_runs(design: object) -> tuple[np.ndarray, list[str] | None]:
    """Return a design's runs in coded units, and its factor names when it has them.

    Raises
    ------
    ValueError
        If the runs are not a two-dimensional table of numbers.
    """
    if isinstance(design, DesignResult):
        frame: pd.DataFrame | None = design.design[list(design.factor_names)]
    elif isinstance(design, pd.DataFrame):
        frame = design[[c for c in design.columns if c not in RESERVED_COLUMN_NAMES]]
    else:
        frame = None
    try:
        runs = np.asarray(design if frame is None else frame, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "compare_designs scores designs in continuous factors, coded from -1 to +1; a design has a column "
            "that is not numeric."
        ) from exc
    if runs.ndim != 2 or runs.shape[1] == 0 or not np.all(np.isfinite(runs)):
        raise ValueError("Each design must be a table of finite numbers, one row per run and one column per factor.")
    return runs, None if frame is None else [str(c) for c in frame.columns]


def _labelled(designs: Mapping[str, Any] | Sequence[Any]) -> list[tuple[str, Any]]:
    """Pair each design with its label: the mapping's key, else its design type."""
    if isinstance(designs, (pd.DataFrame, np.ndarray, DesignResult)):
        raise TypeError("Pass the designs as a mapping {label: design} or a list of DesignResult objects.")
    if hasattr(designs, "items"):
        return [(str(label), design) for label, design in designs.items()]
    labelled = []
    for position, design in enumerate(designs, start=1):
        label = design.design_type if isinstance(design, DesignResult) else f"Design {position}"
        labelled.append((label, design))
    return labelled


def _factor_names(runs: list[tuple[np.ndarray, list[str] | None]]) -> list[str]:
    """Take the factor names from the designs that have them, which must agree.

    Raises
    ------
    ValueError
        If the designs differ in their number of factors, or name them differently.
    """
    counts = {x.shape[1] for x, _ in runs}
    if len(counts) > 1:
        raise ValueError(f"Every design must have the same factors; these have {sorted(counts)} factors.")
    named = [tuple(names) for _, names in runs if names is not None]
    if len(set(named)) > 1:
        raise ValueError(f"The designs name their factors differently: {sorted(set(named))}.")
    k = counts.pop()
    return list(named[0]) if named else [chr(ord("A") + i) if i < 8 else f"X{i + 1}" for i in range(k)]


# --------------------------------------------------------------------------- the measures
def _correlations(x: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Return the absolute correlation between the main effects, squares and interactions."""
    k = x.shape[1]
    pairs = list(itertools.combinations(range(k), 2))
    columns = np.column_stack([x, x**2, *[x[:, i] * x[:, j] for i, j in pairs]]) if pairs else np.hstack([x, x**2])
    labels = [*names, *[f"I({n} ** 2)" for n in names], *[f"{names[i]}:{names[j]}" for i, j in pairs]]
    centred = columns - columns.mean(axis=0)
    norms = np.linalg.norm(centred, axis=0)
    varying = norms > _CONSTANT * math.sqrt(len(x))
    unit = np.divide(centred, norms, out=np.zeros_like(centred), where=varying)
    r = np.clip(np.abs(unit.T @ unit), 0.0, 1.0)
    r[r < _CONSTANT] = 0.0  # rounding error on orthogonal columns
    r[~varying, :] = np.nan
    r[:, ~varying] = np.nan
    return pd.DataFrame(r, index=labels, columns=labels)


def _largest_second_order_r(correlations: pd.DataFrame, k: int) -> float | None:
    """Return the largest |r| between two different second-order terms, or None without a pair."""
    block = correlations.to_numpy()[k:, k:].copy()
    np.fill_diagonal(block, np.nan)
    return None if np.all(np.isnan(block)) else float(np.nanmax(block))


def _criteria(matrix: np.ndarray, model: _Model) -> tuple[dict[str, float], np.ndarray]:
    """Return the five alphabetic criteria of a full-rank model matrix, and the diagonal of M^-1."""
    information = matrix.T @ matrix
    inverse = np.linalg.inv(information)
    p = information.shape[0]
    _, log_det = np.linalg.slogdet(information)
    prediction = np.einsum("ij,jk,ik->i", model.grid, inverse, model.grid)
    criteria = {
        "a": float(np.trace(inverse) / p),
        "i": float(np.sum(inverse * model.moments)),
        "e": float(np.linalg.eigvalsh(information).min()),
        "g": float(prediction.max()),
        "d": float(math.exp(log_det / p)),
    }
    return criteria, np.diag(inverse)


def _power(c: np.ndarray, df_residual: int, model: _Model, effect_size: float, alpha: float) -> dict[str, float | None]:
    """Power for the least precisely estimated term of each type the model has."""
    kinds = np.array(model.kinds)
    power: dict[str, float | None] = {}
    for kind, key in _POWER:
        chosen = c[kinds == kind]
        if chosen.size == 0 or df_residual <= 0:
            power[key] = None
        else:
            power[key] = float(_coefficient_power(effect_size, float(chosen.max()), df_residual, alpha))
    return power


def _inseparable(matrix: np.ndarray, model: _Model) -> str:
    """Say which of the model's terms a rank-deficient design cannot separate."""
    terms = np.array(model.terms)
    centred = matrix - matrix.mean(axis=0)
    norms = np.linalg.norm(centred, axis=0)
    constant = [t for t, n, kind in zip(terms, norms, model.kinds, strict=True) if kind != "intercept" and n < 1e-9]
    unit = np.divide(centred, norms, out=np.zeros_like(centred), where=norms >= 1e-9)
    parallel = np.isclose(np.abs(unit.T @ unit), 1.0) & (norms[:, None] >= 1e-9) & (norms[None, :] >= 1e-9)
    groups = {tuple(terms[np.flatnonzero(row)]) for row in parallel if row.sum() > 1}
    parts = []
    if constant:
        parts.append(f"{', '.join(constant)} take the same value in every run, as the intercept does")
    parts += [f"{' and '.join(group)} cannot be told apart" for group in sorted(groups)]
    return "; ".join(parts) + "." if parts else "Some of its terms are linear combinations of the others."


def _score(x: np.ndarray, model: _Model, effect_size: float, alpha: float) -> tuple[dict[str, Any], list[str]]:
    """Score one design on every measure, with the notes that qualify the numbers."""
    matrix = _expand(x, model.exponents)
    n, p = matrix.shape
    rank = int(np.linalg.matrix_rank(matrix))
    values: dict[str, Any] = {m.key: None for m in MEASURES}
    values.update(runs=n, fits=rank == p, rank=rank, residual_df=n - rank)
    notes = []
    if rank < p:
        notes.append(
            f"Cannot fit the {model.name.replace('_', ' ')} model: its {p} terms have rank {rank} in this "
            f"design. {_inseparable(matrix, model)}"
        )
        return values, notes
    criteria, c = _criteria(matrix, model)
    values.update(criteria)
    values.update(_power(c, n - p, model, effect_size, alpha))
    if n == p:
        notes.append("No degrees of freedom are left to estimate the error, so there is no test and no power.")
    outside = _outside(x, model.region)
    if outside:
        notes.append(outside)
    return values, notes


def _outside(x: np.ndarray, region: DesignRegion | None) -> str | None:
    """Say how far a design's runs leave the cube, or how many leave the region, when any do."""
    if region is None:
        reach = float(np.abs(x).max())
        if reach > 1 + 1e-9:
            return (
                f"Some runs lie outside the cube [-1, 1], up to {reach:.3g} from the center on a factor, "
                "while I and G are computed over the cube."
            )
        return None
    count = int(np.count_nonzero(~region.feasible(x)))
    if not count:
        return None
    verb = "lies" if count == 1 else "lie"
    return f"{count} of its {len(x)} runs {verb} outside the region, while I and G are computed over the region."


def _check_settings(model: str, effect_size: float, significance_level: float) -> None:
    """Refuse a model this comparison does not define, or an impossible effect size or level."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, not {model!r}.")
    if not (math.isfinite(effect_size) and effect_size > 0):
        raise ValueError(f"effect_size must be a positive number of noise standard deviations, not {effect_size}.")
    if not 0 < significance_level < 1:
        raise ValueError(f"significance_level must lie between 0 and 1, not {significance_level}.")


def _as_region(region: object) -> DesignRegion:
    """Return a region given as a :class:`DesignRegion` or as its dict form.

    Raises
    ------
    TypeError
        If ``region`` is neither.
    """
    if isinstance(region, DesignRegion):
        return region
    if isinstance(region, Mapping):
        return DesignRegion.from_dict(dict(region))
    raise TypeError(f"region must be a DesignRegion or its dict form, not {type(region).__name__}.")


def _region_over(region: DesignRegion, names: Sequence[str]) -> DesignRegion | None:
    """Return the region over the designs' factors, in their order, refusing one that cannot be compared over.

    A factor the region does not name is free over its whole range, ``[-1, 1]``.

    Returns
    -------
    DesignRegion or None
        The region, or ``None`` when it has no constraints: it is then the cube, whose
        exact moments are used.

    Raises
    ------
    ValueError
        If the region is a mixture region, names a factor the designs do not have, or
        leaves no setting at all.
    """
    if region.kind == "mixture":
        raise ValueError(
            "compare_designs scores designs over a box of factor settings; a mixture region, of proportions that "
            "sum to 1, cannot be compared over."
        )
    missing = [f.name for f in region.factors if f.name not in names]
    if missing:
        raise ValueError(f"The region names {missing}, which the designs do not have; their factors are {list(names)}.")
    if region.is_empty():
        raise ValueError(
            "No setting in the region meets every one of its constraints, so there is nothing to average the "
            "prediction variance over."
        )
    if not region.is_constrained:
        return None
    numeric = {f.name: f for f in region.factors if f.type != FactorType.categorical}
    if list(numeric) == list(names):
        return region
    return DesignRegion([numeric.get(n) or Factor(name=n, low=-1, high=1) for n in names], region.constraints)


def compare_designs(
    designs: Mapping[str, DesignResult | pd.DataFrame | np.ndarray] | Sequence[DesignResult],
    model: str = "main_quadratic",
    *,
    effect_size: float = 1.0,
    significance_level: float = 0.05,
    region: DesignRegion | Mapping[str, Any] | None = None,
) -> DesignComparison:
    r"""Score designs for one fixed model on the nine measures of design quality.

    The model is fixed first, and every design is scored on it, so the comparison is
    about where each design places its runs. Each design is compared at its own size:
    the table labels it with its run count rather than scaling the measures by it, so
    a larger design scores better on the measures that improve with runs, as it does
    at the bench.

    Write :math:`\mathbf{M} = \mathbf{X}^T\mathbf{X}` for the model matrix
    :math:`\mathbf{X}` of a design with :math:`N` runs, :math:`p` the model's terms,
    and :math:`\mathbf{f}(\mathbf{x})` the model's row at a point
    :math:`\mathbf{x}` of the cube :math:`[-1, 1]^k`. The measures, as the book
    defines them (:data:`BOOK_URL`):

    * :math:`A/p = \mathrm{tr}(\mathbf{M}^{-1})/p`, the average coefficient variance;
    * :math:`I = \mathrm{tr}(\mathbf{M}^{-1}\mathbf{B})`, the average prediction
      variance over the cube, with :math:`\mathbf{B}` the exact averages of the
      products of the model terms there;
    * :math:`E = \lambda_{\min}(\mathbf{M})`, the smallest eigenvalue;
    * :math:`G = \max_{\mathbf{x}} \mathbf{f}^T\mathbf{M}^{-1}\mathbf{f}`, the
      largest prediction variance, over the 7-level grid up to five factors (see
      :attr:`DesignComparison.g_points` above that);
    * :math:`D = |\mathbf{M}|^{1/p}`, the joint precision of the coefficients;
    * max :math:`|r|`, the largest absolute correlation between two second-order
      terms (the squares and two-factor interactions, in the model or not);
    * the power of the test of the least precisely estimated main effect, interaction
      and quadratic in the model: a non-central :math:`F(1, N - p)` with
      non-centrality :math:`\lambda = (|\beta|/\sigma)^2 / c`, where :math:`c` is
      the term's diagonal entry of :math:`\mathbf{M}^{-1}`.

    :math:`A/p`, :math:`I` and :math:`G` are in units of :math:`\sigma^2`, the
    run-to-run noise variance.

    When some settings cannot be run, ``region`` holds the limits on them, and
    :math:`I` and :math:`G` are taken over the settings that can be run rather than
    over the cube: :math:`\mathbf{B}` is averaged over 20000 points drawn uniformly
    from the region (with a fixed seed, so the numbers are the same on every run), and
    :math:`G` is searched over the region's support points (its vertices, and where its
    limits cross the edges of a 3-level grid), the points of the grid above that lie in
    it, and the same sample. The other measures do not depend on where predictions are
    made, and are the same with or without a region. A design with runs outside the
    region is noted. Without a region, or with one that has no constraints, the cube's
    exact moments are used.

    Parameters
    ----------
    designs : mapping of str to design, or sequence of DesignResult
        The designs, in coded units: a :class:`DesignResult` (its coded ``design``), a
        DataFrame of factor columns (``RunOrder`` and ``Block`` are ignored), or an
        array with one column per factor. A mapping's keys label the designs; in a
        sequence each design is labelled with its ``design_type``. Every design has
        the same factors, under the same names.
    model : {"main_quadratic", "main_effects", "interactions", "quadratic"}
        The model every design is scored on. ``"main_quadratic"``, the default and
        the book's model for the nine measures, has the intercept, the main effects
        and the pure quadratics; ``"quadratic"`` adds the two-factor interactions.
    effect_size : float, default 1.0
        The coefficient, in noise standard deviations (:math:`|\beta|/\sigma`), whose
        detection the power measures. With factors coded from -1 to +1, 1.0 is a
        low-to-high change of two standard deviations.
    significance_level : float, default 0.05
        The level of the tests behind the power.
    region : DesignRegion, dict or None
        The settings that can be run: a box region (or its :meth:`DesignRegion.to_dict`
        form) over the designs' factors, whose coded units the designs are in. A factor
        it does not name is free over its whole range. ``None``, the default, is the
        cube :math:`[-1, 1]^k`.

    Returns
    -------
    DesignComparison
        The values, the table in the book's layout (:attr:`DesignComparison.table`),
        notes on each design and its correlation map.

    Raises
    ------
    ValueError
        If ``model`` is not one of :data:`MODELS`, ``effect_size`` is not positive,
        ``significance_level`` is not between 0 and 1, no design is given, the
        designs differ in their factors, or ``region`` is a mixture region, names a
        factor the designs do not have, or leaves no setting to run.
    TypeError
        If a single design is passed instead of a collection of them, or ``region`` is
        neither a :class:`DesignRegion` nor its dict form.

    Examples
    --------
    >>> from process_improve.experiments import Factor, compare_designs, generate_design
    >>> factors = [Factor(name=n, low=-1, high=1) for n in "ABC"]
    >>> comparison = compare_designs({
    ...     "Box-Behnken": generate_design(factors, "box_behnken", n_center_points=3),
    ...     "DSD": generate_design(factors, "dsd"),
    ... })
    >>> comparison.designs
    ('Box-Behnken [n=15]', 'DSD [n=9]')
    >>> round(comparison.values["Box-Behnken [n=15]"]["power_quadratic"], 3)
    0.395
    """
    _check_settings(model, effect_size, significance_level)
    labelled = _labelled(designs)
    if not labelled:
        raise ValueError("Give at least one design to compare.")
    runs = [_coded_runs(design) for _, design in labelled]
    names = _factor_names(runs)
    given = None if region is None else _as_region(region)
    scored = _build_model(len(names), model, names, None if given is None else _region_over(given, names))

    labels: list[str] = []
    values: dict[str, Mapping[str, Any]] = {}
    notes: dict[str, tuple[str, ...]] = {}
    correlations: dict[str, pd.DataFrame] = {}
    for (label, _), (x, _) in zip(labelled, runs, strict=True):
        sized = _unique(f"{label} [n={len(x)}]", labels)
        labels.append(sized)
        design_values, design_notes = _score(x, scored, effect_size, significance_level)
        correlations[sized] = _correlations(x, names)
        design_values["max_r"] = _largest_second_order_r(correlations[sized], len(names))
        if np.isnan(correlations[sized].to_numpy()[len(names) : 2 * len(names)]).all():
            design_notes.append("The squares do not vary in this design, so max |r| is over the interactions only.")
        values[sized] = MappingProxyType(design_values)
        notes[sized] = tuple(design_notes)
    return DesignComparison(
        model=model,
        terms=scored.terms,
        effect_size=float(effect_size),
        significance_level=float(significance_level),
        designs=tuple(labels),
        values=MappingProxyType(values),
        notes=MappingProxyType(notes),
        correlations=MappingProxyType(correlations),
        g_points=scored.g_points,
        region=given,
    )


def _unique(label: str, taken: Sequence[str]) -> str:
    """Return the label, numbered if an earlier design already has it."""
    if label not in taken:
        return label
    return next(f"{label} ({i})" for i in itertools.count(2) if f"{label} ({i})" not in taken)
