# (c) Kevin Dunn, 2010-2026. MIT License.

"""Mixture-process models: mixture components and process factors in one model.

In a mixture-process experiment the proportions ``x_1, ..., x_q`` of the ingredients sum
to 1, while the process factors ``z_1, ..., z_m`` (temperature, mixing speed) range
freely, in coded units from -1 to +1, or take named levels (a catalyst, a supplier). A
model of both is a list of terms, each a product of *atoms*: a component ``x_i``, a
process factor ``z_k``, a square ``z_k^2``, or the cubic difference ``x_i x_j (x_i - x_j)``
of a full cubic Scheffé model. A categorical process factor of ``L`` levels is an atom of
``L - 1`` sum-to-zero columns, and a term's columns are the products of its atoms'
columns. The model matrix
the design engine optimises and the formula recorded for
:func:`~process_improve.experiments.evaluate_design` and
:func:`~process_improve.experiments.analyze_experiment` are both built from that list,
so they cannot disagree.

Three families of model are built (Goos and Jones 2011, section 6.3):

- **By order**, ``"main_effects"``, ``"interactions"`` or ``"quadratic"``: the polynomial
  of that order in all ``q + m`` factors, less the terms that ``sum(x) = 1`` makes
  redundant. The intercept is ``sum(x_i)``, a square ``x_i^2`` is
  ``x_i - sum_{j != i} x_i x_j``, and a process main effect ``z_k`` is ``sum_i x_i z_k``.
  What remains is

  - first order: ``x_i`` and ``z_k``, ``q + m`` terms;
  - with interactions: ``x_i``, ``x_i x_j``, ``x_i z_k`` and ``z_k z_l``;
  - second order: those and ``z_k^2``, the model of Kowalski, Cornell and Vining (2000),
    with ``q + q(q - 1)/2 + qm + m(m - 1)/2 + m`` terms; ``"kcv"`` names it too.

- **Additive**, ``"<mixture model> + <process model>"``: a Scheffé model plus a process
  model without its intercept. ``"scheffe_quadratic + main_effects"`` is the model of
  Goos and Jones's Table 6.2, ``q + q(q - 1)/2 + m`` terms.
- **Crossed**, ``"<mixture model> * <process model>"``: every Scheffé term multiplied by
  every term of the process model, its intercept included.
  ``"scheffe_quadratic * interactions"`` has ``[q + q(q - 1)/2] x [1 + m + m(m - 1)/2]``
  terms.

The mixture model on the left of ``+`` or ``*`` is a Scheffé model name
(``"scheffe_linear"``, ``"scheffe_quadratic"``, ``"scheffe_special_cubic"``,
``"scheffe_cubic"``), and the process model on the right is ``"main_effects"``,
``"interactions"`` or ``"quadratic"``. A categorical factor has no square.

Without an intercept, patsy codes a categorical factor with a column for every level
wherever the term below it is missing from the formula (``spray`` beside the blends, and
``T:catalyst`` in the ``"interactions"`` model, where ``T`` enters only through
``x_i:T``). Those columns add up to columns the blends already span, so the model could
not be estimated. :class:`ReducedRank` keeps every categorical factor at ``L - 1``
columns, and :func:`~process_improve.experiments.evaluate_design` and
:func:`~process_improve.experiments.analyze_experiment` code the formula with it.

The design (:func:`mixture_process_design`) is chosen from every candidate blend of the
constrained simplex (the extreme vertices, edge points, face centroids and centroid of
:func:`~process_improve.experiments.designs_mixture_constrained.mixture_candidates`) at
every candidate process setting (the grid of
:func:`~process_improve.experiments.designs_constrained.build_candidates`), by the
Fedorov exchange the other optimal designs use. The I- and G-criteria average and
maximise over points drawn uniformly from the mixture-process
:class:`~process_improve.experiments.region.DesignRegion`.
"""

from __future__ import annotations

import ast
import itertools
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from patsy import EvalFactor, ModelDesc, Sum
from patsy import Term as PatsyTerm
from patsy import categorical as patsy_categorical

from process_improve._random import check_random_state
from process_improve.experiments._rules import names_categorical
from process_improve.experiments.designs_constrained import (
    ConstrainedOptions,
    _factor_names,
    _Region,
    build_candidates,
    criterion_metadata,
    fedorov_exchange,
    make_criterion,
    nfkc,
    parse_constraint,
)
from process_improve.experiments.designs_mixture_constrained import (
    SCHEFFE_DEGREE,
    _unique_rows,
    cubic_difference_term,
    mixture_candidates,
    mixture_inequalities,
    scheffe_model,
)
from process_improve.experiments.designs_utils import (
    categorical_effect_columns,
    refuse_budget_below_model,
    sorted_level_ranks,
)
from process_improve.experiments.factor import FactorType

if TYPE_CHECKING:
    from process_improve.experiments.factor import Constraint, Factor

logger = logging.getLogger(__name__)

#: The process models, and the order-based mixture-process models, by name.
PROCESS_MODELS = ("main_effects", "interactions", "quadratic")
#: Other names of the order-based models.
_ORDER_ALIASES = {"kcv": "quadratic"}
#: ``"<mixture model> + <process model>"`` (additive) or ``"... * ..."`` (crossed).
_COMBINED = re.compile(r"\s*(\w+)\s*([+*])\s*(\w+)\s*")
#: Most candidate runs (blends times process settings) the exchange searches; more are sampled down to this.
MAX_CANDIDATES = 50_000
#: Points drawn uniformly from the region for the I- and G-criteria.
_N_REGION_SAMPLES = 20_000
#: Runs beyond the model's terms when no budget is given, as for a mixture design.
_EXTRA_RUNS = 3


@dataclass(frozen=True)
class Atom:
    """One factor of a model term: a component or process factor, a square, a cubic difference, or a categorical.

    Parameters
    ----------
    code : str
        How the atom is written in a formula: ``"x1"``, ``"I(T ** 2)"``,
        ``"I(x1 * x2 * (x1 - x2))"``, ``"catalyst"``.
    names : tuple[str, ...]
        The factors it is made of: one for a plain factor, a square or a categorical, two
        for a cubic difference.
    power : {"plain", "square", "cubic", "categorical"}
        Which of the four it is.
    levels : tuple
        A categorical factor's levels, in the order they were declared; empty otherwise.
    """

    code: str
    names: tuple[str, ...]
    power: str = "plain"
    levels: tuple = ()

    @property
    def width(self) -> int:
        """Return the number of model columns: ``L - 1`` for a categorical factor of ``L`` levels, else 1."""
        return len(self.levels) - 1 if self.power == "categorical" else 1

    def columns(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return the atom's model columns at each point, shape ``(n_points, width)``.

        ``data`` maps a factor name to its column: proportions or coded values, or the
        labels of a categorical factor, which are coded sum-to-zero, as
        :func:`~process_improve.experiments.designs_utils.categorical_effect_columns` does.
        """
        if self.power == "categorical":
            labels = np.asarray(data[self.names[0]]).astype(str)
            declared = [str(level) for level in self.levels]
            unknown = sorted(set(labels) - set(declared))
            if unknown:
                raise ValueError(f"Factor {self.names[0]!r} has levels {declared}; got {unknown}.")
            index = np.array([declared.index(label) for label in labels], dtype=int)
            return categorical_effect_columns(sorted_level_ranks(list(self.levels))[index], len(self.levels))
        if self.power == "square":
            values = np.asarray(data[self.names[0]], dtype=float) ** 2
        elif self.power == "cubic":
            a, b = (np.asarray(data[n], dtype=float) for n in self.names)
            values = a * b * (a - b)
        else:
            values = np.asarray(data[self.names[0]], dtype=float)
        return values[:, None]


#: A model term: the product of its atoms. The empty term is the intercept.
Term = tuple[Atom, ...]


def _plain(name: str) -> Atom:
    return Atom(name, (name,))


def _square(name: str) -> Atom:
    return Atom(f"I({name} ** 2)", (name,), "square")


def _cubic(a: str, b: str) -> Atom:
    return Atom(cubic_difference_term(a, b), (a, b), "cubic")


def _process_atom(name: str, levels: Mapping[str, Sequence]) -> Atom:
    """Return a process factor's atom: categorical when ``levels`` holds its levels, plain otherwise."""
    if name in levels:
        return Atom(name, (name,), "categorical", tuple(levels[name]))
    return _plain(name)


def _term_columns(term: Term, data: Mapping[str, np.ndarray], n_points: int) -> np.ndarray:
    """Return a term's model columns: the row-by-row products of its atoms' columns."""
    block = np.ones((n_points, 1))
    for atom in term:
        columns = atom.columns(data)
        block = (block[:, :, None] * columns[:, None, :]).reshape(n_points, -1)
    return block


class ReducedRank:
    """A patsy contrast that codes a categorical factor in ``L - 1`` columns wherever it appears.

    Patsy gives a categorical factor all ``L`` columns where the term below it is absent
    from the formula, which in a model without an intercept makes the columns add up to
    ones the blends already span (see the module docstring). This contrast hands patsy
    the ``L - 1`` columns of ``contrast`` in both cases.

    Parameters
    ----------
    contrast : object or None
        A patsy contrast such as ``Sum()`` or ``Treatment()``. ``None`` means the
        sum-to-zero coding of the design engine, chosen by the number of levels:
        ``Sum(omit=0)`` for two (the first, in sorted order, at -1), ``Sum()`` for more
        (the last at -1 in every column).
    """

    def __init__(self, contrast: object | None = None) -> None:
        self.contrast = contrast

    def code_without_intercept(self, levels: list) -> object:
        """Return the ``L - 1`` columns of the contrast."""
        contrast: Any = self.contrast if self.contrast is not None else (Sum(omit=0) if len(levels) == 2 else Sum())
        return contrast.code_without_intercept(levels)

    def code_with_intercept(self, levels: list) -> object:
        """Return the same ``L - 1`` columns: the blends stand in for the intercept."""
        return self.code_without_intercept(levels)


class _ReducedRankFactor(EvalFactor):
    """A formula factor for a categorical column, whose values carry :class:`ReducedRank` coding."""

    def eval(self, memorize_state: dict, data: Mapping) -> object:
        """Return the column, boxed with its contrast."""
        return patsy_categorical.C(super().eval(memorize_state, data), ReducedRank())


class ReducedRankFormula(str):  # noqa: SLOT000 - a str subtype cannot have non-empty __slots__, and needs its set
    """A model formula whose categorical factors patsy codes in ``L - 1`` columns wherever they appear.

    It is the formula as written, so it reads and validates as a string, and statsmodels
    keeps it as the model's formula: a refit from ``model.formula``, and a prediction at new
    runs, code the factors the same way. Patsy reads it through
    ``__patsy_get_model_desc__``, which swaps each factor that names a column in
    ``categorical`` for one carrying :class:`ReducedRank`. The coefficient names stay
    readable: ``catalyst[S.A]``, ``T:catalyst[S.A]``.

    Parameters
    ----------
    text : str
        The formula, ``"y ~ -1 + x1 + ..."``, or its right-hand side.
    categorical : Iterable[str]
        The names of the categorical columns.
    """

    categorical: frozenset[str]

    def __new__(cls, text: str, categorical: Sequence[str]) -> ReducedRankFormula:  # noqa: PYI034 - Self needs 3.11
        """Return the formula, remembering which of its factors are categorical."""
        formula = super().__new__(cls, text)
        formula.categorical = frozenset(categorical)
        return formula

    def __patsy_get_model_desc__(self, eval_env: object) -> ModelDesc:
        """Return the parsed formula, each categorical factor coded in ``L - 1`` columns."""
        desc = ModelDesc.from_formula(str(self))

        def recoded(term: PatsyTerm) -> PatsyTerm:
            return PatsyTerm(
                [
                    _ReducedRankFactor(f.code, origin=f.origin)
                    if isinstance(f, EvalFactor) and f.code in self.categorical
                    else f
                    for f in term.factors
                ]
            )

        return ModelDesc([recoded(t) for t in desc.lhs_termlist], [recoded(t) for t in desc.rhs_termlist])


@dataclass(frozen=True)
class MixtureProcessModel:
    """A model in mixture components and process factors, as its list of terms.

    Parameters
    ----------
    name : str
        The model's name: ``"interactions"``, ``"scheffe_quadratic + main_effects"``, ...
    terms : tuple[Term, ...]
        The terms, each a product of atoms. No term is an intercept, since the components sum to 1.
    """

    name: str
    terms: tuple[Term, ...]

    @property
    def n_parameters(self) -> int:
        """Number of coefficients: one per term, and ``L - 1`` per categorical factor in it."""
        return sum(int(np.prod([atom.width for atom in term])) for term in self.terms)

    @property
    def has_squares(self) -> bool:
        """True when a term holds a squared process factor, which then needs three levels."""
        return any(atom.power == "square" for term in self.terms for atom in term)

    def formula_rhs(self) -> str:
        """Return the right-hand side of the model's formula, without an intercept: ``"-1 + x1 + x2 + ..."``."""
        return " + ".join(["-1", *(":".join(atom.code for atom in term) for term in self.terms)])

    def matrix(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return the model matrix, a column per term: components as proportions, process factors coded.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            One array per factor, all the same length: labels for a categorical factor.

        Returns
        -------
        np.ndarray
            The model matrix, of shape ``(n_points, n_parameters)``.
        """
        n_points = len(data[next(iter(data))])  # a dict of columns or a DataFrame
        return np.hstack([_term_columns(term, data, n_points) for term in self.terms])


def scheffe_terms(names: list[str], model: str) -> list[Term]:
    """Return a Scheffé model's terms, in the column order of
    :func:`~process_improve.experiments.designs_mixture_constrained.scheffe_matrix`.

    That is the components, their products in pairs, the cubic differences (full cubic
    model only) and the products in threes (special and full cubic models).
    """
    model = scheffe_model(model)
    x = [_plain(n) for n in names]
    terms: list[Term] = [(atom,) for atom in x]
    if SCHEFFE_DEGREE[model] >= 2:
        terms += list(itertools.combinations(x, 2))
    if model == "scheffe_cubic":
        terms += [(_cubic(a, b),) for a, b in itertools.combinations(names, 2)]
    if SCHEFFE_DEGREE[model] == 3:
        terms += list(itertools.combinations(x, 3))
    return terms


def process_terms(names: list[str], model: str, levels: Mapping[str, Sequence] | None = None) -> list[Term]:
    """Return a process model's terms, intercept first: the intercept, main effects, interactions, squares.

    This is the column order of the candidate exchange's model matrix
    (:func:`~process_improve.experiments.designs_constrained.model_matrix`). A factor named
    in ``levels`` is categorical, so it has no square.
    """
    if model not in PROCESS_MODELS:
        raise ValueError(f"Unknown process model {model!r}; choose from {', '.join(PROCESS_MODELS)}.")
    levels = levels or {}
    z = [_process_atom(n, levels) for n in names]
    terms: list[Term] = [(), *((atom,) for atom in z)]
    if model in {"interactions", "quadratic"}:
        terms += list(itertools.combinations(z, 2))
    if model == "quadratic":
        terms += [(_square(n),) for n in names if n not in levels]
    return terms


def mixture_process_model(
    mixture: list[str], process: list[str], model_type: str, levels: Mapping[str, Sequence] | None = None
) -> MixtureProcessModel:
    """Build the mixture-process model ``model_type`` names, for these components and process factors.

    Parameters
    ----------
    mixture : list[str]
        The mixture components' names.
    process : list[str]
        The process factors' names.
    model_type : str
        ``"main_effects"``, ``"interactions"``, ``"quadratic"`` (or ``"kcv"``), or a Scheffé
        model and a process model joined by ``+`` (added) or ``*`` (crossed), such as
        ``"scheffe_quadratic * interactions"``. See the module docstring.
    levels : Mapping[str, Sequence] or None
        The levels of each categorical process factor, keyed by its name; the process
        factors it does not name are continuous.

    Returns
    -------
    MixtureProcessModel
        The model, with its terms in order.

    Raises
    ------
    ValueError
        If ``model_type`` names no such model.

    Examples
    --------
    >>> mixture_process_model(["x1", "x2", "x3"], ["T", "P"], "quadratic").n_parameters
    15
    >>> mixture_process_model(["x1", "x2", "x3"], ["T", "P"], "scheffe_quadratic * interactions").n_parameters
    24
    >>> mixture_process_model(["x1", "x2", "x3"], ["T", "cat"], "interactions", {"cat": "ABC"}).n_parameters
    17
    """
    combined = _COMBINED.fullmatch(model_type)
    if combined is not None:
        left, operator, right = combined.groups()
        blending = scheffe_terms(mixture, left)
        settings = process_terms(process, right, levels)
        if operator == "*":
            terms = [term + other for term in blending for other in settings]
        else:
            terms = blending + settings[1:]
        return MixtureProcessModel(f"{scheffe_model(left)} {operator} {right}", tuple(terms))

    order = _ORDER_ALIASES.get(model_type, model_type)
    if order not in PROCESS_MODELS:
        raise ValueError(
            f"model_type={model_type!r} is not a mixture-process model; choose from "
            f"{', '.join([*PROCESS_MODELS, *_ORDER_ALIASES])}, or join a Scheffé model and a process model with "
            "'+' (added) or '*' (crossed), e.g. 'scheffe_quadratic + main_effects'."
        )
    main_effects = process_terms(process, "main_effects", levels)[1:]
    if order == "main_effects":
        terms = scheffe_terms(mixture, "scheffe_linear") + main_effects
    else:
        # z_k itself is the sum of the x_i z_k, so the process main effects enter through those products.
        linear = scheffe_terms(mixture, "scheffe_linear")
        beyond_main = process_terms(process, order, levels)[1 + len(process) :]
        terms = (
            scheffe_terms(mixture, "scheffe_quadratic")
            + [term + effect for term in linear for effect in main_effects]
            + beyond_main
        )
    return MixtureProcessModel(order, tuple(terms))


def _refuse_rules(constraints: list[Constraint], process: list[Factor]) -> None:
    """Refuse constraints that name categorical factors, or join comparisons with ``and``, ``or``, ``not``."""
    categorical = [f.name for f in process if f.type == FactorType.categorical]
    named = [c.expression for c in constraints if names_categorical(c.expression, categorical)]
    if named:
        raise ValueError(
            f"The constraints {named} name categorical factors or join comparisons with and, or, not, which are "
            "not supported beside mixture components yet; give them for a design of process factors only."
        )


def split_constraints(
    constraints: list[Constraint], mixture: list[Factor], process: list[Factor]
) -> tuple[list[Constraint], list[Constraint]]:
    """Return the constraints on the components and the constraints on the process factors.

    A mixture-process region is the product of the constrained simplex and the
    constrained process box, so each constraint must name factors of one side only. A
    constraint that cannot be parsed, or names no factor, goes to the process side,
    whose parser then gives its usual message.

    Raises
    ------
    ValueError
        If a constraint names both a component and a process factor.
    """
    _refuse_rules(constraints, process)
    on_mixture = {nfkc(f.name) for f in mixture}
    on_process = {nfkc(f.name) for f in process}
    blends: list[Constraint] = []
    settings: list[Constraint] = []
    for constraint in constraints:
        try:
            named = set(_factor_names(ast.parse(constraint.expression, mode="eval")))
        except SyntaxError:
            named = set()
        if named & on_mixture and named & on_process:
            raise ValueError(
                f"Constraint {constraint.expression!r} names both mixture components "
                f"{sorted(named & on_mixture)} and process factors {sorted(named & on_process)}. A constraint must "
                "limit the blends or the process settings, not both together."
            )
        (blends if named & on_mixture else settings).append(constraint)
    return blends, settings


def _candidate_blends(mixture: list[Factor], constraints: list[Constraint]) -> np.ndarray:
    """Return every candidate blend of the constrained simplex: vertices, edge points, face centroids, centroid."""
    kinds = mixture_candidates(*mixture_inequalities(mixture, constraints))
    return _unique_rows(np.vstack(list(kinds.values())))


def _candidate_settings(
    process: list[Factor], constraints: list[Constraint], model: MixtureProcessModel, criterion: str
) -> np.ndarray:
    """Return the candidate process settings: the grid, and where each constraint crosses it, at every level.

    Continuous factors are in coded units, at three levels or more when one is squared in
    the model and two may do otherwise; categorical factors take each of their levels, as
    labels. The rows hold the process factors in their order.
    """
    continuous = [f for f in process if f.type != FactorType.categorical]
    categorical = [f for f in process if f.type == FactorType.categorical]
    names = {f.name for f in continuous}
    inequalities = [g for c in constraints for g in parse_constraint(c.expression, names)]
    coded, cats, _counts = build_candidates(
        _Region(continuous, categorical, inequalities),
        model_type="quadratic" if model.has_squares else "interactions",
        criterion=criterion,
    )
    if len(coded) == 0:
        raise ValueError("No process setting satisfies all the constraints; check them for conflicts.")
    if not categorical:
        return coded
    settings = np.empty((len(coded), len(process)), dtype=object)
    by_name = {f.name: j for j, f in enumerate(process)}
    for j, f in enumerate(continuous):
        settings[:, by_name[f.name]] = coded[:, j]
    for j, f in enumerate(categorical):
        settings[:, by_name[f.name]] = np.asarray(f.levels, dtype=object)[cats[:, j]]
    return settings


def _crossed_points(
    factors: list[Factor], blends: np.ndarray, settings: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return every blend at every setting, as rows in the factors' order, sampled down to ``MAX_CANDIDATES``."""
    index = np.array(list(itertools.product(range(len(blends)), range(len(settings)))))
    if len(index) > MAX_CANDIDATES:
        logger.info("%d candidate runs are too many to search; sampling %d of them.", len(index), MAX_CANDIDATES)
        index = index[np.sort(rng.choice(len(index), size=MAX_CANDIDATES, replace=False))]
    is_mixture = [f.type == FactorType.mixture for f in factors]
    points = np.empty((len(index), len(factors)), dtype=settings.dtype)
    points[:, is_mixture] = blends[index[:, 0]]
    points[:, np.logical_not(is_mixture)] = settings[index[:, 1]]
    return points


def _columns(factors: list[Factor], points: np.ndarray) -> dict[str, np.ndarray]:
    """Return each factor's column of ``points``: labels for a categorical factor, floats otherwise."""
    return {
        f.name: points[:, j] if f.type == FactorType.categorical else points[:, j].astype(float)
        for j, f in enumerate(factors)
    }


def _region_sample(
    factors: list[Factor], constraints: list[Constraint], n: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Draw ``n`` points uniformly from the mixture-process region, each categorical factor at uniform levels."""
    from process_improve.experiments.region import DesignRegion  # noqa: PLC0415  region imports this module

    region = DesignRegion(factors, constraints)
    sample = region.sample(n, rng)
    data: dict[str, np.ndarray] = {name: sample[:, j] for j, name in enumerate(region.names)}
    for f in factors:
        if f.type == FactorType.categorical:
            data[f.name] = rng.choice(np.asarray(f.levels, dtype=object), size=n)
    return data


def category_levels(factors: list[Factor]) -> dict[str, tuple]:
    """Return the levels of each categorical factor, keyed by its name."""
    return {f.name: tuple(f.levels or ()) for f in factors if f.type == FactorType.categorical}


def mixture_process_design(
    factors: list[Factor],
    budget: int | None,
    constraints: list[Constraint] | None = None,
    options: ConstrainedOptions | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate an optimal design for mixture components together with process factors.

    Parameters
    ----------
    factors : list[Factor]
        Mixture components (``low`` and ``high`` their bounds as proportions), and
        continuous and categorical process factors, in any order.
    budget : int or None
        Number of runs. ``None`` means the model's number of terms plus 3. A budget below
        the number of terms raises a ``ValueError``. A candidate run may be chosen more than once.
    constraints : list[Constraint] or None
        Linear inequalities in the proportions, and inequalities in the process factors'
        actual units; each names components only or process factors only.
    options : ConstrainedOptions or None
        ``model_type``, a mixture-process model (see :func:`mixture_process_model`;
        default ``"interactions"``), and ``criterion``: ``"d_optimal"`` (default),
        ``"i_optimal"``, ``"a_optimal"``, ``"e_optimal"``, ``"g_optimal"`` or ``"k_optimal"``.
    random_state : int, numpy.random.Generator or None
        Seed for the exchange's random starts and the region sample.

    Returns
    -------
    tuple[np.ndarray, dict]
        The runs ``(n_runs, n_factors)`` in the factors' order, components as proportions,
        continuous process factors in coded units and categorical ones as level labels
        (the array is then of objects), and metadata. ``formula`` is the model's
        right-hand side, to give :func:`~process_improve.experiments.evaluate_design` or
        :func:`~process_improve.experiments.analyze_experiment`.

    Raises
    ------
    ValueError
        If a constraint names both kinds of factor or a categorical factor, the region is
        empty, or the candidates cannot support the model.
    """
    opts = options if options is not None else ConstrainedOptions(model_type="interactions")
    mixture = [f for f in factors if f.type == FactorType.mixture]
    process = [f for f in factors if f.type != FactorType.mixture]
    constraints = list(constraints or [])
    on_mixture, on_process = split_constraints(constraints, mixture, process)
    levels = category_levels(process)
    model = mixture_process_model([f.name for f in mixture], [f.name for f in process], opts.model_type, levels)
    p = model.n_parameters
    rng = check_random_state(random_state)

    blends = _candidate_blends(mixture, on_mixture)
    settings = _candidate_settings(process, on_process, model, opts.criterion)
    points = _crossed_points(factors, blends, settings, rng)
    f_pool = model.matrix(_columns(factors, points))
    if np.linalg.matrix_rank(f_pool) < p:
        raise ValueError(
            f"The region ({len(blends)} candidate blends at {len(settings)} process settings) cannot support the "
            f"{model.name!r} model with {p} terms. Use a simpler model or loosen the constraints."
        )

    if budget is not None:
        refuse_budget_below_model(budget, p, repr(model.name))
    budget = p + _EXTRA_RUNS if budget is None else budget

    def region_rows() -> np.ndarray:
        return model.matrix(_region_sample(factors, constraints, _N_REGION_SAMPLES, rng))

    criterion = make_criterion(opts.criterion, p, region_rows, f_pool)
    rows, value = fedorov_exchange(f_pool, budget, np.empty((0, p)), rng, criterion)
    meta = {
        "backend": "candidate_exchange",
        "method": f"{opts.criterion}_mixture_process",
        "optimality_criterion": opts.criterion,
        "model_type": model.name,
        "formula": model.formula_rhs(),
        "n_parameters": p,
        "mixture_components": [f.name for f in mixture],
        "process_factors": [f.name for f in process],
        "categorical_process_factors": sorted(levels),
        "n_blends": len(blends),
        "n_process_settings": len(settings),
        "n_candidates": len(points),
        "constraints": [c.expression for c in constraints],
        "constraints_enforced": True,
        **criterion_metadata(criterion, value),
    }
    return points[rows], meta
