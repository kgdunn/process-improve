# (c) Kevin Dunn, 2010-2026. MIT License.

"""Mixture-process models: mixture components and process factors in one model.

In a mixture-process experiment the proportions ``x_1, ..., x_q`` of the ingredients sum
to 1, while the process factors ``z_1, ..., z_m`` (temperature, mixing speed) range
freely, in coded units from -1 to +1. A model of both is a list of terms, each a product
of *atoms*: a component ``x_i``, a process factor ``z_k``, a square ``z_k^2``, or the
cubic difference ``x_i x_j (x_i - x_j)`` of a full cubic Scheffé model. The model matrix
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
``"interactions"`` or ``"quadratic"``.

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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from process_improve._random import check_random_state
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
    """One factor of a model term: a component or process factor, a square, or a cubic difference.

    Parameters
    ----------
    code : str
        How the atom is written in a formula: ``"x1"``, ``"I(T ** 2)"``,
        ``"I(x1 * x2 * (x1 - x2))"``.
    names : tuple[str, ...]
        The factors it is made of: one for a plain factor or a square, two for a cubic difference.
    power : {"plain", "square", "cubic"}
        Which of the three it is.
    """

    code: str
    names: tuple[str, ...]
    power: str = "plain"

    def values(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return the atom's value at each point; ``data`` maps a factor name to its column."""
        if self.power == "square":
            return data[self.names[0]] ** 2
        if self.power == "cubic":
            a, b = (data[n] for n in self.names)
            return a * b * (a - b)
        return data[self.names[0]]


#: A model term: the product of its atoms. The empty term is the intercept.
Term = tuple[Atom, ...]


def _plain(name: str) -> Atom:
    return Atom(name, (name,))


def _square(name: str) -> Atom:
    return Atom(f"I({name} ** 2)", (name,), "square")


def _cubic(a: str, b: str) -> Atom:
    return Atom(cubic_difference_term(a, b), (a, b), "cubic")


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
        """Number of coefficients: one per term."""
        return len(self.terms)

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
            One array per factor, all the same length.

        Returns
        -------
        np.ndarray
            The model matrix, of shape ``(n_points, n_parameters)``.
        """
        return np.column_stack([np.prod([atom.values(data) for atom in term], axis=0) for term in self.terms])


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


def process_terms(names: list[str], model: str) -> list[Term]:
    """Return a process model's terms, intercept first: the intercept, main effects, interactions, squares.

    This is the column order of the candidate exchange's model matrix
    (:func:`~process_improve.experiments.designs_constrained.model_matrix`).
    """
    if model not in PROCESS_MODELS:
        raise ValueError(f"Unknown process model {model!r}; choose from {', '.join(PROCESS_MODELS)}.")
    z = [_plain(n) for n in names]
    terms: list[Term] = [(), *((atom,) for atom in z)]
    if model in {"interactions", "quadratic"}:
        terms += list(itertools.combinations(z, 2))
    if model == "quadratic":
        terms += [(_square(n),) for n in names]
    return terms


def mixture_process_model(mixture: list[str], process: list[str], model_type: str) -> MixtureProcessModel:
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
    """
    combined = _COMBINED.fullmatch(model_type)
    if combined is not None:
        left, operator, right = combined.groups()
        blending = scheffe_terms(mixture, left)
        settings = process_terms(process, right)
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
    main_effects = process_terms(process, "main_effects")[1:]
    if order == "main_effects":
        terms = scheffe_terms(mixture, "scheffe_linear") + main_effects
    else:
        # z_k itself is the sum of the x_i z_k, so the process main effects enter through those products.
        linear = scheffe_terms(mixture, "scheffe_linear")
        beyond_main = process_terms(process, order)[1 + len(process) :]
        terms = (
            scheffe_terms(mixture, "scheffe_quadratic")
            + [term + effect for term in linear for effect in main_effects]
            + beyond_main
        )
    return MixtureProcessModel(order, tuple(terms))


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
    """Return the candidate process settings in coded units: the grid, and where each constraint crosses it.

    Three levels or more when a process factor is squared in the model; two may do otherwise.
    """
    names = {f.name for f in process}
    region = _Region(process, [], [g for c in constraints for g in parse_constraint(c.expression, names)])
    coded, _cats, _counts = build_candidates(
        region, model_type="quadratic" if model.has_squares else "interactions", criterion=criterion
    )
    if len(coded) == 0:
        raise ValueError("No process setting satisfies all the constraints; check them for conflicts.")
    return coded


def _crossed_points(
    factors: list[Factor], blends: np.ndarray, settings: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return every blend at every setting, as rows in the factors' order, sampled down to ``MAX_CANDIDATES``."""
    index = np.array(list(itertools.product(range(len(blends)), range(len(settings)))))
    if len(index) > MAX_CANDIDATES:
        logger.info("%d candidate runs are too many to search; sampling %d of them.", len(index), MAX_CANDIDATES)
        index = index[np.sort(rng.choice(len(index), size=MAX_CANDIDATES, replace=False))]
    is_mixture = [f.type == FactorType.mixture for f in factors]
    points = np.empty((len(index), len(factors)))
    points[:, is_mixture] = blends[index[:, 0]]
    points[:, np.logical_not(is_mixture)] = settings[index[:, 1]]
    return points


def _columns(factors: list[Factor], points: np.ndarray) -> dict[str, np.ndarray]:
    return {f.name: points[:, j] for j, f in enumerate(factors)}


def mixture_process_design(
    factors: list[Factor],
    budget: int | None,
    constraints: list[Constraint] | None = None,
    options: ConstrainedOptions | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate an optimal design for mixture components together with continuous process factors.

    Parameters
    ----------
    factors : list[Factor]
        Mixture components (``low`` and ``high`` their bounds as proportions) and
        continuous process factors, in any order.
    budget : int or None
        Number of runs. ``None`` means the model's number of terms plus 3. A budget below
        the number of terms is raised to it, with a warning, and recorded in
        ``metadata["budget_requested"]``. A candidate run may be chosen more than once.
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
        The runs ``(n_runs, n_factors)`` in the factors' order, components as proportions
        and process factors in coded units, and metadata. ``formula`` is the model's
        right-hand side, to give :func:`~process_improve.experiments.evaluate_design` or
        :func:`~process_improve.experiments.analyze_experiment`.

    Raises
    ------
    ValueError
        If a process factor is categorical, a constraint names both kinds of factor, the
        region is empty, or the candidates cannot support the model.
    """
    opts = options if options is not None else ConstrainedOptions(model_type="interactions")
    mixture = [f for f in factors if f.type == FactorType.mixture]
    process = [f for f in factors if f.type != FactorType.mixture]
    categorical = [f.name for f in process if f.type == FactorType.categorical]
    if categorical:
        raise ValueError(
            f"Process factors of a mixture-process design must be continuous; {categorical} are categorical."
        )
    constraints = list(constraints or [])
    on_mixture, on_process = split_constraints(constraints, mixture, process)
    model = mixture_process_model([f.name for f in mixture], [f.name for f in process], opts.model_type)
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

    requested = budget
    budget = p + _EXTRA_RUNS if budget is None else budget
    if budget < p:
        logger.warning(
            "A budget of %d run(s) cannot estimate the %r model with %d terms; raising the budget to %d.",
            budget,
            model.name,
            p,
            p,
        )
        budget = p

    def region_rows() -> np.ndarray:
        from process_improve.experiments.region import DesignRegion  # noqa: PLC0415  region imports this module

        sample = DesignRegion(factors, constraints).sample(_N_REGION_SAMPLES, rng)
        return model.matrix(_columns(factors, sample))

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
        "n_blends": len(blends),
        "n_process_settings": len(settings),
        "n_candidates": len(points),
        "constraints": [c.expression for c in constraints],
        "constraints_enforced": True,
        **criterion_metadata(criterion, value),
    }
    if requested is not None and requested != budget:
        meta["budget_requested"] = requested
    return points[rows], meta
