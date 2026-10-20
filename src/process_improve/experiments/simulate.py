# (c) Kevin Dunn, 2010-2026. MIT License.

"""How a design would do on a process you can only hypothesise: simulated studies, judged as the next step judges them.

:func:`~process_improve.experiments.compare.compare_designs` and
:func:`~process_improve.experiments.evaluate.evaluate_design` score a design for one fixed model,
analytically. Two things they cannot see decide how a real study goes: terms outside that model
(an interaction a screening design was not built to estimate) bias the effects inside it, and the
analysis has to choose the model from the same noisy runs. :func:`simulate_design_performance`
sees both, by making each design many times on a :class:`~process_improve.experiments.TrueProcess`
and judging every simulated study the way :func:`~process_improve.experiments.recommend_next_step`
judges real ones.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from process_improve._random import check_random_state
from process_improve.experiments.compare import _coded_runs, _factor_names, _labelled, _unique
from process_improve.experiments.factor import DesignResult, Factor
from process_improve.experiments.next_step import (
    _CURVATURE,
    _code,
    _design,
    _optimum,
    _select,
    _Study,
    _with_blocks,
)
from process_improve.experiments.optimization import _extract_b_and_B
from process_improve.experiments.true_process import TrueProcess, _canonical

__all__ = ["simulate_design_performance"]

SimGoal = Literal["learn", "maximize", "minimize", "target"]

#: Two columns whose centred correlation is this close to 1 in size are the same column: exact aliases.
_SAME_COLUMN = 1e-9
#: The term the analysis reports for curvature at the centre runs of a two-level design; it is aliased
#: with every square there.
CURVATURE = "curvature"
#: The largest number of false alarms listed by term for each design.
_LISTED = 5
#: Grid points per factor in the search for the true optimum, and the largest grid searched whole.
_GRID_LEVELS = 5
_LARGEST_GRID = 200_000


def simulate_design_performance(  # noqa: PLR0913 - each setting is a separate input to the simulation
    designs: Mapping[str, DesignResult | pd.DataFrame | np.ndarray]
    | Sequence[DesignResult]
    | DesignResult
    | pd.DataFrame,
    truth: TrueProcess,
    *,
    goal: SimGoal = "learn",
    target: float | None = None,
    n_simulations: int = 200,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = 42,
) -> dict[str, Any]:
    """Simulate studies with each design on a hypothesised process, and report what their analysis finds.

    Each design is made ``n_simulations`` times on ``truth``, with fresh noise each time, and each
    simulated study is judged as :func:`~process_improve.experiments.recommend_next_step` judges
    real runs: the largest standard model the runs estimate, the staged analysis of a definitive
    screening design, Lenth's method for a saturated two-level design, or a stepwise search (AICc)
    when the second-order terms are partly aliased; and a gate on whether anything stands out from
    the noise at all. So the numbers include what an analytic power calculation leaves out: the
    bias that terms outside the model put on the terms inside it, and the chance that the analysis
    picks the wrong model.

    A true term counts as found when the analysis finds it, or a term that the design makes
    identical to it (an exact alias, such as ``D`` for ``A:B`` in a resolution III fraction, or the
    curvature at the centre runs for a square). Every other term found is a false alarm.

    Parameters
    ----------
    designs : mapping of str to design, sequence of DesignResult, or one design
        The designs in coded units, as :func:`~process_improve.experiments.compare_designs` takes
        them: a :class:`DesignResult`, a DataFrame of factor columns (``RunOrder`` and ``Block``
        are ignored) or an array, under a label. One design alone is also taken. Every design has
        the same factors.
    truth : TrueProcess
        The process the runs are made on: its terms in coded units, and the noise. A factor no
        term names is inert. Its terms are the ones a design should find.
    goal : {"learn", "maximize", "minimize", "target"}, default "learn"
        With a goal other than ``"learn"``, each study also predicts its best settings, as the
        next step does before confirming, and the report says how far the true response there
        falls short of the true best inside the ranges.
    target : float or None
        The response wanted, for ``goal="target"``.
    n_simulations : int, default 200
        Simulated studies per design. A share from 200 studies is within about 0.07 of its
        limit (95%), at worst.
    alpha : float, default 0.05
        The level of every test of the judging.
    random_state : int, numpy.random.Generator or None, default 42
        Seed for the noise. Each design gets its own stream, drawn in the order the designs are
        given, so the same arguments give the same report.

    Returns
    -------
    dict[str, Any]
        ``truth`` (the terms and ``sigma``), ``goal``, ``n_simulations``, ``alpha``, ``precision``
        (the widest 95% margin of a share), ``true_optimum`` toward a goal (the coded settings
        and the true response there), and ``designs``: for each design label (with its size, as
        ``"DSD [n=21]"``), a dict with

        * ``n_runs``;
        * ``found``: each true term's share of studies that find it;
        * ``found_all``: the share that find every true term;
        * ``found_exactly``: the share that find every true term and nothing else;
        * ``false_alarms``: the mean number of terms found that are not true, and
          ``false_alarm_terms``, the most frequent of them with their shares;
        * ``nothing_stood_out``: the share in which nothing stood out from the noise;
        * ``judged_by``: how the studies were judged, with the share judged each way;
        * toward a goal, ``shortfall``: how far the true response at each study's predicted best
          settings falls short of the true best (``mean``, ``median``, ``p90``, ``worst``), in the
          response's units and, as ``mean_in_sigma``, in noise standard deviations;
        * ``summary``: one sentence with the main numbers.

    Raises
    ------
    ValueError
        If a design is not a table of finite numbers, the designs differ in their factors, the
        truth uses a factor the designs do not have, ``goal`` is unknown or a target goal has no
        target, ``n_simulations`` is below 1, or ``alpha`` is not between 0 and 1.
    TypeError
        If ``truth`` is not a :class:`TrueProcess`.

    Examples
    --------
    >>> from process_improve.experiments import Factor, TrueProcess, generate_design
    >>> factors = [Factor(name=n, low=-1, high=1) for n in "ABCDEFGH"]
    >>> truth = TrueProcess({"Intercept": 100, "A": 10, "C": 7, "G": -6, "A:C": 8}, sigma=3)
    >>> report = simulate_design_performance(
    ...     {"DSD": generate_design(factors, "dsd", n_center_points=5)}, truth, n_simulations=50
    ... )
    >>> report["designs"]["DSD [n=21]"]["found_all"]
    1.0
    """
    _check_settings(truth, goal, target, n_simulations, alpha)
    labelled = _labelled({"design": designs} if _single(designs) else designs)  # type: ignore[arg-type]
    if not labelled:
        raise ValueError("Give at least one design to simulate.")
    runs = [_coded_runs(design) for _, design in labelled]
    names = _factor_names(runs)
    missing = [f for f in truth.factors if f not in names]
    if missing:
        raise ValueError(f"The process uses the factor(s) {missing}, which the designs do not have: {names}.")
    factors = [Factor(name=n, low=-1, high=1) for n in names]
    best = _true_best(truth, names, goal, target) if goal != "learn" else None
    rng = check_random_state(random_state)
    seeds = rng.integers(2**32, size=len(labelled))
    reports: dict[str, dict[str, Any]] = {}
    for (label, _), (x, _), seed in zip(labelled, runs, seeds, strict=True):
        sized = _unique(f"{label} [n={len(x)}]", list(reports))
        frame = pd.DataFrame(x, columns=names)
        mean = truth.mean(frame)
        studies = [
            _one_study(factors, frame.assign(y=mean + truth.sigma * noise), alpha, goal, target)
            for noise in np.random.default_rng(seed).standard_normal((n_simulations, len(x)))
        ]
        reports[sized] = _report(studies, truth, best, len(x))
    return {
        "truth": {"coefficients": dict(truth.coefficients), "sigma": truth.sigma},
        "goal": goal,
        "n_simulations": n_simulations,
        "alpha": alpha,
        "precision": round(1.96 * 0.5 / math.sqrt(n_simulations), 3),
        **({"true_optimum": best} if best is not None else {}),
        "designs": reports,
    }


def _single(designs: object) -> bool:
    return isinstance(designs, (DesignResult, pd.DataFrame, np.ndarray))


def _check_settings(truth: object, goal: str, target: float | None, n_simulations: int, alpha: float) -> None:
    if not isinstance(truth, TrueProcess):
        raise TypeError(f"truth must be a TrueProcess, not {type(truth).__name__}.")
    if goal not in ("learn", "maximize", "minimize", "target"):
        raise ValueError(f"goal must be 'learn', 'maximize', 'minimize' or 'target', not {goal!r}.")
    if goal == "target" and target is None:
        raise ValueError("A target goal needs its target.")
    if int(n_simulations) != n_simulations or n_simulations < 1:
        raise ValueError(f"n_simulations must be a whole number of at least 1, not {n_simulations!r}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, not {alpha!r}.")


# --------------------------------------------------------------------------- one simulated study
def _one_study(
    factors: list[Factor], runs: pd.DataFrame, alpha: float, goal: SimGoal, target: float | None
) -> dict[str, Any]:
    """Judge one simulated study's ``runs`` (response ``y``): what was found, and where the best settings are."""
    study = _Study(factors, _with_blocks(_code(factors, runs, "y")), "y")
    design = _design(study)
    study.data[_CURVATURE] = design.center.astype(float)
    model = _select(study, design, None, alpha)
    found = model.active(alpha)
    if model.signal and model.p_value(_CURVATURE) < alpha:
        found = [*found, _CURVATURE]
    columns = {term: _column(study, term) for term in found}
    settings = None
    if goal != "learn":
        settings = _optimum(study, model, goal, target, seed=0)  # type: ignore[arg-type]
    return {"method": model.method, "found": columns, "signal": model.signal, "settings": settings, "study": study}


def _column(study: _Study, term: str) -> np.ndarray:
    """Return a term's column on the runs, from its internal name; the curvature term is the centre runs."""
    if term == _CURVATURE:
        return study.data[_CURVATURE].to_numpy(dtype=float)
    _, factors = _canonical(term)
    return np.prod([study.data[f].to_numpy(dtype=float) for f in factors], axis=0)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Say whether two columns are the same up to sign and a constant: exactly aliased on these runs."""
    a, b = a - a.mean(), b - b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < _SAME_COLUMN or nb < _SAME_COLUMN:
        return False  # a column that does not vary is not estimated, so it cannot stand for another
    return abs(float(a @ b)) / (na * nb) > 1 - _SAME_COLUMN


# --------------------------------------------------------------------------- the report for one design
def _report(studies: list[dict[str, Any]], truth: TrueProcess, best: dict[str, Any] | None, n_runs: int) -> dict:
    """Summarise the simulated studies of one design against the truth."""
    study = studies[0]["study"]  # the runs' settings are the same in every study; only the response differs
    truths = {t: _column(study, _internal(study, t)) for t in truth.coefficients if t != "Intercept"}
    n = len(studies)
    hits: Counter[str] = Counter()
    false: Counter[str] = Counter()
    found_all = found_exactly = 0
    for one in studies:
        caught, spurious = _classify(one, truths)
        hits.update(caught)
        false.update(spurious)
        found_all += len(caught) == len(truths)
        found_exactly += len(caught) == len(truths) and not spurious
    report: dict[str, Any] = {
        "n_runs": n_runs,
        "found": {t: hits[t] / n for t in truths},
        "found_all": found_all / n,
        "found_exactly": found_exactly / n,
        "false_alarms": sum(false.values()) / n,
        "false_alarm_terms": {t: c / n for t, c in false.most_common(_LISTED)},
        "nothing_stood_out": sum(not one["signal"] for one in studies) / n,
        "judged_by": {m: c / n for m, c in Counter(one["method"] for one in studies).most_common()},
    }
    if best is not None:
        report["shortfall"] = _shortfall(studies, truth, best)
    report["summary"] = _summary(report, truth)
    return report


def _classify(one: dict[str, Any], truths: dict[str, np.ndarray]) -> tuple[set[str], list[str]]:
    """Return the true terms a study found (itself or through an exact alias), and the other terms it found."""
    found = one["found"]
    caught = {t for t, col in truths.items() if any(_same(col, c) for c in found.values())}
    spurious = [
        CURVATURE if term == _CURVATURE else one["study"].real(term)
        for term, col in found.items()
        if not any(_same(col, t) for t in truths.values())
    ]
    return caught, spurious


def _internal(study: _Study, term: str) -> str:
    """Write a true term in the study's internal names (``x0``, ``x1``, ...)."""
    _, factors = _canonical(term)
    position = {f.name: f"x{i}" for i, f in enumerate(study.factors)}
    return ":".join(position[f] for f in factors)


def _shortfall(studies: list[dict[str, Any]], truth: TrueProcess, best: dict[str, Any]) -> dict[str, float]:
    """Return how far the true response at each study's predicted best falls short of the true best."""
    chosen = pd.DataFrame([one["settings"] for one in studies], columns=list(best["coded"]))
    short = np.maximum(_loss(truth.mean(chosen), best["goal"], best.get("target")) - best["loss"], 0.0)
    return {
        "mean": float(short.mean()),
        "median": float(np.median(short)),
        "p90": float(np.quantile(short, 0.9)),
        "worst": float(short.max()),
        "mean_in_sigma": float(short.mean() / truth.sigma),
    }


def _loss(values: np.ndarray, goal: str, target: float | None) -> np.ndarray:
    """Return how far each true response is from what the goal wants: smaller is better."""
    if goal == "maximize":
        return -values
    if goal == "minimize":
        return values
    return np.abs(values - float(target))  # type: ignore[arg-type]


def _summary(report: dict[str, Any], truth: TrueProcess) -> str:
    """Say the main numbers of one design's report in a sentence."""
    found = ", ".join(f"{t} {100 * s:.0f}%" for t, s in report["found"].items())
    text = (
        f"Finds every true term in {100 * report['found_all']:.0f}% of studies ({found}), with "
        f"{report['false_alarms']:.2g} false alarms per study"
    )
    shortfall = report.get("shortfall")
    if shortfall and shortfall["worst"] == 0:
        text += "; its predicted best settings are the true best in every study"
    elif shortfall:
        text += (
            f"; its predicted best settings fall short of the true best by {shortfall['mean']:.3g} on average "
            f"({shortfall['mean_in_sigma']:.2g} noise standard deviations), and by {shortfall['p90']:.3g} or more "
            "in 1 study in 10"
        )
    return text + "."


# --------------------------------------------------------------------------- the true optimum
def _true_best(
    truth: TrueProcess, names: list[str], goal: SimGoal, target: float | None, span: float = 1.0
) -> dict[str, Any]:
    """Return the coded settings within ``span`` of the centre where the true process best meets the goal.

    A grid of :data:`_GRID_LEVELS` levels per factor (corners included) finds the neighbourhood of
    the best, and L-BFGS-B, with the polynomial's exact gradient, refines the five best grid points.
    The inert factors are left at the centre. ``span`` is 1 for the ranges themselves.
    """
    used = [n for n in names if n in truth.factors]
    terms = [{"term": t, "coefficient": v} for t, v in truth.coefficients.items()]
    b0, b, big_b = _extract_b_and_B(terms, used)

    def loss(z: np.ndarray) -> tuple[float, np.ndarray]:
        value, slope = b0 + z @ b + z @ big_b @ z, b + 2 * big_b @ z
        if goal == "target":
            miss = value - float(target)  # type: ignore[arg-type]
            return miss**2, 2 * miss * slope
        return (-value, -slope) if goal == "maximize" else (value, slope)

    candidates = span * _grid(len(used))
    values = b0 + candidates @ b + np.einsum("ij,jk,ik->i", candidates, big_b, candidates)
    order = np.argsort(_loss(values, goal, target))[:5]
    best_z, best_value = candidates[order[0]], math.inf
    for start in candidates[order] if used else []:  # a process with no factor in it has nothing to search
        found = optimize.minimize(loss, start, jac=True, bounds=[(-span, span)] * len(used), method="L-BFGS-B")
        if float(found.fun) < best_value:
            best_value, best_z = float(found.fun), np.clip(found.x, -span, span)
    coded = dict.fromkeys(names, 0.0) | {n: float(v) for n, v in zip(used, best_z, strict=True)}
    value = float(truth.mean(pd.DataFrame([coded])).item())
    return {
        "coded": coded,
        "response": value,
        "goal": goal,
        "loss": float(_loss(np.array([value]), goal, target)[0]),
        **({"target": target} if goal == "target" else {}),
    }


def _grid(k: int) -> np.ndarray:
    """Return the grid the search for the true optimum starts from: every point when small, else a sample."""
    levels = np.linspace(-1.0, 1.0, _GRID_LEVELS)
    if _GRID_LEVELS**k <= _LARGEST_GRID:
        return np.array(list(itertools.product(levels, repeat=k)))
    rng = np.random.default_rng(0)
    corners = rng.choice([-1.0, 1.0], size=(_LARGEST_GRID // 2, k))
    return np.vstack([corners, rng.choice(levels, size=(_LARGEST_GRID // 2, k))])
