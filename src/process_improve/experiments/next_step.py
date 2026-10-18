# (c) Kevin Dunn, 2010-2026. MIT License.

"""Recommend the next round of a designed experiment from the results of the rounds so far.

A study is a sequence of rounds: screen the factors, resolve what the screen left open (an
effect aliased with an interaction, curvature it could not model), move towards the optimum
when it lies past the ranges studied, and confirm the predicted optimum. After every round the
question is the same: given every run so far, what should the next round be?
:func:`recommend_next_step` answers it with one action, the evidence for it, the runs to make,
and how to judge them once they are in.

The decision is deterministic and follows the evidence in this order:

1. **Confirmation runs** at the predicted optimum are judged by their mean against the
   prediction interval for a mean of that many runs. Confirmed, the study **stops**, unless
   the goal still improves past an edge of a range that may move. Not confirmed, the model
   missed something, and the next round must add the information that shows what.
2. A model that leaves fewer than two **degrees of freedom for error** cannot be judged: add
   centre runs to a two-level design of numeric factors, else replicate the runs.
3. **Nothing stands out from the noise**: widen the ranges, or check the measurement.
4. An active effect is **aliased with a plausible interaction**, so the data cannot say which
   of the two is real: fold the design over, fold it over on one factor, or add runs chosen
   for the model that holds both.
5. **Curvature** (centre runs off the plane of the corners, lack of fit, or a square in the
   model) that the design cannot model in the active factors: add axial runs, which make a
   face-centred central composite design, or runs chosen for the full quadratic model.
6. The best settings sit at an **edge of a range** and the goal improves past it by more than
   twice the noise: when the ranges may move, run the path of steepest ascent beyond it.
7. Otherwise, **confirm** at the predicted optimum.

A factor the selected model leaves out does not change the prediction, so any setting in its
range will do. The runs recommended set it at the end its small, not significant, effect
favours: a setting the study has run, and never a guess.
"""

from __future__ import annotations

import functools
import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize, stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

from process_improve._random import check_random_state
from process_improve.experiments._analyses.aliasing import alias_chains_of
from process_improve.experiments._analyses.curvature import _run_curvature_test
from process_improve.experiments._analyses.lack_of_fit import _run_lack_of_fit
from process_improve.experiments._analyses.model_selection import _run_model_selection
from process_improve.experiments._analyses.prediction import _mean_tests
from process_improve.experiments.augment import augment_design
from process_improve.experiments.factor import Factor, FactorType
from process_improve.experiments.optimization import _extract_b_and_B
from process_improve.experiments.true_process import _canonical

#: The actions :func:`recommend_next_step` can recommend.
ACTIONS = (
    "stop",
    "confirm",
    "add_center_points",
    "replicate",
    "widen_ranges",
    "de_alias",
    "augment_to_rsm",
    "add_runs_for_model",
    "move_region",
    "investigate",
)

Goal = Literal["maximize", "minimize", "target"]

#: A coded setting this close to +/-1 is at the end of its range.
_AT_EDGE = 1e-6

#: An interaction whose column the model's columns reproduce this well (R squared; a variance inflation
#: factor of 10) cannot be told apart from the model's terms by these runs.
_ALIASED = 0.9

#: An alias weight at least this large names a model term the interaction is confused with.
_ALIAS_WEIGHT = 0.25

#: A confirmation setting within this much of the optimum, in coded units, was made at the optimum.
_SAME_SETTING = 0.1

#: The goal must improve past an edge by this many noise standard deviations, per coded unit, to move.
_WORTH_MOVING = 2.0

#: A confirmation mean more than this many half-widths of its interval from the prediction is no chance miss.
_GROSS_MISS = 3.0

#: Steps along the path of steepest ascent beyond the best settings, in coded units of the factor moving most.
_PATH_STEPS = (0.5, 1.0, 1.5, 2.0)

#: The search for the optimum starts from every corner of up to this many factors.
_ALL_CORNERS = 6

#: Centre runs added to a two-level design that has none.
_CENTER_RUNS = 3

#: Runs beyond a model's coefficients that runs chosen for it leave, to estimate the noise.
_SPARE_RUNS = 3

#: The fewest degrees of freedom for error with which the model's terms are judged.
_MIN_ERROR_DF = 2

#: Below this many degrees of freedom for error a real effect can fail to stand out for want of them: the
#: t quantile at 0.05 is 2.78 with four, 4.30 with two, against 2.0 with many.
_FEW_ERROR_DF = 4

#: A range widened for lack of any visible effect grows by this factor about its middle.
_WIDEN = 1.5


# --------------------------------------------------------------------------- the study, coded
@dataclass(frozen=True)
class _Study:
    """The factors, and the runs coded with internal names ``x0, x1, ...`` and the response as ``y``."""

    factors: list[Factor]
    data: pd.DataFrame
    response: str

    @property
    def names(self) -> list[str]:
        return [f"x{i}" for i in range(len(self.factors))]

    @property
    def numeric(self) -> list[bool]:
        return [f.type != FactorType.categorical for f in self.factors]

    def real(self, text: str) -> str:
        """Write a term or formula with the factors' own names in place of the internal ones."""
        for i in reversed(range(len(self.factors))):  # x10 before x1
            text = text.replace(f"x{i}", self.factors[i].name)
        return text.replace("y ~", f"{self.response} ~", 1)

    def actual(self, coded: np.ndarray) -> dict[str, Any]:
        """Return coded settings in the factors' own units, or a categorical factor's level."""
        out: dict[str, Any] = {}
        for f, value in zip(self.factors, coded, strict=True):
            if f.type == FactorType.categorical:
                out[f.name] = f.levels[1] if value >= 0 else f.levels[0]  # type: ignore[index]
            else:
                mid, half = (f.high + f.low) / 2, (f.high - f.low) / 2  # type: ignore[operator]
                out[f.name] = float(mid + value * half)
        return out


def _check_factors(factors: list[Factor]) -> None:
    if not factors:
        raise ValueError("At least one factor is required.")
    for f in factors:
        if f.type == FactorType.mixture:
            raise ValueError(
                f"Factor {f.name!r} is a mixture component; recommend_next_step does not take mixtures, whose "
                "runs are blends."
            )
        if f.type == FactorType.categorical and len(f.levels or []) != 2:
            raise ValueError(
                f"Categorical factor {f.name!r} has {len(f.levels or [])} levels; recommend_next_step takes "
                "categorical factors at two levels."
            )


def _code(factors: list[Factor], frame: pd.DataFrame, response: str | None) -> pd.DataFrame:
    """Return the runs coded to -1/+1 under the internal names, with the response as ``y`` when given."""
    missing = [f.name for f in factors if f.name not in frame.columns]
    if missing:
        raise ValueError(f"The runs have no column for the factor(s) {missing}.")
    coded: dict[str, pd.Series] = {}
    for i, f in enumerate(factors):
        column = frame[f.name]
        if f.type == FactorType.categorical:
            levels = list(f.levels or [])
            unknown = sorted({str(v) for v in column if v not in levels})
            if unknown:
                raise ValueError(f"Factor {f.name!r} has levels {unknown} in the runs, not among {levels}.")
            coded[f"x{i}"] = column.map({levels[0]: -1.0, levels[1]: 1.0}).astype(float)
        else:
            mid, half = (f.high + f.low) / 2, (f.high - f.low) / 2  # type: ignore[operator]
            coded[f"x{i}"] = (pd.to_numeric(column, errors="raise") - mid) / half
    out = pd.DataFrame(coded, index=frame.index)
    if response is not None:
        if response not in frame.columns:
            raise ValueError(f"The runs have no response column {response!r}.")
        out["y"] = pd.to_numeric(frame[response], errors="raise").astype(float)
    return out


# --------------------------------------------------------------------------- what the design can show
@dataclass(frozen=True)
class _Design:
    """What the runs, as a design, can and cannot show."""

    n_runs: int
    center: np.ndarray  # runs with every numeric factor at its middle (and at least one numeric factor)
    two_level: bool  # every run that is not a centre run has every factor at an end
    three_level: bool  # not two-level, and every numeric factor at three levels or more: squares can be fitted
    pure_error_df: int


def _design(study: _Study) -> _Design:
    x = study.data[study.names].to_numpy(dtype=float)
    numeric = np.array(study.numeric)
    center = np.all(np.abs(x[:, numeric]) < _AT_EDGE, axis=1) if numeric.any() else np.zeros(len(x), dtype=bool)
    away = x[~center]
    two_level = bool(len(away)) and bool(np.all(np.abs(np.abs(away) - 1) < _AT_EDGE))
    three_level = (
        not two_level
        and bool(numeric.any())
        and all(len(np.unique(np.round(x[:, j], 9))) >= 3 for j in np.flatnonzero(numeric))
    )
    groups = pd.DataFrame(np.round(x, 9)).value_counts()
    return _Design(
        n_runs=len(x),
        center=center,
        two_level=two_level,
        three_level=three_level,
        pure_error_df=int((groups - 1).sum()),
    )


# --------------------------------------------------------------------------- the model
#: The column marking the centre runs of a two-level design, so the curvature they show is fitted
#: rather than left in the error, where it would hide the effects.
_CURVATURE = "curvature"

#: The fewest contrasts with which Lenth's method judges a saturated two-level design.
_LENTH_MIN_CONTRASTS = 7


@dataclass
class _Model:
    """The model the decision reads, and how its terms were judged.

    ``judged`` holds every term the runs could test, with its p-value; Lenth's method gives
    none, and records 0 for an effect beyond its margin of error and 1 for one within it.
    ``signal`` says whether anything stands out from the noise once the number of terms
    judged is allowed for. ``terms`` is the model for prediction: the active terms and the
    main effects their interactions and squares are built on. ``fit`` is that model's fit,
    with the curvature column when the design has centre runs among two-level runs.
    """

    terms: list[str]
    fit: RegressionResultsWrapper
    method: str
    judged: dict[str, float]
    signal: bool
    noise_sd: float
    noise_df: int
    b0: float = 0.0
    b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    B: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def error_df(self) -> int:
        return self.noise_df

    @property
    def sigma(self) -> float:
        return self.noise_sd

    def p_value(self, term: str) -> float:
        return float(self.judged.get(term, math.nan))

    def active(self, alpha: float) -> list[str]:
        """Return the judged terms, other than the curvature column, that are significant; none without a signal."""
        if not self.signal:
            return []
        return [t for t, p in self.judged.items() if t != _CURVATURE and p < alpha]

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return self.b0 + x @ self.b + np.einsum("ij,jk,ik->i", x, self.B, x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.b + 2 * self.B @ x


def _term_factors(term: str) -> set[str]:
    return set(_canonical(term)[1]) if term != _CURVATURE else set()


def _with_parents(terms: list[str], names: list[str]) -> list[str]:
    """Return the terms after the main effects of every factor their interactions and squares use (heredity)."""
    used = set().union(*(_term_factors(t) for t in terms)) if terms else set()
    mains = [n for n in names if n in used]
    return list(dict.fromkeys([*mains, *terms]))


def _rhs(names: list[str], numeric: list[bool], model: str) -> str:
    """Return the right-hand side of a standard model in the internal names; squares only of numeric factors."""
    return " + ".join(_rhs_terms(names, numeric, model))


def _rhs_terms(names: list[str], numeric: list[bool], model: str) -> list[str]:
    """Return the terms of a standard model, one per column, in the internal names."""
    terms = list(names)
    if model != "main_effects":
        terms += [f"{a}:{b}" for a, b in itertools.combinations(names, 2)]
    if model == "quadratic":
        terms += [f"I({n} ** 2)" for n, num in zip(names, numeric, strict=True) if num]
    return terms


def _ols(data: pd.DataFrame, terms: list[str]) -> RegressionResultsWrapper:
    return smf.ols("y ~ " + (" + ".join(terms) if terms else "1"), data=data).fit()


def _chain_leaders(data: pd.DataFrame, rhs: str) -> list[str] | None:
    """One term per chain of exact aliases in a model; None when even those cannot all be estimated."""
    from patsy import dmatrix  # noqa: PLC0415 - patsy comes with statsmodels; imported where the matrix is built

    matrix = dmatrix(rhs, data, return_type="dataframe")
    names = [str(c) for c in matrix.columns]
    exog = matrix.to_numpy(dtype=float)
    leaders = [chain[0][0] for chain in alias_chains_of(exog, names)]
    kept = exog[:, [names.index(t) for t in leaders]]
    if int(np.linalg.matrix_rank(kept)) < kept.shape[1]:
        return None
    return [t for t in leaders if t != "Intercept"]


def _orthogonal(data: pd.DataFrame, terms: list[str]) -> bool:
    """Whether the terms' columns, centred, are mutually orthogonal: Lenth's method needs independent effects."""
    x = np.asarray(_ols(data, terms).model.exog[:, 1:], dtype=float)
    x = x - x.mean(axis=0)
    gram = x.T @ x
    off = gram - np.diag(np.diag(gram))
    return bool(np.all(np.abs(off) <= 1e-8 * np.abs(np.diag(gram)).max()))


def _t_tests(data: pd.DataFrame, terms: list[str], extra: list[str], method: str, alpha: float) -> dict[str, Any]:
    fit = _ols(data, [*terms, *extra])
    judged = {t: float(fit.pvalues[t]) for t in [*terms, *extra]}
    return {
        "method": method,
        "judged": judged,
        "signal": _signal(fit, alpha),
        "considered": set(terms),
        "noise_sd": float(np.sqrt(fit.scale)),
        "noise_df": int(fit.df_resid),
    }


#: Draws of pure noise behind the critical values of the tests for a signal, and the p-values of Lenth's method.
_NULL_DRAWS = 4_000
_LENTH_DRAWS = 20_000


def _signal(fit: RegressionResultsWrapper, alpha: float) -> bool:
    """Say whether any term of a least-squares fit stands out from the noise (see :func:`_stands_out`)."""
    x = np.asarray(fit.model.exog, dtype=float)
    return _stands_out(x, np.asarray(fit.model.endog, dtype=float), list(range(1, x.shape[1])), alpha)


def _stands_out(x: np.ndarray, y: np.ndarray, tested: list[int], alpha: float) -> bool:
    """Say whether anything stands out from the noise in the regression of ``y`` on ``x``, at level ``alpha``.

    Two tests see different signals: the smallest p-value of the ``tested`` columns times their
    number (Bonferroni) sees one large effect, and the F-test of every column but the first
    (the intercept) sees several moderate ones, which Bonferroni misses with few degrees of
    freedom for error. The statistic is the smaller of the two p-values; its critical value is
    its ``alpha`` quantile over responses of pure noise on these runs, so the false alarm rate
    is ``alpha`` for this design, not up to twice it.
    """
    n, p = x.shape
    df = n - p
    if not tested or df < 1:
        return False
    pinv = np.linalg.pinv(x)
    scale = np.diag(pinv @ pinv.T)[tested]

    def statistic(responses: np.ndarray) -> np.ndarray:
        beta = pinv @ responses
        fitted = x @ beta
        s2 = ((responses - fitted) ** 2).sum(axis=0) / df
        t = np.abs(beta[tested]) / np.sqrt(np.outer(scale, s2))
        bonferroni = 2 * stats.t.sf(t.max(axis=0), df) * len(tested)
        f = ((fitted - responses.mean(axis=0)) ** 2).sum(axis=0) / (p - 1) / s2
        return np.minimum(bonferroni, stats.f.sf(f, p - 1, df))

    null = statistic(np.random.default_rng(n).standard_normal((n, _NULL_DRAWS)))
    return bool(statistic(y[:, None])[0] < np.quantile(null, alpha))


def _pse(contrasts: np.ndarray) -> np.ndarray:
    """Return Lenth's pseudo standard error of each row of absolute contrasts."""
    s0 = 1.5 * np.median(contrasts, axis=-1, keepdims=True)
    return 1.5 * np.nanmedian(np.where(contrasts < 2.5 * s0, contrasts, np.nan), axis=-1)


@functools.lru_cache(maxsize=16)
def _lenth_null(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Lenth's ratios ``|contrast| / PSE`` over ``m`` contrasts of pure noise, each sample sorted.

    The samples are one contrast's ratio, the largest, the second largest, and the smaller
    of the simulated p-values of the largest and the second largest. Lenth's own margins, a t
    quantile on ``m / 3`` degrees of freedom, are approximate (with seven contrasts the
    individual one is 3.76 where the exact value is 2.31), and the simultaneous one is so wide
    that two clear effects in an 8-run design rarely pass it. Simulated quantiles, as Ye and
    Hamada (2000) tabulate them, are exact for any ``m``. The evidence that something is
    active comes from the largest ratio, which sees one large effect, or the second largest,
    which sees two moderate ones, with the critical value set so that the false alarm rate of
    the pair is the significance level.
    """
    z = np.abs(np.random.default_rng(m).standard_normal((_LENTH_DRAWS, m)))
    ratios = z / _pse(z)[:, None]
    ordered = np.sort(ratios, axis=1)
    largest, second = np.sort(ordered[:, -1]), np.sort(ordered[:, -2])
    both = np.minimum(_beyond(largest, ordered[:, -1]), _beyond(second, ordered[:, -2]))
    return np.sort(ratios[:, 0]), largest, second, np.sort(both)  # the contrasts are exchangeable


def _beyond(null: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the share of a sorted null sample at or beyond each value: its simulated p-value."""
    return 1 - np.searchsorted(null, values) / len(null)


def _lenth(data: pd.DataFrame, terms: list[str], alpha: float) -> dict[str, Any] | None:
    """Judge the terms of a saturated two-level design by Lenth's method, on every contrast the runs hold.

    The contrasts are the terms' coefficients and, to make up ``n - 1``, the projections of the
    response on an orthonormal basis of what the model leaves (scaled to the coefficients'
    variance): a Plackett-Burman design's unused columns, or a full factorial's three-factor
    interaction. None when the terms are not balanced and orthogonal, or the contrasts are too few.
    """
    model = _ols(data, terms).model
    x, names = np.asarray(model.exog, dtype=float), list(model.exog_names)[1:]  # patsy orders terms by degree
    n, columns = len(x), x[:, 1:]
    if not np.allclose(columns.T @ columns, n * np.eye(columns.shape[1])) or not np.allclose(columns.sum(axis=0), 0):
        return None
    y = data["y"].to_numpy(dtype=float)
    effects = columns.T @ y / n
    rest = np.linalg.qr(x, mode="complete")[0][:, x.shape[1] :].T @ y / np.sqrt(n)
    m = len(effects) + len(rest)
    pse = float(_pse(np.abs(np.concatenate([effects, rest]))))
    if m < _LENTH_MIN_CONTRASTS or not pse > 0:
        return None
    one, largest, second, both = _lenth_null(m)
    ratios = np.abs(effects) / pse
    p_values = _beyond(one, ratios)
    top = np.sort(ratios)[::-1]  # the terms' ratios only: a signal must be in a term the decision can act on
    observed = min(_beyond(largest, top[:1])[0], _beyond(second, top[1:2])[0] if len(top) > 1 else 1.0)
    signal = bool(observed < np.quantile(both, alpha))
    return {
        "method": f"Lenth's method on the {m} contrasts of the saturated two-level design, with simulated p-values",
        "judged": dict(zip(names, (float(p) for p in p_values), strict=True)),
        "signal": signal,
        "considered": set(terms),
        "noise_sd": pse,
        "noise_df": max(1, round(m / 3)),
    }


def _omars(data: pd.DataFrame, names: list[str], alpha: float) -> dict[str, Any] | None:
    from process_improve.experiments.omars import analyze_omars  # noqa: PLC0415 - one caller

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # its structure check, read from the result instead
        result = analyze_omars(
            data[names],
            data["y"],
            alpha_main=alpha,
            alpha_second_overall=alpha,
            alpha_second_subset=alpha,
            interaction_heredity="strong",  # a DSD's A:B is partly aliased with C:D and E:F; heredity picks A:B
        )
    if not result.success or not all(result.details["design_structure"].values()):
        return None  # its tests hold only for a design with the OMARS structure
    judged = {name: float(p) for name, p in result.main_effect_p_values.items()}
    judged.update(dict.fromkeys(result.active_interactions, 0.0))
    judged.update({f"I({q[:-2]} ** 2)": 0.0 for q in result.active_quadratics})
    return {
        "method": "the staged analysis of a definitive screening design (analyze_omars)",
        "judged": judged,
        "signal": _omars_signal(data, names, alpha),
        "considered": set(_rhs_terms(names, [True] * len(names), "quadratic")),
        "noise_sd": float(result.updated_rmse),
        "noise_df": int(result.updated_error_df),
    }


def _omars_signal(data: pd.DataFrame, names: list[str], alpha: float) -> bool:
    """Say whether a definitive screening design shows anything: its main effects, or its second-order terms.

    The main effects are tested by Bonferroni, and the second-order terms together by the
    F-test, against the error of the full second-order model, as the staged analysis does;
    the pair's critical value is simulated (:func:`_stands_out`). The staged analysis's own
    second-order test pools the main effects it found inactive into the error, which makes
    that error too small and the test fire on noise.
    """
    mains = np.column_stack([np.ones(len(data)), data[names].to_numpy(dtype=float)])
    second = np.asarray(_ols(data, _rhs_terms(names, [True] * len(names), "quadratic")).model.exog, dtype=float)
    second = second - mains @ np.linalg.lstsq(mains, second, rcond=None)[0]
    u, sv, _ = np.linalg.svd(second, full_matrices=False)
    basis = u[:, sv > 1e-8 * sv.max()]
    x = np.hstack([mains, basis])
    return _stands_out(x, data["y"].to_numpy(dtype=float), list(range(1, len(names) + 1)), alpha)


def _judge_terms(study: _Study, design: _Design, alpha: float) -> dict[str, Any] | None:  # noqa: C901, PLR0911 - one return per way of judging, in the order they are tried
    """Judge every term the runs can test, in the largest standard model they estimate.

    Testing the terms a stepwise search picked, on the data that picked them, makes them look
    stronger than they are: in simulated noise it found an effect in most studies. So each
    term is tested in the largest model the runs estimate with at least two degrees of freedom
    for error (one column per chain of exact aliases): the full quadratic, the interactions, or
    the main effects. A two-level design too small for that is judged by Lenth's method on all
    its contrasts. A design whose second-order terms are partly aliased with each other, which
    no standard model separates, is judged by its own staged analysis when it is a definitive
    screening design; with centre runs among two-level runs (a Plackett-Burman design with
    centre runs), a stepwise search (AICc) proposes its interactions, which are then tested
    beside every main effect. Testing the main effects alone would leave an active interaction
    in the error, where it hides the effects. Returns None when the runs are too few to judge
    anything.
    """
    data, names, numeric = study.data, study.names, study.numeric
    extra = [_CURVATURE] if design.two_level and design.center.any() and any(numeric) else []
    n = len(data)

    def enough(n_terms: int) -> bool:
        return n - n_terms - 1 - len(extra) >= _MIN_ERROR_DF

    saturated = design.two_level and not extra  # judged by Lenth's method when no model leaves error
    models = (["quadratic"] if design.three_level else []) + ["interactions", "main_effects"]
    for model in models:
        leaders = _chain_leaders(data, _rhs(names, numeric, model))
        if leaders is None:
            break  # terms partly aliased with each other: judged below
        if enough(len(leaders)):
            return _t_tests(data, leaders, extra, f"t-tests in the {model.replace('_', ' ')} model", alpha)
        judged = _lenth(data, leaders, alpha) if saturated and model == "interactions" else None
        if judged is not None:
            return judged
    else:
        return None  # even the main effects leave too few runs to judge them
    if design.three_level:
        judged = _omars(data, names, alpha)
        if judged is not None:
            return judged
    searched = "quadratic" if design.three_level else "interactions"
    proposed = _run_model_selection(data, "y", names, model=searched)["model_selection"]["selected_terms"]
    second = [t for t in proposed if t not in names]
    if enough(len(names) + len(second)):
        method = f"t-tests of every main effect and the {searched} terms a stepwise search (AICc) proposed"
        judged = _t_tests(data, [*names, *second], extra, method, alpha)
        judged["considered"] = set(_rhs_terms(names, numeric, searched))  # the search weighed every term
        return judged
    if saturated:  # the main effects, and the columns the design leaves for error
        judged = _lenth(data, names, alpha)
        if judged is not None:
            return judged
    mains = _chain_leaders(data, _rhs(names, numeric, "main_effects"))
    if mains is not None and enough(len(mains)):
        return _t_tests(data, mains, extra, "t-tests in the main effects model", alpha)
    return None


def _interaction_candidates(study: _Study, active: list[str]) -> list[str]:
    """Return the two-factor interactions heredity makes plausible: both factors in active terms, not yet active."""
    factors = set().union(*(_term_factors(t) for t in active)) if active else set()
    present = {frozenset(_term_factors(t)) for t in active if ":" in t}
    return [
        f"{a}:{b}"
        for a, b in itertools.combinations(study.names, 2)
        if {a, b} <= factors and frozenset((a, b)) not in present
    ]


def _separable(study: _Study, terms: list[str], candidates: list[str]) -> list[str]:
    """Return the candidate interactions the runs can tell apart from the model's terms (see :func:`_aliasing`)."""
    if not candidates:
        return []
    x = np.asarray(_ols(study.data, terms).model.exog, dtype=float)
    return [c for c in candidates if _explained(x, _product(study, c)) < _ALIASED]


def _product(study: _Study, interaction: str) -> np.ndarray:
    a, b = interaction.split(":")
    return (study.data[a] * study.data[b]).to_numpy(dtype=float)


def _explained(x: np.ndarray, column: np.ndarray) -> float:
    """Return the share of a column's variation about its mean that the columns of ``x`` reproduce (R squared)."""
    centred = column - column.mean()
    total = float(centred @ centred)
    if total <= 0:
        return 1.0  # a constant column: the intercept holds it
    residual = column - x @ np.linalg.lstsq(x, column, rcond=None)[0]
    return 1.0 - float(residual @ residual) / total


def _second_look(study: _Study, judged: dict[str, Any], active: list[str], extra: list[str], alpha: float) -> None:
    """Test the plausible interactions of the active factors that the judging never tested, and add the active ones.

    A Plackett-Burman design judged on its main effects never tests A:C, though A and C are
    active and A:C is nearly orthogonal to every main effect. In the model of the active terms
    it has the degrees of freedom to be tested, and strong heredity keeps the family small.
    """
    terms = _with_parents(active, study.names)
    untested = [c for c in _interaction_candidates(study, active) if c not in judged["considered"]]
    untested = _separable(study, [*terms, *extra], untested)
    if not untested or len(study.data) - len(terms) - len(untested) - 1 - len(extra) < _MIN_ERROR_DF:
        return
    fit = _ols(study.data, [*terms, *untested, *extra])
    found = {c: float(fit.pvalues[c]) for c in untested}
    judged["judged"].update(found)
    judged["considered"] |= set(untested)
    judged["method"] += (
        f"; then the {len(untested)} interaction(s) of active factors it could not test, by t-tests in the model "
        "of the active terms"
    )
    active.extend(c for c, p in found.items() if p < alpha)


def _fit(study: _Study, terms: list[str], judged: dict[str, Any], design: _Design) -> _Model:
    """Fit the model for prediction, and write it as the polynomial ``b0 + x'b + x'Bx`` in coded units."""
    extra = [_CURVATURE] if _CURVATURE in judged["judged"] or (design.two_level and design.center.any()) else []
    extra = [c for c in extra if c in study.data.columns]
    fit = _ols(study.data, [*terms, *extra])
    params = [
        {"term": str(name), "coefficient": float(value)}
        for name, value in fit.params.items()
        if str(name) != _CURVATURE
    ]
    b0, b, B = _extract_b_and_B(params, study.names)
    return _Model(
        terms=terms,
        fit=fit,
        method=judged["method"],
        judged=judged["judged"],
        signal=judged["signal"],
        noise_sd=judged["noise_sd"],
        noise_df=judged["noise_df"],
        b0=b0,
        b=b,
        B=B,
    )


def _select(study: _Study, design: _Design, model_terms: list[str] | None, alpha: float) -> _Model:
    """Judge the terms (see :func:`_judge_terms`), or the caller's model's, and fit the model for prediction."""
    extra = [_CURVATURE] if design.two_level and design.center.any() and any(study.numeric) else []
    if model_terms is not None:
        terms = _internal_terms(study, model_terms)
        return _fit(study, terms, _t_tests(study.data, terms, extra, "t-tests in the model given", alpha), design)
    judged = _judge_terms(study, design, alpha)
    if judged is None:
        empty = {
            "method": "none: too few runs to judge any term",
            "judged": {},
            "signal": False,
            "noise_sd": math.nan,
            "noise_df": 0,
        }
        return _fit(study, [], empty, design)
    active = [t for t, p in judged["judged"].items() if t != _CURVATURE and p < alpha] if judged["signal"] else []
    if active:
        _second_look(study, judged, active, extra, alpha)
    return _fit(study, _with_parents(active, study.names), judged, design)


def _internal_terms(study: _Study, model_terms: list[str]) -> list[str]:
    """Return the caller's model terms in the internal names; the intercept is implied."""
    internal = {f.name: f"x{i}" for i, f in enumerate(study.factors)}
    terms = []
    for term in model_terms:
        canonical, names = _canonical(term)
        unknown = [n for n in names if n not in internal]
        if unknown:
            raise ValueError(f"The model term {term!r} names {unknown}, which are not factors.")
        if canonical == "Intercept":
            continue
        parts = [internal[n] for n in names]
        if "**" in canonical:
            terms.append(f"I({parts[0]} ** 2)")
        else:
            terms.append(":".join(parts))
    return terms


# --------------------------------------------------------------------------- the evidence
def _favoured_ends(study: _Study, sense: float) -> np.ndarray:
    """Each factor's end that its marginal effect favours for the goal (+1 or -1), from the runs at its ends."""
    ends = np.ones(len(study.factors))
    y = study.data["y"].to_numpy(dtype=float)
    for j, name in enumerate(study.names):
        x = study.data[name].to_numpy(dtype=float)
        high, low = y[x > _AT_EDGE], y[x < -_AT_EDGE]
        contrast = (high.mean() - low.mean()) if len(high) and len(low) else 0.0
        ends[j] = 1.0 if contrast * (sense or 1.0) >= 0 else -1.0
    return ends


def _sense(goal: Goal, target: float | None, predicted: float) -> float:
    """Which way the response should move: +1 up, -1 down, 0 for a target already met."""
    if goal == "maximize":
        return 1.0
    if goal == "minimize":
        return -1.0
    assert target is not None
    return float(np.sign(target - predicted))


def _optimum(study: _Study, model: _Model, goal: Goal, target: float | None, seed: int) -> np.ndarray:
    """Return the coded settings inside the ranges at which the model best meets the goal.

    A polynomial of at most second order is searched from the centre, the corner its linear
    terms point to, every corner up to six factors, and twenty random starts. A categorical
    factor is searched at each of its two levels; a factor the model leaves out is set at
    the end it favours (:func:`_favoured_ends`). Many settings can meet a target; the search
    then prefers the one nearest the middle of the ranges.
    """
    in_model = set().union(*(_term_factors(t) for t in model.terms)) if model.terms else set()
    free = [j for j, name in enumerate(study.names) if name in in_model and study.numeric[j]]
    categorical = [j for j, name in enumerate(study.names) if name in in_model and not study.numeric[j]]
    nearest = 1e-6 * float(study.data["y"].var())  # the weight that breaks ties among settings that meet a target

    def loss(x: np.ndarray) -> float:
        value = float(model.value(x)[0])
        if goal == "target":
            return (value - float(target)) ** 2 + nearest * float(x @ x)  # type: ignore[arg-type]
        return -value if goal == "maximize" else value

    rng = check_random_state(seed)
    base = _favoured_ends(study, _sense(goal, target, float(model.b0)))
    best_x, best_loss = base.copy(), math.inf
    for levels in itertools.product((-1.0, 1.0), repeat=len(categorical)):
        start = base.copy()
        start[categorical] = levels
        for s in _starts(model, free, goal, rng) if free else [np.zeros(0)]:

            def partial(z: np.ndarray, start: np.ndarray = start) -> float:
                x = start.copy()
                x[free] = z
                return loss(x)

            if free:
                found = optimize.minimize(partial, s, bounds=[(-1.0, 1.0)] * len(free), method="L-BFGS-B")
                z, value = np.clip(found.x, -1.0, 1.0), float(found.fun)
            else:  # the model holds no numeric factor: nothing to search
                z, value = s, partial(s)
            if value < best_loss - 1e-12:
                best_loss, best_x = value, start.copy()
                best_x[free] = z
    return best_x


def _starts(model: _Model, free: list[int], goal: Goal, rng: np.random.Generator) -> list[np.ndarray]:
    """Return the starting points of the search for the optimum over the ``free`` factors."""
    starts = [np.zeros(len(free)), np.sign(model.b[free] * (-1 if goal == "minimize" else 1))]
    if len(free) <= _ALL_CORNERS:
        starts += [np.array(v, dtype=float) for v in itertools.product((-1.0, 1.0), repeat=len(free))]
    return starts + list(rng.uniform(-1, 1, size=(20, len(free))))


def _curvature(study: _Study, design: _Design, model: _Model, alpha: float) -> dict[str, Any]:
    """Say whether the response curves, and how the runs show it: centre runs, or squares in the model."""
    squares = [t for t in model.terms if "**" in t]
    significant_squares = [t for t in squares if model.p_value(t) < alpha]
    if design.three_level:
        return {
            "source": "squares",
            "significant": bool(significant_squares),
            "terms": significant_squares,
        }
    if design.two_level and design.center.any():
        p = model.p_value(_CURVATURE)
        if np.isfinite(p):  # the curvature column's test in the judging fit, against its residual error
            return {"source": "center_runs", "significant": bool(p < alpha), "p_value": p}
        if design.pure_error_df > 0:  # judged by Lenth's method: the centre runs' own repeats instead
            test = _run_curvature_test(study.data, "y", study.names, alpha)["curvature_test"]
            if "error" not in test:
                return {"source": "center_runs", "significant": bool(test["significant"]), "p_value": test["p_value"]}
    return {"source": None, "significant": False}


def _lack_of_fit(study: _Study, model: _Model, alpha: float) -> dict[str, Any]:
    if model.error_df <= 0:
        return {"tested": False}
    result = _run_lack_of_fit(model.fit, study.data, "y", study.names, alpha)["lack_of_fit"]
    if "error" in result:
        return {"tested": False}
    return {"tested": True, "significant": bool(result["significant"]), "p_value": result["p_value"]}


def _aliasing(study: _Study, model: _Model, active: list[str]) -> list[dict[str, Any]]:
    """List each plausible interaction outside the model that the runs cannot tell apart from its active terms.

    A two-factor interaction is plausible when both its factors are in active terms (strong
    heredity): in a resolution III fraction, main effect C aliased with A:B is in doubt when A
    and B are active, not when they are not. The runs cannot tell it apart from the model when
    the model's columns reproduce its column (R squared of :data:`_ALIASED` or more): wholly
    in a regular fraction, where it is aliased, and not at all in a Plackett-Burman or
    definitive screening design, whose partial aliasing the analysis separates. The active terms
    it is confused with are those its column loads on, with the weight that is each term's
    bias per unit of the interaction's coefficient, ``(X1'X1)^-1 X1' x2``.
    """
    candidates = _interaction_candidates(study, [t for t in model.terms if t in active or ":" in t])
    if not active or not candidates:
        return []
    exog = np.asarray(model.fit.model.exog, dtype=float)
    columns = list(model.fit.model.exog_names)
    found = []
    for candidate in candidates:
        column = _product(study, candidate)
        if _explained(exog, column) < _ALIASED:
            continue
        weights = np.linalg.pinv(exog) @ column
        loads = [j for j, name in enumerate(columns) if name in active and abs(weights[j]) >= _ALIAS_WEIGHT]
        if not loads:  # confused with the model as a whole, not one term: name the term it loads on most
            loads = [max((j for j, name in enumerate(columns) if name in active), key=lambda j: abs(weights[j]))]
        found += [{"term": columns[j], "aliased_with": candidate, "weight": float(weights[j])} for j in loads]
    return found


def _edges(study: _Study, model: _Model, x: np.ndarray, sense: float) -> tuple[list[dict[str, Any]], float]:
    """List the numeric factors at an end of their range at ``x``, and the gain per coded unit past them.

    A factor improves past its end when the gradient there points out of the range, the way
    the goal wants the response to move. The gain is the directional derivative along the
    outward directions of those factors, scaled to one coded unit in the factor that moves
    most: the change the first step of the path of steepest ascent should bring.
    """
    gradient = model.gradient(x)
    in_model = set().union(*(_term_factors(t) for t in model.terms)) if model.terms else set()
    edges, outward = [], np.zeros(len(x))
    for j, name in enumerate(study.names):
        if not study.numeric[j] or name not in in_model or abs(x[j]) < 1 - _AT_EDGE:
            continue
        pushes = sense * gradient[j] * np.sign(x[j])
        improves = bool(pushes > 1e-9)
        edges.append(
            {"factor": study.factors[j].name, "end": "high" if x[j] > 0 else "low", "improves_beyond": improves}
        )
        if improves:
            outward[j] = sense * gradient[j]
    if not outward.any():
        return edges, 0.0
    direction = outward / np.abs(outward).max()
    return edges, float(abs(gradient @ direction))


# --------------------------------------------------------------------------- the runs of each action
def _points(study: _Study, coded: np.ndarray) -> pd.DataFrame:
    """Coded settings as the model's fit reads them, with the curvature column: 1 at the centre of every range."""
    frame = pd.DataFrame(np.atleast_2d(coded), columns=study.names)
    numeric = [n for n, num in zip(study.names, study.numeric, strict=True) if num]
    frame[_CURVATURE] = (frame[numeric].abs() < _AT_EDGE).all(axis=1).astype(float) if numeric else 0.0
    return frame


def _rows(study: _Study, coded: np.ndarray) -> list[dict[str, Any]]:
    return [study.actual(row) for row in np.atleast_2d(coded)]


def _two_level_part(study: _Study, design: _Design) -> pd.DataFrame:
    return study.data.loc[~design.center, study.names]


def _with_others(base: np.ndarray, runs: pd.DataFrame, columns: list[int], study: _Study) -> np.ndarray:
    """Put augment_design's runs over ``columns`` into full coded rows, the other factors at ``base``."""
    out = np.tile(base, (len(runs), 1))
    for j in columns:
        out[:, j] = runs[study.names[j]].to_numpy(dtype=float)
    return out


# --------------------------------------------------------------------------- the context of one decision
@dataclass
class _Context:
    """The study, its model, the evidence and the caller's choices: what every action builder reads."""

    study: _Study
    design: _Design
    model: _Model
    goal: Goal
    target: float | None
    alpha: float
    n_confirmation_runs: int
    ranges_can_move: bool
    seed: int
    history: tuple[str, ...] = ()
    optimum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sense: float = 1.0
    active: list[str] = field(default_factory=list)
    active_factors: list[int] = field(default_factory=list)
    curvature: dict[str, Any] = field(default_factory=dict)
    lack_of_fit: dict[str, Any] = field(default_factory=dict)
    aliasing: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    gain_beyond: float = 0.0

    @property
    def k(self) -> int:
        return len(self.study.factors)

    def name(self, j: int) -> str:
        return self.study.factors[j].name

    def predicted(self) -> float:
        return float(self.model.value(self.optimum)[0])

    def mean_interval(self, n: int) -> tuple[float, float]:
        """Return the prediction interval at the optimum for the mean of ``n`` new runs."""
        frame = self.model.fit.get_prediction(_points(self.study, self.optimum)).summary_frame(alpha=self.alpha).iloc[0]
        if self.model.error_df <= 0:
            return math.nan, math.nan
        t = float(stats.t.ppf(1 - self.alpha / 2, self.model.error_df))
        half = t * float(np.sqrt(frame["mean_se"] ** 2 + self.model.fit.scale / n))
        return float(frame["mean"]) - half, float(frame["mean"]) + half


def _gather(ctx: _Context) -> _Context:
    """Fill in the evidence: the optimum, the active terms and factors, curvature, lack of fit, aliasing, edges."""
    study, model = ctx.study, ctx.model
    ctx.optimum = _optimum(study, model, ctx.goal, ctx.target, ctx.seed)
    ctx.sense = _sense(ctx.goal, ctx.target, ctx.predicted()) or 1.0
    ctx.active = model.active(ctx.alpha)
    factors = set().union(*(_term_factors(t) for t in ctx.active)) if ctx.active else set()
    ctx.active_factors = [j for j, name in enumerate(study.names) if name in factors]
    ctx.curvature = _curvature(study, ctx.design, model, ctx.alpha)
    ctx.lack_of_fit = _lack_of_fit(study, model, ctx.alpha)
    ctx.aliasing = _aliasing(study, model, ctx.active)
    ctx.edges, ctx.gain_beyond = _edges(study, model, ctx.optimum, ctx.sense)
    return ctx


# --------------------------------------------------------------------------- the actions
@dataclass
class _Step:
    """One recommended action: what to do, why, the runs to make (coded), and how to judge them."""

    action: str
    headline: str
    reasons: list[str]
    runs: np.ndarray
    after: str
    extra: dict[str, Any] = field(default_factory=dict)


def _g(value: float) -> str:
    """Write a number to four significant figures."""
    return f"{value:.4g}"


def _names(ctx: _Context, indices: list[int]) -> str:
    return ", ".join(ctx.name(j) for j in indices)


def _confirm(ctx: _Context) -> _Step:
    n = ctx.n_confirmation_runs
    low, high = ctx.mean_interval(n)
    left_out = [j for j in range(ctx.k) if j not in _model_factor_indices(ctx)]
    reasons = [f"The model predicts {_g(ctx.predicted())} at these settings."]
    if np.isfinite(low):
        reasons.append(
            f"The mean of {n} runs there should fall between {_g(low)} and {_g(high)} "
            f"({100 * (1 - ctx.alpha):.0f}% prediction interval for a mean of {n} runs)."
        )
    if left_out:
        reasons.append(
            f"{_names(ctx, left_out)} {'is' if len(left_out) == 1 else 'are'} not in the model, so set at the end "
            "the small, not significant, effect favours; any setting in the range should do as well."
        )
    after = (
        f"If the mean of the {n} runs lies between {_g(low)} and {_g(high)}, accept these settings. If not, the "
        "model has missed something: call recommend_next_step again with these runs as the confirmation runs."
    )
    return _Step(
        "confirm",
        f"Confirm the predicted optimum with {n} runs at these settings.",
        reasons,
        np.tile(ctx.optimum, (n, 1)),
        after,
    )


def _model_factor_indices(ctx: _Context) -> list[int]:
    factors = set().union(*(_term_factors(t) for t in ctx.model.terms)) if ctx.model.terms else set()
    return [j for j, name in enumerate(ctx.study.names) if name in factors]


def _add_center_points(ctx: _Context) -> _Step | None:
    """Three centre runs, for a two-level design of numeric factors that has none."""
    if not all(ctx.study.numeric) or not ctx.design.two_level or ctx.design.center.any():
        return None
    return _Step(
        "add_center_points",
        f"Add {_CENTER_RUNS} runs at the centre of every range.",
        [
            (
                "Every run so far sets each factor at an end of its range, so nothing shows whether the response "
                "curves between the ends, and no run is repeated to estimate the noise."
            ),
        ],
        np.zeros((_CENTER_RUNS, ctx.k)),
        "Analyse again with the centre runs: if their mean is off the plane of the corners "
        "(analysis_type='curvature_test'), the next round adds axial runs; their spread estimates the noise.",
    )


def _replicate(ctx: _Context) -> _Step:
    runs = ctx.study.data[ctx.study.names].to_numpy(dtype=float)
    return _Step(
        "replicate",
        f"Make the {len(runs)} runs again.",
        [
            (
                f"The model leaves {ctx.model.error_df} degree(s) of freedom to estimate the noise, too few to judge "
                "which effects are real."
            ),
        ],
        runs,
        "Analyse every run together: the repeats estimate the noise, and every coefficient's variance halves.",
    )


def _more_error_df(ctx: _Context) -> _Step:
    """Centre runs, or the runs again, when nothing stands out but the noise is estimated from too few runs."""
    step = _add_center_points(ctx) or _replicate(ctx)
    step.reasons = [
        (
            f"No effect stands out from the noise ({ctx.model.method}), but the noise is estimated with only "
            f"{ctx.model.error_df} degree(s) of freedom: a real effect needs to be several times larger to stand out "
            "than it would with more. More runs make the noise estimate firm before the ranges are questioned."
        ),
        *step.reasons,
    ]
    return step


def _widen_ranges(ctx: _Context) -> _Step:
    """Widen the numeric ranges about their middles and run the design again; replicate when they cannot move."""
    spread = float(ctx.study.data["y"].std(ddof=1)) if len(ctx.study.data) > 1 else math.nan
    smallest = min(ctx.model.judged.values(), default=math.nan)
    reasons = [
        (
            f"No effect stands out from the noise ({ctx.model.method}): with {len(ctx.model.judged)} terms judged, the "
            f"smallest p-value, {_g(smallest)}, is what chance alone gives at {ctx.alpha:g} (the response's standard "
            f"deviation over the runs is {_g(spread)})."
        ),
        (
            "Either the ranges are too narrow for the factors to move the response, or the measurement is too "
            "noisy to see them move it: check its repeatability before spending more runs."
        ),
    ]
    if not ctx.ranges_can_move:
        step = _replicate(ctx)
        step.reasons = [*reasons, "The ranges cannot move, so more runs at the same settings average the noise down."]
        return step
    ranges = {}
    for f in ctx.study.factors:
        if f.type == FactorType.categorical:
            continue
        mid, half = (f.high + f.low) / 2, (f.high - f.low) / 2  # type: ignore[operator]
        ranges[f.name] = [mid - _WIDEN * half, mid + _WIDEN * half]
    runs = ctx.study.data[ctx.study.names].to_numpy(dtype=float)
    return _Step(
        "widen_ranges",
        f"Widen every numeric range by half again about its middle, and run the {len(runs)}-run design again.",
        [*reasons, "Wider ranges make each factor's effect larger against the same noise."],
        runs * _WIDEN,  # the same coded design, read on the old scale: the new ends are 1.5 old half-ranges out
        "Analyse the new runs on their own ranges; if still nothing stands out, check the measurement system.",
        {"new_ranges": ranges},
    )


def _de_alias(ctx: _Context) -> _Step | None:
    """Return the runs that separate each ambiguous active term from the interaction it is aliased with."""
    if not ctx.aliasing:
        return None
    pairs = ", ".join(f"{ctx.study.real(a['term'])} with {ctx.study.real(a['aliased_with'])}" for a in ctx.aliasing)
    reasons = [
        (
            f"Active effects are aliased with interactions that heredity makes plausible ({pairs}): the data cannot "
            "say which of each pair is real, and the optimum depends on it."
        ),
    ]
    two_level = _two_level_part(ctx.study, ctx.design)
    mains = any(":" not in a["term"] for a in ctx.aliasing)
    if ctx.design.two_level and mains:
        new = augment_design(two_level, "foldover", factor_names=ctx.study.names)["new_runs"]
        headline = f"Fold the design over: {len(new)} runs, each a run made before with every setting switched."
        reasons.append("The fold-over separates every main effect from every two-factor interaction.")
    else:
        partners = sorted({a["aliased_with"] for a in ctx.aliasing})
        rhs = " + ".join(dict.fromkeys([*ctx.model.terms, *partners]))
        n_terms = len(ctx.model.terms) + len(partners) + 1
        n_new = max(len(partners), n_terms - len(ctx.study.data)) + _SPARE_RUNS
        frame = ctx.study.data[ctx.study.names]
        new = augment_design(
            frame,
            "add_runs_optimal",
            target_model=rhs,
            n_additional_runs=n_new,
            factor_names=ctx.study.names,
            random_state=ctx.seed,
        )["new_runs"]
        headline = (
            f"Add {len(new)} runs chosen to estimate each active term apart from the interaction it is aliased with."
        )
        reasons.append(
            "Runs chosen for the model that holds both terms of each pair (D-optimal) let it tell them apart."
        )
    runs = pd.DataFrame(new)[ctx.study.names].to_numpy(dtype=float)
    return _Step(
        "de_alias",
        headline,
        reasons,
        runs,
        "Analyse every run together: the model can then tell each pair apart, and the next step follows from it.",
    )


def _quadratic_runs(ctx: _Context, factors: list[int]) -> _Step | None:
    """Return runs that let the full quadratic model in ``factors`` be fitted: axial runs, or runs chosen for it.

    None when there is no numeric factor among them, or the runs already fit that model (see
    :func:`_quadratic_fits`). The other numeric factors are held at the middle of their ranges,
    as in a central composite design, so the runs added tell the squares of ``factors`` from
    theirs; categorical factors are held at the level the optimum takes.
    """
    numeric = [j for j in factors if ctx.study.numeric[j]]
    if not numeric or _quadratic_fits(ctx, numeric):
        return None
    names = [ctx.study.names[j] for j in numeric]
    frame = ctx.study.data[names]
    quadratic = _model_columns(frame, "quadratic")
    away = ~ctx.design.center
    two_level_here = bool(np.all(np.abs(np.abs(frame[away].to_numpy(dtype=float)) - 1) < _AT_EDGE))
    interactions = _model_columns(frame, "interactions")
    interactions_fit = len(names) >= 2 and int(np.linalg.matrix_rank(interactions)) == interactions.shape[1]
    if two_level_here and interactions_fit:
        result = augment_design(frame, "upgrade_to_rsm", alpha="face_centered", factor_names=names)
        added = pd.DataFrame(result["new_runs"])
        how = (
            f"Add {len(added)} axial and centre runs in {_names(ctx, numeric)}: with the runs before, a face-centred "
            "central composite design, which fits the full quadratic model and keeps every run inside the ranges."
        )
    else:
        n_new = quadratic.shape[1] - int(np.linalg.matrix_rank(quadratic)) + _SPARE_RUNS
        result = augment_design(
            frame,
            "add_runs_optimal",
            target_model="quadratic",
            n_additional_runs=n_new,
            factor_names=names,
            random_state=ctx.seed,
        )
        added = pd.DataFrame(result["new_runs"])
        how = f"Add {len(added)} runs chosen (D-optimal) to fit the full quadratic model in {_names(ctx, numeric)}."
    held = [j for j in range(ctx.k) if ctx.study.numeric[j] and j not in numeric]
    if held:
        how += f" Hold {_names(ctx, held)} at the middle of {'its range' if len(held) == 1 else 'their ranges'}."
    runs = _with_others(_held_settings(ctx), added, numeric, ctx.study)
    return _Step(
        "augment_to_rsm",
        how,
        [],
        runs,
        "Fit the full quadratic model to every run; its optimum, inside the ranges or on an edge, is "
        "where the confirmation runs go next.",
    )


def _quadratic_fits(ctx: _Context, numeric: list[int]) -> bool:
    """Say whether the runs fit the full quadratic model in these numeric factors, apart from every other square.

    Centre runs among two-level runs give every factor three levels, so the quadratic model in
    any one factor looks estimable; but its square is then the same column as every other
    factor's, and the curvature the centre runs show could be any of theirs. The squares of
    these factors must therefore be told apart from the other factors' squares. (A definitive
    screening design's squares are; that they are partly aliased with its interactions is
    what its own analysis resolves.)
    """
    data = ctx.study.data
    chosen = data[[ctx.study.names[j] for j in numeric]].to_numpy(dtype=float)
    quadratic = _model_columns(pd.DataFrame(chosen), "quadratic")
    if int(np.linalg.matrix_rank(quadratic)) < quadratic.shape[1]:
        return False
    others = [n for j, n in enumerate(ctx.study.names) if ctx.study.numeric[j] and j not in numeric]
    if not others:
        return True
    own = np.column_stack([np.ones(len(chosen)), chosen**2])
    squares = data[others].to_numpy(dtype=float) ** 2
    squares -= squares.mean(axis=0)
    together = int(np.linalg.matrix_rank(np.hstack([own, squares])))
    return together == own.shape[1] + int(np.linalg.matrix_rank(squares))


def _model_columns(frame: pd.DataFrame, model: str) -> np.ndarray:
    x = frame.to_numpy(dtype=float)
    parts = [np.ones((len(x), 1)), x]
    if model in {"interactions", "quadratic"}:
        parts += [x[:, [i]] * x[:, [j]] for i, j in itertools.combinations(range(x.shape[1]), 2)]
    if model == "quadratic":
        parts += [x**2]
    return np.column_stack(parts)


def _held_settings(ctx: _Context) -> np.ndarray:
    """Return where added runs hold the other factors: numeric at the middle, categorical as at the optimum."""
    held = ctx.optimum.copy()
    held[np.array(ctx.study.numeric)] = 0.0
    return held


def _move_region(ctx: _Context) -> _Step | None:
    """Return runs along the path of steepest ascent, past the edges of the ranges the goal improves beyond."""
    if not ctx.ranges_can_move or ctx.gain_beyond <= 0:
        return None
    noise = ctx.model.sigma
    if not np.isfinite(noise) or ctx.gain_beyond < _WORTH_MOVING * noise:
        return None
    gradient = ctx.model.gradient(ctx.optimum)
    outward = np.zeros(ctx.k)
    for j, edge in enumerate(_edge_flags(ctx)):
        if edge:
            outward[j] = ctx.sense * gradient[j]
    direction = outward / np.abs(outward).max()
    runs = np.array([ctx.optimum + t * direction for t in _PATH_STEPS])
    predicted = ctx.model.value(runs)
    moving = [j for j in range(ctx.k) if direction[j] != 0]
    return _Step(
        "move_region",
        f"Move past the ranges of {_names(ctx, moving)}: {len(runs)} runs along the path of steepest ascent.",
        [
            (
                f"The best settings in the ranges sit at their edge, and the model's slope there gains "
                f"{_g(ctx.gain_beyond)} per coded unit beyond it, more than {_WORTH_MOVING:g} times the noise "
                f"(standard deviation {_g(noise)})."
            ),
            (
                "Past the ranges the model is extrapolating: its predictions along the path "
                f"({', '.join(_g(v) for v in predicted)}) say where to look, not what will be found."
            ),
        ],
        runs,
        "Make the runs in order and stop at the first one that does worse than the run before it. Then centre a "
        "new study on the best run, with ranges as wide as these, and start again with its screening or "
        "response surface design.",
        {"path_predictions": [float(v) for v in predicted]},
    )


def _edge_flags(ctx: _Context) -> list[bool]:
    """For each factor, whether it is at an edge the goal improves beyond."""
    beyond = {e["factor"] for e in ctx.edges if e["improves_beyond"]}
    return [f.name in beyond for f in ctx.study.factors]


def _no_effect(ctx: _Context) -> _Step:
    """Stop: nothing stood out again after the ranges were widened, or after the runs were made again."""
    tried = "widened" if "widen_ranges" in ctx.history else "made again"
    return _Step(
        "stop",
        "Stop: no factor moves the response by more than the noise.",
        [
            f"Nothing stands out from the noise ({ctx.model.method}), as before the runs were {tried}.",
            (
                "Over the ranges the process allows, these factors do not change the response by more than its "
                f"run-to-run noise (standard deviation {_g(ctx.model.sigma)}). Check the measurement's repeatability, "
                "and look for factors this study did not vary."
            ),
        ],
        np.zeros((0, ctx.k)),
        "Set the factors for cost or convenience. A new study needs other factors, or a better measurement.",
    )


def _investigate_confirmation(ctx: _Context, why: str) -> _Step:
    """No more runs yet: the confirmation runs miss the prediction by far, though the runs fit the model they need."""
    return _Step(
        "investigate",
        "Look at the process before running more: the confirmation runs miss the prediction by far.",
        [
            why,
            (
                "The runs already fit the model the active factors need, so more runs of the same kind would not "
                "explain the miss. Check whether the process shifted between the rounds (material, equipment, "
                "season), whether a run is an outlier, and whether a factor held constant changed."
            ),
        ],
        np.zeros((0, ctx.k)),
        "Once the cause is found, call recommend_next_step again; a shift between rounds is modelled as a block.",
    )


def _repeat_confirmation(ctx: _Context, why: str) -> _Step:
    """More confirmation runs at the same settings, after a miss small enough to be chance."""
    step = _confirm(ctx)
    n = ctx.n_confirmation_runs
    step.headline = f"Repeat the confirmation: {n} more runs at the same settings."
    step.reasons = [
        why,
        (
            f"The miss is small enough to be chance: a {100 * (1 - ctx.alpha):.0f}% interval misses about one time in "
            f"{round(1 / ctx.alpha)} when the model is right."
        ),
        *step.reasons[:1],
    ]
    step.after = (
        "Add the confirmation runs that missed to the runs, and call recommend_next_step with only these new runs as "
        "the confirmation runs: the model then knows the response at these settings, and the new runs are judged on "
        "their own."
    )
    return step


def _stop(ctx: _Context, judged: dict[str, Any]) -> _Step:
    return _Step(
        "stop",
        "Stop: the confirmation runs agree with the model at its optimum.",
        [
            (
                f"Their mean, {_g(judged['observed_mean'])}, lies inside the {100 * (1 - ctx.alpha):.0f}% prediction "
                f"interval for a mean of {judged['n_runs']} runs, {_g(judged['pi_low'])} to {_g(judged['pi_high'])}."
            ),
        ],
        np.zeros((0, ctx.k)),
        "Use these settings. Run them a few more times in production to see the process hold there.",
    )


# --------------------------------------------------------------------------- the decision
def _curved(ctx: _Context) -> bool:
    """Whether the runs show curvature, or a lack of fit, that the selected model does not capture."""
    if ctx.curvature.get("source") == "center_runs" and ctx.curvature.get("significant"):
        return True
    return bool(ctx.lack_of_fit.get("tested") and ctx.lack_of_fit.get("significant"))


#: Curvature that the runs cannot attribute gets axial runs in every numeric factor up to this many. Near an
#: optimum a factor's linear effect vanishes while its square does not, so the active factors are a poor guide.
_AXIAL_FOR_ALL = 6


def _rsm_step(ctx: _Context, why: str) -> _Step | None:
    numeric = [j for j in range(ctx.k) if ctx.study.numeric[j]]
    attributed = ctx.curvature.get("source") == "squares"
    if len(numeric) <= _AXIAL_FOR_ALL and not attributed:
        factors = numeric
    else:
        factors = ctx.active_factors or _model_factor_indices(ctx) or numeric
    step = _quadratic_runs(ctx, factors)
    if step is not None:
        step.reasons = [why, *step.reasons]
    return step


def _full_quadratic_fits(ctx: _Context) -> bool:
    """Say whether the runs already fit the full quadratic model in the active numeric factors."""
    numeric = [j for j in ctx.active_factors if ctx.study.numeric[j]]
    return bool(numeric) and _quadratic_fits(ctx, numeric)


def _decide(ctx: _Context) -> _Step:  # noqa: C901, PLR0911 - one return per rule, in the order the rules are read
    """Apply the rules of the module docstring, in order, to a study without confirmation runs to judge."""
    if ctx.model.error_df < _MIN_ERROR_DF and ctx.design.pure_error_df < _MIN_ERROR_DF:
        return _add_center_points(ctx) or _replicate(ctx)
    if not ctx.model.signal:
        if ctx.model.error_df < _FEW_ERROR_DF:
            return _more_error_df(ctx)
        if "widen_ranges" in ctx.history or (not ctx.ranges_can_move and "replicate" in ctx.history):
            return _no_effect(ctx)
        return _widen_ranges(ctx)
    de_alias = _de_alias(ctx)
    if de_alias is not None:
        return de_alias
    if _curved(ctx):
        why = (
            f"The centre runs are off the plane of the corners (curvature test p = {_g(ctx.curvature['p_value'])})."
            if ctx.curvature.get("source") == "center_runs"
            else f"The model does not fit the repeats (lack of fit p = {_g(ctx.lack_of_fit['p_value'])})."
        )
        step = _rsm_step(ctx, why)  # None when the runs already fit the full quadratic model
        if step is not None:
            return step
    squares = ctx.curvature.get("terms") or []
    if squares and not _full_quadratic_fits(ctx):
        why = (
            f"Squares are active ({', '.join(ctx.study.real(t) for t in squares)}), but the runs cannot fit the full "
            "quadratic model in the active factors, so the optimum cannot be located."
        )
        step = _rsm_step(ctx, why)
        if step is not None:
            step.action = "add_runs_for_model"
            return step
    step = _move_region(ctx) or _confirm(ctx)
    if ctx.lack_of_fit.get("significant"):
        step.reasons.append(
            f"The lack-of-fit test is significant (p = {_g(ctx.lack_of_fit['p_value'])}), though the runs fit the full "
            "quadratic model: the model is off somewhere in the region, or the repeats agree more closely than the "
            "noise usually allows. Runs at the optimum test it where it matters; look for an outlying run if they miss."
        )
    return step


def _learn_more(ctx: _Context, why: str) -> _Step | None:
    """After a failed confirmation: the runs that add the information the model lacked, or None when none would."""
    for build in (
        _de_alias,
        _add_center_points,
        lambda c: _rsm_step(c, why) if not _full_quadratic_fits(c) else None,
    ):
        step = build(ctx)
        if step is not None:
            step.reasons = [why, *step.reasons]
            return step
    return None


def _judge(ctx: _Context, confirmation: pd.DataFrame, raw: pd.Series) -> dict[str, Any]:
    """Judge the confirmation runs: the mean at each setting against the interval for such a mean."""
    points = _points(ctx.study, confirmation[ctx.study.names].to_numpy(dtype=float))
    summary = ctx.model.fit.get_prediction(points).summary_frame(alpha=ctx.alpha)
    observed = raw.to_numpy(dtype=float)
    tests = _mean_tests(ctx.model.fit, points[ctx.study.names], summary, (observed, observed), ctx.alpha)
    in_model = _model_factor_indices(ctx)
    for test in tests:
        at = np.array([test["settings"][name] for name in ctx.study.names])
        test["at_optimum"] = bool(np.all(np.abs(at[in_model] - ctx.optimum[in_model]) <= _SAME_SETTING))
    at_optimum = [t for t in tests if t["at_optimum"]]
    return {"tests": tests, "at_optimum": at_optimum[0] if at_optimum else None}


# --------------------------------------------------------------------------- the public function
def recommend_next_step(  # noqa: PLR0913
    factors: list[Factor],
    runs: pd.DataFrame,
    response: str,
    *,
    goal: Goal = "maximize",
    target: float | None = None,
    confirmation_runs: pd.DataFrame | None = None,
    model_terms: list[str] | None = None,
    budget: int | None = None,
    ranges_can_move: bool = True,
    n_confirmation_runs: int = 3,
    alpha: float = 0.05,
    random_state: int = 42,
    history: list[str] | None = None,
) -> dict[str, Any]:
    """Recommend the next round of a designed experiment from every run so far.

    Parameters
    ----------
    factors : list[Factor]
        The study's factors: continuous, or categorical at two levels. Their ``low`` and
        ``high`` define the coded scale (-1 to +1) and the region the optimum is sought in.
    runs : pandas.DataFrame
        Every run of the rounds so far, one row each, with a column per factor in its own
        units (a categorical factor's level) and the response. Runs with no response yet are
        left out.
    response : str
        The response column.
    goal : {"maximize", "minimize", "target"}
        What the study wants of the response.
    target : float or None
        The value aimed at, for ``goal="target"``.
    confirmation_runs : pandas.DataFrame or None
        Runs made to confirm a predicted optimum, judged against the model the other runs
        give, with the same columns as ``runs``.
    model_terms : list[str] or None
        The model to use, as terms (``"A"``, ``"A:B"``, ``"I(A ** 2)"``); ``None`` selects one by
        stepwise search with heredity (AICc), over the full quadratic model when the design can
        tell the factors' squares apart and over the two-factor interactions otherwise.
    budget : int or None
        Runs left. An action that needs more falls back to confirmation runs, or to stopping.
    ranges_can_move : bool
        Whether the next round may set factors past their ranges: the path of steepest ascent,
        or wider ranges. ``False`` keeps every recommended run inside them.
    n_confirmation_runs : int
        Runs made to confirm an optimum.
    alpha : float
        Significance level of every test the decision reads.
    random_state : int
        Seed for the search for the optimum and for runs chosen by exchange.
    history : list[str] or None
        The actions of the earlier rounds of this study, oldest first, as this function
        returned them. With ``"widen_ranges"`` among them (or ``"replicate"``, when the ranges
        cannot move), a study in which nothing stands out stops rather than widen the ranges
        again.

    Returns
    -------
    dict[str, Any]
        ``action`` (one of :data:`ACTIONS`), ``headline``, ``reasons`` (the evidence, with its
        numbers), ``runs`` (the next round, in the factors' own units, one dict per run),
        ``n_runs``, ``after`` (how to judge those runs), ``model`` (the terms, coefficients and
        p-values of the model the decision read, in the factors' names), ``evidence`` (active
        factors, curvature, lack of fit, aliasing, the optimum with its prediction and the edges
        it sits on, the confirmation runs' verdict), ``alternatives`` (other actions that apply,
        with their run counts) and, for some actions, ``new_ranges`` or ``path_predictions``.

    Raises
    ------
    ValueError
        For a mixture factor, a categorical factor not at two levels, a factor or response
        missing from the runs, a level not among a factor's levels, a target goal without a
        target, an action in ``history`` not among :data:`ACTIONS`, or fewer than two runs with
        a response.

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.experiments import Factor, TrueProcess, generate_design
    >>> factors = [Factor(name=n, low=-1, high=1) for n in "ABCDEFGH"]
    >>> design = generate_design(factors, design_type="dsd", n_center_points=5).design_actual
    >>> truth = TrueProcess({"Intercept": 100, "A": 10, "C": 7, "G": -6, "A:C": 8}, sigma=3)
    >>> runs = design.assign(y=truth.simulate(design.drop(columns="RunOrder"), random_state=1))
    >>> step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
    >>> step["action"], step["evidence"]["active_terms"]
    ('confirm', ['A', 'C', 'G', 'A:C'])
    """
    _check_factors(factors)
    if goal not in ("maximize", "minimize", "target"):
        raise ValueError(f"goal must be 'maximize', 'minimize' or 'target', not {goal!r}.")
    if goal == "target" and target is None:
        raise ValueError("A target goal needs its target.")
    if n_confirmation_runs < 1:
        raise ValueError(f"n_confirmation_runs must be at least 1, not {n_confirmation_runs}.")
    unknown = sorted(set(history or []) - set(ACTIONS))
    if unknown:
        raise ValueError(f"history holds {unknown}, which are not actions recommend_next_step returns: {ACTIONS}.")
    data = _code(factors, runs, response).dropna(subset=["y"]).reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("At least two runs with a response are needed to recommend the next step.")
    choices = _Choices(goal, target, alpha, n_confirmation_runs, ranges_can_move, random_state, tuple(history or ()))
    ctx = _context(_Study(factors, data, response), model_terms, choices)
    judged = None
    if confirmation_runs is not None and len(confirmation_runs):
        confirming = _code(factors, confirmation_runs, response).dropna(subset=["y"]).reset_index(drop=True)
        judged = _judge(ctx, confirming, confirming["y"])
        verdict = judged["at_optimum"]
        if verdict is not None and verdict["within_PI"]:
            step = _move_region(ctx) or _stop(ctx, verdict)
        else:
            # The confirmation runs are data now: the model is chosen again with them in.
            both = _Study(factors, pd.concat([data, confirming], ignore_index=True), response)
            judged_ctx, ctx = ctx, _context(both, model_terms, choices)
            if verdict is None:
                step = _decide(ctx)
                step.reasons.append(
                    "The confirmation runs were not at the optimum the model now predicts, so they join the other runs."
                )
            else:
                step = _after_a_miss(ctx, judged_ctx, verdict)
    else:
        step = _decide(ctx)
    step = _within_budget(ctx, step, budget)
    return _reply(ctx, step, judged)


def _after_a_miss(ctx: _Context, judged_ctx: _Context, verdict: dict[str, Any]) -> _Step:
    """Return the step after confirmation runs that miss the prediction.

    Runs that add what the design behind the prediction could not show come first. Then, if
    the model chosen again with the missed runs among its data puts its optimum elsewhere (a
    term the first model held by chance, say), that optimum is confirmed. Otherwise a miss
    within :data:`_GROSS_MISS` half-widths of the interval may be chance (the noise estimate
    behind the interval can itself be low), and the confirmation is made again; ``ctx`` holds
    the missed runs as data, so the model already knows the response there, and the new runs
    are judged on their own. Pooling them with the runs that missed would carry the first miss
    into the second test. A larger miss, or a miss after a repeat (two confirmations in
    ``history``), calls for a look at the process instead.
    """
    why = (
        f"The confirmation runs' mean, {_g(verdict['observed_mean'])}, lies outside the prediction interval "
        f"{_g(verdict['pi_low'])} to {_g(verdict['pi_high'])} for a mean of {verdict['n_runs']} runs: the model "
        "missed something, or the miss is chance."
    )
    step = _learn_more(judged_ctx, why)  # what the design the prediction came from could not show
    if step is not None:
        return step
    in_model = _model_factor_indices(ctx)
    if np.any(np.abs(ctx.optimum[in_model] - judged_ctx.optimum[in_model]) > _SAME_SETTING):
        step = _confirm(ctx)
        step.headline = f"Confirm the optimum the model now predicts, with {ctx.n_confirmation_runs} runs there."
        step.reasons = [
            why,
            "With the confirmation runs among its data, the model is chosen again, and its optimum has moved.",
            *step.reasons,
        ]
        return step
    half = (verdict["pi_high"] - verdict["pi_low"]) / 2
    gross = abs(verdict["observed_mean"] - verdict["predicted"]) > _GROSS_MISS * half
    if gross or ctx.history.count("confirm") >= 2:
        return _investigate_confirmation(ctx, why)
    return _repeat_confirmation(ctx, why)


@dataclass(frozen=True)
class _Choices:
    """The caller's choices that every decision reads."""

    goal: Goal
    target: float | None
    alpha: float
    n_confirmation_runs: int
    ranges_can_move: bool
    seed: int
    history: tuple[str, ...]


def _context(study: _Study, model_terms: list[str] | None, choices: _Choices) -> _Context:
    design = _design(study)
    study.data[_CURVATURE] = design.center.astype(float)
    model = _select(study, design, model_terms, choices.alpha)
    ctx = _Context(study, design, model, **vars(choices))
    return _gather(ctx)


def _within_budget(ctx: _Context, step: _Step, budget: int | None) -> _Step:
    """Fall back to confirmation runs, then to stopping, when the step needs more runs than are left."""
    if budget is None or len(step.runs) <= budget:
        return step
    if step.action != "confirm" and ctx.n_confirmation_runs <= budget and ctx.active:
        fallback = _confirm(ctx)
        fallback.reasons.insert(0, f"{step.headline} That needs {len(step.runs)} runs; {budget} are left.")
        return fallback
    return _Step(
        "stop",
        f"Stop: {budget} run(s) are left, fewer than the next step needs.",
        [
            f"{step.headline} That needs {len(step.runs)} runs.",
            f"The best settings the model knows predict {_g(ctx.predicted())}.",
        ],
        np.zeros((0, ctx.k)),
        "Use the best settings the model knows, and plan the next study when runs are available.",
    )


def _alternatives(ctx: _Context, chosen: str) -> list[dict[str, Any]]:
    """Return the other actions that apply to this study, each with its run count and headline."""
    options = []
    for build in (
        _confirm,
        _add_center_points,
        _de_alias,
        lambda c: _quadratic_runs(c, c.active_factors),
        _move_region,
    ):
        try:
            step = build(ctx)
        except ValueError:  # an augmentation the runs cannot take
            continue
        if step is not None and step.action != chosen:
            options.append({"action": step.action, "n_runs": len(step.runs), "headline": step.headline})
    return options


def _reply(ctx: _Context, step: _Step, judged: dict[str, Any] | None) -> dict[str, Any]:
    study, model = ctx.study, ctx.model
    low, high = ctx.mean_interval(ctx.n_confirmation_runs)
    coefficients = [
        {"term": study.real(str(term)), "coefficient": float(value), "p_value": float(model.fit.pvalues[term])}
        for term, value in model.fit.params.items()
    ]
    evidence: dict[str, Any] = {
        "n_runs": ctx.design.n_runs,
        "active_terms": [study.real(t) for t in ctx.active],
        "active_factors": [ctx.name(j) for j in ctx.active_factors],
        "noise_sd": model.sigma,
        "error_df": model.error_df,
        "curvature": {**ctx.curvature, "terms": [study.real(t) for t in ctx.curvature.get("terms", [])]},
        "lack_of_fit": ctx.lack_of_fit,
        "aliasing": [
            {**a, "term": study.real(a["term"]), "aliased_with": study.real(a["aliased_with"])} for a in ctx.aliasing
        ],
        "optimum": {
            "settings": study.actual(ctx.optimum),
            "coded": dict(zip([f.name for f in study.factors], (float(v) for v in ctx.optimum), strict=True)),
            "predicted": ctx.predicted(),
            "interval_for_mean": [low, high],
            "n_runs_in_mean": ctx.n_confirmation_runs,
            "edges": ctx.edges,
        },
    }
    if judged is not None:
        evidence["confirmation"] = [
            {**t, "settings": study.actual(np.array([t["settings"][n] for n in study.names]))} for t in judged["tests"]
        ]
    reply = {
        "action": step.action,
        "headline": step.headline,
        "reasons": step.reasons,
        "runs": _rows(study, step.runs) if len(step.runs) else [],
        "n_runs": len(step.runs),
        "after": step.after,
        "model": {
            "terms": [study.real(t) for t in model.terms],
            "judged_by": model.method,
            "judged": {study.real(t): p for t, p in model.judged.items()},
            "signal": model.signal,
            "coefficients": coefficients,
            "r_squared_adj": float(model.fit.rsquared_adj) if model.error_df > 0 else math.nan,
        },
        "evidence": evidence,
        "alternatives": _alternatives(ctx, step.action),
    }
    reply.update(step.extra)
    return reply
