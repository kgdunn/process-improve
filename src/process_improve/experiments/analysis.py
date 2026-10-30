# (c) Kevin Dunn, 2010-2026. MIT License.

"""Experiment analysis: fit models, ANOVA, diagnostics, residuals.

Provides :func:`analyze_experiment`, the main analytical workhorse for
designed experiments (Tool 3 in the DOE tool architecture).

Uses statsmodels and scipy for the heavy lifting, with thin custom code
for lack-of-fit, curvature test, Lenth's method, pred-R², adequate
precision, and confirmation run testing.
"""

from __future__ import annotations

import itertools
import keyword
import logging
import numbers
import re
import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import ModelDesc, dmatrix
from statsmodels.regression.linear_model import RegressionResultsWrapper

# ENG-02: the per-analysis implementations now live in ``_analyses``; they are
# re-exported here so the ``analyze_experiment`` dispatcher (and external
# importers such as ``tests/test_sec09`` reaching for ``_run_residual_diagnostics``)
# keep working unchanged.
from process_improve.experiments._analyses._shared import (
    BLOCK_COL,
    WHOLE_PLOT_COL,
    _compute_adequate_precision,
    _compute_pred_r_squared,
    _no_intercept_note,
    _sums_of_squares,
    _without_blocks,
    average_fitted_variance,
    with_prior_sigma,
)
from process_improve.experiments._analyses.aliasing import alias_chains_of, chain_reduced_fit
from process_improve.experiments._analyses.box_cox import _run_box_cox
from process_improve.experiments._analyses.curvature import _run_curvature_test
from process_improve.experiments._analyses.diagnostics import _run_residual_diagnostics
from process_improve.experiments._analyses.glm import INFORMATIONS, GlmSpec, check_glm_options, run_glm
from process_improve.experiments._analyses.lack_of_fit import _run_lack_of_fit
from process_improve.experiments._analyses.lenth import _run_lenth_method
from process_improve.experiments._analyses.mixture import (
    run_mixture_anova,
    run_mixture_effects,
    run_mixture_lenth,
    run_mixture_significance,
)
from process_improve.experiments._analyses.model_selection import _run_model_selection
from process_improve.experiments._analyses.ols_extractors import (
    _joint_inference,
    _run_anova,
    _run_coefficients,
    _run_confidence_intervals,
    _run_effects,
    _run_significance,
)
from process_improve.experiments._analyses.prediction import _run_confirmation_test, _run_prediction
from process_improve.experiments._analyses.split_plot import (
    DF_METHODS,
    SplitPlotSpec,
    _readable,
    run_split_plot,
    sum_coded,
    sum_coding_env,
)
from process_improve.experiments._analyses.transforms import transform_response, validate_transform
from process_improve.experiments._lm import suggest_identifier, validate_identifiers_are_safe
from process_improve.experiments.designs_mixture_constrained import SCHEFFE_MODELS, scheffe_formula_rhs
from process_improve.experiments.designs_mixture_process import ReducedRankFormula
from process_improve.experiments.models import validate_formula_is_safe, validate_identifier_is_safe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formula builder
# ---------------------------------------------------------------------------

#: Model names :func:`build_formula` expands; anything else is a formula or a raw RHS.
_NAMED_MODELS = ("main_effects", "interactions", "quadratic", *SCHEFFE_MODELS)

#: How far from 1 a run's proportions may sum and still be read as a blend: proportions printed
#: to three decimals (1/3 as 0.333) sum to 0.999 or 1.001.
_BLEND_SUM_TOLERANCE = 1e-2


def build_formula(
    response: str,
    factors: list[str],
    model: str | None = None,
    categorical: Collection[str] = (),
) -> str:
    """Build a patsy/statsmodels formula string.

    Parameters
    ----------
    response : str
        Name of the response column.
    factors : list[str]
        Factor column names.
    model : str or None
        ``"main_effects"``, ``"interactions"``, ``"quadratic"``, a Scheffé mixture
        model (``"scheffe_linear"``, ``"scheffe_quadratic"``,
        ``"scheffe_special_cubic"``, ``"scheffe_cubic"``), or an explicit formula string.  *None*
        defaults to ``"interactions"``.
    categorical : Collection[str]
        Factors that are categorical. ``"quadratic"`` adds no square for them, since a
        category has no square.

    Returns
    -------
    str
        A formula like ``"Y ~ A + B + A:B"``.
    """
    if model is None:
        model = "interactions"

    if "~" in str(model):
        return model

    joined = " + ".join(factors)

    if model == "main_effects":
        rhs = joined
    elif model == "interactions":
        rhs = f"({joined}) ** 2"
    elif model == "quadratic":
        squared = " + ".join(f"I({f} ** 2)" for f in factors if f not in categorical)
        rhs = f"({joined}) ** 2 + {squared}" if squared else f"({joined}) ** 2"
    elif model in SCHEFFE_MODELS:
        # Mixture components sum to 1, so the intercept is dropped; the fit holds the implicit
        # constant (see _as_mixture), which keeps R-squared and the model df centred.
        rhs = scheffe_formula_rhs(factors, model)
    else:
        # Treat as raw RHS
        rhs = model

    return f"{response} ~ {rhs}"


# ---------------------------------------------------------------------------
# Fitted model container
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Internal, unused container.

    Retained for backwards compatibility with any external caller that
    imports the name from this module. :func:`analyze_experiment` returns
    a plain :class:`dict` (see its ``Returns`` section), not an instance of
    this class, so downstream code that expects the return type of
    ``analyze_experiment`` should key into that dict rather than access
    attributes here.
    """

    ols_result: RegressionResultsWrapper = None
    formula: str = ""
    results: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analysis-type dispatch registry
# ---------------------------------------------------------------------------

_ANALYSIS_REGISTRY: dict[str, str] = {
    "anova": "anova",
    "effects": "effects",
    "coefficients": "coefficients",
    "significance": "significance",
    "residual_diagnostics": "residual_diagnostics",
    "lack_of_fit": "lack_of_fit",
    "curvature_test": "curvature_test",
    "model_selection": "model_selection",
    "box_cox": "box_cox",
    "lenth_method": "lenth_method",
    "confidence_intervals": "confidence_intervals",
    "prediction": "prediction",
    "confirmation_test": "confirmation_test",
    "split_plot": "split_plot",
    "glm": "glm",
}


# ---------------------------------------------------------------------------
# Preparing the data and the model
# ---------------------------------------------------------------------------


def _assemble_data(
    design_matrix: pd.DataFrame,
    responses: pd.DataFrame | pd.Series | None,
    response_column: str | None,
) -> tuple[pd.DataFrame, str]:
    """Join the responses onto the design and name the response column to analyse."""
    df = design_matrix.copy()
    if isinstance(responses, pd.Series):
        if responses.name is None:
            if response_column is None:
                raise ValueError(
                    "responses is an unnamed Series; give it a name (responses.rename('y')) or pass response_column."
                )
            responses = responses.rename(response_column)
        responses = responses.to_frame()
    if responses is not None:
        for col in responses.columns:
            df[col] = responses[col].values

    if response_column is not None:
        return df, response_column
    if responses is None:
        raise ValueError("Must provide either 'responses' or 'response_column'.")
    return df, responses.columns[0]


def _is_numeric(column: pd.Series) -> bool:
    """Whether patsy treats ``column`` as numeric (booleans and strings are categorical)."""
    return pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)


def _looks_like_mixture(df: pd.DataFrame, factor_cols: list[str]) -> bool:
    """Whether every run's factor settings are proportions that sum to 1, to within ``_BLEND_SUM_TOLERANCE``."""
    if len(factor_cols) < 2 or not all(_is_numeric(df[c]) for c in factor_cols):
        return False
    values = df[factor_cols].to_numpy(dtype=float)
    finite = values[np.isfinite(values).all(axis=1)]
    if finite.size == 0:
        return False
    in_unit_interval = bool(np.all((finite >= -1e-9) & (finite <= 1 + 1e-9)))
    return in_unit_interval and bool(np.allclose(finite.sum(axis=1), 1.0, rtol=0.0, atol=_BLEND_SUM_TOLERANCE))


def _mixture_components(df: pd.DataFrame, factor_cols: list[str]) -> list[str]:
    """Return the factors that are mixture components beside other factors, or none.

    They are the numeric factors whose settings all lie between 0 and 1, when there are at
    least two of them, other factors besides, and every run's settings of them sum to 1: the
    data of a mixture-process design. A process factor whose own settings lie between 0 and 1
    hides them, and none are found. Columns that are only ever 0 or 1 blend nothing: they are
    a categorical factor's indicators, one set per run, and none are found either.
    """
    numeric = [c for c in factor_cols if _is_numeric(df[c])]
    values = df[numeric].to_numpy(dtype=float)
    finite = values[np.isfinite(values).all(axis=1)]
    if finite.size == 0:
        return []
    shares = [
        c for c, column in zip(numeric, finite.T, strict=True) if np.all((column >= -1e-9) & (column <= 1 + 1e-9))
    ]
    if len(shares) < 2 or len(shares) == len(factor_cols):
        return []
    settings = finite[:, [numeric.index(c) for c in shares]]
    if np.isin(settings, (0.0, 1.0)).all():
        return []
    return shares if np.allclose(settings.sum(axis=1), 1.0, rtol=0.0, atol=_BLEND_SUM_TOLERANCE) else []


def _as_mixture(
    ols_result: RegressionResultsWrapper, df: pd.DataFrame, factor_cols: list[str], model: str
) -> tuple[RegressionResultsWrapper, bool]:
    """Return the fit, holding a constant when it is a mixture model, and whether to analyse it as one.

    A mixture model has no intercept: the components sum to 1, so their terms make up the
    constant, and R², the model df and the overall F are taken about the mean. statsmodels finds
    that constant by an exact rank test, which blends printed to three decimals (sums of 0.999 or
    1.001) fail. So it is decided here: a fit with no intercept, on blends that sum to 1 to within
    ``_BLEND_SUM_TOLERANCE``, holds the constant when its terms span the constant as closely. The fit
    is analysed as a mixture model (its linear blending tested as one block) for a Scheffe model,
    and for a formula that holds the constant over components alone.
    """
    pure = _looks_like_mixture(df, factor_cols)
    components = factor_cols if pure else _mixture_components(df, factor_cols)
    holds = False
    if components and "Intercept" not in ols_result.model.exog_names:
        exog = np.asarray(ols_result.model.exog, dtype=float)
        ones = np.ones(len(exog))
        gap = ones - exog @ np.linalg.lstsq(exog, ones, rcond=None)[0]
        holds = float(np.abs(gap).max()) <= _BLEND_SUM_TOLERANCE
    if holds and not ols_result.k_constant:
        sums = df[components].to_numpy(dtype=float).sum(axis=1)
        # statsmodels' rank test can miss the constant on blends that sum to 1 in floating point too.
        if not np.allclose(sums, 1.0, rtol=0.0, atol=1e-9):
            warnings.warn(
                f"The blends sum to between {sums.min():.6g} and {sums.max():.6g}, not to 1 exactly, as "
                "proportions rounded for printing do. The fit is taken to hold a constant, as a mixture model does: "
                "r_squared, df_model and the overall F are about the mean, and df_model is one less than the number "
                "of terms.",
                category=UserWarning,
                stacklevel=3,
            )
        ols_result = smf.ols(ols_result.model.formula, data=df, hasconst=True).fit()
    elif model in SCHEFFE_MODELS and not ols_result.k_constant:
        warnings.warn(
            f"The components do not sum to 1 to within {_BLEND_SUM_TOLERANCE:g} in every run, so the Scheffe model "
            "holds no constant: r_squared, df_model and the overall F are about zero, not about the mean.",
            category=UserWarning,
            stacklevel=3,
        )
    return ols_result, model in SCHEFFE_MODELS or (model not in _NAMED_MODELS and pure and holds)


def _coded_formula(formula: str, df: pd.DataFrame, factor_cols: list[str]) -> str:
    """Return ``formula``, beside mixture components with each categorical factor coded in ``L - 1`` columns.

    A model of mixture components has no intercept, so patsy would give a categorical
    factor a column for every level wherever the term below it is absent, and those
    columns would add up to ones the blends already span (see
    :class:`~process_improve.experiments.designs_mixture_process.ReducedRankFormula`).
    Any other formula is returned as it is.
    """
    categorical = [c for c in factor_cols if not _is_numeric(df[c])]
    if not categorical or not _mixture_components(df, factor_cols):
        return formula
    return ReducedRankFormula(formula, categorical)


def _references(rhs: str, column: str) -> bool:
    """Whether the formula right-hand side ``rhs`` names the data column ``column``."""
    return re.search(rf"(?<![\w.]){re.escape(column)}(?!\w)", rhs) is not None


def _resolve_model(
    df: pd.DataFrame,
    model: str | None,
    response_col: str,
    reported_response: str,
    factor_cols: list[str],
) -> tuple[str, list[str]]:
    """Settle the model to fit, and the factor columns it uses.

    ``None`` becomes ``"scheffe_quadratic"`` for mixture data (every run's settings sum
    to 1) and ``"interactions"`` otherwise. An explicit formula must model the response
    being analysed, and is rewritten to the stand-in name when that response is a Python
    keyword. For a formula or a raw right-hand side, only the columns it names are factors.
    """
    if model is None:
        return ("scheffe_quadratic" if _looks_like_mixture(df, factor_cols) else "interactions"), factor_cols
    if model in _NAMED_MODELS:
        return model, factor_cols
    rhs = model
    if "~" in model:
        lhs, rhs = (side.strip() for side in model.split("~", 1))
        if lhs not in (response_col, reported_response):
            raise ValueError(
                f"The formula models {lhs!r}, but the response being analysed is {reported_response!r}. "
                f"Write the formula as '{reported_response} ~ ...', or pass response_column={lhs!r}."
            )
        model = f"{response_col} ~ {rhs}"
    return model, [c for c in factor_cols if _references(rhs, c)]


def _estimable_default(
    df: pd.DataFrame,
    requested: str | None,
    model: str,
    factor_cols: list[str],
    rhs_of: Callable[[str], str],
) -> tuple[str, str, list[str]]:
    """Fit the default interactions model as far as the design can estimate it.

    A design that never ran some combination of two factors' levels (a missing cell, as
    constraints on categorical factors leave) cannot estimate those two factors'
    interaction in full, whatever its size: those interactions are left out and the rest
    are fitted (see :func:`_interactions_the_cells_lose`). A regular fraction aliases its
    interactions in whole chains (``A:B + C:D``), and the analyses test one column per
    chain, so the interactions model stays. A 12-run Plackett-Burman design for eight
    factors only partly aliases its 28 interactions with everything else: no chains form,
    the 37-term model has rank 12, and a fit would report a coefficient for every term,
    most of them arbitrary. Then the main-effects model is fitted. All of this is for the
    default only; a model the caller names is fitted as named. ``rhs_of`` gives the
    right-hand side of a named model's formula.

    Returns
    -------
    tuple[str, str, list[str]]
        The model to fit, a note saying what was done ("" when nothing was), and the
        interactions to leave out of its formula.
    """
    if requested is not None or model != "interactions" or len(factor_cols) < 2:
        return model, "", []

    def matrix(rhs: str) -> tuple[np.ndarray, list[str]]:
        coded = _coded_model_matrix(df, rhs, factor_cols, "interactions")
        return np.asarray(coded, dtype=float), [str(c) for c in coded.columns]

    def estimable(rhs: str) -> bool:
        exog, names = matrix(rhs)
        leaders = [names.index(chain[0][0]) for chain in alias_chains_of(exog, names)]
        return bool(np.linalg.matrix_rank(exog[:, leaders]) == len(leaders))

    full = rhs_of("interactions")
    exog, names = matrix(full)
    if np.linalg.matrix_rank(exog) == exog.shape[1]:
        return model, "", []  # every term is estimable
    lost = _interactions_the_cells_lose(df, factor_cols)
    if lost and estimable(f"{full} - {' - '.join(lost)}"):
        note = (
            f"The default interactions model was fitted without {', '.join(lost)}: "
            f"{_missing_cells_text(df, lost)}, so no design over the combinations this one ran can estimate "
            f"{'that interaction' if len(lost) == 1 else 'those interactions'} in full. Pass model='interactions' "
            "to fit it anyway."
        )
        warnings.warn(note, category=UserWarning, stacklevel=3)
        return model, note, lost
    if estimable(full):
        return model, "", []  # exactly aliased in chains, which the analyses test once each
    main = matrix(rhs_of("main_effects"))[0]
    if np.linalg.matrix_rank(main) < main.shape[1]:
        return model, "", []  # a supersaturated design: the main-effects model is not estimable either
    note = (
        f"The default interactions model has {len(names)} terms, of which this design can estimate "
        f"{np.linalg.matrix_rank(exog)}, and its interactions are not aliased in whole chains, so the main-effects "
        "model was fitted instead. Pass model='interactions' to fit it anyway, or analysis_type='model_selection' "
        "to choose the interactions the data support."
    )
    warnings.warn(note, category=UserWarning, stacklevel=3)
    return "main_effects", note, []


def _without_terms(formula: str, terms: list[str]) -> str:
    """Return ``formula`` with ``terms`` subtracted: the interactions a missing cell costs, in a named model."""
    return f"{formula} - {' - '.join(terms)}" if terms else formula


def _interactions_the_cells_lose(df: pd.DataFrame, factor_cols: list[str]) -> list[str]:
    """Return the two-factor interactions the combinations of levels this design ran cannot estimate in full.

    Each pair of factors is judged on its own: its main effects and interaction, fitted to
    the distinct combinations of the two that were run. A pair whose cells are all there
    (every pair of a Plackett-Burman design shows all four) estimates its interaction,
    however few the runs. A pair with a cell missing does not: catalyst at 3 levels and
    solvent at 2 with one pair never run leave 5 cells for the 6 coefficients, and no
    number of runs over those 5 cells changes that. So does a continuous factor held at
    one value on some level of the other.
    """
    lost = []
    for first, second in itertools.combinations(factor_cols, 2):
        cells = df[[first, second]].drop_duplicates().reset_index(drop=True)
        pair = np.asarray(
            _coded_model_matrix(cells, f"{first} + {second} + {first}:{second}", [first, second], "interactions"),
            dtype=float,
        )
        if np.linalg.matrix_rank(pair) < pair.shape[1]:
            lost.append(f"{first}:{second}")
    return lost


def _missing_cells_text(df: pd.DataFrame, lost: list[str]) -> str:
    """Say which combinations of levels were never run, for each lost interaction of two categorical factors."""
    parts = []
    for term in lost:
        first, second = term.split(":")
        if _is_numeric(df[first]) or _is_numeric(df[second]):
            parts.append(f"{term} has a factor that takes one value on a level of the other")
            continue
        seen = set(df[[first, second]].itertuples(index=False, name=None))
        missing = [
            f"{first} = {a!r} with {second} = {b!r}"
            for a in sorted(df[first].unique(), key=str)
            for b in sorted(df[second].unique(), key=str)
            if (a, b) not in seen
        ]
        parts.append(f"no run has {'; '.join(missing)}")
    return "; ".join(parts)


def _drop_incomplete_runs(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop runs with a missing response or factor setting, warning how many went; refuse when none is left."""
    complete = df[columns].notna().all(axis=1)
    if not complete.any():
        raise ValueError(
            f"No run has a value in every one of {', '.join(map(str, columns))}, so there is nothing to analyse: "
            "each run is missing its response or a factor setting."
        )
    n_dropped = int((~complete).sum())
    if n_dropped:
        warnings.warn(
            f"{n_dropped} run(s) with a missing response or factor setting were left out of the analysis.",
            category=UserWarning,
            stacklevel=3,
        )
        return df.loc[complete].copy()
    return df


def _stand_in(name: str, taken: pd.Index) -> str:
    """Return a name a formula can hold for the response: ``name_``, else ``name_1``, ``name_2``, ...

    The stand-in never gains a second underscore, which the formula check refuses as dunder access (#1002).
    """
    alias, k = f"{name}_", 0
    while alias in taken:
        k += 1
        alias = f"{name}_{k}"
    return alias


def _check_response_scale(response: pd.Series, name: str) -> None:
    """Refuse a response whose sums of squares cannot be held in double precision (#1005).

    Every test, R-squared and p-value is unchanged when the response is divided by a constant,
    so the remedy is a change of units; past this size ``n * max(y)**2`` overflows.
    """
    largest = float(np.max(np.abs(response.to_numpy(dtype=float))))
    if largest > np.sqrt(np.finfo(float).max / len(response)):
        raise ValueError(
            f"The response {name!r} is too large in its units to analyse: its sums of squares overflow, as its "
            f"largest value is {largest:.3g}. Divide it by a power of 10 and analyse it again; the tests, "
            "R-squared and p-values do not change."
        )


def _top_level_terms(rhs: str) -> list[str]:
    """Split a formula's right-hand side at the ``+`` signs outside parentheses."""
    terms, depth, start = [], 0, 0
    for i, char in enumerate(rhs):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if char == "+" and depth == 0:
            terms.append(rhs[start:i].strip())
            start = i + 1
    terms.append(rhs[start:].strip())
    return terms


def _add_block_term(df: pd.DataFrame, formula: str, model: str) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    """Add the blocks to the model as a fixed effect when the design has more than one block.

    Block-to-block differences left out of the model go into the residual and mask real
    effects (Montgomery, *Design and Analysis of Experiments*, chapter 7). The blocks
    enter as sum-coded contrast columns, ``Block1`` to ``Block{b-1}`` for ``b`` blocks
    (in sorted order, the last block coded -1 in every column). An explicit formula gets
    them only where it names ``Block`` as a term of its own, in that term's place, so the
    blocks are a categorical factor there as in a named model, not a number. A Scheffe
    mixture model, and an explicit formula that does not name ``Block``, are not changed;
    a warning says the blocks are not modelled.

    Returns
    -------
    tuple[pandas.DataFrame, str, tuple[str, ...]]
        The data with the contrast columns added, the formula, and the contrast names.
    """
    if BLOCK_COL not in df.columns or df[BLOCK_COL].nunique() < 2:
        return df, formula, ()
    response, rhs = formula.split(" ~ ", 1)
    if model in SCHEFFE_MODELS or model not in _NAMED_MODELS:
        terms = _top_level_terms(rhs)
        if model in SCHEFFE_MODELS or BLOCK_COL not in terms:
            if not _references(rhs, BLOCK_COL):
                kind = "a Scheffe mixture model" if model in SCHEFFE_MODELS else "an explicit formula"
                warnings.warn(
                    f"The design has a {BLOCK_COL!r} column, but blocks are not added to {kind}, so block-to-block "
                    "differences stay in the residual. Drop the column to silence this, or name it in the formula.",
                    category=UserWarning,
                    stacklevel=3,
                )
            return df, formula, ()
        rhs = " + ".join(term for term in terms if term != BLOCK_COL)

    levels = sorted(df[BLOCK_COL].unique(), key=str)
    df = df.copy()
    names: list[str] = []
    for k, level in enumerate(levels[:-1], start=1):
        name = f"{BLOCK_COL}{k}"
        while name in df.columns:
            name += "_"
        df[name] = np.where(df[BLOCK_COL] == level, 1.0, np.where(df[BLOCK_COL] == levels[-1], -1.0, 0.0))
        names.append(name)
    return df, f"{response} ~ {' + '.join([*names, *([rhs] if rhs else [])])}", tuple(names)


def _whole_plot_columns(df: pd.DataFrame, whole_plot: str | Sequence[str] | None, types: list[str]) -> tuple[str, ...]:
    """Name the columns of whole-plot labels: one, or several crossed groupings (a two-way split-plot design).

    A single name, or ``None``, is :func:`_whole_plot_column`'s. Several names must each be a
    column of the data, once each.
    """
    if whole_plot is None or isinstance(whole_plot, str):
        return (_whole_plot_column(df, whole_plot, types),)
    names = tuple(whole_plot)
    missing = [name for name in names if name not in df.columns]
    if not names or missing or len(set(names)) < len(names):
        raise ValueError(
            f"whole_plot={list(names)!r} must name one or more different columns of the data; "
            f"{', '.join(map(repr, missing)) or 'none'} missing."
        )
    return names


def _whole_plot_column(df: pd.DataFrame, whole_plot: str | None, types: list[str]) -> str:
    """Name the column of whole-plot labels; warn when the analysis would treat its runs as independent.

    ``whole_plot`` names the column; by default it is ``WholePlot``, which
    ``generate_design`` adds to a split-plot design, and which need not be present
    unless ``analysis_type="split_plot"`` is asked for. Every other analysis is ordinary
    least squares, which treats each run as independent, so a design with whole plots
    gets a warning.
    """
    column = WHOLE_PLOT_COL if whole_plot is None else whole_plot
    if column not in df.columns:
        if whole_plot is not None:
            raise ValueError(f"whole_plot={whole_plot!r} is not a column of the data.")
        if "split_plot" in types:
            raise ValueError(
                f"analysis_type='split_plot' needs each run's whole plot: a {WHOLE_PLOT_COL!r} column, as "
                "generate_design adds to a split-plot design, or whole_plot='<column>' naming one."
            )
        return column
    if "split_plot" not in types:
        warnings.warn(
            f"The runs are grouped into whole plots by the {column!r} column, but analysis_type={types} fits "
            "ordinary least squares, which treats every run as independent: whole-plot (hard-to-change) effects "
            "then look more significant than they are, and subplot effects less. Add 'split_plot' to "
            "analysis_type for the REML analysis, or drop the column to silence this.",
            category=UserWarning,
            stacklevel=3,
        )
    return column


def _check_term_count(formula: str) -> None:
    """Refuse a formula that expands beyond ``settings.max_formula_terms`` terms (SEC-19), as ``lm`` does."""
    from process_improve.config import settings  # noqa: PLC0415

    n_terms = len(ModelDesc.from_formula(formula).rhs_termlist)
    if n_terms > settings.max_formula_terms:
        raise ValueError(
            f"formula {formula!r} expanded to {n_terms} terms; the SEC-19 cap is "
            f"settings.max_formula_terms={settings.max_formula_terms}."
        )


def _code_factor(column: pd.Series) -> tuple[pd.Series, dict[str, Any] | None]:
    """Code one factor to -1/+1 for effects, returning the coded column and how it was coded.

    A numeric column already in coded units (symmetric about zero, with levels at -1
    and +1) is left alone. Any other numeric column is mapped from its own minimum and
    maximum to -1 and +1. A two-level categorical factor gets -1 for its first level (in
    sorted order) and +1 for its second; one with more levels has no single effect.
    """
    if _is_numeric(column):
        low, high = float(column.min()), float(column.max())
        coded_already = np.isclose(low, -high) and bool(np.isclose(np.abs(column.to_numpy(dtype=float)), 1.0).any())
        if high == low or coded_already:
            return column, None
        return (column - (high + low) / 2) / ((high - low) / 2), {"low": low, "high": high}
    levels = sorted(column.unique(), key=str)
    if len(levels) != 2:
        raise ValueError(
            f"Effects need two-level factors, but {column.name!r} has {len(levels)} levels, so it has no single "
            "effect. Use analysis_type='anova' or 'coefficients' for it."
        )
    return column.map({levels[0]: -1.0, levels[1]: 1.0}).astype(float), {"low": levels[0], "high": levels[1]}


def _coefficient_coding(
    column: pd.Series, bounds: Mapping[str, float] | None
) -> tuple[pd.Series, dict[str, Any] | None]:
    """Code one factor for the coded coefficients: from the range given, else as for the effects.

    A categorical factor without exactly two levels is left as it is: in a named model it
    is sum-coded in the formula instead.
    """
    if bounds is not None:
        low, high = _coding_range(str(column.name), bounds)
        return (column - (high + low) / 2) / ((high - low) / 2), {"low": low, "high": high}
    if not _is_numeric(column) and column.nunique() != 2:
        return column, None
    return _code_factor(column)


def _coding_range(name: str, bounds: object) -> tuple[float, float]:
    """Read one factor's ``{"low": ..., "high": ...}`` coding range, checking it is finite and increasing."""
    pair = [bounds.get(key) for key in ("low", "high")] if isinstance(bounds, Mapping) else []
    values = [float(v) for v in pair if isinstance(v, numbers.Real)]
    if len(values) == 2 and np.isfinite(values).all() and values[0] < values[1]:
        return values[0], values[1]
    raise ValueError(f"coding[{name!r}] must be {{'low': ..., 'high': ...}} with finite low < high; got {bounds!r}.")


def _resolve_coding(
    coding: object, df: pd.DataFrame, factor_cols: list[str], model: str
) -> str | Mapping[str, Mapping[str, float]]:
    """Check ``coding`` and return the scale it means: ``"actual"``, ``"coded"``, or ranges per numeric factor.

    ``"auto"``, the default, is ``"coded"`` except for a mixture model, and for data with
    mixture components beside process factors: components have no coded scale, so these keep
    ``"actual"``. ``None`` means ``"auto"``.
    """
    components = _mixture_components(df, factor_cols)
    if coding is None or coding == "auto":
        return "actual" if model in SCHEFFE_MODELS or components else "coded"
    resolved: str | Mapping[str, Mapping[str, float]]
    if isinstance(coding, Mapping):
        for name, bounds in coding.items():
            if name not in factor_cols:
                raise ValueError(
                    f"coding gives a range for {name!r}, which is not a factor; the factors are {factor_cols}."
                )
            if not _is_numeric(df[str(name)]):
                raise ValueError(
                    f"coding gives a range for {name!r}, a categorical factor; give ranges for numeric ones."
                )
            _coding_range(name, bounds)
        resolved = coding
    elif coding in ("actual", "coded"):
        resolved = str(coding)
    else:
        raise ValueError(f"coding must be 'actual', 'coded', 'auto', or a dict of factor ranges; got {coding!r}.")
    if resolved != "actual" and model in SCHEFFE_MODELS:
        raise ValueError(
            "A Scheffe mixture model has no coded scale: its components are proportions that sum to 1, and "
            "coding each one from -1 to +1 would break that. Use coding='actual', or 'auto'."
        )
    if resolved != "actual" and components:
        raise ValueError(
            f"The mixture components {components} have no coded scale: they are proportions that sum to 1, and "
            "coding each one from -1 to +1 would break that. Use coding='actual', or 'auto', and give the process "
            "factors in the units the coefficients should be in."
        )
    return resolved


@dataclass
class _Fit:
    """A fitted model and what the analyses need from the call that made it."""

    ols: RegressionResultsWrapper
    df: pd.DataFrame
    raw_response: pd.Series
    response_col: str
    factor_cols: list[str]
    model: str
    blocks: tuple[str, ...]
    transform_info: dict[str, Any]
    alpha: float
    whole_plot: str | tuple[str, ...] = WHOLE_PLOT_COL
    df_method: str = "satterthwaite"
    coding: str | Mapping[str, Mapping[str, float]] = "actual"
    _coded: tuple[RegressionResultsWrapper, dict[str, Any]] | None = None
    _coefficient_fit: RegressionResultsWrapper | None = None
    _reduced: tuple[RegressionResultsWrapper, dict[str, str]] | None = None
    glm: dict[str, Any] = field(default_factory=dict)
    #: The largest model model_selection may reach: the model asked for, which is ``model`` unless the
    #: default interactions model was not estimable and main effects were fitted instead.
    searched_model: str | None = None
    #: The response's name as the caller gave it; ``response_col`` can be a stand-in for a keyword.
    response_name: str = ""
    #: The error standard deviation known before the runs, which the standard errors use instead of the
    #: residual's (#789 B6); None for the residual's.
    sigma: float | None = None
    #: Whether the fit is analysed as a mixture model: a Scheffe model, or a formula that holds the constant
    #: over mixture components alone (see :func:`_as_mixture`).
    mixture: bool = False

    @property
    def group_cols(self) -> list[str]:
        """Columns identifying replicated runs: the factors, within each block when blocked."""
        return [*self.factor_cols, BLOCK_COL] if self.blocks else self.factor_cols

    def coded(self) -> tuple[RegressionResultsWrapper, dict[str, Any]]:
        """Return the model refitted on factors coded to -1/+1, the scale the effects are defined on."""
        if self._coded is None:
            coded_df = self.df.copy()
            coding: dict[str, Any] = {}
            for col in self.factor_cols:
                coded_df[col], how = _code_factor(self.df[col])
                if how is not None:
                    coding[col] = how
            if coding:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # the rank warning was already given for the fit
                    refit = smf.ols(self.ols.model.formula, data=coded_df).fit()
                self._coded = (refit, coding)
            else:
                self._coded = (self.ols, {})
        return self._coded

    def centred(self) -> RegressionResultsWrapper:
        """Return the fit with each numeric factor coded to -1/+1, so that its terms are tested at the centre.

        In a model with an interaction or a square, a lower-order term is tested where the
        terms above it are zero. On the factors as given that is wherever their units put
        zero (0 degC, say, far outside the experiment, and somewhere else again in kelvin);
        coded, it is the centre of the design. Only numeric factors are coded, so the term
        names do not change. The fit itself is kept for a mixture model and for data with
        mixture components beside process factors, whose components are proportions, and
        when coding would change the model rather than re-express it, as for an interaction
        without its main effects. When the units are so unequal that
        the fit as given lost rank in floating point, the coded fit, which has not, is used.
        """
        numeric = [c for c in self.factor_cols if _is_numeric(self.df[c])]
        coded_df = self.df.copy()
        for col in numeric:
            coded_df[col] = _code_factor(self.df[col])[0]
        if (
            self.model in SCHEFFE_MODELS
            or _mixture_components(self.df, self.factor_cols)
            or coded_df[numeric].equals(self.df[numeric])
        ):
            return self.ols
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the rank warning was already given for the fit
            refit = smf.ols(self.ols.model.formula, data=coded_df).fit()
        scale = 1.0 + float(np.abs(self.ols.fittedvalues).max())
        same_model = np.allclose(refit.fittedvalues, self.ols.fittedvalues, rtol=0.0, atol=1e-8 * scale)
        return refit if same_model or _rank(refit) > _rank(self.ols) else self.ols

    def factor_coding(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return the data on the coefficients' coded scale, and how each factor not already on it was coded."""
        ranges = self.coding if isinstance(self.coding, Mapping) else {}
        coded_df = self.df.copy()
        coding: dict[str, Any] = {}
        for col in self.factor_cols:
            coded_df[col], how = _coefficient_coding(self.df[col], ranges.get(col))
            if how is not None:
                coding[col] = how
        return coded_df, coding

    def coefficient_fit(self) -> RegressionResultsWrapper:
        """Return the fit the coefficients are read from: the fit itself, or its refit on the coded scale."""
        if self.coding == "actual":
            return self.ols
        if self._coefficient_fit is None:
            coded_df, coding = self.factor_coding()
            formula = self.ols.model.formula
            several = [c for c in self.factor_cols if not _is_numeric(self.df[c]) and self.df[c].nunique() > 2]
            if several and self.model in _NAMED_MODELS:
                formula = sum_coded(formula, several)
            if not coding and formula == self.ols.model.formula:
                self._coefficient_fit = self.ols
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # the rank warning was already given for the fit
                    self._coefficient_fit = smf.ols(formula, data=coded_df, eval_env=sum_coding_env()).fit()
                _warn_if_not_reparametrised(self.ols, self._coefficient_fit)
        return self._coefficient_fit

    def reduced(self) -> tuple[RegressionResultsWrapper, dict[str, str]]:
        """Return the centred fit on one column per alias chain, with the chain name of each retained column."""
        if self._reduced is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._reduced = chain_reduced_fit(self.centred())
        return self._reduced


def _effects(fit: _Fit, *, lenth: bool) -> dict[str, Any]:
    """Effects, or Lenth's method on them, from the -1/+1 coded refit."""
    coded, coding = fit.coded()
    if lenth:  # Lenth's method estimates the error from the effects themselves
        result = _run_lenth_method(coded, fit.alpha, blocks=fit.blocks)
    else:
        result = _run_effects(with_prior_sigma(coded, fit.sigma), blocks=fit.blocks)
    if coding:
        result["effects_coding"] = coding
    return result


def _rank(fit: RegressionResultsWrapper) -> int:
    return int(getattr(fit.model, "rank", np.linalg.matrix_rank(fit.model.exog)))


def _warn_if_not_reparametrised(fit: RegressionResultsWrapper, coded: RegressionResultsWrapper) -> None:
    """Warn when the coded refit fits differently from the fit itself, saying which one to doubt.

    Coding a factor shifts its origin and rescales it. In a hierarchical model that only
    re-expresses the same fit, so the fitted values agree. They differ for one of two
    reasons. Factors in units of very different size (pascals beside kelvin, say) can make
    the fit as given lose rank in floating point, which coding the factors repairs. Or the
    model is not hierarchical: with ``A:B`` and no ``B``, or ``I(A**2)`` and no ``A``, it
    spans a different space once ``A`` and ``B`` are centred.
    """
    scale = 1.0 + float(np.abs(fit.fittedvalues).max())
    if np.allclose(coded.fittedvalues, fit.fittedvalues, rtol=0.0, atol=1e-8 * scale):
        return
    if _rank(coded) > _rank(fit):
        message = (
            f"The factors' units are scaled so differently that the fit as given lost rank in floating point "
            f"({_rank(fit)} of {_rank(coded)} columns), so it, its model_summary and the analyses other than the "
            "coefficients are unreliable; the coded coefficients are not affected. Code the factors to -1/+1 "
            "before analysing."
        )
    else:
        message = (
            "On the coded scale this model fits the data differently, so the coded coefficients do not describe "
            "the fit in model_summary. A model with an interaction or a square but not the main effects below it "
            "is not the same model once its factors are centred; add those main effects, or use coding='actual'."
        )
    warnings.warn(message, category=UserWarning, stacklevel=4)


def _coefficients(fit: _Fit, analysis: str) -> dict[str, Any]:
    """Return the coefficients, or their confidence intervals, on the scale ``coding`` asks for."""
    ols = fit.coefficient_fit()
    shown = with_prior_sigma(ols, fit.sigma)
    extract = _run_coefficients if analysis == "coefficients" else _run_confidence_intervals
    result = extract(shown, fit.alpha)
    for row in result[analysis]:
        row["term"] = _readable(row["term"])
    result["coding"] = "actual" if fit.coding == "actual" else "coded"
    if fit.model not in SCHEFFE_MODELS and (coding := fit.factor_coding()[1]):
        result["factor_coding"] = coding
    if analysis == "coefficients":
        result |= _joint_inference(shown, fit.alpha, _readable)
        if ols is not fit.ols:
            # The equation in the factors' own units, for whoever needs to evaluate it there.
            actual = with_prior_sigma(fit.ols, fit.sigma)
            result["coefficients_actual"] = _run_coefficients(actual, fit.alpha)["coefficients"]
    return result


def _split_plot(fit: _Fit, points: pd.DataFrame | None) -> dict[str, Any]:
    """REML analysis with the whole plots as a random effect; categorical factors sum-coded in a named model.

    The result carries the names and the scale of its coefficients, as ``"coefficients"`` does, so
    it can be passed to ``optimize_responses`` as a fitted model. The fit is on the runs as given:
    its coefficients are coded when the factors are already coded, and in actual units otherwise.
    """
    formula = fit.ols.model.formula
    if fit.model in _NAMED_MODELS and fit.model not in SCHEFFE_MODELS:
        formula = sum_coded(formula, [c for c in fit.factor_cols if not _is_numeric(fit.df[c])])
    spec = SplitPlotSpec(
        whole_plot=fit.whole_plot,
        alpha=fit.alpha,
        blocks=fit.blocks,
        df_method=fit.df_method,
        information=fit.glm["information"],
    )
    result = run_split_plot(fit.df, formula, spec, new_points=points)
    coding = fit.factor_coding()[1] if fit.model not in SCHEFFE_MODELS else {}
    result["split_plot"] = {
        "response_name": fit.response_name or fit.response_col,
        "factor_names": list(fit.factor_cols),
        "coding": "actual" if coding else "coded",
        **({"factor_coding": coding} if coding else {}),
        **result["split_plot"],
    }
    return result


def _mixture_handlers(fit: _Fit) -> dict[str, Callable[[], dict[str, Any]]]:
    """Analyses that differ for a Scheffe mixture model, keyed by analysis type."""
    return {
        "anova": lambda: _noted(run_mixture_anova(fit.ols, fit.factor_cols), _no_intercept_note(fit.ols)),
        "significance": lambda: run_mixture_significance(fit.ols, fit.factor_cols, fit.alpha),
        "effects": lambda: run_mixture_effects(fit.ols, fit.factor_cols),
        "lenth_method": run_mixture_lenth,
    }


def _handlers(
    fit: _Fit,
    new_points: pd.DataFrame | None,
    observed_at_new: list[float] | None,
) -> dict[str, Callable[[], dict[str, Any]]]:
    """Every analysis, keyed by analysis type, each run only when requested."""

    # Block contrasts a new point does not give are zero: the average over the blocks.
    points = None if new_points is None else new_points.assign(**{b: 0.0 for b in fit.blocks if b not in new_points})

    def prediction() -> dict[str, Any]:
        if points is None:
            return {"prediction": {"error": "new_points is required for prediction."}}
        return _run_prediction(with_prior_sigma(fit.ols, fit.sigma), points, fit.alpha, fit.transform_info)

    def confirmation() -> dict[str, Any]:
        if points is None or observed_at_new is None:
            return {"confirmation_test": {"error": "new_points and observed_at_new are required."}}
        known = with_prior_sigma(fit.ols, fit.sigma)
        return _run_confirmation_test(known, points, observed_at_new, fit.alpha, fit.transform_info)

    def box_cox() -> dict[str, Any]:
        raw = fit.df.assign(**{fit.response_col: fit.raw_response})
        return _run_box_cox(raw, fit.response_col, exog=fit.ols.model.exog, alpha=fit.alpha)

    handlers: dict[str, Callable[[], dict[str, Any]]] = {
        "anova": lambda: _noted(
            _run_anova(fit.reduced()[0], labels=fit.reduced()[1], blocks=fit.blocks), _no_intercept_note(fit.ols)
        ),
        "effects": lambda: _effects(fit, lenth=False),
        "coefficients": lambda: _coefficients(fit, "coefficients"),
        "significance": lambda: _run_significance(
            with_prior_sigma(fit.reduced()[0], fit.sigma), fit.alpha, labels=fit.reduced()[1], blocks=fit.blocks
        ),
        "residual_diagnostics": lambda: _run_residual_diagnostics(fit.ols),
        "lack_of_fit": lambda: _run_lack_of_fit(fit.ols, fit.df, fit.response_col, fit.group_cols, fit.alpha),
        "curvature_test": lambda: _run_curvature_test(
            fit.df, fit.response_col, fit.factor_cols, fit.alpha, group_cols=fit.group_cols
        ),
        "model_selection": lambda: _run_model_selection(
            fit.df, fit.response_col, fit.factor_cols, model=fit.searched_model or fit.model
        ),
        "box_cox": box_cox,
        "lenth_method": lambda: _effects(fit, lenth=True),
        "confidence_intervals": lambda: _coefficients(fit, "confidence_intervals"),
        "prediction": prediction,
        "confirmation_test": confirmation,
        "split_plot": lambda: _split_plot(fit, points),
        "glm": lambda: {
            "glm": run_glm(fit.df, fit.ols.model.formula, GlmSpec(**fit.glm), alpha=fit.alpha, new_points=points)
        },
    }
    if fit.mixture:
        # The usual ANOVA, significance and effects test each blending coefficient
        # against zero, which means nothing for a mixture.
        handlers.update(_mixture_handlers(fit))
    elif _mixture_components(fit.df, fit.factor_cols):
        handlers.update(dict.fromkeys(("effects", "lenth_method"), _no_effects_beside_components))
    return handlers


def _no_effects_beside_components() -> dict[str, Any]:
    """Refuse effects for data with mixture components beside process factors."""
    raise ValueError(
        "Effects are measured between each factor's -1 and +1 settings, and mixture components, which are "
        "proportions that sum to 1, have none. Use analysis_type='coefficients' for the blending coefficients "
        "in proportions, or 'anova'."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_experiment(  # noqa: PLR0913
    design_matrix: pd.DataFrame,
    responses: pd.DataFrame | pd.Series | None = None,
    model: str | None = None,
    analysis_type: str | list[str] = "anova",
    significance_level: float = 0.05,
    transform: str | None = None,
    coding: str | Mapping[str, Mapping[str, float]] | None = "auto",
    new_points: pd.DataFrame | None = None,
    observed_at_new: list[float] | None = None,
    response_column: str | None = None,
    whole_plot: str | Sequence[str] | None = None,
    family: str | None = None,
    link: str | None = None,
    trials: str | float | Sequence[float] | None = None,
    dispersion: str | float | None = None,
    information: str = "expected",
    df_method: str = "satterthwaite",
    sigma: float | None = None,
) -> dict[str, Any]:
    """Fit models, run ANOVA, compute effects, diagnose residuals.

    Parameters
    ----------
    design_matrix : DataFrame
        Factor settings per run.  May also contain the response column(s). A
        ``RunOrder`` column is ignored. A ``Block`` column with two or more blocks
        is added to the named non-mixture models as a fixed effect (sum-coded contrast
        columns ``Block1``, ...; one ``"Block"`` row in the ANOVA), so block-to-block
        differences do not inflate the error; drop the column to analyse without it. An
        explicit formula gets the same fixed effect where it names ``Block`` as a term
        (``"y ~ Block + A + B + A:B"``).
        A ``WholePlot`` column, which ``generate_design`` adds to a split-plot design,
        labels each run's whole plot: it is used by ``analysis_type="split_plot"``,
        and any other analysis warns that it treats the runs as independent.
        Runs with a missing response or factor setting are left out, with a warning.
    responses : DataFrame, Series, or None
        Response column(s).  If *None*, ``response_column`` must name a
        column already present in *design_matrix*. When a DataFrame with
        more than one column is passed, only the first column is analysed;
        the rest are added to the working frame but ignored by every
        subsequent step. Use ``response_column`` to pick a specific column
        explicitly. An unnamed Series takes the name ``response_column``.
    model : str or None
        ``"main_effects"``, ``"interactions"``, ``"quadratic"``, a Scheffé mixture
        model (``"scheffe_linear"``, ``"scheffe_quadratic"``,
        ``"scheffe_special_cubic"``, ``"scheffe_cubic"``, fitted without an intercept since the
        components sum to 1), an explicit formula whose left-hand side is the
        response, or *None*: ``"scheffe_quadratic"`` when every run's factor
        settings are proportions summing to 1, and ``"interactions"`` otherwise.
        Blends that sum to 1 only to within 0.01, as proportions printed to three
        decimals do, are blends too, with a warning. A formula with no intercept, over
        blends alone, whose terms make up the constant is analysed as a Scheffé model.
        That default becomes ``"main_effects"``, with a warning and
        ``model_summary["model_note"]``, when the design cannot estimate the
        interactions and does not alias them in whole chains (a Plackett-Burman or
        definitive screening design); ``"model_selection"`` still searches the
        interactions. The model may expand to at most ``settings.max_formula_terms`` terms.
    analysis_type : str or list[str]
        One or more of: ``"anova"``, ``"effects"``, ``"coefficients"``,
        ``"significance"``, ``"residual_diagnostics"``, ``"lack_of_fit"``,
        ``"curvature_test"``, ``"model_selection"``, ``"box_cox"``,
        ``"lenth_method"``, ``"confidence_intervals"``, ``"prediction"``,
        ``"confirmation_test"``, ``"split_plot"``, ``"glm"``.

        The ANOVA uses Type II sums of squares (reported as ``anova_type``). The ANOVA
        and significance test each term at the centre of the design: they are computed
        with the numeric factors coded to -1/+1, so they do not depend on the factors'
        units. On the factors as given, a main effect alongside its interaction or square
        would be tested where those are zero (0 degC, say), and so would differ between
        degC and kelvin. A model that coding would change, such as an interaction without
        its main effects, is tested as given. The
        ANOVA and significance list exactly aliased terms once, as their alias chain
        (``"A:B + C:D"``), as the effects do; the coefficients keep one entry per term.
        A factor nested in another, whose levels each fall in one of that other's, leaves
        the model matrix short of full rank: the ANOVA then tests each term on the
        degrees of freedom it adds, gives a term that adds none no test, and says so in
        ``anova_note``, naming the strata for
        :func:`~process_improve.experiments.multistratum_anova`. Effects and Lenth's
        method use the factors coded to -1/+1: a numeric factor not already coded is
        mapped from its own minimum and maximum, and a two-level categorical factor
        from its first level to its second (sorted); the mapping is reported under
        ``effects_coding``. An effect is the change in response from the lowest to the
        highest value of its model column: twice the coefficient for a main effect or
        interaction, the coefficient itself for a square such as ``I(A**2)``. The curvature test uses
        pure error from the replicated runs.

        Each of the ``"coefficients"`` carries its ``std_error``, ``t_value``, ``p_value``, its
        interval ``ci_low`` to ``ci_high`` at the ``confidence_level`` of 1 - ``significance_level``,
        and its ``standardised_coefficient``: the coefficient
        times the standard deviation of its column of the model matrix, over the standard
        deviation of the response (the "beta" coefficient, MYSTAT's STD COEF). It has no
        units, so terms in different units compare; the intercept's is 0. Like the
        coefficient, it depends on ``coding`` when the model has interactions or squares.
        The ``"coefficients"`` analysis also gives ``estimate_correlations``, the correlation
        of each pair of estimates, from :math:`(X'X)^{-1}` alone, and
        ``joint_confidence_region``: its ``sum_of_squares`` bounds the values of all the
        coefficients together at ``1 - significance_level``, :math:`S_R (1 + p F / (n - p))`
        with :math:`S_R` the ``residual_sum_of_squares`` (Box, Hunter and Hunter, eq. 14.45);
        with a ``sigma``, :math:`S_R` plus sigma squared times the upper chi-square point on
        :math:`p` degrees of freedom.

        ``"split_plot"`` is for a design whose hard-to-change factors change only
        between whole plots. Every other analysis is ordinary least squares, which
        treats the runs as independent, so it makes the whole-plot effects look more
        significant than they are and the subplot effects less. This one fits the
        model with a random whole-plot effect, by REML, and tests each term on
        Satterthwaite's denominator degrees of freedom, so a whole-plot term is judged
        against the whole-plot error and a subplot term against the run-to-run error.
        In a balanced design this is the classical split-plot ANOVA. The tests are
        marginal (Type III), with every categorical factor coded to sum-to-zero
        contrasts for them whatever coding the formula gives it, so a main effect is
        averaged over the factors it interacts with. The coefficients are in the
        formula's coding; a named model sum-codes its categorical factors. The result,
        under ``"split_plot"``, has the ``variance_components`` (``whole_plot``,
        ``residual`` and their ratio ``eta``, which a split-plot design is built
        for), the ``error_df`` of the two strata, ``coefficients`` and per-term
        ``tests`` (each with its ``df``, ``stratum`` and p-value), and the
        ``significant_terms``. The degrees of freedom use the expected information,
        so they equal Kenward and Roger's (as R's pbkrtest gives them); R's lmerTest
        uses the observed information, and gives slightly larger ones in an
        unbalanced design. ``df_method="kenward_roger"`` also applies Kenward and
        Roger's inflation of the standard errors and their scaling of the F tests, and
        ``information="observed"`` computes the df from the observed information: the
        two together are JMP's analysis. In a balanced design all of these agree. A
        design run in random blocks is analysed the same way, the blocks named as the
        whole plots. With ``new_points``, the result also has ``predictions`` from the
        generalized least-squares fit: the ``predicted`` mean with its ``std_error``,
        ``df`` and interval (``ci_low``, ``ci_high``), and the interval for one new
        run there (``pi_low``, ``pi_high``, with ``pi_std_error`` and ``pi_df``), which
        falls in a new whole plot, so its variance adds every variance component. The
        result carries ``response_name``, ``factor_names`` and ``coding`` as
        ``"coefficients"`` does, so it can be passed to ``optimize_responses`` as a
        fitted model. The fit is made on the runs as given: its coefficients are coded
        when the factors are already coded, and in actual units otherwise.
        ``"glm"`` fits a generalized linear model of the ``family`` given, for a response that
        is a count of defects (``"poisson"``), a number of successes out of a number of
        ``trials`` (``"binomial"``) or a positive amount whose spread grows with its size
        (``"gamma"``). The coefficients are found by maximum likelihood and tested by Wald's z;
        the interval for a mean is worked on the linear predictor's scale and carried through the
        inverse link, so it stays in range. The result, under ``"glm"``, has the
        ``coefficients`` (with ``exp_coefficient``, an odds ratio or a rate ratio, under a logit
        or log link), the ``log_likelihood``, ``deviance`` and ``pearson_chi2``, the
        ``goodness_of_fit`` tests (Hosmer and Lemeshow's for the binomial family), and the
        ``fitted`` mean of each run with its interval; with ``new_points``, ``predictions``
        there as well. See :func:`~process_improve.experiments._analyses.glm.run_glm`.
    significance_level : float
        Default 0.05. Used by every test's ``significant`` flag and every interval.
    transform : str or None
        ``"log"`` (natural), ``"log10"``, ``"sqrt"``, ``"inverse"``, ``"box_cox"``, or ``None``.
        The two logs give the same F tests and p-values; the sums of squares, coefficients and
        predictions are on the scale chosen. The model is
        fitted on the transformed response, so predictions and intervals are on that
        scale (reported as ``prediction_scale`` / ``scale``), and confirmation runs are
        transformed the same way before they are compared. Box-Cox chooses ``lambda``
        by the profile likelihood of the model. A response outside the transform's
        domain raises ``ValueError``.
    coding : {"auto", "actual", "coded"}, dict or None, default "auto"
        The scale of the ``"coefficients"`` and the ``"confidence_intervals"``; the other
        analyses do not use it.

        - ``"auto"`` (default): ``"coded"``, except for a Scheffe mixture model, whose
          components are proportions and which keeps ``"actual"``. ``None`` means the same.
          Before 2.0.0 the default was ``"actual"``; pass it to keep the factors' own scale.
        - ``"actual"``: the factors as given.
        - ``"coded"``: the model refitted with each numeric factor mapped from its minimum
          and maximum to -1 and +1 (a factor already in coded units is left as it is), a
          two-level categorical factor from its first level to its second, as for the
          effects, and, in a named model, a categorical factor with more levels given
          sum-to-zero contrasts.
        - A dict such as ``{"T": {"low": 150, "high": 200}}``: as ``"coded"``, but each
          factor it names is mapped from that range, the same form as
          ``optimize_responses``' ``factor_ranges``. Use it when the data's extremes are not
          the levels that should be -1 and +1, such as a central composite design's axial
          runs.

        Whenever the coefficients are refitted on the coded scale, the result also holds
        them on the factors' own scale, as ``coefficients_actual``: the equation to use for
        predictions in actual units.

        On the coded scale a coefficient is half the change in response across its factor's
        range, and in a model with interactions a main effect is tested at the centre of the
        design. On the actual scale it is tested where the other factors are zero, which can
        be far outside the experiment (0 degC, say), so its sign and p-value can differ.
        ``optimize_responses`` evaluates models at coded settings, so pass it coded
        coefficients. The result reports the scale as ``coding``, and, as ``factor_coding``,
        the ``low`` and ``high`` that map to -1 and +1 for each factor not already coded.
    new_points : DataFrame or None
        For prediction or confirmation testing, and for the predictions of
        ``"split_plot"`` and ``"glm"``. In a blocked model, points without the block
        contrast columns are predicted for the average block.
    observed_at_new : list[float] or None
        Observed values at *new_points* (for confirmation testing), one per row.
    response_column : str or None
        Name of the response column when it lives inside *design_matrix*.
    whole_plot : str, sequence of str, or None
        The column labelling each run's whole plot, for ``"split_plot"``. ``None``
        (default) uses the ``WholePlot`` column when there is one. The column is never
        a factor. Several columns are several crossed random effects: the rows and the
        columns of a two-way split-plot (strip-plot) design, whose runs share a row's
        first-stage setting and a column's second-stage processing. The result then has
        one variance component, error df and set of BLUPs per column, named by it.
    family : str or None
        For ``"glm"``: ``"binomial"``, ``"poisson"``, ``"gamma"``, ``"gaussian"`` or
        ``"inverse_gaussian"``.
    link : str or None
        For ``"glm"``: ``"logit"``, ``"probit"``, ``"cloglog"``, ``"log"``, ``"identity"``,
        ``"inverse"``, ``"inverse_squared"`` or ``"sqrt"``. ``None`` (default) uses the family's
        canonical link.
    trials : str, float, sequence of float or None
        For a binomial ``"glm"``: the number of trials of each run, whose response is then its
        number of successes. A column of *design_matrix* (never a factor), one number for every
        run, or one per run. ``None`` (default) takes the response as 0 or 1.
    dispersion : {"pearson", "deviance", "ml"}, float or None
        For ``"glm"``: how the dispersion is found. ``None`` (default) fixes it at 1 for the
        binomial and Poisson families and estimates it by Pearson's statistic over the residual
        degrees of freedom for the others, as R's ``glm`` does; ``"ml"`` estimates it by maximum
        likelihood, as SAS's GENMOD does. A number fixes it there.
    information : {"expected", "observed"}
        For ``"glm"``: whether the covariance of the coefficients inverts the expected
        information (default, as R does) or the observed information (as SAS's GENMOD does).
        The two are the same under each family's canonical link. For ``"split_plot"``: the
        REML information the degrees of freedom are computed from, the expected (default,
        as SAS and R's pbkrtest do) or the observed (as JMP and R's lmerTest do).
    df_method : {"satterthwaite", "kenward_roger"}
        For ``"split_plot"``: ``"satterthwaite"`` (default) tests each term on
        Satterthwaite's df with the model-based standard errors; ``"kenward_roger"`` adjusts
        the covariance of the estimates for the uncertainty in the variance components and
        scales the F tests, as Kenward and Roger (1997) do. A single coefficient's df are the
        same either way.
    sigma : float or None
        The error standard deviation when it is known before these runs, from past records
        or earlier experiments. The standard errors of the ``"coefficients"``, ``"effects"``,
        ``"confidence_intervals"``, ``"significance"``, ``"prediction"`` and
        ``"confirmation_test"`` then use it instead of the residual, and their tests and
        intervals read the normal distribution, not the t, since sigma is not estimated from
        the runs: each ``t_value`` is a z value. A saturated design gets standard errors this
        way too. The ANOVA, the lack-of-fit test, Lenth's method and a mixture model's
        ``"significance"`` and ``"effects"`` keep the residual. Give sigma in the units of the
        response as it is fitted, after ``transform``. None (the default) uses the residual.

    Returns
    -------
    dict[str, Any]
        Results keyed by analysis type. Always includes ``"response_name"``,
        ``"factor_names"`` (the factor columns, in order), so that the result
        with ``"coefficients"`` can be passed to ``optimize_responses`` as a
        fitted model, and ``"model_summary"`` with the keys:

        - ``formula`` - the resolved patsy formula that was fitted.
        - ``model`` - the model fitted: a model name, or ``"formula"``.
        - ``transform`` - the response transform, or ``None``; with
          ``box_cox_lambda`` for Box-Cox.
        - ``r_squared`` - R^2 of the fit.
        - ``r_squared_adj`` - adjusted R^2.
        - ``r_squared_pred`` - prediction R^2 (leave-one-out style).
        - ``adequate_precision`` - signal-to-noise ratio (>= 4 is
          considered adequate): the range of the fitted values over the square root of
          ``average_fitted_variance``.
        - ``average_fitted_variance`` - the average variance of the fitted values over the
          runs, p s²/n with s² the residual mean square (p sigma²/n with a ``sigma``): the
          runs' leverages sum to p, the number of terms the model estimates, whatever the
          design (Box and Wetz). Its square root is the fitted values' average standard error.
        - ``n_obs`` - number of observations used to fit.
        - ``n_terms`` - number of columns in the model matrix.
        - ``model_rank`` - rank of the model matrix with the numeric
          factors coded to -1/+1, the number of terms the design can
          estimate whatever the units; less than ``n_terms`` implies
          aliasing / rank deficiency.
        - ``rank_deficient`` - ``True`` if ``model_rank < n_terms``.
        - ``ill_conditioned`` - ``True`` if, in the factors' own units, the
          model matrix lost rank in floating point although the design did
          not: the units are too unequal, and the fit is unreliable.
        - ``df_model`` - model degrees of freedom.
        - ``df_residual`` - residual degrees of freedom.
        - ``mse_residual`` - mean squared error of the residuals.
        - ``prior_sigma`` - the ``sigma`` given, when one is: the standard errors,
          ``adequate_precision`` and ``average_fitted_variance`` use it, not the square root of
          ``mse_residual``.
        - ``f_statistic`` and ``f_p_value`` - the overall F test of the model against the
          intercept alone (against zero, on uncorrected sums of squares, with no intercept).
        - ``log_likelihood``, ``aic`` and ``bic`` - the Gaussian log-likelihood and the criteria
          that count the error variance as a parameter, as R's ``AIC`` does; ``aic_step``,
          n ln(RSS/n) + 2p, as R's ``step`` prints it.
        - ``ss_total``, ``ss_mean`` and ``ss_total_uncorrected`` - the total sum of squares
          about the mean, the correction for the mean (n times the mean squared), and the
          crude total, the sum of the squared responses, which is the other two added.
        - ``ss_model`` - the model's sum of squares on the basis of ``r_squared`` and
          ``f_statistic``: about the mean, or about zero for a model whose terms do not span
          a constant; ``ss_model_uncorrected`` - the sum of the squared fitted values, the
          model's sum about zero; ``ss_residual`` - the residual sum of squares.
        - ``coefficient_of_variation`` - 100 times the residual standard deviation over the
          mean response, Design-Expert's "C.V. %"; ``None`` when the mean is zero.
        - With blocks in the model, ``ss_blocks`` (the blocks' sum of squares, fitted first)
          and ``r_squared_without_blocks``, ``r_squared_adj_without_blocks`` and
          ``r_squared_pred_without_blocks``: the three R-squared values over the total less
          the blocks' sum of squares, as Design-Expert reports them.

        Notes from individual analyses are kept apart, as ``anova_note``,
        ``effects_note``, ``lenth_note``, ``significance_note`` and
        ``split_plot_note``. The ``model_summary`` is of the least-squares fit, also
        for ``"split_plot"``.

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.experiments.analysis import analyze_experiment
    >>> df = pd.DataFrame({
    ...     "A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1],
    ...     "y": [28, 36, 18, 31],
    ... })
    >>> result = analyze_experiment(df, response_column="y", analysis_type="coefficients")
    >>> result["coefficients"][0]["term"]
    'Intercept'

    With the error standard deviation known from earlier runs, the four runs, which leave
    no residual, still give each coefficient a standard error, sigma over the square root of 4:

    >>> known = analyze_experiment(df, response_column="y", analysis_type="coefficients", sigma=2.0)
    >>> [round(row["std_error"], 6) for row in known["coefficients"]]
    [1.0, 1.0, 1.0, 1.0]
    """
    df, response_col = _assemble_data(design_matrix, responses, response_column)

    # User-supplied names (response_column / design_matrix dict keys) are
    # interpolated into the patsy formula, so reject anything that is not a
    # plain identifier before it can become an injection vector (SEC-14).
    validate_identifier_is_safe(response_col)

    if response_col not in df.columns:
        raise ValueError(f"Response column '{response_col}' not found in data.")
    reported_response = response_col
    if keyword.iskeyword(response_col) or response_col in {"C", "I", "Q"}:
        # "yield" is the commonest response in chemistry, and a Python keyword, which a
        # formula cannot name; a column named C, I or Q hides patsy's function of that
        # name, so I(x1 ** 2) fails. Fit under a stand-in name and report the real one.
        response_col = _stand_in(response_col, df.columns)
        df = df.rename(columns={reported_response: response_col})

    # Factor columns = everything except the response and the design-bookkeeping
    # columns. A DesignResult carries "RunOrder" (and optionally "Block"); if a
    # caller passes the whole design frame with the response joined, these must
    # not become factors. This mirrors evaluate_design's filtering so the two
    # public consumers agree on what counts as a factor.
    types = [analysis_type] if isinstance(analysis_type, str) else list(analysis_type)
    unknown = [t for t in types if t not in _ANALYSIS_REGISTRY]
    if unknown:
        available = sorted(_ANALYSIS_REGISTRY.keys())
        raise ValueError(f"Unknown analysis_type(s): {unknown}. Available: {available}")
    glm = {"family": family, "link": link, "trials": trials, "dispersion": dispersion, "information": information}
    _check_options(df_method, sigma)
    whole_plot_cols, trials_col = _side_columns(df, whole_plot, types, transform, glm)

    _NON_FACTOR_COLS = {response_col, "RunOrder", BLOCK_COL, WHOLE_PLOT_COL, *whole_plot_cols, trials_col}
    factor_cols = [c for c in df.columns if c not in _NON_FACTOR_COLS]
    validate_identifiers_are_safe(factor_cols)
    for col in factor_cols:
        if keyword.iskeyword(col):
            raise ValueError(
                f"Factor column {col!r} is a Python keyword, which a model formula cannot use; "
                f"rename it, for example to {suggest_identifier(col)!r}."
            )

    requested_model = model
    model, factor_cols = _resolve_model(df, model, response_col, reported_response, factor_cols)
    used = [
        response_col,
        *factor_cols,
        *([BLOCK_COL] if BLOCK_COL in df.columns else []),
        *(whole_plot_cols if "split_plot" in types else []),
        *([trials_col] if trials_col else []),
    ]
    df = _drop_incomplete_runs(df, used)
    _check_response_scale(df[response_col], reported_response)
    coding = _resolve_coding(coding, df, factor_cols, model)

    categorical = [c for c in factor_cols if not _is_numeric(df[c])]
    searched_model = model  # what model_selection searches, even when the default falls back below
    model, model_note, left_out = _estimable_default(
        df,
        requested_model,
        model,
        factor_cols,
        lambda m: build_formula(response_col, factor_cols, m, categorical).split("~", 1)[1],
    )
    formula = _without_terms(build_formula(response_col, factor_cols, model, categorical), left_out)
    # Patsy evaluates formula terms as Python, so a custom ``model`` string is a
    # code-execution vector. Permit only a safe Wilkinson formula, optionally
    # with I()/Q() over data columns (the ``quadratic`` shorthand needs it).
    validate_formula_is_safe(formula, df.columns, allow_transforms=True)
    df, formula, blocks = _add_block_term(df, formula, model)
    _check_term_count(formula)

    raw_response = df[response_col].astype(float)
    rhs = formula.split("~", 1)[1]
    transformed, transform_info = transform_response(
        raw_response.to_numpy(),
        transform,
        exog=lambda: np.asarray(dmatrix(_coded_formula(rhs, df, factor_cols), df), dtype=float),
    )
    df[response_col] = transformed
    ols_result, mixture = _as_mixture(
        smf.ols(_coded_formula(formula, df, factor_cols), data=df).fit(), df, factor_cols, model
    )
    logger.debug("analyze_experiment: fitted %r on %d observations; analyses=%s", formula, len(df), types)

    results: dict[str, Any] = {
        # So the result can be passed to optimize_responses as a fitted model as it is.
        "response_name": reported_response,
        "factor_names": list(factor_cols),
        "model_summary": _model_summary(ols_result, formula, reported_response, model, transform_info)
        | _without_blocks(ols_result, blocks)
        | _estimability(ols_result, _design_rank(df, rhs, factor_cols, model))
        | ({"model_note": model_note} if model_note else {})
        | _fitted_noise(ols_result, sigma),
    }

    fit = _Fit(
        ols=ols_result,
        df=df,
        raw_response=raw_response,
        response_col=response_col,
        factor_cols=factor_cols,
        model=model,
        blocks=blocks,
        transform_info=transform_info,
        alpha=significance_level,
        whole_plot=whole_plot_cols if len(whole_plot_cols) > 1 else whole_plot_cols[0],
        df_method=df_method,
        coding=coding,
        glm=glm,
        searched_model=searched_model,
        response_name=reported_response,
        sigma=sigma,
        mixture=mixture,
    )
    handlers = _handlers(fit, new_points, observed_at_new)
    for t in types:
        results.update(handlers[t]())
    if "model_selection" in results:
        chosen = results["model_selection"]
        chosen["selected_formula"] = chosen["selected_formula"].replace(
            f"{response_col} ~", f"{reported_response} ~", 1
        )
    return results


def _side_columns(
    df: pd.DataFrame,
    whole_plot: str | Sequence[str] | None,
    types: list[str],
    transform: str | None,
    glm: dict[str, Any],
) -> tuple[tuple[str, ...], str | None]:
    """Check the transform and a generalized linear model's options; return the whole-plot and trials columns."""
    validate_transform(transform)
    if "split_plot" in types and glm["information"] not in INFORMATIONS:
        raise ValueError(
            f"information must be one of {', '.join(map(repr, INFORMATIONS))}; got {glm['information']!r}."
        )
    return _whole_plot_columns(df, whole_plot, types), _trials_column(df, glm, types, transform)


def _fitted_noise(ols_result: RegressionResultsWrapper, sigma: float | None) -> dict[str, Any]:
    """Return the fitted values' average variance, and with a prior ``sigma`` adequate precision read against it.

    Both are worked from the residual mean square unless ``sigma`` is given (#797 R1);
    ``adequate_precision`` then replaces the one ``_model_summary`` worked from the residual.
    """
    if sigma is None:
        return {"average_fitted_variance": average_fitted_variance(ols_result)}
    variance = float(sigma) ** 2
    return {
        "prior_sigma": float(sigma),
        "adequate_precision": _compute_adequate_precision(ols_result, variance),
        "average_fitted_variance": average_fitted_variance(ols_result, variance),
    }


def _check_options(df_method: str, sigma: float | None) -> None:
    """Raise ``ValueError`` for a degrees-of-freedom method the split-plot analysis does not offer, or a bad sigma."""
    if df_method not in DF_METHODS:
        raise ValueError(f"df_method must be one of {', '.join(map(repr, DF_METHODS))}; got {df_method!r}.")
    if sigma is not None and not (
        isinstance(sigma, numbers.Real) and not isinstance(sigma, bool) and np.isfinite(sigma) and sigma > 0
    ):
        raise ValueError(
            f"sigma must be a positive number, the error standard deviation known before these runs; got {sigma!r}."
        )


def _trials_column(df: pd.DataFrame, glm: dict[str, Any], types: list[str], transform: str | None) -> str | None:
    """Check a generalized linear model's options; return the column holding the trials, if one does."""
    if "glm" not in types:
        return None
    family, trials = glm["family"], glm["trials"]
    check_glm_options(family, glm["link"], glm["dispersion"], glm["information"])
    if transform is not None:
        raise ValueError(
            "A generalized linear model relates the mean to the factors through its link; "
            "fit the response as it is, without a transform."
        )
    if trials is not None and family != "binomial":
        raise ValueError(f"trials are for the binomial family, not {family!r}.")
    if isinstance(trials, str):
        validate_identifier_is_safe(trials)
        if trials not in df.columns:
            raise ValueError(f"The trials column {trials!r} is not in the data.")
        return trials
    return None


def _design_rank(df: pd.DataFrame, rhs: str, factor_cols: list[str], model: str) -> int:
    """Return how many terms the design can estimate: the model matrix's rank with numeric factors coded.

    The rank of the model matrix in the factors' own units can come out lower than the
    design's in floating point when the units are very unequal (pascals beside kelvin)
    or a span is tiny next to its distance from zero. Coding each numeric factor to
    -1/+1 removes both without changing which terms the runs can separate, as long as
    the coding re-expresses the model (see :func:`_coding_re_expresses`). A model that is
    not hierarchical can span other columns once coded: ``A:B`` without ``A`` and ``B``,
    which coded varies where ``A * B`` is the same in every run, or a factor nested in
    another through the other's 0/1 indicators, ``I1:B + I2:B``, which coded are
    ``(2*I1 - 1)*B`` and its negative. Its rank is then its own, of its columns as given
    scaled to unit length, which no choice of units changes. A mixture model's components
    are proportions and are not coded.
    """
    coded = np.asarray(_coded_model_matrix(df, rhs, factor_cols, model), dtype=float)
    rank = int(np.linalg.matrix_rank(coded))
    if model in SCHEFFE_MODELS or _coding_re_expresses(df, rhs, factor_cols, coded):
        return rank
    return int(np.linalg.matrix_rank(_unit_columns(np.asarray(dmatrix(rhs, df, return_type="dataframe"), dtype=float))))


#: The golden ratio, whose multiples give each factor a different second coding (see :func:`_coding_re_expresses`).
_GOLDEN_RATIO = (1 + 5**0.5) / 2


def _coding_re_expresses(df: pd.DataFrame, rhs: str, factor_cols: list[str], coded: np.ndarray) -> bool:
    """Whether coding the numeric factors re-expresses the model, rather than changing it.

    A hierarchical model spans the same columns whatever each factor's origin and scale:
    ``(A - a)*(B - b)`` lies in the span of 1, ``A``, ``B`` and ``A:B``. Another model's span
    moves with them, and the factors' own units are one more origin and scale. Comparing the
    coded columns with those as given cannot tell the two apart, since it is the columns as
    given that may have lost rank in floating point. So the model matrix ``coded`` is compared
    with a second coding instead, as well conditioned as the first: the i-th numeric factor's
    -1/+1 coding ``x`` becomes ``(1 + g) * (x + g)``, with ``g`` the fractional part of ``i``
    times the golden ratio. A span that moves keeps to this coding only if the runs line up
    with it by chance. When the coded columns already span every run, no coding can add to
    them, the comparison cannot tell, and the model is taken as changed.
    """
    rank = int(np.linalg.matrix_rank(coded))
    if rank == len(coded):
        # shortcut: a saturated design is then ranked as given, which a span tiny next to its distance from zero can
        # lower; compare the codings on more points than the runs if such a design is met.
        return False
    second = df.copy()
    numeric = [col for col in factor_cols if _is_numeric(df[col])]
    for i, col in enumerate(numeric, start=1):
        g = i * _GOLDEN_RATIO % 1
        second[col] = (1 + g) * (_code_factor(df[col])[0] + g)
    moved = _unit_columns(
        np.asarray(dmatrix(_coded_formula(rhs, second, factor_cols), second, return_type="dataframe"), dtype=float)
    )
    both = np.hstack([_unit_columns(coded), moved])
    return int(np.linalg.matrix_rank(moved)) == rank == int(np.linalg.matrix_rank(both))


def _unit_columns(matrix: np.ndarray) -> np.ndarray:
    """Return a matrix with each column scaled to unit length, so its rank does not depend on the units."""
    norms = np.linalg.norm(matrix, axis=0)
    return matrix / np.where(norms > 0, norms, 1.0)


def _coded_model_matrix(df: pd.DataFrame, rhs: str, factor_cols: list[str], model: str) -> pd.DataFrame:
    """Return the model matrix of ``rhs`` with each numeric factor coded to -1/+1 (a mixture's left as is)."""
    coded = df.copy()
    if model not in SCHEFFE_MODELS:
        for col in factor_cols:
            if _is_numeric(df[col]):
                coded[col] = _code_factor(df[col])[0]
    return dmatrix(_coded_formula(rhs, coded, factor_cols), coded, return_type="dataframe")


def _estimability(ols_result: RegressionResultsWrapper, design_rank: int) -> dict[str, Any]:
    """Return how many of the model's terms the design can estimate, warning when it cannot, or the units cannot."""
    # A rank-deficient model matrix is still "fitted" by the pseudo-inverse, and a
    # coefficient is reported for every requested term, but only `model_rank` of them
    # are determined by the data: the rest are one arbitrary solution out of infinitely
    # many. Economical designs that carry structured aliasing (definitive screening,
    # OMARS and other foldovers) land here routinely, so say so rather than letting the
    # caller read confident-looking output. The rank is the design's, from the coded
    # factors; when the factors' own units lose rank in floating point as well, that is
    # a separate warning, so as not to blame the design for the units.
    n_terms = int(np.shape(ols_result.model.exog)[1])
    model_rank = design_rank
    fit_rank = int(getattr(ols_result.model, "rank", np.linalg.matrix_rank(ols_result.model.exog)))
    rank_deficient = model_rank < n_terms
    ill_conditioned = fit_rank < model_rank
    if ill_conditioned:
        message = (
            f"The design can estimate all {model_rank} of its independent terms, but in the factors' units the "
            f"model matrix lost rank in floating point ({fit_rank} of {model_rank}): the units are very unequal, "
            "or a factor's span is tiny next to its distance from zero. The fit, its coefficients, model_summary "
            "and the analyses built on it are numerically unreliable. Code the factors to -1/+1 (or shift and "
            "rescale them) before analysing."
        )
        logger.warning("analyze_experiment: %s", message)
        warnings.warn(message, category=RuntimeWarning, stacklevel=3)
    if rank_deficient:
        message = (
            f"The model matrix has {n_terms} terms but rank {model_rank}, so "
            f"{n_terms - model_rank} of them are not estimable from this design. A "
            "coefficient is still reported for every term, but those are one solution "
            "out of infinitely many: predictions at the design points are unaffected, "
            "while individual coefficients and predictions elsewhere are not "
            "determined. Fit fewer terms, or add runs."
        )
        logger.warning("analyze_experiment: %s", message)
        warnings.warn(message, category=RuntimeWarning, stacklevel=3)
    return {
        "n_terms": n_terms,
        "model_rank": model_rank,
        "rank_deficient": rank_deficient,
        "ill_conditioned": ill_conditioned,
    }


def _model_summary(
    ols_result: RegressionResultsWrapper,
    formula: str,
    reported_response: str,
    model: str,
    transform_info: dict[str, Any],
) -> dict[str, Any]:
    """Fit statistics of the fitted model."""
    return {
        # Report the formula under the caller's response name, not the fitted alias.
        "formula": f"{reported_response} ~{formula.split('~', 1)[1]}",
        "model": model if model in _NAMED_MODELS else "formula",
        **transform_info,
        "r_squared": float(ols_result.rsquared),
        "r_squared_adj": float(ols_result.rsquared_adj),
        "r_squared_pred": _compute_pred_r_squared(ols_result),
        "adequate_precision": _compute_adequate_precision(ols_result),
        "n_obs": int(ols_result.nobs),
        "df_model": int(ols_result.df_model),
        "df_residual": int(ols_result.df_resid),
        "mse_residual": float(ols_result.mse_resid),
        **_sums_of_squares(ols_result),
        **_fit_criteria(ols_result),
    }


def _noted(anova: dict[str, Any], note: str | None) -> dict[str, Any]:
    """Return an ANOVA result with ``note`` added to its ``anova_note``, when there is a note and a table."""
    if note is None or not anova.get("anova_table"):
        return anova
    return anova | {"anova_note": f"{anova['anova_note']} {note}" if anova.get("anova_note") else note}


def _fit_criteria(ols_result: RegressionResultsWrapper) -> dict[str, float | None]:
    """Return the overall F test of the model and its likelihood criteria, as R's summary, AIC and step give them.

    The F test compares the model with the intercept alone (with no intercept, with zero, on the
    uncorrected sums of squares, as R does). ``aic`` and ``bic`` count the error variance among the
    parameters, -2 ln L + 2(p + 1), as R's ``AIC`` and the library's model selection do;
    ``aic_step`` is n ln(RSS/n) + 2p, which R's ``step`` and ``extractAIC`` print. The two
    AICs differ by a constant for a given number of runs, so they rank models alike. ``aicc``
    corrects ``aic`` for a small number of runs, AIC + 2k(k + 1)/(n - k - 1) with k = p + 1, as JMP
    and the library's model selection compute it; it is ``None`` when n <= p + 2, where the
    correction is undefined.
    """
    n, p = float(ols_result.nobs), float(ols_result.df_model + ols_result.k_constant)
    rss = max(float(ols_result.ssr), 1e-300)
    log_likelihood = -n / 2 * (np.log(2 * np.pi * rss / n) + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_statistic, f_p_value = float(ols_result.fvalue), float(ols_result.f_pvalue)
    aic, k = -2 * log_likelihood + 2 * (p + 1), p + 1  # k counts the error variance among the parameters
    return {
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
        "log_likelihood": float(log_likelihood),
        "aic": float(aic),
        "aicc": float(aic + 2 * k * (k + 1) / (n - k - 1)) if n - k - 1 > 0 else None,
        "bic": float(-2 * log_likelihood + np.log(n) * (p + 1)),
        "aic_step": float(n * np.log(rss / n) + 2 * p),
    }
