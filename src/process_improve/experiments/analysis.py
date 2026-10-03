# (c) Kevin Dunn, 2010-2026. MIT License.

"""Experiment analysis: fit models, ANOVA, diagnostics, residuals.

Provides :func:`analyze_experiment`, the main analytical workhorse for
designed experiments (Tool 3 in the DOE tool architecture).

Uses statsmodels and scipy for the heavy lifting, with thin custom code
for lack-of-fit, curvature test, Lenth's method, pred-R², adequate
precision, and confirmation run testing.
"""

from __future__ import annotations

import keyword
import logging
import re
import warnings
from collections.abc import Callable, Collection
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
    _compute_adequate_precision,
    _compute_pred_r_squared,
)
from process_improve.experiments._analyses.aliasing import chain_reduced_fit
from process_improve.experiments._analyses.box_cox import _run_box_cox
from process_improve.experiments._analyses.curvature import _run_curvature_test
from process_improve.experiments._analyses.diagnostics import _run_residual_diagnostics
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
    _run_anova,
    _run_coefficients,
    _run_confidence_intervals,
    _run_effects,
    _run_significance,
)
from process_improve.experiments._analyses.prediction import _run_confirmation_test, _run_prediction
from process_improve.experiments._analyses.transforms import transform_response, validate_transform
from process_improve.experiments.designs_mixture_constrained import SCHEFFE_MODELS, scheffe_formula_rhs
from process_improve.experiments.models import validate_formula_is_safe, validate_identifier_is_safe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formula builder
# ---------------------------------------------------------------------------

#: Model names :func:`build_formula` expands; anything else is a formula or a raw RHS.
_NAMED_MODELS = ("main_effects", "interactions", "quadratic", *SCHEFFE_MODELS)


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
        ``"scheffe_special_cubic"``), or an explicit formula string.  *None*
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
        # Mixture components sum to 1, so the intercept is dropped; statsmodels detects
        # the implicit constant, which keeps R-squared and the model df centred.
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
    """Whether every run's factor settings are proportions that sum to 1, as in a mixture design."""
    if len(factor_cols) < 2 or not all(_is_numeric(df[c]) for c in factor_cols):
        return False
    values = df[factor_cols].to_numpy(dtype=float)
    finite = values[np.isfinite(values).all(axis=1)]
    if finite.size == 0:
        return False
    in_unit_interval = bool(np.all((finite >= -1e-9) & (finite <= 1 + 1e-9)))
    return in_unit_interval and bool(np.allclose(finite.sum(axis=1), 1.0, atol=1e-6))


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


def _drop_incomplete_runs(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop runs with a missing response or factor setting, warning how many went."""
    complete = df[columns].notna().all(axis=1)
    n_dropped = int((~complete).sum())
    if n_dropped:
        warnings.warn(
            f"{n_dropped} run(s) with a missing response or factor setting were left out of the analysis.",
            category=UserWarning,
            stacklevel=3,
        )
        return df.loc[complete].copy()
    return df


def _add_block_term(df: pd.DataFrame, formula: str, model: str) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    """Add the blocks to the model as a fixed effect when the design has more than one block.

    Block-to-block differences left out of the model go into the residual and mask real
    effects (Montgomery, *Design and Analysis of Experiments*, chapter 7). The blocks
    enter as sum-coded contrast columns, ``Block1`` to ``Block{b-1}`` for ``b`` blocks
    (in sorted order, the last block coded -1 in every column). A Scheffe mixture model
    and an explicit formula are not changed; a warning says the blocks are not modelled.

    Returns
    -------
    tuple[pandas.DataFrame, str, tuple[str, ...]]
        The data with the contrast columns added, the formula, and the contrast names.
    """
    if BLOCK_COL not in df.columns or df[BLOCK_COL].nunique() < 2:
        return df, formula, ()
    if model in SCHEFFE_MODELS or model not in _NAMED_MODELS:
        if not _references(formula.split("~", 1)[1], BLOCK_COL):
            kind = "a Scheffe mixture model" if model in SCHEFFE_MODELS else "an explicit formula"
            warnings.warn(
                f"The design has a {BLOCK_COL!r} column, but blocks are not added to {kind}, so block-to-block "
                "differences stay in the residual. Drop the column to silence this, or name it in the formula.",
                category=UserWarning,
                stacklevel=3,
            )
        return df, formula, ()

    levels = sorted(df[BLOCK_COL].unique(), key=str)
    df = df.copy()
    names: list[str] = []
    for k, level in enumerate(levels[:-1], start=1):
        name = f"{BLOCK_COL}{k}"
        while name in df.columns:
            name += "_"
        df[name] = np.where(df[BLOCK_COL] == level, 1.0, np.where(df[BLOCK_COL] == levels[-1], -1.0, 0.0))
        names.append(name)
    response, rhs = formula.split(" ~ ", 1)
    return df, f"{response} ~ {' + '.join(names)} + {rhs}", tuple(names)


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
    _coded: tuple[RegressionResultsWrapper, dict[str, Any]] | None = None
    _reduced: tuple[RegressionResultsWrapper, dict[str, str]] | None = None

    @property
    def group_cols(self) -> list[str]:
        """Columns identifying replicated runs: the factors, within each block when blocked."""
        return [*self.factor_cols, BLOCK_COL] if self.blocks else self.factor_cols

    def coded(self) -> tuple[RegressionResultsWrapper, dict[str, Any]]:
        """Return the model refitted on factors coded to -1/+1, where an effect is twice a coefficient."""
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

    def reduced(self) -> tuple[RegressionResultsWrapper, dict[str, str]]:
        """Return the fit on one column per alias chain, with the chain name of each retained column."""
        if self._reduced is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._reduced = chain_reduced_fit(self.ols)
        return self._reduced


def _effects(fit: _Fit, *, lenth: bool) -> dict[str, Any]:
    """Effects, or Lenth's method on them, from the -1/+1 coded refit."""
    coded, coding = fit.coded()
    result = _run_lenth_method(coded, fit.alpha, blocks=fit.blocks) if lenth else _run_effects(coded, blocks=fit.blocks)
    if coding:
        result["effects_coding"] = coding
    return result


def _mixture_handlers(fit: _Fit) -> dict[str, Callable[[], dict[str, Any]]]:
    """Analyses that differ for a Scheffe mixture model, keyed by analysis type."""
    return {
        "anova": lambda: run_mixture_anova(fit.ols, fit.factor_cols),
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
        return _run_prediction(fit.ols, points, fit.alpha, fit.transform_info)

    def confirmation() -> dict[str, Any]:
        if points is None or observed_at_new is None:
            return {"confirmation_test": {"error": "new_points and observed_at_new are required."}}
        return _run_confirmation_test(fit.ols, points, observed_at_new, fit.alpha, fit.transform_info)

    def box_cox() -> dict[str, Any]:
        raw = fit.df.assign(**{fit.response_col: fit.raw_response})
        return _run_box_cox(raw, fit.response_col, exog=fit.ols.model.exog, alpha=fit.alpha)

    handlers: dict[str, Callable[[], dict[str, Any]]] = {
        "anova": lambda: _run_anova(fit.reduced()[0], labels=fit.reduced()[1], blocks=fit.blocks),
        "effects": lambda: _effects(fit, lenth=False),
        "coefficients": lambda: _run_coefficients(fit.ols),
        "significance": lambda: _run_significance(
            fit.reduced()[0], fit.alpha, labels=fit.reduced()[1], blocks=fit.blocks
        ),
        "residual_diagnostics": lambda: _run_residual_diagnostics(fit.ols),
        "lack_of_fit": lambda: _run_lack_of_fit(fit.ols, fit.df, fit.response_col, fit.group_cols, fit.alpha),
        "curvature_test": lambda: _run_curvature_test(
            fit.df, fit.response_col, fit.factor_cols, fit.alpha, group_cols=fit.group_cols
        ),
        "model_selection": lambda: _run_model_selection(fit.df, fit.response_col, fit.factor_cols, model=fit.model),
        "box_cox": box_cox,
        "lenth_method": lambda: _effects(fit, lenth=True),
        "confidence_intervals": lambda: _run_confidence_intervals(fit.ols, fit.alpha),
        "prediction": prediction,
        "confirmation_test": confirmation,
    }
    if fit.model in SCHEFFE_MODELS:
        # The usual ANOVA, significance and effects test each blending coefficient
        # against zero, which means nothing for a mixture.
        handlers.update(_mixture_handlers(fit))
    return handlers


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
    coding: str = "coded",
    new_points: pd.DataFrame | None = None,
    observed_at_new: list[float] | None = None,
    response_column: str | None = None,
) -> dict[str, Any]:
    """Fit models, run ANOVA, compute effects, diagnose residuals.

    Parameters
    ----------
    design_matrix : DataFrame
        Factor settings per run.  May also contain the response column(s). A
        ``RunOrder`` column is ignored. A ``Block`` column with two or more blocks
        is added to the named non-mixture models as a fixed effect (sum-coded contrast
        columns ``Block1``, ...; one ``"Block"`` row in the ANOVA), so block-to-block
        differences do not inflate the error; drop the column to analyse without it.
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
        ``"scheffe_special_cubic"``, fitted without an intercept since the
        components sum to 1), an explicit formula whose left-hand side is the
        response, or *None*: ``"scheffe_quadratic"`` when every run's factor
        settings are proportions summing to 1, and ``"interactions"`` otherwise.
        The model may expand to at most ``settings.max_formula_terms`` terms.
    analysis_type : str or list[str]
        One or more of: ``"anova"``, ``"effects"``, ``"coefficients"``,
        ``"significance"``, ``"residual_diagnostics"``, ``"lack_of_fit"``,
        ``"curvature_test"``, ``"model_selection"``, ``"box_cox"``,
        ``"lenth_method"``, ``"confidence_intervals"``, ``"prediction"``,
        ``"confirmation_test"``.

        The ANOVA uses Type II sums of squares (reported as ``anova_type``). The
        ANOVA and significance list exactly aliased terms once, as their alias chain
        (``"A:B + C:D"``), as the effects do; the coefficients keep one entry per term. Effects and Lenth's
        method are twice the coefficients of factors coded to -1/+1: a numeric
        factor not already coded is mapped from its own minimum and maximum, and a
        two-level categorical factor from its first level to its second (sorted);
        the mapping is reported under ``effects_coding``. The curvature test uses
        pure error from the replicated runs.
    significance_level : float
        Default 0.05. Used by every test's ``significant`` flag and every interval.
    transform : str or None
        ``"log"``, ``"sqrt"``, ``"inverse"``, ``"box_cox"``, or ``None``. The model is
        fitted on the transformed response, so predictions and intervals are on that
        scale (reported as ``prediction_scale`` / ``scale``), and confirmation runs are
        transformed the same way before they are compared. Box-Cox chooses ``lambda``
        by the profile likelihood of the model. A response outside the transform's
        domain raises ``ValueError``.
    coding : str, default ``"coded"``
        Reserved for a future coded/actual factor-scale switch. Currently
        accepted for API stability but not consumed by the analysis.
    new_points : DataFrame or None
        For prediction or confirmation testing. In a blocked model, points without
        the block contrast columns are predicted for the average block.
    observed_at_new : list[float] or None
        Observed values at *new_points* (for confirmation testing), one per row.
    response_column : str or None
        Name of the response column when it lives inside *design_matrix*.

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
          considered adequate).
        - ``n_obs`` - number of observations used to fit.
        - ``n_terms`` - number of columns in the model matrix.
        - ``model_rank`` - numerical rank of the model matrix; less
          than ``n_terms`` implies aliasing / rank deficiency.
        - ``rank_deficient`` - ``True`` if ``model_rank < n_terms``.
        - ``df_model`` - model degrees of freedom.
        - ``df_residual`` - residual degrees of freedom.
        - ``mse_residual`` - mean squared error of the residuals.

        Notes from individual analyses are kept apart, as ``anova_note``,
        ``effects_note``, ``lenth_note`` and ``significance_note``.

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
    """
    df, response_col = _assemble_data(design_matrix, responses, response_column)

    # User-supplied names (response_column / design_matrix dict keys) are
    # interpolated into the patsy formula, so reject anything that is not a
    # plain identifier before it can become an injection vector (SEC-14).
    validate_identifier_is_safe(response_col)

    if response_col not in df.columns:
        raise ValueError(f"Response column '{response_col}' not found in data.")
    reported_response = response_col
    if keyword.iskeyword(response_col):
        # "yield" is the commonest response in chemistry, and a Python keyword, which a
        # formula cannot name. Fit under a stand-in name and report the real one.
        alias = f"{response_col}_"
        while alias in df.columns:
            alias += "_"
        df = df.rename(columns={response_col: alias})
        response_col = alias

    # Factor columns = everything except the response and the design-bookkeeping
    # columns. A DesignResult carries "RunOrder" (and optionally "Block"); if a
    # caller passes the whole design frame with the response joined, these must
    # not become factors. This mirrors evaluate_design's filtering so the two
    # public consumers agree on what counts as a factor.
    _NON_FACTOR_COLS = {response_col, "RunOrder", BLOCK_COL}
    factor_cols = [c for c in df.columns if c not in _NON_FACTOR_COLS]
    for col in factor_cols:
        validate_identifier_is_safe(col)
        if keyword.iskeyword(col):
            raise ValueError(f"Factor column {col!r} is a Python keyword, which a model formula cannot use; rename it.")

    types = [analysis_type] if isinstance(analysis_type, str) else list(analysis_type)
    unknown = [t for t in types if t not in _ANALYSIS_REGISTRY]
    if unknown:
        available = sorted(_ANALYSIS_REGISTRY.keys())
        raise ValueError(f"Unknown analysis_type(s): {unknown}. Available: {available}")
    validate_transform(transform)

    model, factor_cols = _resolve_model(df, model, response_col, reported_response, factor_cols)
    used = [response_col, *factor_cols, *([BLOCK_COL] if BLOCK_COL in df.columns else [])]
    df = _drop_incomplete_runs(df, used)

    categorical = [c for c in factor_cols if not _is_numeric(df[c])]
    formula = build_formula(response_col, factor_cols, model, categorical)
    # Patsy evaluates formula terms as Python, so a custom ``model`` string is a
    # code-execution vector. Permit only a safe Wilkinson formula, optionally
    # with I()/Q() over data columns (the ``quadratic`` shorthand needs it).
    validate_formula_is_safe(formula, df.columns, allow_transforms=True)
    df, formula, blocks = _add_block_term(df, formula, model)
    _check_term_count(formula)

    raw_response = df[response_col].astype(float)
    rhs = formula.split("~", 1)[1]
    transformed, transform_info = transform_response(
        raw_response.to_numpy(), transform, exog=lambda: np.asarray(dmatrix(rhs, df), dtype=float)
    )
    df[response_col] = transformed
    ols_result = smf.ols(formula, data=df).fit()
    logger.debug("analyze_experiment: fitted %r on %d observations; analyses=%s", formula, len(df), types)

    results: dict[str, Any] = {
        # So the result can be passed to optimize_responses as a fitted model as it is.
        "response_name": reported_response,
        "factor_names": list(factor_cols),
        "model_summary": _model_summary(ols_result, formula, reported_response, model, transform_info),
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


def _model_summary(
    ols_result: RegressionResultsWrapper,
    formula: str,
    reported_response: str,
    model: str,
    transform_info: dict[str, Any],
) -> dict[str, Any]:
    """Fit statistics and estimability of the fitted model, warning when it is rank deficient."""
    # Estimability. A rank-deficient model matrix is still "fitted" by the
    # pseudo-inverse, and a coefficient is reported for every requested term, but only
    # `model_rank` of them are determined by the data: the rest are one arbitrary
    # solution out of infinitely many. Economical designs that carry structured
    # aliasing (definitive screening, OMARS and other foldovers) land here routinely,
    # so say so rather than letting the caller read confident-looking output.
    n_terms = int(np.shape(ols_result.model.exog)[1])
    model_rank = int(getattr(ols_result.model, "rank", np.linalg.matrix_rank(ols_result.model.exog)))
    rank_deficient = model_rank < n_terms
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
        # Report the formula under the caller's response name, not the fitted alias.
        "formula": f"{reported_response} ~{formula.split('~', 1)[1]}",
        "model": model if model in _NAMED_MODELS else "formula",
        **transform_info,
        "r_squared": float(ols_result.rsquared),
        "r_squared_adj": float(ols_result.rsquared_adj),
        "r_squared_pred": _compute_pred_r_squared(ols_result),
        "adequate_precision": _compute_adequate_precision(ols_result),
        "n_obs": int(ols_result.nobs),
        "n_terms": n_terms,
        "model_rank": model_rank,
        "rank_deficient": rank_deficient,
        "df_model": int(ols_result.df_model),
        "df_residual": int(ols_result.df_resid),
        "mse_residual": float(ols_result.mse_resid),
    }
