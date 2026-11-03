# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Curve responses in a designed experiment: FPCA of the curves, a model of each score, bands on predicted curves.

Each run's response is a curve (a time course, a profile, a spectrum). :class:`~process_improve.multivariate.FPCA`
reduces the curves to a few scores, and each score is modelled against the design factors with
:func:`~process_improve.experiments.analyze_experiment`, so coding, mixture models and split-plot whole plots
behave as they do for a scalar response. Because the eigenfunctions rebuild the curve, a prediction of the
scores is a prediction of the whole curve. See ``docs/development/functional_doe.rst``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from patsy import DesignInfo, PatsyError, build_design_matrices, dmatrix
from scipy import stats
from sklearn.base import clone

from process_improve._random import check_random_state
from process_improve.experiments.analysis import analyze_experiment
from process_improve.multivariate._fpca import FPCA  # the module, not the package: multivariate imports experiments

#: Components are kept up to this fraction of the curves' variance.
VARIANCE_KEPT = 0.99
#: Later components, up to this fraction, are kept as well when the design factors explain their scores. A factor
#: can act on a shape that carries little of the variance; a rule on the variance alone would drop it.
VARIANCE_TESTED = 0.999


@dataclass(frozen=True)
class _Projection:
    """The model matrix of the runs, and what it takes to test a score against it."""

    hat: np.ndarray
    df_model: int
    df_residual: int


def _projection(matrix: np.ndarray) -> _Projection:
    """Return the hat matrix of a model matrix, and the degrees of freedom of the F-test of a score on it.

    The scores are centred, so the test is against the mean when the model can fit a constant (an intercept,
    or mixture components that sum to one), and against zero when it cannot.
    """
    rank = int(np.linalg.matrix_rank(matrix))
    hat = matrix @ np.linalg.pinv(matrix)
    ones = np.ones(matrix.shape[0])
    has_constant = np.allclose(hat @ ones, ones)
    return _Projection(hat=hat, df_model=rank - int(has_constant), df_residual=matrix.shape[0] - rank)


def _component_table(model: FPCA, projection: _Projection, significance_level: float) -> pd.DataFrame:
    """Return each component's share of the variance, its F-test, and whether it is kept, up to ``VARIANCE_TESTED``."""
    ratio = model.explained_variance_ratio_.to_numpy()
    cumulative = np.cumsum(ratio)
    by_variance = int(np.searchsorted(cumulative, VARIANCE_KEPT - 1e-12) + 1)
    last = max(by_variance, int(np.searchsorted(cumulative, VARIANCE_TESTED - 1e-12) + 1))
    last = min(last, len(ratio))
    scores = model.scores_.to_numpy()[:, :last]
    explained = ((projection.hat @ scores) ** 2).sum(axis=0)
    unexplained = ((scores - projection.hat @ scores) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_statistic = (explained / projection.df_model) / (unexplained / projection.df_residual)
    p_value = stats.f.sf(f_statistic, projection.df_model, projection.df_residual)
    order = np.arange(last)
    kept = (order < by_variance) | (p_value < significance_level)
    return pd.DataFrame(
        {
            "variance_ratio": ratio[:last],
            "cumulative": cumulative[:last],
            "f_statistic": f_statistic,
            "p_value": p_value,
            "kept": kept,
            "reason": np.where(order < by_variance, "variance", np.where(kept, "factor test", "")),
        },
        index=model.explained_variance_ratio_.index[:last],
    )


def _kept_scores(model: FPCA, kept: np.ndarray) -> np.ndarray:
    """Return the scores with the components not kept set to zero, so the curves they rebuild leave those out."""
    scores = model.scores_.to_numpy().copy()
    mask = np.zeros(scores.shape[1], dtype=bool)
    mask[: kept.size] = kept
    scores[:, ~mask] = 0.0
    return scores


@dataclass(frozen=True)
class FunctionalFit:
    """
    Curves from a designed experiment, fitted by :func:`analyze_functional_experiment`.

    Attributes
    ----------
    fpca : FPCA
        The functional PCA of the curves, with every component the curves support.
    components : pandas.DataFrame
        One row per component up to 99.9% of the variance: ``variance_ratio``, ``cumulative``, the F-test of its
        score on the design factors (``f_statistic``, ``p_value``), ``kept``, and the ``reason`` it is kept:
        ``"variance"`` up to 99% of the variance, ``"factor test"`` after that.
    scores : dict[str, dict]
        For each kept component, the result of :func:`~process_improve.experiments.analyze_experiment` on its
        score: the ANOVA, the coefficients, the lack of fit, and the split-plot analysis when the design has a
        ``WholePlot`` column.
    residual_covariance : pandas.DataFrame
        The covariance of the kept scores' residuals. The scores are uncorrelated across runs, but their
        residuals after the model is fitted are not.
    formula : str
        The right-hand side of the model fitted to each score.
    """

    fpca: FPCA
    components: pd.DataFrame
    scores: dict[str, dict[str, Any]]
    residual_covariance: pd.DataFrame
    formula: str
    _design: DesignInfo = field(repr=False)
    _model_matrix: np.ndarray = field(repr=False)
    _curves: pd.DataFrame = field(repr=False)
    _groups: np.ndarray = field(repr=False)
    _significance_level: float = field(repr=False)

    def predict(
        self,
        new_points: pd.DataFrame,
        *,
        n_bootstrap: int = 400,
        level: float = 0.95,
        random_state: int | np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Return the predicted curve at each new point, with pointwise and simultaneous bands.

        The bands come from a wild bootstrap. Each run's whole residual curve, scaled by 1/sqrt(1 - h) for the
        run's leverage h, is multiplied by +1 or -1 at random and added back to its fitted curve; the FPCA, the
        choice of components and the models of the scores are then fitted again, and the new points predicted.
        The design is kept fixed. With a ``WholePlot`` column the sign is drawn once per whole plot, so runs
        that share a whole plot keep their shared error. A band that treated the eigenfunctions as known
        would leave out their uncertainty, and cover far less often than it claims.

        The bootstrap measures variance, not bias. The smoothing still pulls each curve a little near the ends
        of the grid, so the bands cover somewhat less often than ``level`` there: in a simulation of an 18-run
        design with a correct model, 95% bands covered the true curve 87% of the time pointwise and 88% of
        the time over the whole curve.

        Parameters
        ----------
        new_points : pandas.DataFrame
            Factor settings, one row per curve to predict, with the factors the model uses.
        n_bootstrap : int, default=400
            Number of bootstrap replicates. The simultaneous band rests on a high quantile of a maximum,
            which a few hundred replicates are needed to estimate.
        level : float, default=0.95
            Coverage of the bands.
        random_state : int, numpy.random.Generator, or None, default=None
            Seeds the bootstrap, per the reproducibility contract.

        Returns
        -------
        dict
            ``curve``, ``standard_error``, ``pointwise_low``, ``pointwise_high``, ``simultaneous_low`` and
            ``simultaneous_high``: DataFrames with a row per new point and a column per grid point.
            ``critical_value``: each new point's multiple of the standard error that the simultaneous band
            spans. ``level`` and ``n_bootstrap``, as used.
        """
        if int(n_bootstrap) < 2:
            raise ValueError(f"n_bootstrap must be at least 2; got {n_bootstrap}.")
        if not 0 < level < 1:
            raise ValueError(f"level must lie between 0 and 1; got {level}.")
        rng = check_random_state(random_state)
        try:
            (new_matrix,) = build_design_matrices([self._design], new_points, return_type="matrix")
        except PatsyError as error:
            raise ValueError(f"new_points cannot be read by the model {self.formula!r}: {error}") from None
        projection = _projection(self._model_matrix)
        to_new = np.asarray(new_matrix) @ np.linalg.pinv(self._model_matrix)
        kept = self.components["kept"].to_numpy()
        estimate = self._curves_at(self.fpca, kept, to_new)
        fitted = self._curves_at(self.fpca, kept, projection.hat)
        leverage = np.clip(np.diag(projection.hat), 0.0, 1.0)
        # OLS residuals are smaller than the errors by sqrt(1 - h); a run fitted exactly (h = 1) has none to resample.
        scale = np.divide(1.0, np.sqrt(1.0 - leverage), out=np.zeros_like(leverage), where=leverage < 1.0)
        residuals = (self._curves.to_numpy() - fitted) * scale[:, None]
        replicates = np.stack(
            [self._replicate(fitted, residuals, projection, to_new, rng) for _ in range(int(n_bootstrap))]
        )
        return _bands(estimate, replicates, projection.df_residual, level, (new_points.index, self.fpca.grid_))

    def _curves_at(self, model: FPCA, kept: np.ndarray, to_points: np.ndarray) -> np.ndarray:
        """Return the curves the fitted scores predict, through a linear map from the runs to the points."""
        return model.inverse_transform(to_points @ _kept_scores(model, kept), grid=self.fpca.grid_).to_numpy()

    def _replicate(
        self,
        fitted: np.ndarray,
        residuals: np.ndarray,
        projection: _Projection,
        to_new: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return the new points' curves predicted from one wild-bootstrap replicate of the runs."""
        signs = rng.choice([-1.0, 1.0], size=int(self._groups.max()) + 1)[self._groups]
        resampled = pd.DataFrame(fitted + signs[:, None] * residuals, columns=self._curves.columns)
        # shortcut: the smoothing penalty is not chosen again, so the bands leave out the uncertainty of GCV's
        # choice; choose it per replicate if that ever matters.
        model = clone(self.fpca).set_params(n_components=None, smoothing=self.fpca.smoothing_).fit(resampled)
        table = _component_table(model, projection, self._significance_level)
        return self._curves_at(model, table["kept"].to_numpy(), to_new)


def _bands(
    estimate: np.ndarray, replicates: np.ndarray, df_residual: int, level: float, labels: tuple[pd.Index, np.ndarray]
) -> dict[str, Any]:
    """Return the bands from the bootstrap replicates of the predicted curves, labelled by new point and grid point."""
    error = replicates.std(axis=0, ddof=1)
    quantile = stats.t.ppf(0.5 + level / 2, max(df_residual, 1))
    # The simultaneous band's multiple is the level-quantile of each replicate's largest standardised departure.
    standardised = np.divide(np.abs(replicates - estimate), error, out=np.zeros_like(replicates), where=error > 0)
    # shortcut: the bootstrap's maximum is a large-sample quantity; it is widened by the same ratio of the t to the
    # normal quantile that the pointwise band carries, for the few residual degrees of freedom of a designed
    # experiment. A studentised bootstrap would do this exactly, at the cost of a bootstrap in each replicate.
    widening = quantile / stats.norm.ppf(0.5 + level / 2)
    critical = widening * np.quantile(standardised.max(axis=2), level, axis=0)
    frames = {
        "curve": estimate,
        "standard_error": error,
        "pointwise_low": estimate - quantile * error,
        "pointwise_high": estimate + quantile * error,
        "simultaneous_low": estimate - critical[:, None] * error,
        "simultaneous_high": estimate + critical[:, None] * error,
    }
    index, grid = labels
    out: dict[str, Any] = {key: pd.DataFrame(value, index=index, columns=grid) for key, value in frames.items()}
    out["critical_value"] = pd.Series(critical, index=index, name="critical_value")
    out["level"], out["n_bootstrap"] = level, replicates.shape[0]
    return out


def analyze_functional_experiment(
    design_matrix: pd.DataFrame,
    curves: pd.DataFrame,
    model: str | None = None,
    *,
    fpca: FPCA | None = None,
    significance_level: float = 0.05,
) -> FunctionalFit:
    """Model curves measured in a designed experiment: FPCA of the curves, then a model of each score.

    The curves are reduced by functional PCA. The components are kept up to 99% of the curves' variance, and
    any later component up to 99.9% is kept too when the design factors explain its score (an F-test at
    ``significance_level``): a factor can act on a shape that carries little of the variance. Each kept
    score is then analysed with :func:`~process_improve.experiments.analyze_experiment`, under the same
    ``model``. :meth:`FunctionalFit.predict` predicts whole curves at new factor settings, with bands.

    Parameters
    ----------
    design_matrix : pandas.DataFrame
        Factor settings, one row per run. A ``WholePlot`` column marks the whole plots of a split-plot design:
        each score then also gets the split-plot analysis, and the bootstrap of the bands draws one sign per
        whole plot.
    curves : pandas.DataFrame
        One curve per run, with the same index as ``design_matrix``, and one column per grid point (see
        :class:`~process_improve.multivariate.FPCA`); ``NaN`` where a run was not measured.
    model : str or None, default=None
        The model of each score, as :func:`~process_improve.experiments.analyze_experiment` takes it: a named
        model or a right-hand side.
    fpca : FPCA or None, default=None
        The functional PCA to use, unfitted, for its smoothing settings; its ``n_components`` is ignored, as
        the components are chosen here. ``FPCA()`` when not given.
    significance_level : float, default=0.05
        Level of the F-test that keeps a later component, and of the score analyses.

    Returns
    -------
    FunctionalFit
        The FPCA, the components kept and why, each kept score's analysis, and the residual covariance.

    Raises
    ------
    ValueError
        If the design and the curves are not the same runs, if the model has no residual degrees of freedom
        left, or if the model's terms cannot be read from the design.
    """
    if not curves.index.equals(design_matrix.index):
        raise ValueError("design_matrix and curves must be the same runs, with the same index in the same order.")
    fitted = clone(fpca if fpca is not None else FPCA()).set_params(n_components=None).fit(curves)
    first = fitted.scores_.iloc[:, 0].rename("fpc_1")
    probe = analyze_experiment(design_matrix, first, model=model, analysis_type="coefficients")
    formula = probe["model_summary"]["formula"].split("~", 1)[1].strip()
    try:
        matrix = dmatrix(formula, design_matrix, return_type="matrix")
    except PatsyError as error:
        raise ValueError(f"The model {formula!r} cannot be built from the design's columns: {error}") from None
    projection = _projection(np.asarray(matrix))
    if projection.df_residual < 1:
        raise ValueError(
            f"The model has {projection.df_model + 1} estimable terms but there are {len(curves)} runs, so no "
            "residual degrees of freedom are left to test or bound the scores. Fit a smaller model, or add runs."
        )
    table = _component_table(fitted, projection, significance_level)
    return FunctionalFit(
        fpca=fitted,
        components=table,
        scores=_score_analyses(design_matrix, fitted, table, model, significance_level),
        residual_covariance=_residual_covariance(fitted, table, projection),
        formula=formula,
        _design=matrix.design_info,
        _model_matrix=np.asarray(matrix),
        _curves=curves,
        _groups=_bootstrap_groups(design_matrix),
        _significance_level=significance_level,
    )


def _score_analyses(
    design_matrix: pd.DataFrame, fitted: FPCA, table: pd.DataFrame, model: str | None, significance_level: float
) -> dict[str, dict[str, Any]]:
    """Return each kept score's analysis, under the score's component name."""
    analyses = ["anova", "coefficients", "lack_of_fit"] + (["split_plot"] if "WholePlot" in design_matrix else [])
    out = {}
    for position, name in enumerate(table.index):
        if table["kept"].iloc[position]:
            score = fitted.scores_.iloc[:, position].rename(f"fpc_{position + 1}")
            out[name] = analyze_experiment(
                design_matrix, score, model=model, analysis_type=analyses, significance_level=significance_level
            )
    return out


def _residual_covariance(fitted: FPCA, table: pd.DataFrame, projection: _Projection) -> pd.DataFrame:
    """Return the covariance of the kept scores' residuals after the model."""
    names = table.index[table["kept"].to_numpy()]
    scores = fitted.scores_[names].to_numpy()
    residuals = scores - projection.hat @ scores
    return pd.DataFrame(residuals.T @ residuals / projection.df_residual, index=names, columns=names)


def _bootstrap_groups(design_matrix: pd.DataFrame) -> np.ndarray:
    """Return the groups the bootstrap draws one sign for: the whole plots, or else each run alone."""
    if "WholePlot" in design_matrix:
        return pd.factorize(design_matrix["WholePlot"])[0]
    return np.arange(len(design_matrix))
