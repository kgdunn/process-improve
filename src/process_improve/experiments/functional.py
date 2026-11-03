# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Curve responses in a designed experiment: FPCA of the curves, a model of each score, bands on predicted curves.

Each run's response is a curve (a time course, a profile, a spectrum). :class:`~process_improve.multivariate.FPCA`
reduces the curves to a few scores, and each score is modelled against the design factors with
:func:`~process_improve.experiments.analyze_experiment`, so coding, mixture models and split-plot whole plots
behave as they do for a scalar response. Because the eigenfunctions rebuild the curve, a prediction of the
scores is a prediction of the whole curve. See ``docs/development/functional_doe.rst``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from patsy import DesignInfo, PatsyError, build_design_matrices, dmatrix
from scipy import optimize, stats
from scipy.interpolate import BSpline
from sklearn.base import clone

from process_improve._random import check_random_state
from process_improve.experiments.analysis import (
    BLOCK_COL,
    _add_block_term,
    _looks_like_mixture,
    _mixture_components,
    _references,
    analyze_experiment,
)
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


@dataclass(frozen=True)
class _Components:
    """A functional PCA as B-spline coefficients: the mean's, each eigenfunction's, the scores and variance shares."""

    mean_coef: np.ndarray
    eigen_coef: np.ndarray
    scores: np.ndarray
    ratio: np.ndarray


def _components_of(model: FPCA) -> _Components:
    """Return a fitted FPCA's components."""
    ratio = model.explained_variance_ratio_.to_numpy()
    return _Components(model._mean_coef, model._eigen_coef, model.scores_.to_numpy(), ratio)


def _refit(model: FPCA, curves: np.ndarray) -> _Components:
    """Return the components of curves on a fitted FPCA's grid, basis and smoothing: its coefficients, then one SVD.

    A bootstrap replicate keeps all three, so a full refit, which builds the basis again and runs a PCA with
    its diagnostics, does ten times the work. The SVD fixes each component only up to sign, which a curve
    rebuilt from a score times its eigenfunction does not see.
    """
    coefficients = model._coefficients(curves, model.smoothing_)
    mean = coefficients.mean(axis=0)
    left, sizes, right = np.linalg.svd((coefficients - mean) @ model._root_gram, full_matrices=False)
    keep = min(curves.shape[0] - 1, sizes.size)  # centred curves span one dimension fewer than there are curves
    total = float(np.sum(sizes**2))
    ratio = sizes[:keep] ** 2 / total if total > 0 else np.zeros(keep)
    return _Components(mean, model._inverse_root_gram @ right[:keep].T, left[:, :keep] * sizes[:keep], ratio)


def _component_table(components: _Components, projection: _Projection, significance_level: float) -> pd.DataFrame:
    """Return each component's share of the variance, its F-test, and whether it is kept, up to ``VARIANCE_TESTED``."""
    ratio = components.ratio
    cumulative = np.cumsum(ratio)
    by_variance = int(np.searchsorted(cumulative, VARIANCE_KEPT - 1e-12) + 1)
    last = max(by_variance, int(np.searchsorted(cumulative, VARIANCE_TESTED - 1e-12) + 1))
    last = min(last, len(ratio))
    scores = components.scores[:, :last]
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
        index=pd.Index([f"FPC {a + 1}" for a in range(last)]),
    )


def _full_scores(model: FPCA, position: list[int], values: np.ndarray) -> np.ndarray:
    """Return one row of scores on every component, holding ``values`` at the kept ``position`` and zero elsewhere."""
    full = np.zeros((1, model.n_components_))
    full[0, position] = values
    return full


def _kept_scores(scores: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Return the scores with the components not kept set to zero, so the curves they rebuild leave those out."""
    scores = scores.copy()
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
    _blocks: tuple[str, ...] = field(default=(), repr=False)
    _settings: pd.DataFrame | None = field(default=None, repr=False)

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
            Factor settings, one row per curve to predict, with the factors the model uses. When the model has
            blocks, each curve is predicted for the average over the blocks.
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
        if not 0 < level < 1:
            raise ValueError(f"level must lie between 0 and 1; got {level}.")
        if self._blocks and BLOCK_COL in new_points:
            raise ValueError(
                f"new_points has a {BLOCK_COL!r} column, but a curve is predicted for the average over the blocks; "
                "drop the column."
            )
        points = new_points.assign(**{name: 0.0 for name in self._blocks if name not in new_points})
        try:
            (new_matrix,) = build_design_matrices([self._design], points, return_type="matrix")
        except PatsyError as error:
            raise ValueError(f"new_points cannot be read by the model {self.formula!r}: {error}") from None
        rows = np.asarray(new_matrix)
        maps = self.curve_maps(n_bootstrap=n_bootstrap, random_state=random_state)
        estimate = maps["mean"] + rows @ maps["map"]
        replicates = maps["replicate_means"][:, None, :] + np.einsum("np,bpg->bng", rows, maps["replicate_maps"])
        return _bands(estimate, replicates, maps["df_residual"], level, (new_points.index, self.fpca.grid_))

    def curve_maps(
        self,
        *,
        grid: np.ndarray | None = None,
        n_bootstrap: int = 400,
        random_state: int | np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Return the fitted curve, and each wild-bootstrap replicate of it, as an affine map of a model row.

        At settings whose row of the model matrix is ``x``, the predicted curve is ``mean + x @ map`` and
        replicate ``b``'s is ``replicate_means[b] + x @ replicate_maps[b]``. The bootstrap is the one
        :meth:`predict` describes, and :meth:`predict` forms its bands from these maps. A caller that
        predicts at many settings, as a page does while a reader moves them, needs only these numbers and
        the row.

        Parameters
        ----------
        grid : numpy.ndarray or None, default=None
            Points to give the curves at, within the fitted range; the FPCA's grid when not given. A coarser
            grid keeps the maps small for a curve measured at many points, such as a spectrum.
        n_bootstrap : int, default=400
            Number of bootstrap replicates.
        random_state : int, numpy.random.Generator, or None, default=None
            Seeds the bootstrap, per the reproducibility contract.

        Returns
        -------
        dict
            ``terms``, the model matrix's columns; ``mean``, one value per grid point; ``map``, a row per term;
            ``replicate_means`` and ``replicate_maps``, the same for each replicate; and ``df_residual``.
        """
        if int(n_bootstrap) < 2:
            raise ValueError(f"n_bootstrap must be at least 2; got {n_bootstrap}.")
        rng = check_random_state(random_state)
        projection = _projection(self._model_matrix)
        to_coefficients = np.linalg.pinv(self._model_matrix)
        kept = self.components["kept"].to_numpy()
        fitted = self._curves_at(self.fpca, kept, projection.hat)
        leverage = np.clip(np.diag(projection.hat), 0.0, 1.0)
        # OLS residuals are smaller than the errors by sqrt(1 - h); a run fitted exactly (h = 1) has none to resample.
        scale = np.divide(1.0, np.sqrt(1.0 - leverage), out=np.zeros_like(leverage), where=leverage < 1.0)
        residuals = (self._curves.to_numpy() - fitted) * scale[:, None]
        at = self.fpca.grid_ if grid is None else np.asarray(grid, dtype=float)
        if at.min() < self.fpca.grid_[0] or at.max() > self.fpca.grid_[-1]:
            raise ValueError(f"grid must lie within the fitted range [{self.fpca.grid_[0]}, {self.fpca.grid_[-1]}].")
        basis = BSpline.design_matrix(at, self.fpca.knots_, self.fpca.degree).toarray()
        mean, linear = _affine(_components_of(self.fpca), kept, to_coefficients, basis)
        replicates = [
            self._replicate(fitted, residuals, projection, to_coefficients, rng, basis) for _ in range(int(n_bootstrap))
        ]
        return {
            "terms": list(self._design.column_names),
            "mean": mean,
            "map": linear,
            "replicate_means": np.stack([m for m, _ in replicates]),
            "replicate_maps": np.stack([m for _, m in replicates]),
            "df_residual": projection.df_residual,
        }

    def match_target(
        self,
        target: pd.Series | np.ndarray,
        *,
        bounds: Mapping[str, tuple[float, float] | float | str] | None = None,
        n_starts: int = 20,
        random_state: int | np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Return the factor settings whose predicted curve comes closest to a target curve.

        The eigenfunctions are orthonormal, so the L2 distance between two curves in their span is the
        distance between their scores: the target is scored by the FPCA, and the search moves the predicted
        scores of the kept components towards it. The part of the target outside the kept components is out
        of reach of any setting, and is the floor of the distance. The search is a bounded quasi-Newton
        search from ``n_starts`` random starts, of which the best is kept; for mixture components it is held
        to blends whose proportions sum to one. Different settings can reproduce one target equally well,
        and the search returns one of them.

        Parameters
        ----------
        target : pandas.Series or numpy.ndarray
            The target curve on the grid the curves were measured on; ``NaN`` where it has no value.
        bounds : mapping, optional
            For each factor, a ``(low, high)`` range to search, or one value to hold it at. A numeric
            factor not named is searched over the range the design covers; a categorical factor must be
            given its level.
        n_starts : int, default=20
            Number of random starting settings.
        random_state : int, numpy.random.Generator, or None, default=None
            Seeds the starting settings, per the reproducibility contract.

        Returns
        -------
        dict
            ``settings`` (a Series of the factors' values), ``curve`` (the predicted curve there),
            ``distance`` (the L2 distance from the target over the grid), ``scores`` and ``target_scores``
            (the kept components' scores, predicted and of the target).
        """
        if int(n_starts) < 1:
            raise ValueError(f"n_starts must be at least 1; got {n_starts}.")
        rng = check_random_state(random_state)
        kept = self.components["kept"].to_numpy()
        names = self.components.index[kept]
        goal = self.fpca.transform(np.asarray(target, dtype=float).reshape(1, -1)).iloc[0][names].to_numpy()
        coefficients = np.linalg.pinv(self._model_matrix) @ _kept_scores(self.fpca.scores_.to_numpy(), kept)
        searched, held = self._search_space(bounds)
        position = np.flatnonzero(kept).tolist()

        def scores_at(x: np.ndarray) -> np.ndarray:
            point = pd.DataFrame([{**held, **dict(zip(searched, x, strict=True))}])
            point = point.assign(**dict.fromkeys(self._blocks, 0.0))
            (row,) = build_design_matrices([self._design], point, return_type="matrix")
            return (np.asarray(row) @ coefficients)[0, position]

        def distance(x: np.ndarray) -> float:
            return float(np.sum((scores_at(x) - goal) ** 2))

        box = list(searched.values())
        starts = rng.uniform([low for low, _ in box], [high for _, high in box], size=(int(n_starts), len(box)))
        blend = self._blend(searched, held)
        if blend is None:
            best = min(
                (optimize.minimize(distance, x0, bounds=box, method="L-BFGS-B") for x0 in starts), key=lambda r: r.fun
            )
        else:  # mixture components stay a blend: their proportions, with any held, sum to one
            share, total = blend
            starts[:, share] = rng.dirichlet(np.ones(len(share)), size=int(n_starts)) * total
            constraint = {"type": "eq", "fun": lambda x: x[share].sum() - total}
            best = min(
                (
                    optimize.minimize(distance, x0, bounds=box, method="SLSQP", constraints=[constraint])
                    for x0 in starts
                ),
                key=lambda r: r.fun,
            )
        settings = pd.Series({**held, **dict(zip(searched, best.x, strict=True))}, name="settings")
        curve = self.fpca.inverse_transform(_full_scores(self.fpca, position, scores_at(best.x))).iloc[0]
        gap = (curve.to_numpy() - np.asarray(target, dtype=float)) ** 2
        measured = np.isfinite(gap)
        return {
            "settings": settings,
            "curve": curve.rename("curve"),
            "distance": float(np.sqrt(np.trapezoid(gap[measured], self.fpca.grid_[measured]))),
            "scores": pd.Series(scores_at(best.x), index=names, name="scores"),
            "target_scores": pd.Series(goal, index=names, name="target_scores"),
        }

    def _blend(self, searched: dict[str, tuple[float, float]], held: dict[str, Any]) -> tuple[list[int], float] | None:
        """Return the positions of the searched mixture components and the share left to them, or None."""
        settings = self._settings if self._settings is not None else pd.DataFrame()
        factors = [*searched, *held]
        components = factors if _looks_like_mixture(settings, factors) else _mixture_components(settings, factors)
        if not components:
            return None
        names = list(searched)
        share = [names.index(c) for c in components if c in searched]
        return share, 1.0 - sum(float(held[c]) for c in components if c in held)

    def _search_space(
        self, bounds: Mapping[str, tuple[float, float] | float | str] | None
    ) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
        """Split the model's factors into those searched, each with its range, and those held at one value."""
        settings = self._settings if self._settings is not None else pd.DataFrame()
        given = dict(bounds or {})
        factors = [c for c in settings.columns if _references(self.formula, c) and c not in self._blocks]
        unknown = sorted(set(given) - set(factors))
        if unknown:
            raise ValueError(f"bounds names {unknown}, which the model {self.formula!r} does not use.")
        searched, held = {}, {}
        for name in factors:
            value = given.get(name)
            if isinstance(value, tuple):
                searched[name] = (float(value[0]), float(value[1]))
            elif value is not None:
                held[name] = value
            elif pd.api.types.is_numeric_dtype(settings[name]):
                searched[name] = (float(settings[name].min()), float(settings[name].max()))
            else:
                raise ValueError(f"The categorical factor {name!r} needs a level in bounds, such as {{{name!r}: ...}}.")
        if not searched:
            raise ValueError("Every factor is held at one value, so there is nothing to search.")
        return searched, held

    def _curves_at(self, model: FPCA, kept: np.ndarray, to_points: np.ndarray) -> np.ndarray:
        """Return the curves the fitted scores predict, through a linear map from the runs to the points."""
        scores = to_points @ _kept_scores(model.scores_.to_numpy(), kept)
        return model.inverse_transform(scores, grid=self.fpca.grid_).to_numpy()

    def _replicate(
        self,
        fitted: np.ndarray,
        residuals: np.ndarray,
        projection: _Projection,
        to_coefficients: np.ndarray,
        rng: np.random.Generator,
        basis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one wild-bootstrap replicate of the runs' fit, as its mean curve and its map (see :func:`_affine`)."""
        signs = rng.choice([-1.0, 1.0], size=int(self._groups.max()) + 1)[self._groups]
        # shortcut: the smoothing penalty is not chosen again, so the bands leave out the uncertainty of GCV's
        # choice; choose it per replicate if that ever matters.
        components = _refit(self.fpca, fitted + signs[:, None] * residuals)
        table = _component_table(components, projection, self._significance_level)
        return _affine(components, table["kept"].to_numpy(), to_coefficients, basis)


def _affine(
    components: _Components, kept: np.ndarray, to_coefficients: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fit's mean curve, through the B-splines ``basis`` at its points, and the map from a model row.

    The rebuilt curve is affine in the scores, and the predicted scores are linear in the row, so the curve
    at a row is the mean plus the row times the coefficients of the kept scores times their eigenfunctions.
    """
    shapes = (basis @ components.eigen_coef).T
    return basis @ components.mean_coef, to_coefficients @ _kept_scores(components.scores, kept) @ shapes


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
    # The blocks enter the model as the sum-coded contrast columns analyze_experiment adds; build them the same way.
    working, _, blocks = _add_block_term(design_matrix, "score ~ 1", "main_effects")
    try:
        matrix = dmatrix(formula, working, return_type="matrix")
    except PatsyError as error:
        raise ValueError(f"The model {formula!r} cannot be built from the design's columns: {error}") from None
    projection = _projection(np.asarray(matrix))
    if projection.df_residual < 1:
        raise ValueError(
            f"The model has {projection.df_model + 1} estimable terms but there are {len(curves)} runs, so no "
            "residual degrees of freedom are left to test or bound the scores. Fit a smaller model, or add runs."
        )
    table = _component_table(_components_of(fitted), projection, significance_level)
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
        _blocks=blocks if any(name in formula for name in blocks) else (),
        _settings=design_matrix,
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
