# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

# ENG-13 (#295): plotting deps live in the ``[plotting]`` extra. The
# ``_MissingExtra`` stand-in lets module-import succeed for the algorithm
# surface (``PCA``, ``PLS``, ...) while any actual plot call raises a
# clear "install the extra" ImportError.
try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra
    go = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]

try:
    import ridgeplot
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra
    ridgeplot = _MissingExtra("ridgeplot", "plotting")  # type: ignore[assignment]
from sklearn.base import BaseEstimator, clone
from tqdm import tqdm

from .._linalg import safe_inverse
from .._random import check_random_state
from ..univariate.metrics import detect_outliers_esd
from ..visualization.themes import REFERENCE_LINE_COLOR

# ENG-01: shared primitives now live in ``_common`` (re-exported here for
# backward compatibility; see the module docstring above).
from ._common import (
    SpecificationWarning,
    _nz,  # noqa: F401  # re-exported for back-compat: tests import _nz from this module
    epsqrt,
)
from ._diagnostics import (
    eigenvalue_summary,
    observation_contributions,
    project_variables,
    rv2_coefficient,
    rv_coefficient,
    squared_cosine,
    vip,
)
from ._limits import (
    ellipse_coordinates,
    hotellings_t2_limit,
    score_limit,
    spe_calculation,
    spe_limit,
)
from ._mbpca import MBPCA
from ._mbpls import MBPLS, randomization_test_mbpls
from ._nipals import (
    internal_pls_nipals_fit_one_pc,
    nan_to_zeros,
    quick_regress,
    regress_a_space_on_b_row,
    ssq,
    terminate_check,
)
from ._pca import PCA
from ._pls import PLS
from ._preprocessing import MCUVScaler, center, scale
from ._tpls import TPLS, DataFrameDict
from .plots import (
    Plot,
    coefficient_plot,
    correlation_loadings_plot,
    explained_variance_plot,
    loading_plot,
    predictions_vs_observed_plot,
    score_plot,
    spe_plot,
    t2_plot,
)


class Resampler:
    """Base class for resampling methods."""

    def __init__(  # noqa: PLR0913
        self,
        estimator: BaseEstimator,
        x: DataFrameDict,
        accessor: Callable,
        use_jackknife: bool = True,
        bootstrap_rounds: int = 0,
        fraction_excluded: float = 0.0,
        random_state: int | np.random.Generator | None = None,
    ):
        """Initialize the resampling method.

        The `accessor` is a callable that takes an estimator and returns the parameters of interest.

        Mutually exclusive parameters:
            * `use_jackknife` flag indicates whether to use jackknife resampling (leave out one sample; rebuild)
            * `bootstrap_rounds` specifies the number of bootstrap rounds if applicable (resample data with replacement)
            * `fraction_excluded` specifies the fraction of data to exclude in each resample (for fractional resampling)

        Only one of these parameters should be set at a time.

        Parameters
        ----------
        random_state : int, np.random.Generator, or None, optional
            Seeds the RNG used by ``bootstrap()`` and ``fractional()``;
            see ``docs/development/reproducibility.rst`` (ENG-08). Pass
            the same int twice to get bit-identical resamples; pass
            ``None`` for fresh entropy on each call. ``jackknife()``
            is deterministic and ignores this parameter.
        """
        if not isinstance(estimator, BaseEstimator):
            raise TypeError("estimator must be a BaseEstimator instance.")
        self.estimator = estimator

        if not isinstance(x, DataFrameDict):
            raise TypeError("x must be a DataFrameDict instance.")
        self.x = x

        if not callable(accessor):
            raise TypeError("accessor must be a callable function.")
        self.accessor = accessor

        self.use_jackknife = use_jackknife
        self.bootstrap_rounds = int(bootstrap_rounds)
        self.fraction_excluded = float(fraction_excluded)
        if self.use_jackknife and self.bootstrap_rounds > 0 and self.fraction_excluded > 0.0:
            raise ValueError(
                (
                    "`use_jackknife`, `bootstrap_rounds`, and `fraction_excluded` are mutually exclusive. ",
                    "Set only one of them.",
                )
            )

        # Resolve random_state up front so the same instance can be
        # called twice and produce bit-identical resamples (ENG-08).
        # Keep the original value for repr / debugging.
        self.random_state = random_state
        self._rng = check_random_state(random_state)

        self.parameters: list = []
        self.n_resamples = 0

    def resample(self, show_progress: bool = True) -> Resampler:
        """Perform the resampling."""
        if self.use_jackknife:
            return self.jackknife(show_progress=show_progress)
        elif self.bootstrap_rounds > 0:
            return self.bootstrap(show_progress=show_progress)
        elif self.fraction_excluded > 0.0:
            return self.fractional(show_progress=show_progress)
        else:
            raise ValueError("Either use_jackknife or bootstrap_rounds must be set.")

    def jackknife(self, show_progress: bool) -> Resampler:
        """Perform jackknife resampling on the given estimator."""
        self.parameters = []
        indices = np.arange(len(self.x))
        for i in tqdm(range(len(self.x)), desc="Jackknife Resampling", disable=not show_progress):
            leave_one_out_indices = indices[indices != i]
            x_train = self.x[leave_one_out_indices]
            parameter = self.accessor(clone(self.estimator).fit(x_train))
            self.parameters.append(parameter)

        self.n_resamples = len(self.parameters)
        if self.n_resamples == 0:
            raise ValueError("No resamples were generated. Check your data and parameters.")
        return self

    def bootstrap(self, show_progress: bool) -> Resampler:
        """Perform bootstrap resampling on the given estimator."""
        self.parameters = []

        # Generate bootstrap samples, resample with replacement, in a loop of self.bootstrap_rounds iterations.
        # The shared ``self._rng`` is seeded via the constructor's ``random_state`` (ENG-08).
        for _ in tqdm(range(self.bootstrap_rounds), desc="Bootstrap Resampling", disable=not show_progress):
            # Resample indices with replacement

            indices = self._rng.choice(len(self.x), size=len(self.x), replace=True)
            x_train = self.x[indices]
            parameter = self.accessor(clone(self.estimator).fit(x_train))
            self.parameters.append(parameter)

        self.n_resamples = len(self.parameters)
        if self.n_resamples == 0:
            raise ValueError("No resamples were generated. Check your data and parameters.")

        return self

    def fractional(self, show_progress: bool) -> Resampler:
        """Perform fractional resampling on the given estimator.

        Will repeat N times (N = number of rows in x), each time leaving out a fraction of the data as specified by
        self.fraction_excluded.
        """
        self.parameters = []

        # The shared ``self._rng`` is seeded via the constructor's ``random_state`` (ENG-08).
        # Re-validate here: the __init__ guard can be bypassed by mutating
        # ``fraction_excluded`` to 0 (or out of range) before calling fractional().
        if not 0.0 < self.fraction_excluded < 1.0:
            raise ValueError(
                f"`fraction_excluded` must be in the open interval (0, 1) to perform fractional "
                f"resampling, got {self.fraction_excluded}."
            )
        n_groups = int(1 / self.fraction_excluded)
        for _ in tqdm(range(len(self.x)), desc="Fractional Resampling", disable=not show_progress):
            # Find the indices to leave out
            all_indices = np.arange(len(self.x))
            self._rng.shuffle(all_indices)
            groups = np.array_split(all_indices, n_groups)
            rows_to_drop = groups[0]
            train_indices = np.setdiff1d(all_indices, rows_to_drop)
            x_train = self.x[train_indices]
            parameter = self.accessor(clone(self.estimator).fit(x_train))
            self.parameters.append(parameter)

        self.n_resamples = len(self.parameters)
        if self.n_resamples == 0:
            raise ValueError("No resamples were generated. Check your data and parameters.")

        return self

    def plot_results(self, cutoff: float | None = None) -> go.Figure:
        """
        Plot the results of the resampling.

        A vertical line can be added at the specified cutoff value. If `cutoff` is None, no vertical line is added.
        """
        parameters = pd.DataFrame(self.parameters)
        size_per_sample = len(self.parameters[0])

        # Resort the columns of the parameters DataFrame by the .median() value of each column
        parameters = parameters.reindex(parameters.median().sort_values(ascending=False).index, axis=1)

        fig = ridgeplot.ridgeplot(
            samples=parameters.to_numpy().T.reshape((size_per_sample, 1, self.n_resamples)),
            # bandwidth=4,
            kde_points=np.linspace(0, 2, 500),
            colorscale="viridis",
            colormode="row-index",
            opacity=0.6,
            labels=parameters.columns.tolist(),
            spacing=0.1,
            norm="probability",
        )
        if cutoff is not None:
            fig.add_vline(
                x=cutoff, line_color="red", line_dash="dash", annotation_text="Cutoff", annotation_position="top left"
            )
        fig.update_layout(
            font_size=16,
            plot_bgcolor="white",
            xaxis=dict(
                title="Parameter Value",
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                title="Parameter Index",
                showgrid=True,
                zeroline=False,
                showticklabels=True,
            ),
            title="Resampling Results",
        )
        return fig



# ENG-23 (#305): explicit ``__all__`` so the thin re-exporter ``methods.py``
# can do ``from ._pca_pls import *`` without triggering CodeQL's
# py/polluting-import warning. List enumerated to mirror the public surface
# the prior ``methods.py`` exposed -- every name visible at module level,
# minus stdlib / 3rd-party imports and underscore-prefixed helpers.
__all__ = [
    "MBPCA",
    "MBPLS",
    "PCA",
    "PLS",
    "REFERENCE_LINE_COLOR",
    "TPLS",
    "DataFrameDict",
    "MCUVScaler",
    "Plot",
    "Resampler",
    "SpecificationWarning",
    "center",
    "check_random_state",
    "coefficient_plot",
    "correlation_loadings_plot",
    "detect_outliers_esd",
    "eigenvalue_summary",
    "ellipse_coordinates",
    "epsqrt",
    "explained_variance_plot",
    "hotellings_t2_limit",
    "internal_pls_nipals_fit_one_pc",
    "loading_plot",
    "nan_to_zeros",
    "observation_contributions",
    "predictions_vs_observed_plot",
    "project_variables",
    "quick_regress",
    "randomization_test_mbpls",
    "regress_a_space_on_b_row",
    "rv2_coefficient",
    "rv_coefficient",
    "safe_inverse",
    "scale",
    "score_limit",
    "score_plot",
    "spe_calculation",
    "spe_limit",
    "spe_plot",
    "squared_cosine",
    "ssq",
    "t2_plot",
    "terminate_check",
    "vip",
]
