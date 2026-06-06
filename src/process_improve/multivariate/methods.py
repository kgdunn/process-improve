"""Public re-exporter for ``process_improve.multivariate``.

As of ENG-01 the implementation is split across focused submodules
(:mod:`~process_improve.multivariate._pca`, ``_pls``, ``_tpls``, ``_mbpls``,
``_mbpca``, ``_preprocessing``, ``_nipals``, ``_limits``, ``_diagnostics``,
``_resampling`` and ``plots``). This module aggregates their public surface so
every name remains importable from a single, stable path::

    from process_improve.multivariate.methods import PCA, PLS, MCUVScaler

The older :mod:`process_improve.multivariate._pca_pls` path is kept as a thin
backward-compatibility shim that re-exports from here.
"""

from .._linalg import safe_inverse
from .._random import check_random_state
from ..univariate.metrics import detect_outliers_esd
from ..visualization.themes import REFERENCE_LINE_COLOR
from ._common import NotEnoughVarianceError, SpecificationWarning, epsqrt
from ._diagnostics import (
    eigenvalue_summary,
    observation_contributions,
    project_variables,
    rv2_coefficient,
    rv_coefficient,
    spe_contributions,
    squared_cosine,
    t2_contributions,
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
from ._resampling import Resampler
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

__all__ = [
    "MBPCA",
    "MBPLS",
    "PCA",
    "PLS",
    "REFERENCE_LINE_COLOR",
    "TPLS",
    "DataFrameDict",
    "MCUVScaler",
    "NotEnoughVarianceError",
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
    "spe_contributions",
    "spe_limit",
    "spe_plot",
    "squared_cosine",
    "ssq",
    "t2_contributions",
    "t2_plot",
    "terminate_check",
    "vip",
]
