"""Multivariate analysis: PCA, PLS, TPLS, scaling, and diagnostic plots."""

from typing import NoReturn

from process_improve.multivariate.methods import (
    OPLS,
    PCA,
    PLS,
    TPLS,
    AdaptivePCA,
    AdaptivePLS,
    MCUVScaler,
    center,
    check_predictive_signal,
    class_enrichment,
    count_discoveries_under_null,
    eigenvalue_summary,
    group_contributions,
    observation_contributions,
    project_variables,
    rv2_coefficient,
    rv_coefficient,
    scale,
    score_contributions,
    spe_contributions,
    squared_cosine,
    t2_contributions,
    vip,
)
from process_improve.multivariate.plots import (
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
    "OPLS",
    "PCA",
    "PLS",
    "TPLS",
    "AdaptivePCA",
    "AdaptivePLS",
    "MCUVScaler",
    "center",
    "check_predictive_signal",
    "class_enrichment",
    "coefficient_plot",
    "correlation_loadings_plot",
    "count_discoveries_under_null",
    "eigenvalue_summary",
    "explained_variance_plot",
    "group_contributions",
    "loading_plot",
    "observation_contributions",
    "predictions_vs_observed_plot",
    "project_variables",
    "rv2_coefficient",
    "rv_coefficient",
    "scale",
    "score_contributions",
    "score_plot",
    "spe_contributions",
    "spe_plot",
    "squared_cosine",
    "t2_contributions",
    "t2_plot",
    "vip",
]

# ---------------------------------------------------------------------------
# Migration helpers - old names raise helpful errors
# ---------------------------------------------------------------------------

_RENAMED = {
    "permutation_q2": "check_predictive_signal",
    "pipeline_null": "count_discoveries_under_null",
}


def __getattr__(name: str) -> NoReturn:
    """Raise a helpful error when a renamed module attribute is accessed."""
    if name in _RENAMED:
        new = _RENAMED[name]
        raise AttributeError(f"{name!r} has been renamed to {new!r}. Use: from {__name__} import {new}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
