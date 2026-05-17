"""Multivariate analysis: PCA, PLS, TPLS, scaling, and diagnostic plots."""

from process_improve.multivariate.methods import (
    PCA,
    PLS,
    TPLS,
    MCUVScaler,
    center,
    eigenvalue_summary,
    observation_contributions,
    project_variables,
    rv2_coefficient,
    rv_coefficient,
    scale,
    squared_cosine,
    vip,
)
from process_improve.multivariate.plots import (
    loading_plot,
    score_plot,
    spe_plot,
    t2_plot,
)

__all__ = [
    "PCA",
    "PLS",
    "TPLS",
    "MCUVScaler",
    "center",
    "eigenvalue_summary",
    "loading_plot",
    "observation_contributions",
    "project_variables",
    "rv2_coefficient",
    "rv_coefficient",
    "scale",
    "score_plot",
    "spe_plot",
    "squared_cosine",
    "t2_plot",
    "vip",
]
