"""Robust regression methods, and least squares for a model nonlinear in its parameters."""

from process_improve.regression.methods import (
    OLS,
    fit_robust_lm,
    multiple_linear_regression,
    repeated_median_slope,
    robust_regression,
)
from process_improve.regression.nonlinear import nonlinear_regression

__all__ = [
    "OLS",
    "fit_robust_lm",
    "multiple_linear_regression",
    "nonlinear_regression",
    "repeated_median_slope",
    "robust_regression",
]
