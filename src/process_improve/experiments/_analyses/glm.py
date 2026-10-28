# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Generalized linear models: a response that is a count, a proportion or a positive skewed amount.

Least squares assumes the response varies by the same amount about every fitted value. A count of
defects (Poisson), a number of successes out of a number of trials (binomial), or a time to
failure that spreads in proportion to its size (gamma) does not. A generalized linear model fits

.. math::

    g(\mu_i) = x_i^\top \beta, \qquad \operatorname{Var}(y_i) = \phi \, V(\mu_i),

with a link :math:`g` that keeps the fitted mean :math:`\mu` in range and a variance function
:math:`V` of the family. The coefficients are found by maximum likelihood (iteratively reweighted
least squares), and every test and interval is a Wald one, on the normal distribution: a
coefficient's z is its estimate over its standard error, and the interval for a mean is worked on
the scale of the linear predictor and carried through the inverse link, so it stays in range and
is not symmetric about the fitted mean.

Two conventions differ between packages, and both are offered:

- The dispersion :math:`\phi` is 1 for the binomial and Poisson families. For the others it is
  estimated: by Pearson's statistic over the residual degrees of freedom (R's ``glm``, the
  default), by the deviance over them, or by maximum likelihood (SAS's GENMOD).
- The covariance of the coefficients is the inverse of the expected information (R, the default)
  or of the observed information (SAS's GENMOD). They are the same for the canonical link of each
  family (logit, log, inverse), and differ otherwise, as for the gamma family with a log link.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrices
from scipy import optimize, stats

#: The families offered, each with its canonical link (the default).
FAMILIES: dict[str, tuple[type, str]] = {
    "binomial": (sm.families.Binomial, "logit"),
    "poisson": (sm.families.Poisson, "log"),
    "gamma": (sm.families.Gamma, "inverse"),
    "gaussian": (sm.families.Gaussian, "identity"),
    "inverse_gaussian": (sm.families.InverseGaussian, "inverse_squared"),
}

#: The links offered, by name.
LINKS: dict[str, type] = {
    "logit": sm.families.links.Logit,
    "probit": sm.families.links.Probit,
    "cloglog": sm.families.links.CLogLog,
    "log": sm.families.links.Log,
    "identity": sm.families.links.Identity,
    "inverse": sm.families.links.InversePower,
    "inverse_squared": sm.families.links.InverseSquared,
    "sqrt": sm.families.links.Sqrt,
}

DISPERSIONS = ("pearson", "deviance", "ml")
INFORMATIONS = ("expected", "observed")
_FIXED_DISPERSION = ("binomial", "poisson")
#: At most this many distinct runs are each their own Hosmer-Lemeshow group; more are put in ten.
_HL_GROUPS = 10


def check_glm_options(family: str | None, link: str | None, dispersion: str | float | None, information: str) -> None:
    """Raise ``ValueError`` for a family, link, dispersion or information the GLM fit does not offer."""
    if family is None:
        raise ValueError(f"analysis_type='glm' needs a family: one of {', '.join(map(repr, FAMILIES))}.")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {', '.join(map(repr, FAMILIES))}; got {family!r}.")
    if link is not None and link not in LINKS:
        raise ValueError(f"link must be one of {', '.join(map(repr, LINKS))}; got {link!r}.")
    if isinstance(dispersion, str) and dispersion not in DISPERSIONS:
        raise ValueError(f"dispersion must be a positive number or one of {', '.join(map(repr, DISPERSIONS))}.")
    if not isinstance(dispersion, str) and dispersion is not None and not dispersion > 0:
        raise ValueError(f"A dispersion given as a number must be positive; got {dispersion!r}.")
    if information not in INFORMATIONS:
        raise ValueError(f"information must be one of {', '.join(map(repr, INFORMATIONS))}; got {information!r}.")


def _trials(df: pd.DataFrame, trials: str | float | Sequence[float] | None) -> np.ndarray | None:
    """Return the number of trials of each run: a column of the data, one number for every run, or one per run."""
    if trials is None:
        return None
    if isinstance(trials, str):
        found = df[trials].to_numpy(dtype=float)  # a missing count already dropped its run
    else:
        found = np.broadcast_to(np.asarray(trials, dtype=float), (len(df),)).copy()
    if np.any(found <= 0):  # a column too (#1001): a zero reached statsmodels, which divided by it
        raise ValueError("Every run's number of trials must be positive.")
    return found


def _ml_dispersion(family: sm.families.Family, y: np.ndarray, mu: np.ndarray, weights: np.ndarray) -> float:
    """Return the dispersion that maximises the likelihood at the fitted means (SAS GENMOD's estimate)."""

    def negative(log_phi: float) -> float:
        return -float(family.loglike(y, mu, var_weights=weights, scale=np.exp(log_phi)))

    start = np.log(max(float(np.mean(weights * (y - mu) ** 2 / family.variance(mu))), 1e-12))
    found = optimize.minimize_scalar(negative, bounds=(start - 10, start + 10), method="bounded")
    return float(np.exp(found.x))


def _bernoulli_log_likelihood(successes: np.ndarray, trials: np.ndarray, p: np.ndarray) -> float:
    """Return the log-likelihood of the individual trials, which leaves out the binomial coefficients (Minitab's)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(successes > 0, successes * np.log(p), 0.0) + np.where(
            trials - successes > 0, (trials - successes) * np.log1p(-p), 0.0
        )
    return float(terms.sum())


def _hosmer_lemeshow(successes: np.ndarray, trials: np.ndarray, p: np.ndarray, patterns: np.ndarray) -> dict:
    """Return Hosmer and Lemeshow's test: observed against expected successes in groups of similar fitted risk.

    With no more than ten distinct runs (settings), each is its own group; with more, the runs are
    sorted by fitted probability and cut into ten groups of about equal numbers of trials. The
    statistic is referred to chi-squared on the groups less two degrees of freedom.
    """
    keys = pd.Series([tuple(row) for row in patterns])
    codes = pd.factorize(keys)[0]
    if codes.max() + 1 <= _HL_GROUPS:
        groups = codes
    else:
        order = np.argsort(p, kind="stable")
        cumulative = np.cumsum(trials[order]) / trials.sum()
        groups = np.empty(len(p), dtype=int)
        groups[order] = np.minimum((cumulative * _HL_GROUPS - 1e-9).astype(int), _HL_GROUPS - 1)
    statistic = 0.0
    labels = np.unique(groups)
    for g in labels:
        at = groups == g
        n, observed, expected = trials[at].sum(), successes[at].sum(), (trials[at] * p[at]).sum()
        statistic += (observed - expected) ** 2 / (expected * (1 - expected / n))
    df = len(labels) - 2
    return {"statistic": float(statistic), "df": df, "p_value": float(stats.chi2.sf(statistic, df)) if df > 0 else None}


@dataclass(frozen=True)
class GlmSpec:
    """The generalized linear model to fit: its family, link, and how its trials, dispersion and covariance are found.

    :func:`run_glm` documents each field under its *spec* parameter.
    """

    family: str
    link: str | None = None
    trials: str | float | Sequence[float] | None = None
    dispersion: str | float | None = None
    information: str = "expected"


def run_glm(
    df: pd.DataFrame,
    formula: str,
    spec: GlmSpec | str,
    alpha: float = 0.05,
    new_points: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fit a generalized linear model and report its coefficients, fit and fitted means with their intervals.

    Parameters
    ----------
    df : DataFrame
        The runs: the factors, the response and, for a binomial count, the trials.
    formula : str
        A patsy formula, ``"y ~ A + B + A:B"``.
    spec : GlmSpec or str
        The model: a :class:`GlmSpec`, or the family's name alone for its canonical link and
        the defaults below.

        - ``family``: ``"binomial"``, ``"poisson"``, ``"gamma"``, ``"gaussian"`` or ``"inverse_gaussian"``.
        - ``link``: the link function; ``None`` uses the family's canonical link (``"logit"``, ``"log"``,
          ``"inverse"``, ``"identity"``, ``"inverse_squared"``).
        - ``trials``: for the binomial family, the number of trials of each run, whose response is
          then the number of successes: a column of *df*, one number for every run, or one per run.
          ``None`` takes the response as 0 or 1 (one trial per run).
        - ``dispersion``: ``"pearson"``, ``"deviance"``, ``"ml"`` or a number; ``None`` fixes it at 1
          for the binomial and Poisson families and estimates it by Pearson's statistic for the
          others. A number fixes it there.
        - ``information``: ``"expected"`` or ``"observed"``, the information matrix the covariance
          of the coefficients inverts.
    alpha : float
        One less the confidence level of the intervals.
    new_points : DataFrame or None
        Settings at which to predict the mean as well, with its interval.

    Returns
    -------
    dict[str, Any]
        ``family``, ``link``, ``dispersion`` and ``dispersion_method``, ``information``;
        ``coefficients``, each with its ``std_error``, Wald ``z_value`` and ``p_value``, ``ci_low``
        and ``ci_high``, and, under a log or logit link, the ``exp_coefficient`` with its interval
        (a rate ratio or an odds ratio per unit of the term); ``log_likelihood``, ``deviance``,
        ``null_deviance``, ``pearson_chi2``, ``df_residual``, ``aic``; ``goodness_of_fit``, with the
        ``pearson`` and ``deviance`` tests and, for the binomial family, ``hosmer_lemeshow``; and
        ``fitted``, the mean of each run with its interval, and ``predictions`` at *new_points*.
        For the binomial family with trials, ``log_likelihood_trials`` is the log-likelihood of the
        individual trials (without the binomial coefficients), which Minitab prints.
    """
    spec = GlmSpec(spec) if isinstance(spec, str) else spec
    family, link, trials, dispersion, information = (
        spec.family,
        spec.link,
        spec.trials,
        spec.dispersion,
        spec.information,
    )
    family_class, canonical = FAMILIES[family]
    link_name = link or canonical
    glm_family = family_class(LINKS[link_name]())
    response, exog = dmatrices(formula, df, return_type="dataframe")
    y = response.iloc[:, 0].to_numpy(dtype=float)
    n = _trials(df, trials)
    weights = np.ones(len(y)) if n is None else n
    endog = y if n is None else np.column_stack([y, n - y])
    model = sm.GLM(endog, exog, family=glm_family)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit()
    mu = np.asarray(fitted.mu, dtype=float)
    proportion = y if n is None else y / n
    pearson = float(np.sum(weights * (proportion - mu) ** 2 / glm_family.variance(mu)))
    phi, method = _dispersion(dispersion, family, pearson, fitted, lambda: (glm_family, proportion, mu, weights))
    params = fitted.params.to_numpy(dtype=float)
    cov = np.linalg.inv(-model.hessian(params, scale=phi, observed=information == "observed"))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    def means(matrix: np.ndarray) -> list[dict[str, float]]:
        eta = matrix @ params
        half = z_crit * np.sqrt(np.einsum("ij,jk,ik->i", matrix, cov, matrix))
        inverse = glm_family.link.inverse
        ends = np.sort(np.column_stack([inverse(eta - half), inverse(eta + half)]), axis=1)
        return [
            {"predicted": float(m), "ci_low": float(lo), "ci_high": float(hi)}
            for m, (lo, hi) in zip(inverse(eta), ends, strict=True)
        ]

    log_likelihood = float(glm_family.loglike(proportion, mu, var_weights=weights, scale=phi))
    result: dict[str, Any] = {
        "family": family,
        "link": link_name,
        "dispersion": phi,
        "dispersion_method": method,
        "information": information,
        "coefficients": _coefficients(list(exog.columns), params, cov, z_crit, link_name in ("log", "logit")),
        "log_likelihood": log_likelihood,
        "deviance": float(fitted.deviance),
        "null_deviance": float(fitted.null_deviance),
        "pearson_chi2": pearson,
        "df_residual": float(fitted.df_resid),
        "aic": float(-2 * log_likelihood + 2 * (len(params) + (method != "fixed"))),
        "goodness_of_fit": _chi_squared_tests(pearson, float(fitted.deviance), float(fitted.df_resid)),
        "fitted": means(exog.to_numpy(dtype=float)),
    }
    if family == "binomial":
        result["goodness_of_fit"]["hosmer_lemeshow"] = _hosmer_lemeshow(
            proportion * weights, weights, mu, exog.to_numpy(dtype=float)
        )
        if n is not None:
            result["log_likelihood_trials"] = _bernoulli_log_likelihood(y, n, mu)
    if new_points is not None:
        result["predictions"] = means(np.asarray(build_design_matrices([exog.design_info], new_points)[0], dtype=float))
    return result


def _dispersion(
    dispersion: str | float | None,
    family: str,
    pearson: float,
    fitted: Any,  # noqa: ANN401 - statsmodels' GLM results
    ml_inputs: Callable[[], tuple[sm.families.Family, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, str]:
    """Return the dispersion and how it was found: fixed, or estimated by Pearson, the deviance or likelihood."""
    if isinstance(dispersion, (int, float)) and not isinstance(dispersion, bool):
        return float(dispersion), "fixed"
    if dispersion is None and family in _FIXED_DISPERSION:
        return 1.0, "fixed"
    method = dispersion if isinstance(dispersion, str) else "pearson"
    df_resid = float(fitted.df_resid)
    if method == "pearson":
        return pearson / df_resid, method
    if method == "deviance":
        return float(fitted.deviance) / df_resid, method
    return _ml_dispersion(*ml_inputs()), method


def _coefficients(
    terms: list[str], params: np.ndarray, cov: np.ndarray, z_crit: float, exponentiate: bool
) -> list[dict[str, Any]]:
    """Return each coefficient with its Wald test and interval, and, under a log or logit link, their exponentials."""
    rows = []
    for term, b, s in zip(terms, params, np.sqrt(np.diag(cov)), strict=True):
        row: dict[str, Any] = {
            "term": term,
            "coefficient": float(b),
            "std_error": float(s),
            "z_value": float(b / s),
            "p_value": float(2 * stats.norm.sf(abs(b / s))),
            "ci_low": float(b - z_crit * s),
            "ci_high": float(b + z_crit * s),
        }
        if exponentiate:  # an odds ratio (logit) or a rate ratio (log) per unit of the term
            row |= {
                "exp_coefficient": float(np.exp(b)),
                "exp_ci_low": float(np.exp(b - z_crit * s)),
                "exp_ci_high": float(np.exp(b + z_crit * s)),
            }
        rows.append(row)
    return rows


def _chi_squared_tests(pearson: float, deviance: float, df_resid: float) -> dict[str, dict[str, float]]:
    """Return Pearson's and the deviance goodness-of-fit tests, each on the residual degrees of freedom."""
    return {
        name: {"statistic": value, "df": df_resid, "p_value": float(stats.chi2.sf(value, df_resid))}
        for name, value in (("pearson", pearson), ("deviance", deviance))
    }
