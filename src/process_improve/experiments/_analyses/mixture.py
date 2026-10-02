# (c) Kevin Dunn, 2010-2026. MIT License.

"""ANOVA, significance and effects for Scheffe mixture models.

In a mixture the components sum to 1, so a component's blending coefficient is not
compared with zero: ``beta_i = 0`` says only that a pure blend of component ``i``
gives a response of zero, which is rarely a question anyone asks. The questions that
mean something are:

- **Do the components blend differently at all?** The linear block is tested as one
  hypothesis, ``beta_1 = beta_2 = ... = beta_q``, which is the mixture analogue of
  testing every main effect at once (Cornell, *Experiments with Mixtures*).
- **Is there non-linear blending?** Each ``x_i x_j`` (and ``x_i x_j x_k``) term is
  tested against zero, as in any regression; those hypotheses are meaningful.

Effects are reported in the Cox direction: moving from the centroid towards a pure
component while the others keep their relative proportions. For the linear blending
coefficients the effect of component ``i`` is ``beta_i - mean(beta_j, j != i)``
(Cornell 2002, section 5.10). The "effect = 2 x coefficient" of a coded factor, and
Lenth's method built on it, have no meaning for mixtures and are not computed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper


def _linear_names(ols_result: RegressionResultsWrapper, components: list[str]) -> list[str]:
    names = [str(n) for n in ols_result.params.index]
    return [c for c in components if c in names]


def _linear_block_test(ols_result: RegressionResultsWrapper, components: list[str]) -> dict[str, Any]:
    """F-test of ``beta_1 = ... = beta_q`` over the linear blending coefficients."""
    names = [str(n) for n in ols_result.params.index]
    linear = _linear_names(ols_result, components)
    restrictions = np.zeros((len(linear) - 1, len(names)))
    last = names.index(linear[-1])
    for row, component in enumerate(linear[:-1]):
        restrictions[row, names.index(component)] = 1.0
        restrictions[row, last] = -1.0
    test = ols_result.f_test(restrictions)
    f_value = float(np.squeeze(test.fvalue))
    df = len(linear) - 1
    return {
        "source": "Linear mixture",
        "df": df,
        "sum_sq": f_value * df * float(ols_result.mse_resid),
        "mean_sq": f_value * float(ols_result.mse_resid),
        "F": f_value,
        "p_value": float(test.pvalue),
    }


def _nonlinear_terms(ols_result: RegressionResultsWrapper, components: list[str]) -> list[str]:
    linear = set(components)
    return [str(n) for n in ols_result.params.index if str(n) not in linear]


def run_mixture_anova(ols_result: RegressionResultsWrapper, components: list[str]) -> dict[str, Any]:
    """Mixture ANOVA: the linear blending block, each non-linear blending term, and the residual."""
    if ols_result.df_resid <= 0:
        return {"anova_table": [], "note": "Saturated model - no residual degrees of freedom for ANOVA."}
    mse = float(ols_result.mse_resid)
    rows = [_linear_block_test(ols_result, components)]
    for term in _nonlinear_terms(ols_result, components):
        t = float(ols_result.tvalues[term])
        rows.append(
            {
                "source": term,
                "df": 1,
                "sum_sq": t**2 * mse,
                "mean_sq": t**2 * mse,
                "F": t**2,
                "p_value": float(ols_result.pvalues[term]),
            }
        )
    rows.append(
        {
            "source": "Residual",
            "df": int(ols_result.df_resid),
            "sum_sq": float(ols_result.ssr),
            "mean_sq": mse,
            "F": None,
            "p_value": None,
        }
    )
    return {
        "anova_table": rows,
        "note": (
            "Mixture model: the linear blending terms are tested together (H0: all components blend "
            "equally), not each against zero. Non-linear blending terms are tested individually."
        ),
    }


def run_mixture_significance(
    ols_result: RegressionResultsWrapper, components: list[str], alpha: float = 0.05
) -> dict[str, Any]:
    """Significant non-linear blending terms, and whether the components blend differently at all."""
    linear = _linear_block_test(ols_result, components) if ols_result.df_resid > 0 else None
    nonlinear = _nonlinear_terms(ols_result, components)
    pvals = ols_result.pvalues
    return {
        "significant_terms": [t for t in nonlinear if pvals[t] < alpha],
        "not_significant_terms": [t for t in nonlinear if pvals[t] >= alpha],
        "linear_blending_differs": None if linear is None else bool(linear["p_value"] < alpha),
        "linear_blending_p_value": None if linear is None else linear["p_value"],
        "significance_level": alpha,
    }


def run_mixture_effects(ols_result: RegressionResultsWrapper, components: list[str]) -> dict[str, Any]:
    """Cox-direction effects of the linear blending coefficients."""
    linear = _linear_names(ols_result, components)
    beta = np.array([float(ols_result.params[c]) for c in linear])
    q = len(beta)
    effects = {c: float(beta[i] - (beta.sum() - beta[i]) / (q - 1)) for i, c in enumerate(linear)}
    return {
        "effects": effects,
        "effect_direction": "cox",
        "note": (
            "Mixture model: each effect is the linear blending coefficient minus the mean of the others "
            "(Cox direction from the centroid). Effects of non-linear blending terms are not defined."
        ),
    }


def run_mixture_lenth() -> dict[str, Any]:
    """Lenth's method is for coded two-level factors; refuse it for mixtures, with the reason."""
    return {
        "lenth_method": None,
        "note": (
            "Lenth's method needs effects of coded two-level factors; mixture components have none. "
            "Use the mixture ANOVA (analysis_type='anova') instead."
        ),
    }
