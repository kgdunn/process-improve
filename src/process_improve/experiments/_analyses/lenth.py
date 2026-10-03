# (c) Kevin Dunn, 2010-2026. MIT License.
"""Lenth's method (PSE) for unreplicated factorials (ENG-02).

Lenth (1989), "Quick and easy analysis of unreplicated factorials", Technometrics
31(4), 469-473: with ``m`` effects ``c_j``,

- ``s0 = 1.5 median |c_j|``,
- ``PSE = 1.5 median { |c_j| : |c_j| < 2.5 s0 }``,
- ``ME = t(1 - alpha/2; m/3) PSE`` and ``SME = t(gamma; m/3) PSE`` with
  ``gamma = (1 + (1 - alpha)**(1/m)) / 2``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

from ._shared import is_block_term
from .aliasing import estimable_effects


def _run_lenth_method(
    ols_result: RegressionResultsWrapper,
    alpha: float = 0.05,
    blocks: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Lenth's method (PSE) for unreplicated factorials.

    ``ols_result`` must be fitted on factors coded to -1/+1, so that an effect is twice
    the coefficient. The method assumes each effect is a separate estimate, so exactly
    aliased terms enter once, as their alias chain (#16), not once per term with the
    chain's effect shared out between them. The ``blocks`` contrasts are not factor
    effects and are left out.
    """
    params = estimable_effects(ols_result).coefficients
    params = params[[label for label in params.index if not is_block_term(label, blocks)]]
    effects = 2.0 * params.to_numpy()  # coded ±1 → effect = 2 * coefficient
    abs_effects = np.abs(effects)

    # Step 1: initial median
    s0 = 1.5 * np.median(abs_effects)

    # Step 2: pseudo standard error - median of |effects| strictly below 2.5 * s0
    trimmed = abs_effects[abs_effects < 2.5 * s0]
    pse = s0 if len(trimmed) == 0 else 1.5 * np.median(trimmed)

    # Margin of error and simultaneous margin of error (Lenth's gamma, a Sidak quantile)
    m = len(effects)
    t_val = stats.t.ppf(1 - alpha / 2, df=m / 3)
    t_val_sim = stats.t.ppf((1 + (1 - alpha) ** (1 / m)) / 2, df=m / 3)
    me = t_val * pse
    sme = t_val_sim * pse

    term_names = list(params.index)
    effect_list = [
        {
            "term": str(name),
            "effect": float(effects[i]),
            "active_ME": bool(abs(effects[i]) > me),
            "active_SME": bool(abs(effects[i]) > sme),
        }
        for i, name in enumerate(term_names)
    ]

    return {
        "lenth_method": {
            "PSE": float(pse),
            "ME": float(me),
            "SME": float(sme),
            "significance_level": alpha,
            "effects": effect_list,
        }
    }
