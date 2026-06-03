# (c) Kevin Dunn, 2010-2026. MIT License.
"""Lenth's method (PSE) for unreplicated factorials (ENG-02)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper


def _run_lenth_method(ols_result: RegressionResultsWrapper) -> dict[str, Any]:
    """Lenth's method (PSE) for unreplicated factorials.

    Not available in mainstream Python libraries - custom (~30 lines).
    """
    params = ols_result.params.drop("Intercept", errors="ignore")
    effects = 2.0 * params.values  # coded ±1 → effect = 2 * coefficient
    abs_effects = np.abs(effects)

    # Step 1: initial median
    s0 = 1.5 * np.median(abs_effects)

    # Step 2: pseudo standard error - median of |effects| <= 2.5 * s0
    trimmed = abs_effects[abs_effects <= 2.5 * s0]
    pse = s0 if len(trimmed) == 0 else 1.5 * np.median(trimmed)

    # Margin of error and simultaneous margin of error
    m = len(effects)
    t_val = stats.t.ppf(1 - 0.025, df=m / 3)
    t_val_sim = stats.t.ppf(1 - 0.025 / m, df=m / 3)  # Bonferroni-like
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
            "effects": effect_list,
        }
    }
