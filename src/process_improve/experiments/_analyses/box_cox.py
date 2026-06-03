# (c) Kevin Dunn, 2010-2026. MIT License.
"""Box-Cox transformation diagnostic (ENG-02)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _run_box_cox(design_df: pd.DataFrame, response_col: str) -> dict[str, Any]:
    """Box-Cox transformation using scipy."""
    from scipy.stats import boxcox  # noqa: PLC0415

    y = np.asarray(design_df[response_col], dtype=float)

    if np.any(y <= 0):
        return {"box_cox": {"error": "Box-Cox requires all positive response values."}}

    y_transformed, lmbda = boxcox(y)

    return {
        "box_cox": {
            "lambda": float(lmbda),
            "transformed_values": [float(v) for v in y_transformed],
            "recommendation": (
                "log transform" if abs(lmbda) < 0.05
                else "square root" if abs(lmbda - 0.5) < 0.05
                else "inverse" if abs(lmbda - (-1.0)) < 0.05
                else f"power transform (lambda={lmbda:.3f})"
            ),
        }
    }
