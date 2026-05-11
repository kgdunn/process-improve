"""Generate the canonical centered-with-NaN matrix for the mixOmics cross-check.

This script is the *single source of truth* for the matrix that both
:mod:`process_improve` and R's :func:`mixOmics::nipals` see. It

1. loads the raw ``linnerud.csv`` (20 samples; 3 exercise + 3 physiological
   columns; vendored from scikit-learn's :func:`load_linnerud`),
2. injects MCAR NaN at a fixed ratio with a fixed seed,
3. column-mean-centers the result using ``nanmean`` (mixOmics' NIPALS
   convention),
4. writes ``linnerud_centered_with_nan.csv``.

The R reference script and the pytest cross-check both consume the
written CSV, so the two sides cannot drift on preprocessing or on the
NaN mask.

Run from the repository root::

    uv run python tests/fixtures/mixomics_nipals/prepare_fixture.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
SOURCE_CSV = HERE / "linnerud.csv"
CANONICAL_CSV = HERE / "linnerud_centered_with_nan.csv"

NAN_RATIO = 0.10
SEED = 20260511
N_COMPONENTS = 3


def build_canonical_matrix() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV)
    rng = np.random.default_rng(SEED)
    mask = rng.random(df.shape) < NAN_RATIO
    # Guarantee at least one observation in every row and every column so
    # NIPALS' degeneracy guards stay happy on this small 20x6 matrix.
    for r in range(df.shape[0]):
        if mask[r].all():
            mask[r, 0] = False
    for c in range(df.shape[1]):
        if mask[:, c].all():
            mask[0, c] = False
    arr = df.to_numpy(dtype=float)
    arr[mask] = np.nan
    centered = arr - np.nanmean(arr, axis=0, keepdims=True)
    return pd.DataFrame(centered, columns=df.columns)


def main() -> None:
    canonical = build_canonical_matrix()
    canonical.to_csv(CANONICAL_CSV, index=False, float_format="%.10f")
    print(f"wrote {CANONICAL_CSV}")
    print(f"  shape={canonical.shape}, NaN count={int(canonical.isna().sum().sum())}")


if __name__ == "__main__":
    main()
