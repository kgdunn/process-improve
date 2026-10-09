"""Input validation for :class:`TPLS`: what a fit accepts, and how it refuses the rest.

Every test here uses a 20-blend, 4-material synthetic set that fits in milliseconds, so
the guards are pinned without loading the pyphi example data.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import TPLS, DataFrameDict


def _synthetic(seed: int = 0) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
    """Return a D matrix and its F, Z and Y blocks: 20 blends of 4 materials, one group "G"."""
    rng = np.random.default_rng(seed)
    n_blends, n_materials = 20, 4
    index = [f"blend{i}" for i in range(n_blends)]
    materials = [f"mat{j}" for j in range(n_materials)]
    d_matrix = {"G": pd.DataFrame(rng.normal(size=(n_materials, 3)), index=materials, columns=list("pqr"))}
    blocks = {
        "F": {"G": pd.DataFrame(rng.random((n_blends, n_materials)), index=index, columns=materials)},
        "Z": {"Z1": pd.DataFrame(rng.normal(size=(n_blends, 2)), index=index, columns=["z1", "z2"])},
        "Y": {"Y1": pd.DataFrame({"y": rng.normal(size=n_blends)}, index=index)},
    }
    return d_matrix, blocks


@pytest.mark.parametrize("skip_f", [False, True], ids=["F-scaled", "F-as-is"])
def test_int64_columns_fit_like_the_same_values_as_float64(skip_f: bool) -> None:
    """int64 columns in every block are accepted, and fit exactly as their float64 copies do.

    The dtype check promised "float64 or int64" but tested for ``np.dtypes.IntDType``,
    which is the C ``int`` (int32), so every int64 column was refused. Accepting them
    also needs float working copies: with ``skip_f_matrix_preprocessing`` an all-integer
    F group reached the in-place deflation as an integer array.
    """
    d_matrix, blocks = _synthetic()
    rng = np.random.default_rng(1)
    d_matrix["G"]["grade"] = rng.integers(1, 4, size=4)
    f_group = blocks["F"]["G"]
    blocks["F"]["G"] = pd.DataFrame(
        rng.integers(0, 5, size=f_group.shape), index=f_group.index, columns=f_group.columns
    )
    blocks["Z"]["Z1"]["setting"] = rng.integers(0, 5, size=20)
    blocks["Y"]["Y1"]["rating"] = rng.integers(1, 10, size=20)
    assert set(blocks["F"]["G"].dtypes) == {np.dtype("int64")}

    as_float_d = {key: frame.astype(float) for key, frame in d_matrix.items()}
    as_float = {block: {key: frame.astype(float) for key, frame in groups.items()} for block, groups in blocks.items()}

    with_ints = TPLS(2, d_matrix, skip_f_matrix_preprocessing=skip_f).fit(DataFrameDict(copy.deepcopy(blocks)))
    with_floats = TPLS(2, as_float_d, skip_f_matrix_preprocessing=skip_f).fit(DataFrameDict(as_float))

    np.testing.assert_allclose(with_ints.t_scores_super, with_floats.t_scores_super, rtol=1e-12)
    np.testing.assert_allclose(
        with_ints.diagnose(DataFrameDict(blocks)).hat["Y1"],
        with_floats.diagnose(DataFrameDict(as_float)).hat["Y1"],
        rtol=1e-12,
    )


def test_a_text_column_is_still_refused_by_name() -> None:
    """Only float64 and int64 are numbers TPLS can use: a text column is named in the error."""
    d_matrix, blocks = _synthetic()
    blocks["Z"]["Z1"]["operator"] = ["A", "B"] * 10
    with pytest.raises(ValueError, match=r"must be of type float64 or int64\. Bad columns: \['operator'\]"):
        TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(blocks))
