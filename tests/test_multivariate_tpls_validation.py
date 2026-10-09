"""Input validation for :class:`TPLS`: what a fit accepts, and how it refuses the rest.

Every test here uses a 20-blend, 4-material synthetic set that fits in milliseconds, so
the guards are pinned without loading the pyphi example data.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import TPLS, DataFrameDict, MCUVScaler, make_tpls_scorer


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


def _with_first_column_renamed(block: str, group: str) -> DataFrameDict:
    """Return the synthetic blocks, with the first column of one group renamed to "renamed"."""
    _, blocks = _synthetic()
    frame = blocks[block][group]
    blocks[block][group] = frame.rename(columns={frame.columns[0]: "renamed"})
    return DataFrameDict(blocks)


def _fit(model: TPLS, data: object) -> object:
    return model.fit(data)


def _diagnose(model: TPLS, data: object) -> object:
    return model.diagnose(data)


def _score(model: TPLS, data: object) -> object:
    return make_tpls_scorer()(model, data)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"n_components": 0}, ValueError, r"^n_components must be positive; got 0\.$"),
        ({"d_matrix": [pd.DataFrame()]}, TypeError, r"^d_matrix must be a dict of DataFrames; got list\.$"),
        ({"d_matrix": {"G": np.eye(2)}}, TypeError, r"^d_matrix must contain pandas DataFrames as values\.$"),
        ({"max_iter": 0}, ValueError, r"^max_iter must be positive; got 0\.$"),
    ],
    ids=["no-components", "d-matrix-not-a-dict", "d-matrix-holds-an-array", "no-iterations"],
)
def test_the_constructor_refuses_an_invalid_argument(arguments: dict, error: type[Exception], message: str) -> None:
    """Each argument is checked when the model is built, before any data is seen."""
    with pytest.raises(error, match=message):
        TPLS(**{"n_components": 2, "d_matrix": _synthetic()[0], **arguments})


@pytest.mark.parametrize(
    ("action", "data", "error", "message"),
    [
        (_fit, _synthetic()[1], TypeError, r"^X must be a DataFrameDict; got dict\.$"),
        (_diagnose, _synthetic()[1], TypeError, r"^X must be a DataFrameDict; got dict\.$"),
        (
            _fit,
            DataFrameDict({**_synthetic()[1], "F": {**_synthetic()[1]["F"], "H": _synthetic()[1]["F"]["G"]}}),
            ValueError,
            r"^The keys in F must match the keys in D\.$",
        ),
        (
            _diagnose,
            _with_first_column_renamed("F", "G"),
            ValueError,
            r"^Columns in block F, group \[G\] must match training data column names for each material\.$",
        ),
        (
            _diagnose,
            _with_first_column_renamed("Z", "Z1"),
            ValueError,
            r"^Column names in block Z, group \[Z1\] must match training data column names\.$",
        ),
        (
            _score,
            DataFrameDict({**_synthetic()[1], "Y": {}}),
            ValueError,
            r'^X\["Y"\] must contain at least one block to score a TPLS model\.$',
        ),
    ],
    ids=[
        "fit-a-plain-dict",
        "diagnose-a-plain-dict",
        "fit-an-F-group-that-D-lacks",
        "diagnose-a-renamed-F-column",
        "diagnose-a-renamed-Z-column",
        "score-without-a-Y-block",
    ],
)
def test_malformed_data_is_refused(
    action: Callable[[TPLS, object], object], data: object, error: type[Exception], message: str
) -> None:
    """Each entry point names what is wrong with the data it was handed."""
    d_matrix, blocks = _synthetic()
    model = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(blocks))
    with pytest.raises(error, match=message):
        action(model, data)


def test_skipping_f_preprocessing_uses_the_f_block_as_given() -> None:
    """A pre-scaled F with ``skip_f_matrix_preprocessing`` fits and diagnoses like raw F without it."""
    d_matrix, blocks = _synthetic()
    prescaled = copy.deepcopy(blocks)
    prescaled["F"]["G"] = MCUVScaler().fit_transform(blocks["F"]["G"])

    internal = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(copy.deepcopy(blocks)))
    as_given = TPLS(n_components=2, d_matrix=d_matrix, skip_f_matrix_preprocessing=True)
    as_given.fit(DataFrameDict(copy.deepcopy(prescaled)))

    np.testing.assert_allclose(as_given.t_scores_super, internal.t_scores_super, rtol=1e-10)
    np.testing.assert_allclose(
        as_given.diagnose(DataFrameDict(prescaled)).t_scores_super,
        internal.diagnose(DataFrameDict(blocks)).t_scores_super,
        rtol=1e-10,
    )


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


def test_a_non_string_group_name_fits_when_d_and_f_use_the_same_one() -> None:
    """A group name is a label, so an integer one works as long as D and F agree on it.

    The key check compared ``str(key)`` from D against F while the per-group check and
    the fit itself used the raw key, so no non-string group name could ever fit: matching
    integer keys failed the first check, and a string "1" in F against an integer 1 in D
    failed the second with a message saying F had no group '1'.
    """
    d_matrix, blocks = _synthetic()
    numbered_d = {1: d_matrix["G"]}
    numbered = {**blocks, "F": {1: blocks["F"]["G"]}}

    model = TPLS(n_components=2, d_matrix=numbered_d).fit(DataFrameDict(numbered))
    reference = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(blocks))
    np.testing.assert_allclose(model.t_scores_super, reference.t_scores_super, rtol=1e-12)
    np.testing.assert_allclose(model.diagnose(DataFrameDict(numbered)).hat["Y1"], reference.hat["Y1"], rtol=1e-10)

    # The string "1" is a different label from the integer 1, and is refused as one.
    mismatched = {**blocks, "F": {"1": blocks["F"]["G"]}}
    with pytest.raises(ValueError, match=r"^The keys in F must match the keys in D\.$"):
        TPLS(n_components=2, d_matrix=numbered_d).fit(DataFrameDict(mismatched))
