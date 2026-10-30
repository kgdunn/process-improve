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


def test_a_row_with_no_y_value_is_predicted_from_its_materials() -> None:
    """#709: a material not yet tested for anything fitted to 500 iterations and turned every value NaN.

    Each product is one material (F is the identity), and M7 has no Y value at all. Its row
    now carries no Y information (u = 0), so the fit converges, and M7 is predicted from its
    D row through F, as a row with only some Y values missing already was.
    """
    rng = np.random.default_rng(0)
    materials = [f"M{i}" for i in range(8)]
    d = pd.DataFrame(rng.normal(size=(8, 5)), index=materials, columns=[f"p{j}" for j in range(5)])
    f = pd.DataFrame(np.eye(8), index=materials, columns=materials)
    y = pd.DataFrame(rng.normal(size=(8, 2)), index=materials, columns=["y1", "y2"])
    y.loc["M7"] = np.nan
    model = TPLS(n_components=1, d_matrix={"G": d}).fit(DataFrameDict({"F": {"G": f}, "Z": {}, "Y": {"Q": y}}))
    assert model.fitting_statistics["iterations"][0] < model.max_iter
    assert np.isfinite(model.hat["Q"].to_numpy()).all()


def test_a_row_with_no_y_value_shapes_no_weight() -> None:
    """#709: the first component's weights and scores are those of the fit without the row.

    F is passed as given and there is no Z block, so the extra row changes no scaling; Y's
    centre and scale are worked over the observed values only. The row still has a score.
    """
    d_matrix, blocks = _synthetic()
    blocks["Z"] = {}
    blocks["F"]["G"] = MCUVScaler().fit_transform(blocks["F"]["G"])
    gap = copy.deepcopy(blocks)
    gap["Y"]["Y1"].loc["blend19", "y"] = np.nan
    fewer = {name: {key: frame.drop(index="blend19") for key, frame in block.items()} for name, block in blocks.items()}

    with_gap = TPLS(n_components=1, d_matrix=d_matrix, skip_f_matrix_preprocessing=True).fit(DataFrameDict(gap))
    without = TPLS(n_components=1, d_matrix=d_matrix, skip_f_matrix_preprocessing=True).fit(DataFrameDict(fewer))

    np.testing.assert_allclose(with_gap.r_loadings_f["G"], without.r_loadings_f["G"], atol=1e-12)
    np.testing.assert_allclose(with_gap.s_loadings_d["G"], without.s_loadings_d["G"], atol=1e-12)
    np.testing.assert_allclose(with_gap.w_loadings_super, without.w_loadings_super, atol=1e-12)
    shared = without.t_scores_super.index
    np.testing.assert_allclose(with_gap.t_scores_super.loc[shared], without.t_scores_super, atol=1e-12)
    assert np.isfinite(with_gap.t_scores_super.loc["blend19"]).all()


def _without(blocks: dict, row: str) -> dict:
    """Return the blocks with one row dropped from every group."""
    return {name: {key: frame.drop(index=row) for key, frame in block.items()} for name, block in blocks.items()}


def test_a_blend_with_nothing_observed_in_f_scores_zero_and_leaves_the_rest_alone() -> None:
    """#513: a row with no F value gets a zero score, as PCA and PLS give an all-missing row.

    Its regression had an empty denominator, which gave NaN, and the NaN spread to every
    score, so the fit was refused (#709). Now the row has no weight, and the other rows'
    scores and the weights are those of the fit without it. Its Y is the mean of the
    other rows', so the only change to Y's preprocessing is its scale, which no score depends on.
    """
    d_matrix, blocks = _synthetic()
    blocks["Z"] = {}
    fewer = _without(blocks, "blend3")
    gap = copy.deepcopy(blocks)
    gap["F"]["G"].loc["blend3"] = np.nan
    gap["Y"]["Y1"].loc["blend3", "y"] = fewer["Y"]["Y1"]["y"].mean()

    with_gap = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(gap))
    without = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(fewer))

    np.testing.assert_array_equal(with_gap.t_scores_super.loc["blend3"], [0.0, 0.0])
    shared = without.t_scores_super.index
    np.testing.assert_allclose(with_gap.t_scores_super.loc[shared], without.t_scores_super, atol=1e-10)
    np.testing.assert_allclose(with_gap.r_loadings_f["G"], without.r_loadings_f["G"], atol=1e-10)
    np.testing.assert_allclose(with_gap.w_loadings_super, without.w_loadings_super, atol=1e-10)


def test_a_material_with_nothing_observed_in_d_has_a_zero_loading_and_no_weight() -> None:
    """#513: a material not yet tested for anything gets a zero loading in every component.

    It used to make every score NaN, and the fit was refused (#709). Its F column now has
    no part in any score, so every prediction is that of the fit without the material.
    Leaving it out also changes D's block scale, which moves the scores by one factor
    and the Y loadings by its inverse, so the predictions are what can be compared.
    """
    d_matrix, blocks = _synthetic()
    blocks["Z"] = {}
    untested = {"G": d_matrix["G"].copy()}
    untested["G"].loc["mat2"] = np.nan
    fewer_d = {"G": d_matrix["G"].drop(index="mat2")}
    fewer = copy.deepcopy(blocks)
    fewer["F"]["G"] = fewer["F"]["G"].drop(columns="mat2")

    with_it = TPLS(n_components=2, d_matrix=untested).fit(DataFrameDict(copy.deepcopy(blocks)))
    without = TPLS(n_components=2, d_matrix=fewer_d).fit(DataFrameDict(fewer))

    np.testing.assert_array_equal(with_it.r_loadings_f["G"].loc["mat2"], [0.0, 0.0])
    np.testing.assert_allclose(with_it.hat["Y1"], without.hat["Y1"], atol=1e-10)


def test_a_blend_with_nothing_observed_in_z_fits_and_diagnoses_alike() -> None:
    """#513: a row with no Z value has a zero Z score, in the fit and in diagnose alike.

    The fit used to be refused with a message that named no row (#709). Now it converges,
    and diagnose, which already gave such a row a zero Z score, reproduces every fitted score.
    """
    d_matrix, blocks = _synthetic()
    blocks["Z"]["Z1"].loc["blend3"] = np.nan
    model = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(copy.deepcopy(blocks)))
    assert max(model.fitting_statistics["iterations"]) < model.max_iter
    np.testing.assert_allclose(model.diagnose(DataFrameDict(blocks)).t_scores_super, model.t_scores_super, atol=1e-10)


def test_a_block_with_no_variation_still_stops_the_fit_at_once() -> None:
    """#709: a NaN in the iteration never leaves it, so the fit stops instead of running on.

    An empty row no longer causes one (#513), but a Y block with no variation leaves the
    iteration nothing to follow, and that is still refused, with a message that says where to look.
    """
    d_matrix, blocks = _synthetic()
    blocks["Y"]["Y1"]["y"] = 3.0
    with (
        pytest.warns(UserWarning, match="no variance"),
        pytest.raises(
            ValueError,
            match=r"^TPLS cannot fit component 1: its scores are not finite, because a block has no variation",
        ),
    ):
        TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(blocks))


def test_a_constant_f_column_fits_as_an_unobserved_one() -> None:
    """#513: a column excluded for having no variance is absent from the fit, not a column of zeros.

    Scaling turns a constant column into NaN, which is then zeroed, but the presence maps
    were taken from the raw blocks, so its cells still counted in every row's denominator
    and shrank every score. A material used at a fixed fraction in every blend now fits as
    the same material left unobserved does, in fit and in diagnose.
    """
    d_matrix, blocks = _synthetic()
    constant, unobserved = copy.deepcopy(blocks), copy.deepcopy(blocks)
    constant["F"]["G"]["mat3"] = 0.25
    unobserved["F"]["G"]["mat3"] = np.nan

    with pytest.warns(UserWarning, match=r"Block F\[\"G\"\] has columns with no variance.*'mat3'"):
        fixed = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(copy.deepcopy(constant)))
    absent = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(unobserved))

    np.testing.assert_allclose(fixed.t_scores_super, absent.t_scores_super, atol=1e-12)
    np.testing.assert_allclose(fixed.hat["Y1"], absent.hat["Y1"], atol=1e-12)
    np.testing.assert_allclose(fixed.diagnose(DataFrameDict(constant)).t_scores_super, fixed.t_scores_super, atol=1e-10)


def test_diagnose_scores_a_row_with_a_missing_z_cell_as_the_fit_does() -> None:
    """#513: diagnose corrects a Z score for its missing cells, as the fit does.

    The fit divides each row's Z score by the squared weights of the cells it has. diagnose
    computed that same denominator and never divided by it, so the scores of the training
    rows, diagnosed again, differed in exactly the rows with a missing Z cell.
    """
    d_matrix, blocks = _synthetic()
    blocks["Z"]["Z1"].loc["blend3", "z1"] = np.nan
    model = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(copy.deepcopy(blocks)))
    np.testing.assert_allclose(model.diagnose(DataFrameDict(blocks)).t_scores_super, model.t_scores_super, atol=1e-10)


@pytest.mark.parametrize("n_components", [1, 3])
def test_a_fitted_model_works_from_its_fitted_components_whatever_n_components_says(n_components: int) -> None:
    """``set_params(n_components=...)`` without a refit leaves a fitted TPLS as it was (#1028).

    scikit-learn's convention: what a fitted model does depends on its fitted attributes only.
    diagnose, the summary, the T² limit and the ellipse read the constructor's ``n_components``
    instead, so on a two-component fit 3 indexed past the fitted loadings, and 1 failed in
    diagnose and moved the T² limit and the ellipse to a one-component model's.
    """
    d_matrix, blocks = _synthetic()
    data = DataFrameDict(blocks)
    model = TPLS(n_components=2, d_matrix=d_matrix).fit(data)
    result, summary = model.diagnose(data), model.display_results()
    limit, ellipse = model.hotellings_t2_limit(0.99), model.ellipse_coordinates(1, 2)

    model.set_params(n_components=n_components)
    assert model.n_components_ == 2
    again = model.diagnose(data)
    pd.testing.assert_frame_equal(again.t_scores_super, result.t_scores_super)
    pd.testing.assert_frame_equal(again.hotellings_t2, result.hotellings_t2)
    pd.testing.assert_frame_equal(again.hat["Y1"], result.hat["Y1"])
    for block, group in (("F", "G"), ("Z", "Z1")):
        pd.testing.assert_frame_equal(again.spe[block][group], result.spe[block][group])
    assert model.display_results() == summary
    assert model.hotellings_t2_limit(0.99) == limit
    for coordinates, expected in zip(model.ellipse_coordinates(1, 2), ellipse, strict=True):
        np.testing.assert_array_equal(coordinates, expected)
