"""Input validation for :class:`TPLS`: what a fit accepts, and how it refuses the rest.

Every test here uses a 20-blend, 4-material synthetic set that fits in milliseconds, so
the guards are pinned without loading the pyphi example data.
"""

from __future__ import annotations

import copy
import re
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from process_improve.multivariate.methods import TPLS, DataFrameDict, MCUVScaler, epsqrt, make_tpls_scorer


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


def _score_given_y(model: TPLS, data: object, y: object) -> object:
    # The scorer is built inside the test, so the mutation run counts the test as one that reaches it.
    return make_tpls_scorer()(model, data, y)


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
            _fit,
            DataFrameDict({**_synthetic()[1], "Y": {**_synthetic()[1]["Y"], "Y2": _synthetic(1)[1]["Y"]["Y1"]}}),
            ValueError,
            r"^TPLS fits one Y block; got \['Y1', 'Y2'\]\.$",
        ),
        (
            _fit,
            DataFrameDict({**_synthetic()[1], "Y": {}}),
            ValueError,
            r"^TPLS fits one Y block; got \[\]\.$",
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
        "fit-two-Y-blocks",
        "fit-no-Y-block",
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
            match=(
                r"^TPLS cannot fit component 1: its scores are not finite, because a block has no variation for "
                r"the iteration to follow\. Check that at least one column of the first Y block, and at least one "
                r"column of each D and F group, varies and is observed\.$"
            ),
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
    with pytest.warns(UserWarning, match=r"Block F\[\"G\"\] has columns with no observed values.*'mat3'"):
        absent = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(unobserved))

    np.testing.assert_allclose(fixed.t_scores_super, absent.t_scores_super, atol=1e-12)
    np.testing.assert_allclose(fixed.hat["Y1"], absent.hat["Y1"], atol=1e-12)
    np.testing.assert_allclose(fixed.diagnose(DataFrameDict(constant)).t_scores_super, fixed.t_scores_super, atol=1e-10)


@pytest.mark.parametrize(
    ("block", "group", "column"),
    [("D", "G", "r"), ("F", "G", "mat3"), ("Z", "Z1", "z2"), ("Y", "Y1", "y_unlogged")],
)
def test_an_all_nan_column_is_named_in_a_warning_as_a_constant_one_is(block: str, group: str, column: str) -> None:
    """#982: a column with no observed value is excluded from the fit with a warning that names it.

    A constant column got that warning (#513); an all-NaN column, which carries no more
    information, used to be left out without a word.
    """
    d_matrix, blocks = _synthetic()
    frame = d_matrix[group] if block == "D" else blocks[block][group]
    frame[column] = np.nan
    pattern = rf"Block {block}\[\"{group}\"\] has columns with no observed values.*'{column}'"
    with pytest.warns(UserWarning, match=pattern):
        TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(blocks))


def test_a_column_with_one_observed_value_is_named_as_having_no_variance() -> None:
    """#982: a single observed value has a NaN standard deviation, which was also excluded silently."""
    d_matrix, blocks = _synthetic()
    blocks["Z"]["Z1"]["z2"] = np.nan
    blocks["Z"]["Z1"].loc["blend0", "z2"] = 1.0
    with pytest.warns(UserWarning, match=r"Block Z\[\"Z1\"\] has columns with no variance.*'z2'"):
        TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(blocks))


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


def test_a_fitted_model_keeps_the_settings_it_was_fitted_with() -> None:
    """``set_params`` without a refit changes neither the summary nor diagnose (#1028).

    display_results read the constructor's ``max_iter`` after fit, both to flag a component
    that ran out of iterations and to print the setting, and diagnose and the summary read
    ``skip_f_matrix_preprocessing``: a model fitted with ``max_iter=1`` lost its flags once
    ``max_iter`` was raised, and diagnose stopped scaling F once the skip was switched on.
    """
    d_matrix, blocks = _synthetic()
    data = DataFrameDict(blocks)
    capped = TPLS(n_components=2, d_matrix=d_matrix, max_iter=1).fit(data)
    summary = capped.display_results()
    assert summary.count("** (max iter reached)") == 2
    assert "\nmax_iter: 1\nskip_f_matrix_preprocessing: False\n" in summary
    result = capped.diagnose(data)

    capped.set_params(max_iter=500, skip_f_matrix_preprocessing=True)
    assert capped.display_results() == summary
    again = capped.diagnose(data)
    pd.testing.assert_frame_equal(again.t_scores_super, result.t_scores_super)
    pd.testing.assert_frame_equal(again.hat["Y1"], result.hat["Y1"])
    pd.testing.assert_frame_equal(again.spe["F"]["G"], result.spe["F"]["G"])


def test_the_defaults_are_those_of_the_other_iterative_estimators() -> None:
    """max_iter is 500 and tol epsqrt, as on PCA, PLS, MBPCA and MBPLS; F is centred and scaled by default."""
    d_matrix, _ = _synthetic()
    assert TPLS(2, d_matrix).get_params() == {
        "n_components": 2,
        "d_matrix": d_matrix,
        "max_iter": 500,
        "skip_f_matrix_preprocessing": False,
        "tol": epsqrt,
    }


def test_a_model_never_fitted_says_so() -> None:
    """Every method that needs a fit raises NotFittedError before one, as PCA's and PLS's do.

    ``__init__`` sets attributes that end in an underscore, so scikit-learn's check_is_fitted
    passed on a model never fitted, and diagnose failed later with an AttributeError.
    """
    d_matrix, blocks = _synthetic()
    model = TPLS(n_components=2, d_matrix=d_matrix)
    assert model.is_fitted_ is False
    data = DataFrameDict(blocks)
    for call in (model.diagnose, model.score, lambda x: make_tpls_scorer()(model, x), lambda _: model.vip()):
        with pytest.raises(NotFittedError):
            call(data)
    with pytest.raises(NotFittedError):
        _ = model.d_block_scaling_
    with pytest.raises(RuntimeError, match=r"^The model is not fitted yet\. Please call `fit` first\.$"):
        model.display_results()


def test_a_fit_that_fails_leaves_the_model_unfitted() -> None:
    """A refit that fails part way does not leave the previous fit standing with some of its parts replaced."""
    d_matrix, blocks = _synthetic()
    model = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(copy.deepcopy(blocks)))
    blocks["Y"]["Y1"]["y"] = 3.0
    with pytest.warns(UserWarning, match="no variance"), pytest.raises(ValueError, match="cannot fit component 1"):
        model.fit(DataFrameDict(blocks))
    for call in (model.diagnose, model.score, lambda _: model.vip(), lambda _: model.d_block_scaling_):
        with pytest.raises(NotFittedError):
            call(DataFrameDict(blocks))


def test_an_infinite_value_is_refused() -> None:
    """A missing value is allowed in any block, an infinite one in none."""
    d_matrix, blocks = _synthetic()
    blocks["Z"]["Z1"].iloc[2, 0] = np.inf
    with pytest.raises(ValueError, match="Input contains infinity"):
        TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(blocks))


def test_a_column_is_left_out_only_when_its_values_are_all_equal() -> None:
    """Each column is centred on its mean and scaled by its sd (ddof=1); one of equal values is left out.

    [-epsqrt, 0, epsqrt] has a mean of exactly 0 and an sd of exactly epsqrt (2**-26), and
    [5, 5, 5 + 1e-9] one of 1e-9 / sqrt(3): both vary, so both stay, in any units. Their sds
    were below the floor of epsqrt the rule used to be, which left out 'c', and 'kept' would
    have gone in units half as large. [0.1, 0.1, 0.1] has an sd of rounding noise, not 0, and
    one observed value is no spread either: both are left out. A block of one row has no sd:
    its scale is 1.
    """
    model = TPLS(n_components=1, d_matrix=_synthetic()[0])
    block = pd.DataFrame(
        {
            "kept": [-epsqrt, 0.0, epsqrt],
            "a": [2.0, 2.0, 2.0],
            "b": [1.0, 4.0, 7.0],
            "c": [5.0, 5.0, 5.0 + 1e-9],
            "d": [0.1, 0.1, 0.1],
            "e": [np.nan, 3.0, np.nan],
        },
        index=["r0", "r1", "r2"],
    )
    assert block["d"].std() > 0
    message = (
        r'^Block Z\["Z1"\] has columns with no variance, which carry no information and are excluded from the '
        r"model: 'a', 'd', 'e'\. Drop them, or check the data for a stuck sensor\.$"
    )
    with pytest.warns(UserWarning, match=message):
        centre, scale = model._learn_center_and_scaling_parameters(block, 'Z["Z1"]')
    pd.testing.assert_series_equal(centre, block.mean())
    pd.testing.assert_series_equal(
        scale,
        pd.Series([epsqrt, np.nan, 3.0, np.std([5.0, 5.0, 5.0 + 1e-9], ddof=1), np.nan, np.nan], index=block.columns),
    )

    one_row = block.iloc[[1]]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        centre, scale = model._learn_center_and_scaling_parameters(one_row, "Y")
    pd.testing.assert_series_equal(centre, one_row.iloc[0].rename(None))
    pd.testing.assert_series_equal(scale, pd.Series(1.0, index=block.columns))

    two_rows = block.iloc[[0, 2]][["kept", "a", "b"]]
    with pytest.warns(UserWarning, match=r"no variance.*: 'a'\. Drop"):
        _, scale = model._learn_center_and_scaling_parameters(two_rows, "Y")
    pd.testing.assert_series_equal(
        scale, pd.Series([np.sqrt(2) * epsqrt, np.nan, np.sqrt(18.0)], index=["kept", "a", "b"])
    )


def test_a_column_in_small_units_fits_as_it_does_in_large_ones() -> None:
    """Scaled to unit variance, the column is the same in any units (#1029).

    A Z column 1e9 times smaller has an sd near 1e-9, below the floor of epsqrt that marked a
    column as having no variance: it was left out with a warning, and the scores moved by 4.
    """
    d_matrix, blocks = _synthetic()
    small = copy.deepcopy(blocks)
    small["Z"]["Z1"]["z1"] *= 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(small))
    reference = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(copy.deepcopy(blocks)))
    np.testing.assert_allclose(model.t_scores_super, reference.t_scores_super, rtol=1e-9, atol=1e-9)


def test_a_column_with_nothing_observed_is_named_on_its_own() -> None:
    """A column with no observed value gets its own warning, and only that one; it is left out (#982).

    A column with one observed value has no sd (NaN), so it is named as having no variance. In
    a block of one row every scale is 1, except that of a column with nothing in it, which is NaN.
    """
    model = TPLS(n_components=1, d_matrix=_synthetic()[0])
    block = pd.DataFrame(
        {"never": [np.nan] * 3, "b": [1.0, 4.0, 7.0], "once": [np.nan, 3.0, np.nan], "a": [2.0, 2.0, 2.0]},
        index=["r0", "r1", "r2"],
    )
    never = (
        "has columns with no observed values, which carry no information and are excluded from the model: "
        "'never'. Drop them, or check the data for a sensor that was never logged."
    )
    no_variance = (
        "has columns with no variance, which carry no information and are excluded from the model: "
        "'once', 'a'. Drop them, or check the data for a stuck sensor."
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        centre, scale = model._learn_center_and_scaling_parameters(block, 'F["G"]')
    assert [(w.category, str(w.message)) for w in caught] == [
        (UserWarning, f'Block F["G"] {never}'),
        (UserWarning, f'Block F["G"] {no_variance}'),
    ]
    pd.testing.assert_series_equal(centre, pd.Series([np.nan, 4.0, 3.0, 2.0], index=block.columns))
    pd.testing.assert_series_equal(scale, pd.Series([np.nan, 3.0, np.nan, np.nan], index=block.columns))

    one_row = block.iloc[[1]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        centre, scale = model._learn_center_and_scaling_parameters(one_row, "Y")
    assert [str(w.message) for w in caught] == [f"Block Y {never}"]
    pd.testing.assert_series_equal(centre, one_row.iloc[0].rename(None))
    pd.testing.assert_series_equal(scale, pd.Series([np.nan, 1.0, 1.0, 1.0], index=block.columns))


@pytest.mark.parametrize(
    ("action", "where"),
    [
        (lambda model, data, y: model.fit(data, y), "TPLS.fit"),
        (lambda model, data, y: model.score(data, y), "TPLS.score"),
        (_score_given_y, "A TPLS scorer"),
    ],
    ids=["fit", "score", "scorer"],
)
def test_a_separate_y_is_refused_in_full(action: Callable, where: str) -> None:
    """The response travels in X["Y"]; a separate y is refused, and the message says what to do instead."""
    d_matrix, blocks = _synthetic()
    data = DataFrameDict(blocks)
    model = TPLS(n_components=1, d_matrix=d_matrix).fit(data)
    message = (
        f'{where} does not take a separate `y`: a T-shaped model carries its response inside X["Y"]. Put the '
        'response in the "Y" block of the DataFrameDict and leave `y` as None. For cross-validation with a named '
        "metric, pass scoring=make_tpls_scorer('r2') rather than scoring='r2'."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        action(model, data, blocks["Y"]["Y1"])


def test_one_blend_is_refused() -> None:
    """Its Y column centres to 0, so the first regression on it is 0 / 0: no component can be fitted."""
    d_matrix, blocks = _synthetic()
    one = {name: {key: frame.iloc[:1] for key, frame in block.items()} for name, block in blocks.items()}
    with pytest.raises(ValueError, match=r"^TPLS cannot fit component 1: its scores are not finite"):
        TPLS(n_components=1, d_matrix=d_matrix).fit(DataFrameDict(one))


def test_an_unknown_metric_and_an_empty_y_are_refused_in_full() -> None:
    """make_tpls_scorer names the metrics it knows; score names what it needs."""
    message = (
        "Unknown metric 'accuracy' for make_tpls_scorer. Choose one of ['neg_mean_absolute_error', "
        "'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2'], or pass a callable metric(y_true, y_pred)."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        make_tpls_scorer("accuracy")
    d_matrix, blocks = _synthetic()
    model = TPLS(n_components=1, d_matrix=d_matrix).fit(DataFrameDict(blocks))
    message = "y_actual is empty: X['Y'] must contain at least one block to compute a score."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        model.score(DataFrameDict({**blocks, "Y": {}}))


def _with_a_second_z_group() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
    """Return the synthetic blocks with a second Z group, "Z2", of three conditions."""
    d_matrix, blocks = _synthetic()
    index = blocks["F"]["G"].index
    values = np.random.default_rng(5).normal(size=(len(index), 3))
    blocks["Z"]["Z2"] = pd.DataFrame(values, index=index, columns=["a", "b", "c"])
    return d_matrix, blocks


def test_several_z_groups_fit_and_diagnose_alike() -> None:
    """#981: each Z group is its own block in the super-level, in the fit and in diagnose alike.

    The fit gives each Z group its own weights and its own block score, so the super weights
    have one row per Z group, then one for F. They were labelled for a single Z group, so a
    second one raised a shape error, and diagnose pooled every Z group into one score.
    """
    d_matrix, blocks = _with_a_second_z_group()
    blocks["Z"]["Z2"].loc["blend4", "a"] = np.nan
    blocks["Z"]["Z2"].loc["blend7"] = np.nan
    model = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(copy.deepcopy(blocks)))

    assert model.w_loadings_super.index.to_list() == ['Z["Z1"]', 'Z["Z2"]', "F"]
    result = model.diagnose(DataFrameDict(blocks))
    np.testing.assert_allclose(result.t_scores_super, model.t_scores_super, atol=1e-10)
    np.testing.assert_allclose(result.hat["Y1"], model.hat["Y1"], atol=1e-10)


def test_the_order_of_the_z_groups_does_not_change_the_fit() -> None:
    """#981: the Z groups enter the super-level as a set, so listing them in another order changes nothing."""
    d_matrix, blocks = _with_a_second_z_group()
    swapped = copy.deepcopy(blocks)
    swapped["Z"] = {"Z2": swapped["Z"]["Z2"], "Z1": swapped["Z"]["Z1"]}
    model = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(blocks))
    other = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(swapped))

    np.testing.assert_allclose(other.t_scores_super, model.t_scores_super, atol=1e-10)
    np.testing.assert_allclose(other.hat["Y1"], model.hat["Y1"], atol=1e-10)
    np.testing.assert_allclose(other.w_loadings_super.loc[model.w_loadings_super.index], model.w_loadings_super)
