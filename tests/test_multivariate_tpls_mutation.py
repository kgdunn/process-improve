"""Details of TPLS that the nightly mutation run found no test for (#1092).

Each test fits a small synthetic T-shaped data set in milliseconds: 24 blends of two materials of four
lots each, with three properties per lot, under two process conditions, with two quality columns. Three
components, so that the last one is not also the second. Each test fits its own model, so that the
mutation run counts every test as one that reaches the fit.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn

from process_improve.multivariate._limits import ellipse_coordinates, hotellings_t2_limit, spe_calculation
from process_improve.multivariate.methods import TPLS, DataFrameDict, NotEnoughVarianceError

LOTS = {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]}
CONDITIONS = ["temperature", "time"]


def _blocks(seed: int = 2) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
    """Return the D matrix and the F, Z and Y blocks of 24 blends."""
    rng = np.random.default_rng(seed)
    n_blends = 24
    index = pd.Index([f"blend{i}" for i in range(n_blends)], name="blend")
    d_matrix = {
        group: pd.DataFrame(rng.normal(size=(len(lots), 3)), index=lots, columns=["p1", "p2", "p3"])
        for group, lots in LOTS.items()
    }
    formulas = {
        group: pd.DataFrame(rng.dirichlet(np.ones(len(lots)), n_blends), index=index, columns=lots)
        for group, lots in LOTS.items()
    }
    conditions = pd.DataFrame(rng.normal(size=(n_blends, 2)), index=index, columns=CONDITIONS)
    quality = (
        formulas["A"].to_numpy() @ [3.0, 1.0, 2.0, 0.0]
        + formulas["B"].to_numpy() @ [1.0, 2.0, 0.5, 1.5]
        + 0.5 * conditions["temperature"].to_numpy()
    )
    y = pd.DataFrame(
        {"q1": quality + 0.1 * rng.normal(size=n_blends), "q2": 2.0 - quality + 0.1 * rng.normal(size=n_blends)},
        index=index,
    )
    return d_matrix, {"F": formulas, "Z": {"Conditions": conditions}, "Y": {"Quality": y}}


def _fitted() -> tuple[TPLS, dict[str, dict[str, pd.DataFrame]]]:
    """Return a three-component model of the 24 blends, and its blocks."""
    d_matrix, blocks = _blocks()
    return TPLS(n_components=3, d_matrix=d_matrix).fit(DataFrameDict(blocks)), blocks


def _inputs(blocks: dict[str, dict[str, pd.DataFrame]], rows: slice) -> DataFrameDict:
    """Return the F and Z blocks of some of the blends, as diagnose reads them."""
    return DataFrameDict({block: {g: frame.iloc[rows] for g, frame in blocks[block].items()} for block in ("F", "Z")})


# --------------------------------------------------------------------------- fitted or not, and what fit refuses
def test_a_model_never_fitted_says_false_as_a_bool() -> None:
    """``__sklearn_is_fitted__`` is annotated ``bool``: an unfitted model gives ``False``, not another falsy value."""
    assert TPLS(n_components=1, d_matrix={}).__sklearn_is_fitted__() is False


def test_a_d_matrix_that_is_not_a_dict_is_named_by_its_type_when_sklearn_skips_its_checks() -> None:
    """Scikit-learn refuses a list for ``d_matrix`` first, unless told to skip its checks: fit then names its type."""
    _, blocks = _blocks()
    with (
        sklearn.config_context(skip_parameter_validation=True),
        pytest.raises(TypeError, match=r"^d_matrix must be a dict of DataFrames; got list\.$"),
    ):
        TPLS(n_components=1, d_matrix=[]).fit(DataFrameDict(blocks))  # type: ignore[arg-type]


def test_a_blend_made_only_of_a_lot_with_no_observed_property_is_refused_at_the_component_it_stops() -> None:
    """Such a blend has no F score, as its only lot weighs 0: the super scores are not finite, and the fit says where.

    The F weights are finite, so it is the check of the super scores that refuses it, naming component 1.
    """
    d_matrix, blocks = _blocks()
    d_matrix["A"].loc["A4"] = np.nan
    blocks["F"]["A"].iloc[0] = [np.nan, np.nan, np.nan, 1.0]
    blocks["F"]["B"].iloc[0] = np.nan
    message = (
        "TPLS cannot fit component 1: its scores are not finite, because a block has no variation for the "
        "iteration to follow. Check that each group of F covaries with the first Y block, or fit fewer components."
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the inner PLS of a NaN score overflows on its way
        with pytest.raises(NotEnoughVarianceError, match=f"^{re.escape(message)}$"):
            TPLS(n_components=3, d_matrix=d_matrix).fit(DataFrameDict(blocks))


# --------------------------------------------------------------------------- the T² ellipse
@pytest.mark.parametrize("judged_rows", ["fitted", "new"])
def test_the_ellipse_judges_the_rows_it_is_asked_to(judged_rows: str) -> None:
    """``judged_rows`` reaches the limit the ellipse is drawn at, and silences the FutureWarning of the default."""
    model, _ = _fitted()
    with pytest.warns(FutureWarning):
        default = model.ellipse_coordinates(1, 2, conf_level=0.9)
    expected = ellipse_coordinates(
        1, 2, 0.9, 100, 3, model.scaling_factor_for_scores, n_rows=24, judged_rows=judged_rows
    )

    coordinates = model.ellipse_coordinates(1, 2, conf_level=0.9, judged_rows=judged_rows)

    for got, want, before in zip(coordinates, expected, default, strict=True):
        np.testing.assert_array_equal(got, want)
        assert np.array_equal(got, before) == (judged_rows == "new")


# --------------------------------------------------------------------------- diagnose
def test_diagnose_names_tpls_when_its_deprecated_t2_is_read() -> None:
    model, blocks = _fitted()
    out = model.diagnose(_inputs(blocks, slice(0, 3)))
    with pytest.warns(
        DeprecationWarning,
        match=r"^process_improve\.multivariate\.TPLS\.diagnose\(\.\.\.\)\.hotellings_t2 is deprecated since 2\.4\.0 ",
    ):
        _ = out.hotellings_t2


def test_diagnose_gives_the_cumulative_t2_of_the_fit_for_the_rows_it_was_fitted_to() -> None:
    model, blocks = _fitted()

    t2 = model.diagnose(_inputs(blocks, slice(None))).cumulative_hotellings_t2

    pd.testing.assert_frame_equal(t2, model.hotellings_t2, rtol=1e-8)


def test_diagnose_returns_each_groups_residuals_whose_length_is_its_spe() -> None:
    """For every F and Z group: a frame on the rows and columns of the input, its row lengths the last SPE."""
    model, blocks = _fitted()
    rows = _inputs(blocks, slice(3, 9))

    out = model.diagnose(rows)

    assert list(out.residuals) == ["F", "Z"]
    for block, groups in (("F", LOTS), ("Z", {"Conditions": CONDITIONS})):
        assert list(out.residuals[block]) == list(groups)
        for group, columns in groups.items():
            residual = out.residuals[block][group]
            pd.testing.assert_index_equal(residual.index, rows[block][group].index)
            assert list(residual.columns) == columns
            np.testing.assert_allclose(
                np.linalg.norm(residual.to_numpy(), axis=1), out.spe[block][group].iloc[:, -1], rtol=1e-12
            )
            assert (residual.abs().to_numpy() > 0).any()


# --------------------------------------------------------------------------- what optimal inversion reads
def test_the_inversion_probe_names_the_lots_then_the_conditions() -> None:
    model, _ = _fitted()

    probe = model._inversion_probe(0.9)

    assert model._inversion_columns() == [
        ("F", "A", LOTS["A"]),
        ("F", "B", LOTS["B"]),
        ("Z", "Conditions", CONDITIONS),
    ]
    assert probe.names == (*LOTS["A"], *LOTS["B"], *CONDITIONS)
    assert probe.y_names == ("q1", "q2")
    assert probe.groups == {"A": tuple(LOTS["A"]), "B": tuple(LOTS["B"])}
    assert probe.block_names == ("F/A", "F/B", "Z/Conditions")


def test_the_inversion_probe_carries_the_limits_and_spreads_at_its_confidence_level() -> None:
    """The limits are those at the level asked for; the T² limit is the one for new rows, as an input is, unasked."""
    model, blocks = _fitted()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        probe = model._inversion_probe(0.9)

    spe = [model.spe["F"]["A"], model.spe["F"]["B"], model.spe["Z"]["Conditions"]]
    assert probe.block_limits == tuple(float(spe_calculation(s.to_numpy(), 0.9)) for s in spe)
    assert probe.t2_limit == hotellings_t2_limit(0.9, 3, 24, judged_rows="new")
    np.testing.assert_array_equal(probe.score_sd, model.scaling_factor_for_scores.to_numpy())
    inputs = pd.concat([*blocks["F"].values(), blocks["Z"]["Conditions"]], axis=1)
    np.testing.assert_allclose(probe.centre, inputs.mean(), rtol=1e-12)
    np.testing.assert_allclose(probe.spread, inputs.std(), rtol=1e-12)
    np.testing.assert_allclose(probe.y_scale, blocks["Y"]["Quality"].std(), rtol=1e-12)


def test_the_inversion_probe_reads_0_for_a_centre_and_1_for_a_spread_a_column_does_not_have() -> None:
    """A lot never used has no centre and no spread, and a constant column no spread: the probe reads 0 and 1."""
    d_matrix, blocks = _blocks()
    blocks["F"]["B"]["B4"] = np.nan
    blocks["Z"]["Conditions"]["time"] = 5.0
    blocks["Y"]["Quality"]["q3"] = 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = TPLS(n_components=3, d_matrix=d_matrix).fit(DataFrameDict(blocks))
    assert [str(warning.message).split(" has ")[0] for warning in caught] == [
        'Block F["B"]',
        'Block Y["Quality"]',
        'Block Z["Conditions"]',
    ]

    probe = model._inversion_probe(0.9)

    inputs = pd.concat([*blocks["F"].values(), blocks["Z"]["Conditions"]], axis=1)
    centre, spread = inputs.mean().to_numpy(), inputs.std().to_numpy()
    assert np.isnan(centre).sum() == 1
    assert (np.isnan(spread) | (spread == 0)).sum() == 2
    np.testing.assert_allclose(probe.centre, np.where(np.isnan(centre), 0.0, centre), rtol=1e-12)
    np.testing.assert_allclose(probe.spread, np.where(spread > 0, spread, 1.0), rtol=1e-12)
    y_spread = blocks["Y"]["Quality"].std().to_numpy()
    np.testing.assert_allclose(probe.y_scale, np.where(y_spread > 0, y_spread, 1.0), rtol=1e-12)


def test_the_inversion_probe_reads_each_lots_centre_and_spread_by_its_name() -> None:
    """F may list a material's lots in another order than D does: the probe reads each lot's centre by its name."""
    d_matrix, blocks = _blocks()
    blocks["F"]["A"] = blocks["F"]["A"][LOTS["A"][::-1]]
    model = TPLS(n_components=3, d_matrix=d_matrix).fit(DataFrameDict(blocks))

    probe = model._inversion_probe(0.9)

    assert probe.names[:4] == tuple(LOTS["A"])
    inputs = pd.concat([*blocks["F"].values(), blocks["Z"]["Conditions"]], axis=1)[list(probe.names)]
    np.testing.assert_allclose(probe.centre, inputs.mean(), rtol=1e-12)
    np.testing.assert_allclose(probe.spread, inputs.std(), rtol=1e-12)


def test_the_inversion_probe_diagnoses_rows_of_its_inputs_as_the_model_does() -> None:
    model, blocks = _fitted()
    rows = _inputs(blocks, slice(2, 7))
    out = model.diagnose(rows)
    groups = [("F", "A"), ("F", "B"), ("Z", "Conditions")]

    scores, y_hat, residuals, t2, spe = model._inversion_probe(0.9).diagnose(
        np.hstack([rows[block][group].to_numpy() for block, group in groups])
    )

    np.testing.assert_array_equal(scores, out.t_scores_super)
    np.testing.assert_array_equal(y_hat, out.hat["Quality"])
    assert len(residuals) == len(groups)
    for residual, (block, group) in zip(residuals, groups, strict=True):
        np.testing.assert_array_equal(residual, out.residuals[block][group])
    np.testing.assert_array_equal(t2, out.cumulative_hotellings_t2.iloc[:, -1])
    np.testing.assert_array_equal(spe, np.column_stack([out.spe[block][group].iloc[:, -1] for block, group in groups]))


def test_the_inversion_probe_refuses_two_groups_that_share_a_name() -> None:
    d_matrix, blocks = _blocks()
    d_matrix["B"] = d_matrix["B"].rename(index={"B1": "A1"})
    blocks["F"]["B"] = blocks["F"]["B"].rename(columns={"B1": "A1"})
    model = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(blocks))

    message = "Two TPLS groups share a lot or condition name; optimal inversion needs every input named once."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        model._inversion_probe(0.95)
