# (c) Kevin Dunn, 2010-2026. MIT License.
"""Tests for PLS model inversion and the null space (``PLS.invert``).

The reference numbers come from García-Carrión et al., "On the equivalence
between null space and orthogonal space in latent variable regression modeling",
Journal of Chemometrics, 39 (2025): e70057, DOI: 10.1002/cem.70057. Case Study 1
of that paper (a 6x5 simulated dataset) has published direct-inversion and
null-space values that we reproduce here.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import PLS, MCUVScaler


@pytest.fixture
def case_study_1() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paper Case Study 1: X [6x5] and single response y [6x1] (Table 1)."""
    X = pd.DataFrame(
        [
            [5.43, 7.54, 125.64, 58.51, 50.49],
            [5.43, 15.97, 126.20, 258.48, 74.44],
            [99.23, 7.54, 9893.38, 59.29, 737.15],
            [99.23, 15.97, 9765.16, 254.11, 1576.28],
            [52.33, 11.76, 2787.64, 139.21, 538.76],
            [52.33, 11.76, 2849.95, 135.67, 630.73],
        ],
        columns=["x1", "x2", "x3", "x4", "x5"],
    )
    y = pd.DataFrame({"y": [61.85, 278.99, 307.89, 436.4, 266.08, 260.52]})
    return X, y


def test_invert_reproduces_paper_case_study_1(case_study_1: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Direct-inversion score and null-space basis match the paper's Table 3."""
    X, y = case_study_1
    model = PLS(n_components=2).fit(X, y)
    result = model.invert(168.23)

    # Table 3: tau_DI = (-1.4189, -0.2231); the published X are rounded, so a
    # loose tolerance is appropriate.
    assert result.scores.to_numpy() == pytest.approx([-1.4189, -0.2231], abs=5e-3)

    # Table 3: null-space basis G_NS = (-0.1553, 0.9879). A basis vector is only
    # defined up to sign, so compare the absolute direction.
    g_ns = result.null_space_basis.to_numpy().ravel()
    assert np.abs(g_ns) == pytest.approx([0.1553, 0.9879], abs=5e-3)
    assert result.null_space_dimension == 1

    # The prediction at the solution equals the requested target.
    assert float(result.y_hat.iloc[0]) == pytest.approx(168.23, abs=1e-6)


def test_invert_round_trip(case_study_1: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """predict(invert(y_des).x_new) recovers y_des."""
    X, y = case_study_1
    model = PLS(n_components=2).fit(X, y)
    for target in (100.0, 168.23, 400.0):
        result = model.invert(target)
        y_back = float(model.predict(result.x_new.to_frame().T).iloc[0, 0])
        assert y_back == pytest.approx(target, abs=1e-6)


def test_null_space_moves_leave_prediction_unchanged(case_study_1: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Moving along the null space changes the inputs but not the prediction."""
    X, y = case_study_1
    model = PLS(n_components=2).fit(X, y)
    base = model.invert(168.23)
    for coord in (-2.0, 0.5, 3.7):
        moved = model.invert(168.23, null_space_coordinates=[coord])
        # Inputs differ...
        assert not np.allclose(moved.x_new.to_numpy(), base.x_new.to_numpy())
        # ...but the predicted response is identical.
        y_moved = float(model.predict(moved.x_new.to_frame().T).iloc[0, 0])
        assert y_moved == pytest.approx(168.23, abs=1e-6)


def test_invert_accepts_scalar_series_and_array(case_study_1: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A scalar, a 1-element array, and a named Series give the same result."""
    X, y = case_study_1
    model = PLS(n_components=2).fit(X, y)
    from_scalar = model.invert(168.23).x_new
    from_array = model.invert(np.array([168.23])).x_new
    from_series = model.invert(pd.Series({"y": 168.23})).x_new
    assert from_array.to_numpy() == pytest.approx(from_scalar.to_numpy())
    assert from_series.to_numpy() == pytest.approx(from_scalar.to_numpy())


def test_invert_accepts_dataframe_and_dict(case_study_1: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A one-row DataFrame and a dict give the same result as a scalar."""
    X, y = case_study_1
    model = PLS(n_components=2).fit(X, y)
    from_scalar = model.invert(168.23).x_new
    from_frame = model.invert(pd.DataFrame({"y": [168.23]})).x_new
    from_dict = model.invert({"y": 168.23}).x_new
    assert from_frame.to_numpy() == pytest.approx(from_scalar.to_numpy())
    assert from_dict.to_numpy() == pytest.approx(from_scalar.to_numpy())


def test_invert_wrong_target_count_raises(case_study_1: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Passing the wrong number of target values is a clear error."""
    X, y = case_study_1
    model = PLS(n_components=2).fit(X, y)  # single target
    with pytest.raises(ValueError, match="expected 1"):
        model.invert(np.array([1.0, 2.0]))


def test_invert_missing_target_value_raises(case_study_1: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A NaN in the desired response is rejected."""
    X, y = case_study_1
    model = PLS(n_components=2).fit(X, y)
    with pytest.raises(ValueError, match="missing value"):
        model.invert(np.nan)


def test_invert_on_prescaled_model_scale_false(case_study_1: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Inversion works when the model was fit with scale=False on pre-scaled data.

    Exercises the un-scaled code path (no internal X/Y scaler). The desired
    response must be supplied on the same (already-scaled) Y scale the model
    saw, and the round-trip must still hold.
    """
    X, y = case_study_1
    x_scaler = MCUVScaler().fit(X)
    y_scaler = MCUVScaler().fit(y)
    Xs = x_scaler.transform(X)
    ys = y_scaler.transform(y)
    model = PLS(n_components=2, scale=False).fit(Xs, ys)

    target_scaled = float(y_scaler.transform(pd.DataFrame({"y": [168.23]})).iloc[0, 0])
    result = model.invert(target_scaled)
    y_back = float(model.predict(result.x_new.to_frame().T).iloc[0, 0])
    assert y_back == pytest.approx(target_scaled, abs=1e-6)
    assert result.null_space_dimension == 1


def test_invert_reports_hotellings_t2(case_study_1: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The solution's T2 is the sum of squared standardized scores and is finite."""
    X, y = case_study_1
    model = PLS(n_components=2).fit(X, y)
    result = model.invert(168.23)
    expected_t2 = float(
        np.sum((result.scores.to_numpy() / model.scaling_factor_for_scores_.to_numpy()) ** 2)
    )
    assert result.hotellings_t2 == pytest.approx(expected_t2)
    assert result.hotellings_t2 >= 0.0


def test_invert_wrong_coordinate_length_raises(case_study_1: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A mismatched null_space_coordinates length is a clear error."""
    X, y = case_study_1
    model = PLS(n_components=2).fit(X, y)  # null-space dimension is 1
    with pytest.raises(ValueError, match="null-space dimension"):
        model.invert(168.23, null_space_coordinates=[1.0, 2.0])


def test_invert_single_component_has_empty_null_space() -> None:
    """With one component the inversion is unique: a zero-width null space."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((30, 4)), columns=[f"x{i}" for i in range(4)])
    beta = rng.standard_normal((4, 1))
    y = pd.DataFrame(X.to_numpy() @ beta, columns=["y"])
    model = PLS(n_components=1).fit(X, y)
    result = model.invert(1.0)
    assert result.null_space_dimension == 0
    assert result.null_space_basis.shape[1] == 0


def test_invert_cheddar_cheese_holdout() -> None:
    """Design toward the taste of held-out cheeses; round-trip must hold.

    Mirrors the pid-book worked example: train on cheeses 5-30, then invert the
    two-component model toward each of the four held-out cheeses' taste scores.
    """
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
    path = folder / "cheddar-cheese.csv"
    if not path.exists():
        pytest.skip("cheddar-cheese.csv fixture not present")
    cheese = pd.read_csv(path)
    x_cols = ["Acetic", "H2S", "Lactic"]
    train = cheese.iloc[4:]
    hold = cheese.iloc[:4]
    model = PLS(n_components=2).fit(train[x_cols], train[["Taste"]])
    for i in range(len(hold)):
        target = float(hold["Taste"].iloc[i])
        result = model.invert(target)
        assert result.null_space_dimension == 1
        y_back = float(model.predict(result.x_new.to_frame().T).iloc[0, 0])
        assert y_back == pytest.approx(target, abs=1e-6)


def test_invert_solvents_two_response_design() -> None:
    """Design a solvent with a target logP and Solubility (a two-response case).

    The solvents dataset has seven physical properties and two responses (logP
    and Solubility). Inverting toward a desired (logP, Solubility) is a genuine
    multi-response design: the null-space dimension is A - rank(Y), the design
    round-trips to both targets, and moving along the null space leaves both
    predictions unchanged. O-PLS's single-division inversion does not apply here,
    which is why the null-space / orthogonal-space equivalence is single-response
    only.
    """
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
    path = folder / "solvents.csv"
    if not path.exists():
        pytest.skip("solvents.csv fixture not present")
    solvents = pd.read_csv(path).dropna()
    x_cols = ["MeltingPoint", "BoilingPoint", "Dielectric", "DipoleMoment", "RefractiveIndex", "ET30", "Density"]
    y_cols = ["logP", "Solubility"]
    X, Y = solvents[x_cols], solvents[y_cols]

    n_components = 3
    model = PLS(n_components=n_components).fit(X, Y)
    rank_y = np.linalg.matrix_rank((Y - Y.mean()).to_numpy())
    assert rank_y == 2

    desired = pd.Series({"logP": 0.5, "Solubility": 0.0})
    result = model.invert(desired)
    assert result.null_space_dimension == n_components - rank_y  # 3 - 2 = 1

    y_back = model.predict(result.x_new.to_frame().T)
    assert y_back[y_cols].to_numpy().ravel() == pytest.approx(desired.to_numpy(), abs=1e-6)

    # Both target responses are unchanged when moving along the null space.
    moved = model.invert(desired, null_space_coordinates=[2.0])
    y_moved = model.predict(moved.x_new.to_frame().T)
    assert y_moved[y_cols].to_numpy().ravel() == pytest.approx(desired.to_numpy(), abs=1e-6)
    assert not np.allclose(moved.x_new.to_numpy(), result.x_new.to_numpy())


def test_invert_multi_response_null_space_dimension() -> None:
    """On the real LDPE data the null-space dimension is A - rank(Y).

    LDPE has five (collinear) quality variables. Inverting toward a full
    five-target vector gives a null space of dimension A - rank(Y), and the
    reconstructed inputs reproduce the requested targets.
    """
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
    values = pd.read_csv(folder / "LDPE" / "LDPE.csv", index_col=0)
    X = values.iloc[:, :14]
    Y = values.iloc[:, 14:]
    Xs = MCUVScaler().fit_transform(X)
    Ys = MCUVScaler().fit_transform(Y)
    n_components = 6
    model = PLS(n_components=n_components).fit(Xs, Ys)

    # A desired target taken as one of the training rows' responses.
    y_desired = Ys.iloc[0]
    result = model.invert(y_desired)

    rank_y = np.linalg.matrix_rank(Ys.to_numpy())
    assert result.null_space_dimension == n_components - rank_y

    y_back = model.predict(result.x_new.to_frame().T).to_numpy().ravel()
    assert y_back == pytest.approx(y_desired.to_numpy(), abs=1e-6)
