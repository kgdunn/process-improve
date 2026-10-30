# (c) Kevin Dunn, 2010-2026. MIT License.

import copy
import inspect
import io
import itertools
import logging
import pathlib
import re
import time
import urllib.request
import warnings
from collections.abc import Callable
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist
from sklearn import config_context
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    LeaveOneOut,
    PredefinedSplit,
    RepeatedKFold,
    ShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

from process_improve.multivariate._common import _nz
from process_improve.multivariate._cv_criteria import _CompareSettings, _PseudoValidationSettings
from process_improve.multivariate._diagnostics import cumulative_block_t2
from process_improve.multivariate._impute import impute_low_rank
from process_improve.multivariate._pca import _leverage_corrected_press, _low_rank_reconstruction, _pca_ekf_press
from process_improve.multivariate._pls import _direct_weights, _held_out_scores
from process_improve.multivariate._projection import operator_for_pattern, project_rows
from process_improve.multivariate.methods import (
    MBPCA,
    MBPLS,
    OPLS,
    PCA,
    PLS,
    PRM,
    TPLS,
    DataFrameDict,
    MCUVScaler,
    NotEnoughVarianceError,
    SpecificationWarning,
    UncentredDataWarning,
    center,
    coefficient_plot,
    compare_cv_criteria,
    correlation_loadings_plot,
    eigenvalue_summary,
    ellipse_coordinates,
    epsqrt,
    explained_variance_plot,
    group_contributions,
    hotellings_t2_limit,
    internal_pls_nipals_fit_one_pc,
    make_tpls_scorer,
    nan_to_zeros,
    observation_contributions,
    predictions_vs_observed_plot,
    project_variables,
    pseudo_validation_set,
    quick_regress,
    regress_a_space_on_b_row,
    rv2_coefficient,
    rv_coefficient,
    scale,
    score_contributions,
    score_limit,
    selectivity_ratio,
    spe_calculation,
    spe_contributions,
    squared_cosine,
    ssq,
    t2_contributions,
    target_projection,
    terminate_check,
    vip,
)
from process_improve.multivariate.plots import _x_space_loadings

pd.options.plotting.backend = "plotly"
pd.options.display.max_columns = 20
pd.options.display.width = 200
pio.renderers.default = "browser"


def test_nan_to_zeros() -> None:
    """Test the `nan_to_zeros` function."""
    in_array = np.array([[1, 2, np.nan], [4, 5, 6], [float("nan"), 8, 9]])
    out_array = nan_to_zeros(in_array)
    assert pytest.approx(out_array) == np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]])


def test_terminate_check_off_by_one_fixed() -> None:
    """SEC-33 (#282) sub-item 1: the NIPALS terminate_check uses ``>=`` so the
    iterative algorithm runs exactly ``md_max_iter`` iterations rather than
    ``md_max_iter + 1``.
    """

    # Unconverged scores: the relative change ||diff|| / ||t_a|| is ~4.6e-4,
    # far above md_tol, so the tolerance branch is False.
    t_a_guess = np.array([1.0, 2.0, 3.0])
    t_a = np.array([1.001, 2.001, 3.001])
    settings = {"md_tol": 1e-9, "md_max_iter": 5}

    # iterations < md_max_iter -> not terminated.
    assert terminate_check(t_a_guess, t_a, iterations=4, settings=settings) is False
    # iterations == md_max_iter -> terminated (post-fix).
    assert terminate_check(t_a_guess, t_a, iterations=5, settings=settings) is True


def test_terminate_check_is_scale_relative() -> None:
    """#504: terminate_check compares the successive-iteration change relative
    to the norm of the current score vector, so the convergence decision is
    invariant to a global rescaling of the vectors.
    """
    settings = {"md_tol": 1e-6, "md_max_iter": 500}

    # A relative change of 1e-3 in each element: not converged at md_tol=1e-6,
    # at any magnitude. The old absolute criterion declared the tiny pair
    # (scale 1e-12) converged instantly.
    base_guess = np.array([1.0, 2.0, 3.0])
    base_new = base_guess * (1 + 1e-3)
    for magnitude in (1e-12, 1.0, 1e9):
        assert terminate_check(magnitude * base_guess, magnitude * base_new, iterations=1, settings=settings) is False

    # A relative change of 1e-9: converged at md_tol=1e-6, at any magnitude.
    # The old absolute criterion could never reach 1e-6 at scale 1e9 (the
    # absolute gap is ~3.7), so it burned all md_max_iter iterations there.
    tiny_new = base_guess * (1 + 1e-9)
    for magnitude in (1e-12, 1.0, 1e9):
        assert terminate_check(magnitude * base_guess, magnitude * tiny_new, iterations=1, settings=settings) is True

    # A fully-deflated, all-zero score vector: the floored denominator reports
    # convergence instead of dividing by zero.
    zeros = np.zeros(3)
    assert terminate_check(zeros, zeros, iterations=1, settings=settings) is True


def test_terminate_check_needs_the_change_strictly_below_md_tol() -> None:
    """Convergence is a relative change *below* ``md_tol``; a change equal to it is not enough.

    ``||(1.5, 0) - (1, 0)|| / ||(1, 0)||`` is exactly 0.5 in binary floating point.
    """
    settings = {"md_tol": 0.5, "md_max_iter": 10}
    assert terminate_check(np.array([1.5, 0.0]), np.array([1.0, 0.0]), iterations=1, settings=settings) is False
    assert terminate_check(np.array([1.25, 0.0]), np.array([1.0, 0.0]), iterations=1, settings=settings) is True


def test_nipals_fit_is_scale_invariant() -> None:
    """#504: fitting on ``X`` and on ``1000 * X`` takes the same number of
    NIPALS iterations, and gives identical loadings and scores up to the
    scale factor, for both PCA and PLS.
    """
    rng = np.random.default_rng(42)
    N, K, M, A = 40, 6, 2, 3
    t_true = rng.standard_normal((N, 3))
    p_true = rng.standard_normal((3, K))
    X = pd.DataFrame(t_true @ p_true + 0.1 * rng.standard_normal((N, K)))
    X = X - X.mean()  # mean-centre only: no unit-variance scaling, on purpose
    Y = pd.DataFrame(X.values @ rng.standard_normal((K, M)) + 0.1 * rng.standard_normal((N, M)))
    Y = Y - Y.mean()
    factor = 1000.0

    pca_base = PCA(n_components=A, algorithm="nipals").fit(X)
    pca_scaled = PCA(n_components=A, algorithm="nipals").fit(X * factor)
    assert pca_base.fitting_info_["iterations"] == pytest.approx(pca_scaled.fitting_info_["iterations"])
    assert pca_scaled.loadings_.values == pytest.approx(pca_base.loadings_.values, rel=1e-8, abs=1e-10)
    assert pca_scaled.scores_.values == pytest.approx(factor * pca_base.scores_.values, rel=1e-8)

    # PLS with scale=False so the rescale actually reaches the NIPALS loop.
    pls_base = PLS(n_components=A, scale=False).fit(X, Y)
    pls_scaled = PLS(n_components=A, scale=False).fit(X * factor, Y * factor)
    assert pls_base.fitting_info_["iterations"] == pytest.approx(pls_scaled.fitting_info_["iterations"])
    assert pls_scaled.x_loadings_.values == pytest.approx(pls_base.x_loadings_.values, rel=1e-8, abs=1e-10)
    assert pls_scaled.scores_.values == pytest.approx(factor * pls_base.scores_.values, rel=1e-8)


def test_spe_calculation_handles_empty_negative_slice() -> None:
    """SEC-33 (#282) sub-item 2: ``np.var`` on an empty negative-only slice
    returns NaN and emits a RuntimeWarning; the sign-flip comparison now
    guards with ``.size > 0`` so all-non-negative input takes the non-flip
    branch cleanly.
    """
    import warnings

    # All-positive spe_values -> ``neg.size == 0``.
    spe_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        limit = spe_calculation(spe_values, conf_level=0.95)
    assert np.isfinite(limit)
    assert limit > 0.0


def test_regress_y_space_on_x() -> None:
    """Test the `regress_y_space_on_x` function."""
    x_space = np.array([[1, 2, 3, 4]])
    y_space = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, float("NaN"), 4],
            [float("NaN"), float("NaN"), 3, float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), 4],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [6, 4, 2, 0],
        ]
    )
    present_map = np.logical_not(np.isnan(y_space))
    y_space_filled = nan_to_zeros(y_space)
    regression_vector = regress_a_space_on_b_row(y_space_filled, x_space, present_map)
    # The row with nothing present has no regression: 0, with no weight anywhere (#513), not NaN.
    assert pytest.approx(np.array([[1, 1, 1, 1, 0, 2 / 3]]).T) == regression_vector

    # Present cells that all meet a zero in the row vector are a degenerate direction, and stay NaN.
    degenerate = regress_a_space_on_b_row(np.array([[0.0, 5.0]]), np.array([[1.0, 0.0]]), np.array([[False, True]]))
    assert np.isnan(degenerate).all()


def test_regress_a_space_on_b_row_tells_an_empty_row_from_a_degenerate_one_in_one_call() -> None:
    """An all-missing row gives 0 and a degenerate row NaN, side by side, and neither raises a RuntimeWarning."""
    a_space = np.array([[0.0, 0.0], [0.0, 5.0], [2.0, 4.0]])
    present = np.array([[False, False], [False, True], [True, True]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coefficients = regress_a_space_on_b_row(a_space, np.array([[1.0, 0.0]]), present)
    assert coefficients.shape == (3, 1)
    assert coefficients[0, 0] == 0.0
    assert np.isnan(coefficients[1, 0])
    assert coefficients[2, 0] == 2.0  # (2 * 1 + 4 * 0) / (1^2 + 0^2)


@pytest.mark.slow
def test_pca_spe_limits() -> None:
    """Simulate data and see if SPE limit cuts off at 5%."""
    N = 1000
    repeats = 50
    outliers_95 = []
    outliers_99 = []
    # Seeded: unseeded, the 99% mean (10.07, sd 0.29 over 60 runs) left 10 +/- 1 about
    # once in a thousand runs of each CI job, as it did on macOS in #682.
    rng = np.random.default_rng(0)
    for _ in range(repeats):
        # The desired mean values of the sample.
        mu = np.array([0.0, 0.0, 0.0])

        # The desired covariance matrix.
        r = np.array([[5.20, -4.98, -1.00], [-4.98, 5.50, 2.94], [-1.00, 2.94, 2.77]])

        X = pd.DataFrame(rng.multivariate_normal(mu, r, size=N))
        scaler = MCUVScaler().fit(X)
        mcuv = scaler.fit_transform(X)

        A = 2
        pca = PCA(n_components=A).fit(mcuv)
        spe_limit_95 = pca.spe_limit(conf_level=0.95)
        spe_limit_99 = pca.spe_limit(conf_level=0.99)

        outliers_95.append((pca.spe_.iloc[:, A - 1] > spe_limit_95).sum())
        outliers_99.append((pca.spe_.iloc[:, A - 1] > spe_limit_99).sum())

    assert np.mean(outliers_95) == pytest.approx(0.05 * N, rel=0.1)
    assert np.mean(outliers_99) == pytest.approx(0.01 * N, rel=0.1)


@pytest.mark.dataset
def test_pca_foods() -> None:
    """Arrays with no variance should not be able to have variance extracted."""

    url = "https://openmv.net/file/food-texture.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Python/process-improve-tests"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            foods = pd.read_csv(io.StringIO(response.read().decode())).drop(
                [
                    "Unnamed: 0",
                ],
                axis=1,
            )
    except (urllib.error.URLError, TimeoutError):
        pytest.skip("Cannot reach openmv.net - skipping network-dependent test")
    scaler = MCUVScaler().fit(foods)
    foods_mcuv = scaler.fit_transform(foods)

    A = 2
    pca = PCA(n_components=A).fit(foods_mcuv)

    assert np.linalg.norm(
        np.diag(pca.scores_.T @ pca.scores_) / (pca.n_samples_ - 1) - pca.explained_variance_
    ) == pytest.approx(0, abs=epsqrt)

    hotellings_t2_limit_95 = pca.hotellings_t2_limit(0.95)
    assert hotellings_t2_limit_95 == pytest.approx(6.64469, rel=1e-3)

    pca.spe_limit(conf_level=0.95)

    ellipse_x, ellipse_y = pca.ellipse_coordinates(score_horiz=1, score_vert=2, conf_level=0.95, n_points=100)
    assert ellipse_x[-1] == pytest.approx(4.48792, rel=1e-5)
    assert ellipse_y[-1] == pytest.approx(0, rel=1e-7)


@pytest.fixture
def fixture_kamyr_data_missing_value() -> pd.DataFrame:
    """Load the fixture."""
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
    return pd.read_csv(
        folder / "kamyr.csv",
        index_col=None,
        header=None,
    )


def test_pca_missing_data(fixture_kamyr_data_missing_value: pd.DataFrame) -> None:
    """Testing PCA with the Kamyr data set."""
    X_mcuv = MCUVScaler().fit_transform(fixture_kamyr_data_missing_value)

    # Build the model
    A = 2
    pca = PCA(n_components=A)
    assert pca.missing_data_settings is None

    # Check that default auto algorithm was used (NIPALS for missing data)
    model = pca.fit(X_mcuv)
    assert model.algorithm_ == "nipals"
    assert model.has_missing_data_ is True

    assert np.linalg.norm((model.loadings_.T @ model.loadings_) - np.eye(model.n_components)) == pytest.approx(
        0, abs=1e-2
    )


def test_pca_missing_data_as_numpy(fixture_kamyr_data_missing_value: pd.DataFrame) -> None:
    """Test the PCA model with missing data."""
    X_mcuv = MCUVScaler().fit_transform(fixture_kamyr_data_missing_value.values)

    # Build the model
    A = 2
    pca = PCA(n_components=A)
    assert pca.missing_data_settings is None

    # Check that default auto algorithm was used (NIPALS for missing data). MCUVScaler
    # hands back a DataFrame even for array input, so convert: this test is about an ndarray.
    model = pca.fit(X_mcuv.to_numpy())
    assert model.algorithm_ == "nipals"
    assert model.has_missing_data_ is True

    assert np.linalg.norm((model.loadings_.T @ model.loadings_) - np.eye(model.n_components)) == pytest.approx(
        0, abs=1e-2
    )


@pytest.fixture
def fixture_mv_utilities() -> tuple[np.ndarray, np.ndarray]:
    """
    Multivariate methods depend on an internal regression and Sum of Squares
    calculations. This code tests those crucial steps.
    """
    x = np.asarray([1, 2, 3, 4, 5, 6]).reshape(6, 1)
    Y = np.asarray(
        [
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, np.nan, 3, np.nan, 5, np.nan],
        ]
    )
    Y = Y.T
    return x, Y


def test_ssq(fixture_mv_utilities: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the sum-of-squares calculation."""
    x, _ = fixture_mv_utilities
    assert pytest.approx(ssq(x), abs=1e-9) == (1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6)


def test_ssq_per_row_and_column_skips_missing_cells() -> None:
    """Per-column and per-row sums skip NaN; an all-missing row or column sums to 0."""
    X = np.array([[1.0, np.nan, 2.0], [3.0, np.nan, np.nan]])
    np.testing.assert_array_equal(ssq(X, axis=0), [10.0, 0.0, 4.0])
    np.testing.assert_array_equal(ssq(X, axis=1), [5.0, 9.0])
    assert ssq(X) == 14.0


def test_ssq_refuses_an_axis_a_matrix_does_not_have() -> None:
    """A matrix has axes 0 and 1 (or None for the total); any other axis used to give a silent 0.0."""
    with pytest.raises(ValueError, match=r"^axis must be 0, 1 or None; got 2\.$"):
        ssq(np.ones((3, 2)), axis=2)


def test_quick_regress(fixture_mv_utilities: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the quick_regress function."""
    x, Y = fixture_mv_utilities
    out = quick_regress(Y, x).ravel()
    assert pytest.approx(out[0], abs=1e-9) == 1
    assert pytest.approx(out[1], abs=1e-8) == 0.61538462
    assert pytest.approx(out[2], abs=1e-9) == 0

    # Checked against R: summary(lm(c(1,1,1,1,1,1) ~ seq(6) + 0))
    assert pytest.approx(out[3], abs=1e-6) == 0.23077

    # Checked against what is expected: (1 + 3^2 + 5^2)/(1 + 3^2 + 5^2)
    assert pytest.approx(out[4], abs=1e-14) == 1.0


def test_quick_regress_degeneracy_guard_is_scale_free() -> None:
    """#513: the zero-denominator guard is relative to ssq(x), not an absolute epsqrt.

    A coefficient `(x'y) / (x'x)` is invariant under rescaling both inputs, so the same
    data expressed in different units must give the same answer. The guard used to be an
    absolute comparison against `epsqrt` (~1.5e-8), which is a statement about the scale
    of the data rather than its conditioning: at a scale of 1e-5 the denominator falls to
    ~1e-9 and every coefficient was silently returned as 0.0.
    """
    rng = np.random.default_rng(0)
    n_rows, n_cols = 10, 3
    for magnitude in (1.0, 1e-3, 1e-5, 1e-12):
        x = rng.standard_normal((n_rows, 1)) * magnitude
        Y = rng.standard_normal((n_rows, n_cols)) * magnitude
        coefficients = quick_regress(Y, x).ravel()
        # Same numbers, expressed in units where the old absolute guard never fired.
        rescaled = quick_regress(Y / magnitude, x / magnitude).ravel()
        assert coefficients == pytest.approx(rescaled, rel=1e-12)
        assert np.isfinite(coefficients).all()
        assert not np.allclose(coefficients, 0.0), f"all coefficients zeroed at magnitude {magnitude}"


def test_quick_regress_still_zeroes_a_genuine_degeneracy() -> None:
    """#513: the relative guard must not stop catching the cases the absolute one caught."""
    rng = np.random.default_rng(1)
    n_rows = 8

    # `x` carries no signal at all: every coefficient is undefined.
    assert np.all(quick_regress(rng.standard_normal((n_rows, 2)), np.zeros((n_rows, 1))) == 0.0)

    # One column of `Y` is entirely missing, so the mask empties `x` for that column only.
    Y = rng.standard_normal((n_rows, 2))
    Y[:, 1] = np.nan
    coefficients = quick_regress(Y, rng.standard_normal((n_rows, 1))).ravel()
    assert coefficients[0] != 0.0
    assert coefficients[1] == 0.0


def test_quick_regress_refuses_arrays_that_share_no_dimension() -> None:
    """The vector `x` must match Y's rows (to regress its columns) or its columns (to regress its rows)."""
    with pytest.raises(ValueError, match=r"^The dimensions of the input arrays are not compatible\.$"):
        quick_regress(np.ones((3, 2)), np.ones((4, 1)))


def _rotation(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def _inner_loop_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "process_improve.multivariate._nipals"]


def test_inner_pls_nipals_stops_at_its_iteration_cap(caplog: pytest.LogCaptureFixture) -> None:
    """Two nearly tied directions in X'Y make the iteration creep; the loop stops after exactly 500 steps.

    With X the identity, each step of the inner loop is a step of the power method on YY':
    ``w`` is ``u`` normalised, ``t = w``, ``q = Y't`` and the next ``u`` is ``YY'w / |Y'w|^2``.
    So the ``w`` of step ``n`` is ``(YY')^(n - 1) u_0`` normalised. With singular values 1 and
    0.99 each step shrinks the change by only about 2%, so it is still far above the tolerance
    after 500 steps; one step more or fewer moves ``w`` by 3e-7. The loop ran 502 steps
    under its cap of 500 until the cap was counted right.
    """
    x_space = np.eye(2)
    y_space = _rotation(np.pi / 5) @ np.diag([1.0, 0.99]) @ _rotation(np.pi / 7)
    everything = np.ones((2, 2), dtype=bool)
    with caplog.at_level(logging.DEBUG, logger="process_improve.multivariate._nipals"):
        fitted = internal_pls_nipals_fit_one_pc(x_space, y_space, everything, everything)

    assert _inner_loop_messages(caplog) == ["PLS NIPALS inner loop stopped after 500 iterations (max_iter=500)"]
    # YY' = R diag(1, 0.99^2) R', with R the left rotation.
    power = _rotation(np.pi / 5) @ np.diag([1.0, 0.99 ** (2 * 499)]) @ _rotation(np.pi / 5).T
    expected = power @ y_space[:, 0]
    np.testing.assert_allclose(fitted["w_i"].ravel(), expected / np.linalg.norm(expected), rtol=0, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_inner_pls_nipals_stops_once_u_moves_less_than_epsqrt_relative(
    scale: float, caplog: pytest.LogCaptureFixture
) -> None:
    """The inner loop stops at the first step whose change in ``u``, relative to ``|u|``, is below epsqrt.

    With X the identity the steps are the power method on YY' (see the test above), worked here
    in closed form to find that step. Singular values 1 and 0.5 shrink the change fourfold per
    step, so it comes after about a dozen steps. Scaling X and Y by ``c`` scales ``u`` by ``c``
    and leaves every relative change as it was, so the step is the same in any units (#504).
    """
    y_space = _rotation(np.pi / 5) @ np.diag([1.0, 0.5]) @ _rotation(np.pi / 7)
    u = y_space[:, 0]
    for step in range(1, 100):  # noqa: B007 - read after the loop
        unit = u / np.linalg.norm(u)
        u_next = y_space @ y_space.T @ unit / float(np.sum((y_space.T @ unit) ** 2))
        if np.linalg.norm(u - u_next) / np.linalg.norm(u) < epsqrt:
            break
        u = u_next
    assert 10 <= step <= 15

    everything = np.ones((2, 2), dtype=bool)
    with caplog.at_level(logging.DEBUG, logger="process_improve.multivariate._nipals"):
        internal_pls_nipals_fit_one_pc(scale * np.eye(2), scale * y_space, everything, everything)
    assert _inner_loop_messages(caplog) == [f"PLS NIPALS inner loop stopped after {step} iterations (max_iter=500)"]


@pytest.mark.parametrize(
    ("scores", "flipped"),
    [
        ([-1.0, -5.0, 1.0, 2.0], True),  # the negative side spreads 4.0, the other 0.25
        ([0.0, 10.0, 10.5, -1.0, -2.0], False),  # 0.25 against 23.4: a zero score sits on the non-negative side
        ([0.0, 1.0, 2.0, -10.0, -10.5], False),  # 0.0625 against 0.667: and not on the negative side
        ([-1.0, -2.0, 1.0, 2.0], False),  # a tie, 0.25 each, does not flip
        ([1.0, 2.0, 3.0], False),  # no negative score: nothing to compare
        ([-1.0, -2.0, -4.0], False),  # no non-negative score: likewise
    ],
)
def test_inner_pls_nipals_sign_convention(scores: list[float], flipped: bool) -> None:
    """The component flips when its negative scores spread more (``np.var``) than its non-negative ones.

    With one X column and Y equal to it, the loop converges at once to ``w = q = 1`` and ``t = u = x``,
    so the scores are the data and the two variances can be worked by hand. A side with no score
    is not compared, and asks ``np.var`` for nothing, so no RuntimeWarning is raised (SEC-33, #282).
    """
    x = np.array(scores)[:, np.newaxis]
    present = np.ones_like(x, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted = internal_pls_nipals_fit_one_pc(x, x.copy(), present, present)
    sign = -1.0 if flipped else 1.0
    for name, expected in (("t_i", x), ("u_i", x), ("w_i", np.ones((1, 1))), ("q_i", np.ones((1, 1)))):
        np.testing.assert_array_equal(fitted[name], sign * expected, err_msg=name)


@pytest.mark.parametrize("seed", range(6))
def test_inner_pls_nipals_flips_u_with_the_rest_of_the_component(seed: int) -> None:
    """#513: the sign convention flips ``u_i`` too, so it is still ``Yq / q'q`` for the returned ``q_i``.

    Only ``t_i``, ``w_i`` and ``q_i`` were negated, which left ``u_i`` pointing against the
    score on every component the convention flipped, as PLS and MBPLS never do. Some of
    these seeds flip and some do not; the relation holds for all of them.
    """
    rng = np.random.default_rng(seed)
    x_space = rng.standard_normal((15, 4))
    y_space = x_space @ rng.standard_normal((4, 2)) + 0.3 * rng.standard_normal((15, 2))
    everything = np.ones((15, 4), dtype=bool)
    fitted = internal_pls_nipals_fit_one_pc(x_space, y_space, everything, np.ones((15, 2), dtype=bool))
    q_i = fitted["q_i"]
    np.testing.assert_allclose(fitted["u_i"], y_space @ q_i / float(np.sum(q_i**2)), atol=1e-6)


def test_tpls_super_weights_point_the_way_of_the_super_score() -> None:
    """#513: with ``u`` flipped along with the score, every block score points the super score's way.

    A block's super weight is its score's regression on ``u``, so each is non-negative once
    ``u`` agrees with the super score; with no Z block the one F weight is +1. The
    un-flipped ``u`` made it -1 on the components the sign convention flipped. The
    predictions are the same either way.
    """
    rng = np.random.default_rng(3)
    n_blends, n_materials = 20, 4
    index = [f"blend{i}" for i in range(n_blends)]
    materials = [f"mat{j}" for j in range(n_materials)]
    d_matrix = {"G": pd.DataFrame(rng.normal(size=(n_materials, 3)), index=materials, columns=list("pqr"))}
    blocks = {
        "F": {"G": pd.DataFrame(rng.random((n_blends, n_materials)), index=index, columns=materials)},
        "Z": {"Z1": pd.DataFrame(rng.normal(size=(n_blends, 2)), index=index, columns=["z1", "z2"])},
        "Y": {"Y1": pd.DataFrame({"y": rng.normal(size=n_blends)}, index=index)},
    }
    with_z = TPLS(n_components=3, d_matrix=d_matrix).fit(DataFrameDict(copy.deepcopy(blocks)))
    assert (with_z.w_loadings_super.to_numpy() >= 0).all()

    blocks["Z"] = {}
    no_z = TPLS(n_components=3, d_matrix=d_matrix).fit(DataFrameDict(blocks))
    np.testing.assert_allclose(no_z.w_loadings_super.to_numpy(), 1.0)


def test_nipals_unit_normalisation_is_floored() -> None:
    """#513: `_pls` and `_pca` now floor the norm they divide by, as `_mbpls`/`_mbpca` do.

    A loading or weight vector that collapses during NIPALS makes the unguarded
    `v / sqrt(ssq(v))` a 0/0, and the resulting NaN propagates through deflation into
    every later component. `_nz` floors the denominator to the smallest positive float,
    which leaves every well-conditioned norm untouched and turns the degenerate case into
    a finite (zero) vector.
    """
    collapsed = np.zeros((4, 1))
    with np.errstate(invalid="ignore"):
        unguarded = collapsed / np.sqrt(ssq(collapsed))
    assert np.isnan(unguarded).all(), "the unguarded form is what produced the NaNs"
    guarded = collapsed / _nz(float(np.sqrt(ssq(collapsed))))
    assert np.isfinite(guarded).all()
    assert np.all(guarded == 0.0)

    # A well-conditioned vector is untouched by the floor: still exactly unit length.
    ordinary = np.array([[3.0], [4.0]])
    assert float(np.sqrt(ssq(ordinary / _nz(float(np.sqrt(ssq(ordinary))))))) == pytest.approx(1.0, rel=1e-15)


@pytest.fixture
def fixture_tablet_spectra_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Verify the PCA model for the case of no missing data.

    # R code:
    # -------
    # Read large data file
    file <- 'http://openmv.net/file/tablet-spectra.csv'
    spectra <- read.csv(file, header = FALSE, row.names = 1)
    # Only extract 4 components, but
    # center and scale the data before
    # calculation the components
    model.pca <- prcomp(spectra,
                        center = TRUE,
                        scale =TRUE,
                        rank. = 4)
    summary(model.pca)

    Importance of first k=4 (out of 460) components:
                               PC1     PC2     PC3     PC4
    Standard deviation     21.8835 10.9748 3.60075 3.27081
    Proportion of Variance  0.7368  0.1853 0.01995 0.01646
    Cumulative Proportion   0.7368  0.9221 0.94200 0.95846


    # T' * T on the scores matrix T:
    t(model.pca$x) %*% model.pca$x

                  PC1          PC2           PC3          PC4
    PC1  2.198092e+05 6.885159e-11 -1.134026e-11 3.454659e-11
    PC2  6.885159e-11 5.528459e+04  2.042206e-10 5.821477e-11
    PC3 -1.134026e-11 2.042206e-10  5.951125e+03 7.815970e-13
    PC4  3.454659e-11 5.821477e-11  7.815970e-13 4.910481e+03
    """
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
    spectra = pd.read_csv(
        folder / "tablet-spectra.csv",
        index_col=0,
        header=None,
    )

    # Ignoring values < 1E-8 (round them to zero) from the R output above.
    known_scores_covar = np.array(
        [
            [2.198092e05, 0, 0, 0],
            [0, 5.528459e04, 0, 0],
            [0, 0, 5.951125e03, 0],
            [0, 0, 0, 4.910481e03],
        ]
    )
    return spectra, known_scores_covar


def test_mcuv_centering(fixture_tablet_spectra_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Mean centering of the testing data."""

    spectra, _ = fixture_tablet_spectra_data
    X_mcuv = MCUVScaler().fit_transform(spectra)
    assert pytest.approx(np.max(np.abs(X_mcuv.mean(axis=0))), rel=1e-9) == 0.0


def test_mcuv_scaling(fixture_tablet_spectra_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Scaling by standard deviation."""

    spectra, _ = fixture_tablet_spectra_data
    X_mcuv = MCUVScaler().fit_transform(spectra)

    assert pytest.approx(np.min(np.abs(X_mcuv.std(axis=0))), 1e-10) == 1
    assert pytest.approx(X_mcuv.std(), 1e-10) == 1


def test_mcuv_keeps_integer_column_labels(fixture_tablet_spectra_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Integer labels survive: the spectra's columns are 1..650, which once came out as 0..649.

    sklearn sets ``feature_names_in_`` only for string labels, so the labels must come from the frame.
    """
    spectra, _ = fixture_tablet_spectra_data
    assert spectra.columns[0] == 1
    scaler = MCUVScaler().fit(spectra)
    X_mcuv = scaler.transform(spectra)

    pd.testing.assert_index_equal(scaler.center_.index, spectra.columns)
    pd.testing.assert_index_equal(X_mcuv.columns, spectra.columns)
    pd.testing.assert_index_equal(scaler.inverse_transform(X_mcuv).columns, spectra.columns)
    pd.testing.assert_index_equal(PCA(n_components=1).fit(X_mcuv).loadings_.index, spectra.columns)
    # An ndarray has no labels, so its columns are numbered from 0.
    assert list(MCUVScaler().fit_transform(spectra.to_numpy()[:, :3]).columns) == [0, 1, 2]


def test_mcuv_round_trips_a_single_column_target_given_as_a_series_and_a_1d_array() -> None:
    """A Series transforms as its one-column frame, and the 1-D array a regressor predicts inverts to that column."""
    y = pd.Series([3.0, 5.0, 4.0, 8.0, 10.0], index=list("abcde"), name="yield")
    scaler = MCUVScaler().fit(y)

    scaled = scaler.transform(y)
    pd.testing.assert_frame_equal(scaled, scaler.transform(y.to_frame()))
    assert list(scaled.columns) == ["yield"]
    assert list(scaled.index) == list("abcde")

    restored = scaler.inverse_transform(scaled["yield"].to_numpy())
    assert restored.shape == (5, 1)
    np.testing.assert_allclose(restored["yield"].to_numpy(), y.to_numpy())


def test_scale_ddof_matches_mcuvscaler() -> None:
    """``scale(center(X), ddof=1)`` must reproduce ``MCUVScaler`` (both use N-1)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(8, 4)))

    from_scale = np.asarray(scale(center(X), ddof=1), dtype=float)
    from_mcuv = np.asarray(MCUVScaler().fit_transform(X), dtype=float)
    assert np.allclose(from_scale, from_mcuv)

    # The default (ddof=0) divides by N, i.e. the population standard deviation.
    _, divisor = scale(center(X), extra_output=True)
    assert np.allclose(1.0 / divisor, np.asarray(center(X)).std(axis=0, ddof=0))


def test_scale_with_custom_func() -> None:
    """A custom scaling ``func`` is honoured and ``ddof`` is ignored for it."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(6, 3)))

    # Scale each column by its mean absolute value rather than the std. ``ddof``
    # only applies to the default ``np.std`` path, so passing it here is a no-op
    # and must not be forwarded to the (ddof-unaware) custom function.
    mad = lambda col: np.abs(col).mean()  # noqa: E731
    scaled, divisor = scale(X, func=mad, extra_output=True, ddof=1)

    expected_divisor = X.apply(mad, axis=0).to_numpy()
    assert np.allclose(1.0 / divisor, expected_divisor)
    assert np.allclose(np.asarray(scaled, dtype=float), X.to_numpy() / expected_divisor)


def test_scale_zero_variance_column_is_finite() -> None:
    """A constant (zero-variance) column is left as-is, not turned into inf/NaN."""
    X = pd.DataFrame({"varies": [1.0, 2.0, 3.0, 4.0], "constant": [5.0, 5.0, 5.0, 5.0]})
    scaled, divisor = scale(center(X), extra_output=True)

    assert np.isfinite(np.asarray(scaled, dtype=float)).all()
    # The constant column maps to a scaling factor of 1.0 (divisor stored as 1/std).
    assert divisor[1] == pytest.approx(1.0)
    # ... and is therefore unchanged from its (already centred) input: all zeros.
    assert np.allclose(np.asarray(scaled, dtype=float)[:, 1], 0.0)


def test_mcuv_by_hand_keeps_the_labels_and_round_trips() -> None:
    """Column a: mean 3, sd sqrt((4 + 1 + 9) / 2) = sqrt(7). Column b: mean 4, sd 1."""
    X = pd.DataFrame({"a": [1.0, 2.0, 6.0], "b": [3.0, 5.0, 4.0]}, index=["r1", "r2", "r3"])
    scaler = MCUVScaler().fit(X)
    pd.testing.assert_series_equal(scaler.center_, pd.Series([3.0, 4.0], index=["a", "b"]))
    pd.testing.assert_series_equal(scaler.scale_, pd.Series([np.sqrt(7.0), 1.0], index=["a", "b"]))

    expected = pd.DataFrame({"a": np.array([-2.0, -1.0, 3.0]) / np.sqrt(7.0), "b": [-1.0, 1.0, 0.0]}, index=X.index)
    scaled = scaler.transform(X)
    pd.testing.assert_frame_equal(scaled, expected)
    pd.testing.assert_frame_equal(scaler.inverse_transform(scaled), X)

    # An array has no row labels, so the rows are numbered; numbers held as objects come back as floats.
    from_array = scaler.inverse_transform(expected.to_numpy().astype(object))
    pd.testing.assert_frame_equal(from_array, X.reset_index(drop=True))

    # Fitted on an array, the output names are the ones the caller supplies, else sklearn's x0, x1.
    unnamed = MCUVScaler().fit(X.to_numpy())
    assert unnamed.get_feature_names_out(["a", "b"]).tolist() == ["a", "b"]
    assert unnamed.get_feature_names_out().tolist() == ["x0", "x1"]


def test_mcuv_declares_that_nan_passes_through() -> None:
    """A Pipeline reads the ``allow_nan`` tag of sklearn to decide whether NaN may reach this step."""
    assert MCUVScaler().__sklearn_tags__().input_tags.allow_nan is True


def test_mcuv_input_contract() -> None:
    """Dense, at least two rows, NaN allowed but not inf, and transform only on the columns it was fitted on."""
    with pytest.raises(TypeError, match=r"^MCUVScaler does not accept sparse input: "):
        MCUVScaler().fit(csr_matrix(np.eye(3)))
    with pytest.raises(ValueError, match=r"1 sample\(s\) \(shape=\(1, 2\)\) while a minimum of 2 is required"):
        MCUVScaler().fit(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="Input X contains infinity"):
        MCUVScaler().fit(np.array([[1.0], [np.inf]]))
    # Two rows are enough for a sample standard deviation: mean 2, sd sqrt(2). A NaN is a missing value.
    for rows in ([[1.0], [3.0]], [[1.0], [3.0], [np.nan]]):
        fitted = MCUVScaler().fit(np.array(rows))
        assert fitted.center_.tolist() == [2.0]
        assert fitted.scale_.tolist() == [np.sqrt(2.0)]

    from_array = MCUVScaler().fit(np.array([[1.0, 3.0], [2.0, 5.0]]))
    with pytest.raises(TypeError, match="Sparse data was passed for X, but dense data is required"):
        from_array.transform(csr_matrix(np.eye(2)))
    with pytest.raises(ValueError, match="X has 3 features, but MCUVScaler is expecting 2 features as input"):
        from_array.transform(np.ones((2, 3)))
    from_frame = MCUVScaler().fit(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [3.0, 5.0, 4.0]}))
    with pytest.raises(ValueError, match="The feature names should match those that were passed during fit"):
        from_frame.transform(pd.DataFrame({"a": [1.0], "c": [2.0]}))
    with pytest.raises(NotFittedError):
        MCUVScaler().transform(np.ones((2, 2)))
    with pytest.raises(NotFittedError):
        MCUVScaler().inverse_transform(np.ones((2, 2)))


def test_mcuv_constant_single_value_and_empty_columns() -> None:
    """A constant column is centred on its value and not scaled; one observed value or none gives no spread either.

    Three times 0.1 averages to 0.10000000000000002 in floating point, so centring a constant on its
    mean would leave a residue of 2e-17 where the column should be exactly 0.
    """
    X = np.array([[0.1, 7.0, np.nan], [0.1, np.nan, np.nan], [0.1, np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scaler = MCUVScaler().fit(X)
        out = scaler.transform(X).to_numpy()
    assert scaler.center_.tolist() == [0.1, 7.0, 0.0]
    assert scaler.scale_.tolist() == [1.0, 1.0, 1.0]
    np.testing.assert_array_equal(out, [[0.0, 0.0, np.nan], [0.0, np.nan, np.nan], [0.0, np.nan, np.nan]])


def test_mcuv_does_not_divide_by_a_spread_at_or_below_sqrt_tiny() -> None:
    """The square root of the smallest normal float is 2**-511; a standard deviation that small scales by 1.

    The first column, 2**-510, 0 and 2**-511, has mean 2**-511 and deviations of +-2**-511 and 0, so its
    sample standard deviation is exactly 2**-511. The second, a thousand times larger, keeps its own.
    """
    X = np.array([[2.0**-510, 2.0**-500], [0.0, 0.0], [2.0**-511, 2.0**-501]])
    scaler = MCUVScaler().fit(X)
    assert scaler.scale_.iloc[0] == 1.0
    assert scaler.scale_.iloc[1] == pytest.approx(np.std(X[:, 1], ddof=1), rel=1e-15)


def test_center_and_scale_by_hand_along_either_axis() -> None:
    """Down the columns: means 2, 3.5, 5 and population sds 1, 1.5, 1. Across the rows: means 3, 4."""
    X = pd.DataFrame([[1.0, 2.0, 6.0], [3.0, 5.0, 4.0]])
    centred, means = center(X, extra_output=True)
    np.testing.assert_array_equal(means, [2.0, 3.5, 5.0])
    np.testing.assert_array_equal(np.asarray(centred), [[-1.0, -1.5, 1.0], [1.0, 1.5, -1.0]])
    np.testing.assert_array_equal(np.asarray(center(X)), np.asarray(centred))
    by_row, row_means = center(X, axis=1, extra_output=True)
    np.testing.assert_array_equal(row_means, [[3.0], [4.0]])
    np.testing.assert_array_equal(np.asarray(by_row), [[-2.0, -1.0, 3.0], [-1.0, 1.0, 0.0]])

    scaled, multipliers = scale(X, extra_output=True)
    np.testing.assert_array_equal(multipliers, [1.0, 1.0 / 1.5, 1.0])
    np.testing.assert_allclose(np.asarray(scaled), X.to_numpy() * [1.0, 1.0 / 1.5, 1.0], rtol=1e-15)
    np.testing.assert_array_equal(np.asarray(scale(X)), np.asarray(scaled))
    # Row 1 is 1, 2, 6: mean 3, population sd sqrt(14 / 3). Row 2 is 3, 5, 4: mean 4, sd sqrt(2 / 3).
    by_row, row_multipliers = scale(X, axis=1, extra_output=True)
    np.testing.assert_allclose(row_multipliers, [[np.sqrt(3.0 / 14.0)], [np.sqrt(1.5)]], rtol=1e-15)
    np.testing.assert_allclose(np.asarray(by_row), X.to_numpy() * row_multipliers, rtol=1e-15)


def test_scale_leaves_a_column_unchanged_when_the_measure_gives_no_spread() -> None:
    """A custom measure can give 0 for a column that is not constant: its multiplier is 1, as for a constant.

    The median absolute deviation of 0, 0, 0, 0, 1 is 0; of 2, 4, 6, 8, 10 it is median(4, 2, 0, 2, 4) = 2.
    """
    from scipy.stats import median_abs_deviation

    X = pd.DataFrame({"mostly_zero": [0.0, 0.0, 0.0, 0.0, 1.0], "even": [2.0, 4.0, 6.0, 8.0, 10.0]})
    scaled, multipliers = scale(X, func=median_abs_deviation, extra_output=True)
    np.testing.assert_array_equal(multipliers, [1.0, 0.5])
    np.testing.assert_array_equal(np.asarray(scaled), X.to_numpy() * [1.0, 0.5])


@pytest.mark.slow
def test_pca_tablet_spectra(fixture_tablet_spectra_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    r"""
    Check PCA characteristics.

    1. model's loadings must be orthogonal if there are no missing data.
        P.T * P = I
    2. model's loadings must be of unit length (norm = 1.0)
        P.T * P = I
    3. model's scores must be orthogonal
        T.T * T is a diagonal matrix when there's no missing data
    4. each earlier score's variance must be >= variance of later score


    PCA models have the following properties:

    * :math:`p_i'p_j' = 0` for :math:`i\neq j`; i.e. :math:`p_i \perp p_j`
    * :math:`t_i't_j' = 0` for :math:`i\neq j`; i.e. :math:`t_i \perp t_j`
    * :math:`P'P = I_A` when extracting :math:`A` components
    * :math:`P_{all} \text{ is a } \min(N,K) \times \min(N,K)` matrix, for all components
    * :math:`T_{all} \text{ is a } \min(N,K) \times \min(N,K)` matrix, for all components
    (it is just a rearrangement of X)
    * :math:`\text{SVD}(X): UDV' = X` and :math:`V' = P'` and :math:`UD = T`
    """

    spectra, known_scores_covar = fixture_tablet_spectra_data

    # Number of components to calculate
    model = PCA(n_components=2)
    model.fit(scale(center(spectra)))

    # P'P = identity matrix of size A x A
    orthogonal_check = model.loadings_.T @ model.loadings_
    assert pytest.approx(np.linalg.norm(orthogonal_check - np.eye(model.n_components)), rel=1e-9) == 0.0

    # Check the R2 value against the R software output
    assert model.r2_cumulative_[1] == pytest.approx(0.7368, rel=1e-3)
    assert model.r2_cumulative_[2] == pytest.approx(0.9221, rel=1e-2)

    # Unit length: actually checked above, via subtraction with I matrix.
    # Check if scores are orthogonal
    scores_covar = pd.DataFrame(model.scores_.T @ model.scores_).astype(float)
    for i in range(model.n_components):
        for j in range(model.n_components):
            # Technically not need, but more explict this way.
            if i == j:
                assert scores_covar.iloc[i, j] == pytest.approx(known_scores_covar[i, j], rel=1e-2)
            else:
                assert scores_covar.iloc[i, j] == pytest.approx(known_scores_covar[i, j], abs=1e-4)

                # if i >= 1: Creates unnecessary type errors and not such a useful check.
                #    assert scores_covar.iloc[j, j] > scores_covar.iloc[i, i]

    # Check the model against an SVD: this raw data set has no missing
    # data, so the SVD should be faster and more accurate than NIPALS
    autoscaled_X = scale(center(spectra))
    _u, _s, v = np.linalg.svd(autoscaled_X)

    loadings_delta = np.linalg.norm(np.abs(v[0 : model.n_components, :]) - np.abs(model.loadings_.T))
    assert loadings_delta == pytest.approx(0, abs=1e-8)

    # It is not possible, it seems, to get the scores to match the SVD
    # scores. Numerical error?


def test_pca_errors_no_variance_to_start() -> None:
    """Arrays with no variance should seem to work, but should have no variability explained."""
    K, N, A = 17, 12, 5
    data = pd.DataFrame(np.zeros((N, K)))
    model = PCA(n_components=A)
    # with pytest.raises(RuntimeError):
    model.fit(data)
    assert np.sum(model.scores_.values) == pytest.approx(0, abs=epsqrt)
    assert model.r2_cumulative_.sum() == pytest.approx(0, abs=epsqrt)
    assert np.isnan(model.r2_cumulative_[A - 1])


def test_pca_invalid_calls() -> None:
    """Tests various invalid calls, and corresponding error messages."""
    K, N, A = 4, 3, 5
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.uniform(low=-1, high=1, size=(N, K)))
    model = PCA(n_components=A)
    with pytest.warns(
        SpecificationWarning,
        match=r"The requested number of components is more than can be computed from data(.*)",
    ):
        model.fit(data)
    data.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Tolerance must exceed machine precision"):
        _ = PCA(n_components=A, algorithm="nipals", tol=0.0).fit(data)

    with pytest.raises(ValueError, match=r"Algorithm .* is not recognized(.*)"):
        _ = PCA(n_components=A, algorithm="SCP").fit(data)

    # Plain numpy IS accepted by design (sklearn compatibility), so the rejection
    # below is about sparse input specifically, not about DataFrames. PCA rejects it
    # before validate_data does, so the message can name the ColumnTransformer knobs
    # that avoid the sparse round trip instead of sklearn's generic `.toarray()` (#399).
    sparse_data = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    with pytest.raises(TypeError, match=r"PCA does not accept sparse input.*sparse_threshold=0"):
        PCA(n_components=2).fit(sparse_data)


def test_pca_svd_refuses_missing_data() -> None:
    """The SVD needs every cell, so asking for it on gappy data names the algorithms that cope."""
    gappy = pd.DataFrame(np.random.default_rng(3).standard_normal((20, 4)))
    gappy.iloc[0, 0] = np.nan
    with pytest.raises(
        ValueError, match=r"^SVD algorithm cannot handle missing data\. Use 'nipals', 'tsr', or 'auto'\.$"
    ):
        PCA(n_components=2, algorithm="svd").fit(gappy)


#: Each PCA entry point that accepts a plain array, reduced to the numbers it returns.
_PCA_ARRAY_ENTRY_POINTS = {
    "fit": lambda _, X: PCA(n_components=2).fit(X).scores_.to_numpy(),
    "project": lambda model, X: model.project(X).scores.to_numpy(),
    "minka_mle": lambda _, X: PCA.minka_mle(X),
    "parallel_analysis": lambda _, X: PCA.parallel_analysis(X, n_simulations=20, random_state=0).null_threshold,
    "select_n_components": lambda _, X: PCA.select_n_components(X, random_state=0).press.to_numpy(),
}


@pytest.mark.parametrize("entry_point", list(_PCA_ARRAY_ENTRY_POINTS))
def test_pca_entry_points_read_an_array_as_its_dataframe(entry_point: str) -> None:
    """A plain array gives the same numbers as the same values in a DataFrame."""
    frame = pd.DataFrame(np.random.default_rng(5).standard_normal((30, 5)) * [1.0, 2.0, 3.0, 4.0, 5.0] + 10.0)
    model = PCA(n_components=2).fit(frame)
    call = _PCA_ARRAY_ENTRY_POINTS[entry_point]
    np.testing.assert_allclose(call(model, frame.to_numpy()), call(model, frame))


def test_pca_columns_with_no_variance() -> None:
    """Create a column with no variance. That column's loadings should be 0."""
    K = 14
    N = 29
    A = 4
    rng = np.random.default_rng(0)
    cols_with_no_variance = [10, 3]
    T = rng.uniform(low=-1, high=1, size=(N, A))
    P = rng.uniform(low=-1, high=1, size=(K, A))
    X = T @ P.T
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0, ddof=0)
    x_to_fit = pd.DataFrame((X - meanX) / stdX)
    x_to_fit.iloc[:, cols_with_no_variance] = 0

    m = PCA(n_components=2)
    m.fit(x_to_fit)

    # `loadings_` is a K by A matrix.  Check sum of loadings in rows with
    # no variance must be zero
    assert np.sum(np.abs(m.loadings_.iloc[cols_with_no_variance, :].values)) == pytest.approx(0, abs=1e-14)
    # The loadings must still be orthonormal though:
    assert np.sum(np.identity(m.n_components) - m.loadings_.values.T @ m.loadings_.values) == pytest.approx(
        0, abs=1e-14
    )

    # Are scores orthogonal?
    covmatrix = m.scores_.values.T @ m.scores_.values
    assert (np.sum(np.abs(covmatrix - np.diag(np.diag(covmatrix))))) == pytest.approx(0, abs=1e-6)


@pytest.fixture
def fixture_pca_pca_wold_etal_paper() -> pd.DataFrame:
    """
    Return data from the PCA paper by Wold, Esbensen and Geladi, 1987.

    Principal Component Analysis, Chemometrics and Intelligent Laboratory
    Systems, v 2, p37-52; http://dx.doi.org/10.1016/0169-7439(87)80084-9
    """
    return pd.DataFrame(np.array([[3, 4, 2, 2], [4, 3, 4, 3], [5.0, 5, 6, 4]]))


def test_pca_wold_centering(fixture_pca_pca_wold_etal_paper: pd.DataFrame) -> None:
    """Checks the centering step."""
    _, centering = center(fixture_pca_pca_wold_etal_paper, extra_output=True)
    assert centering == pytest.approx([4, 4, 4, 3], rel=1e-8)


def test_pca_wold_scaling(fixture_pca_pca_wold_etal_paper: pd.DataFrame) -> None:
    """Checks the scaling step. Page 40 of the above paper."""

    _, scaling = scale(center(fixture_pca_pca_wold_etal_paper), extra_output=True, ddof=1)
    assert scaling == pytest.approx([1, 1, 0.5, 1])


def test_pca_wold_model_results(fixture_pca_pca_wold_etal_paper: pd.DataFrame) -> None:
    """Check if the PCA model matches the results in the paper."""

    X_preproc = scale(center(fixture_pca_pca_wold_etal_paper))
    pca_1 = PCA(n_components=1)
    pca_1.fit(X_preproc.copy())

    # The remaining sum of squares, page 43. The paper's SS is on data scaled with
    # ddof=1 (unit-variance total SS = N - 1 = 2 per column); `X_preproc` above uses
    # scale()'s default ddof=0 (total SS = N = 3), so the two residual SS differ by
    # the uniform factor N / (N - 1) = 1.5 that ddof applies identically to every
    # column. Refit on the ddof=1 preprocessing to compare like with like; the
    # dimensionless R2 (pca_1.r2_per_variable_ below) is unaffected either way.
    X_preproc_ddof1 = scale(center(fixture_pca_pca_wold_etal_paper), ddof=1)
    pca_1_ddof1 = PCA(n_components=1).fit(X_preproc_ddof1.copy())
    residuals = X_preproc_ddof1.to_numpy() - pca_1_ddof1.scores_.to_numpy() @ pca_1_ddof1.loadings_.to_numpy().T
    ss_x = np.sum(residuals**2, axis=0)
    assert ss_x == pytest.approx([0.0551, 1.189, 0.0551, 0.0551], abs=1e-3)

    # With 2 components, the loadings are, page 40
    # P.T = [ 0.5410, 0.3493,  0.5410,  0.5410],
    #      [-0.2017, 0.9370, -0.2017, -0.2017]
    X_preproc = scale(center(fixture_pca_pca_wold_etal_paper))
    pca_2 = PCA(n_components=2)
    pca_2.fit(X_preproc)
    assert np.abs(pca_2.loadings_.values[:, 0]) == pytest.approx([0.5410, 0.3493, 0.5410, 0.5410], abs=1e-4)
    assert np.abs(pca_2.loadings_.values[:, 1]) == pytest.approx([0.2017, 0.9370, 0.2017, 0.2017], abs=1e-4)

    # Scores. The scaling is off here by a constant factor of 0.8165
    # assert np.all(pca_2.x_scores["1"] == pytest.approx([-1.6229, -0.3493, 1.9723], rel=1e-3))
    # assert np.all(pca_2.x_scores["2"] == pytest.approx([0.6051, -0.9370, 0.3319], rel=1e-4))

    # R2 values, given on page 43
    assert pca_2.r2_per_component_.values == pytest.approx([0.831, 0.169], rel=1e-2)

    # SS values, on page 43
    # SS_X = np.sum(X_preproc ** 2, axis=0)
    # assert SS_X == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-9)

    # Testing data:
    # X_test = Block(np.array([[3, 4, 3, 4], [1, 2, 3, 4.0]]))
    # X.preprocess(X_test)
    # compare_entries(X_test.data, np.array([[-1, 0, -0.5, 1],
    # [-3, -2, -0.5, 1]])
    # #testing = PCA_model.apply(X_test)
    # compare_entries(testing.T, np.array([[-0.2075, 0.1009],
    # [-2.0511, -1.3698]])


def test_pca_score() -> None:
    """Test PCA score() method returns negative MSE reconstruction error."""
    rng = np.random.default_rng(99)
    X = pd.DataFrame(rng.standard_normal((30, 5)), columns=[f"V{i}" for i in range(1, 6)])
    X = MCUVScaler().fit_transform(X)

    pca_2 = PCA(n_components=2).fit(X)
    pca_4 = PCA(n_components=4).fit(X)

    s2 = pca_2.score(X)
    s4 = pca_4.score(X)

    # Returns a negative float (negative MSE)
    assert isinstance(s2, float)
    assert s2 < 0
    assert np.isfinite(s2)

    # More components → better reconstruction → higher (less negative) score
    assert s4 > s2

    # Works with numpy arrays too
    s2_np = pca_2.score(X.values)
    assert s2_np == pytest.approx(s2, abs=1e-12)


def test_pca_score_refuses_a_1d_row() -> None:
    """score() takes a 2-D X, as transform() does: a bare 1-D row is refused, not reshaped."""
    X = pd.DataFrame(np.random.default_rng(0).standard_normal((20, 4)))
    model = PCA(n_components=2).fit(X)
    with pytest.raises(ValueError, match=r"Expected 2D array, got 1D array instead"):
        model.score(X.iloc[0].to_numpy())


def test_pca_select_n_components() -> None:
    """Test PRESS-based component selection on synthetic data with known structure."""
    rng = np.random.default_rng(77)
    N, K = 30, 50

    # 2 strong latent components driving 50 measured variables
    # N < K: the classic chemometrics scenario where PRESS cross-validation
    # excels - noise components overfit because there aren't enough samples
    # to reliably estimate them.
    T_true = rng.standard_normal((N, 2)) * np.array([10.0, 6.0])
    P_true = rng.standard_normal((2, K))
    P_true /= np.linalg.norm(P_true, axis=1, keepdims=True)
    noise = rng.standard_normal((N, K)) * 1.0
    # Raw block: the default scale_inside_folds=True fits the scaling on each
    # element-fold's in-fold cells, which is the usage the library recommends.
    X = pd.DataFrame(T_true @ P_true + noise)

    max_comp = 6
    result = PCA.select_n_components(X, max_components=max_comp, cv=5, random_state=0)

    # Returns a Bunch including both the back-compat fields and the new
    # ekf bookkeeping (per_fold_press, se_press, cv_scheme, selection_rule).
    assert isinstance(result, Bunch)
    assert {"n_components", "press", "press_ratio", "q2", "cv_scores"} <= set(result.keys())
    assert {"per_fold_press", "se_press", "cv_scheme", "selection_rule"} <= set(result.keys())
    assert result.cv_scheme == "ekf"
    assert result.selection_rule == "min"

    # q2_se is the per-component Q2-scale standard error (the +/-1 SE band): an
    # exact rescale of se_press by the null-model sum of squares, which the
    # folds measure in the space they were fitted in (#546), so it comes back
    # from the helper rather than being recomputed here.
    assert "q2_se" in result
    assert (result.q2_se >= 0).all()
    X_arr = np.asarray(X, dtype=float)
    null_ss = _pca_ekf_press(
        X_arr, max_comp, n_folds=5, random_state=0, n_repeats=1, ekf_max_iter=50, ekf_tol=1e-6, scale_inside_folds=True
    ).null_model_ss
    np.testing.assert_allclose(result.q2_se.to_numpy(), result.se_press.to_numpy() / null_ss, rtol=1e-9)

    # With 2 true components and N < K, should recommend 2 (or at most 3)
    assert 2 <= result.n_components <= 3

    # PRESS is a Series indexed 1..max_components
    assert isinstance(result.press, pd.Series)
    assert len(result.press) == max_comp
    assert list(result.press.index) == list(range(1, max_comp + 1))
    assert all(result.press > 0)

    # PRESS should decrease substantially for the 2nd component
    assert result.press[2] < result.press[1]

    # PRESS ratio is indexed 2..max_components
    assert isinstance(result.press_ratio, pd.Series)
    assert len(result.press_ratio) == max_comp - 1
    assert list(result.press_ratio.index) == list(range(2, max_comp + 1))

    # Q2 (cross-validated R2 of X) is a Series indexed 1..max_components,
    # finite, bounded above by 1, and increasing as the first true components
    # are added (mirrors PLS.select_n_components's r2y_validated).
    assert isinstance(result.q2, pd.Series)
    assert len(result.q2) == max_comp
    assert list(result.q2.index) == list(range(1, max_comp + 1))
    assert np.isfinite(result.q2.to_numpy()).all()
    assert (result.q2 <= 1.0 + 1e-9).all()
    assert result.q2[2] > result.q2[1]
    # Q2 is exactly the normalised PRESS under ekf, against the reference the
    # folds measured, so it stays directly comparable to r2_cumulative_.
    assert result.q2.to_numpy() == pytest.approx(1.0 - result.press.to_numpy() / null_ss)

    # Every variable's own Q2 is reported beside the pooled figure, which is
    # what shows whether one column is carrying it (#546).
    assert result.q2_per_variable.shape == (max_comp, X.shape[1])
    assert list(result.q2_per_variable.index) == list(range(1, max_comp + 1))

    # cv_scores aliases per_fold_press under ekf; shape (A, n_folds).
    assert isinstance(result.cv_scores, pd.DataFrame)
    assert result.cv_scores.shape == (max_comp, 5)
    assert isinstance(result.per_fold_press, pd.DataFrame)
    assert result.per_fold_press.shape == (max_comp, 5)

    # Works with an integer for n_folds (and ignores splitter objects under ekf).
    result2 = PCA.select_n_components(X, max_components=4, cv=3, random_state=0)
    assert isinstance(result2.n_components, int)
    assert result2.per_fold_press.shape == (4, 3)


def test_pca_select_n_components_ekf_recovers_known_rank() -> None:
    """Ekf does not over-select on a clean low-rank dataset (the row-wise pathology)."""
    rng = np.random.default_rng(2025)
    N, K, true_rank = 50, 30, 3
    T = rng.standard_normal((N, true_rank)) * np.array([8.0, 5.0, 3.0])
    P = rng.standard_normal((true_rank, K))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    X = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)))

    result = PCA.select_n_components(X, max_components=10, cv=5, random_state=0)
    # ekf with GlobalMin recovers the true rank (or one beside it).
    assert true_rank - 1 <= result.n_components <= true_rank + 1
    # PRESS rises again past the true rank when extra components fit noise.
    a_star = int(np.argmin(result.press.to_numpy())) + 1
    assert a_star <= true_rank + 1


@pytest.mark.parametrize("shape", [(12, 30), (30, 12), (9, 9), (1, 5), (5, 1)])
@pytest.mark.parametrize("rank", [1, 3, 50])
def test_low_rank_reconstruction_matches_the_truncated_svd(shape: tuple[int, int], rank: int) -> None:
    """The Gram-matrix projector gives the truncated SVD's reconstruction, wide or tall, rank capped at full."""
    X = np.random.default_rng(4).standard_normal(shape)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank, *shape)
    expected = (U[:, :r] * S[:r]) @ Vt[:r]
    np.testing.assert_allclose(_low_rank_reconstruction(X, rank), expected, atol=1e-10)


def test_pca_select_n_components_ekf_handles_missing_cells() -> None:
    """A block that already has missing cells cross-validates, and agrees with the complete one.

    ``PCA.fit`` has always taken a block with missing cells through NIPALS, but the
    ekf scheme built its fold matrix from the raw array: the column centre and scale
    came from ``mean()``/``std()`` over columns holding NaN, and only the held-out
    cells were filled before the SVD, so the caller's own NaNs reached
    ``np.linalg.svd`` and it raised ``LinAlgError: SVD did not converge``.
    An unmeasured cell has no true value to predict, so it is now imputed by EM
    alongside the held-out cells and never scored.
    """
    rng = np.random.default_rng(11)
    N, K, true_rank = 60, 8, 2
    T = rng.standard_normal((N, true_rank))
    P = rng.standard_normal((true_rank, K))
    complete = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)))

    holed = complete.copy()
    holed.iloc[rng.choice(N, 12, replace=False), 1] = np.nan
    assert holed.isna().sum().sum() == 12

    intact = PCA.select_n_components(complete, max_components=4, cv=5, random_state=0)
    with_gaps = PCA.select_n_components(holed, max_components=4, cv=5, random_state=0)

    # Both find the true rank, and the missing cells cost accuracy rather than
    # changing the answer: the Q2 curves peak at the same component count.
    assert intact.n_components == true_rank
    assert with_gaps.n_components == true_rank
    assert int(np.argmax(intact.q2.to_numpy())) == int(np.argmax(with_gaps.q2.to_numpy()))
    assert with_gaps.q2.to_numpy()[true_rank - 1] == pytest.approx(intact.q2.to_numpy()[true_rank - 1], abs=0.1)
    # Nothing is scored against a cell that was never measured, so no NaN leaks
    # into the PRESS or the per-column split.
    assert np.isfinite(with_gaps.press.to_numpy()).all()
    assert np.isfinite(with_gaps.q2.to_numpy()).all()


def test_pca_select_n_components_ekf_q2_is_negative_on_noise() -> None:
    """Pure noise has nothing to predict, so cumulative Q2 stays below zero.

    This is the property the row-wise scheme removed in 2.0 violated: it scored a
    held-out row through its own values, so its "Q2" climbed towards 1 as components
    approached the variable count even when the data carried no structure at all.
    """
    rng = np.random.default_rng(4)
    noise = pd.DataFrame(rng.standard_normal((46, 8)))

    ekf = PCA.select_n_components(noise, max_components=7, cv=5, random_state=0)
    assert (ekf.q2.to_numpy() < 0).all()


@pytest.mark.parametrize(
    ("old_spelling", "replacement"),
    [
        ({"n_iter": 5}, "ekf_max_iter"),
        ({"tol": 1e-4}, "ekf_tol"),
        ({"threshold": 0.95}, "selection_rule"),
        ({"algorithm": "nipals"}, None),
    ],
)
def test_pca_select_n_components_refuses_the_removed_keywords(old_spelling: dict, replacement: str | None) -> None:
    """``n_iter``, ``tol``, ``threshold`` and ``**pca_kwargs`` were removed in 2.0.

    ``**pca_kwargs`` reached ``PCA`` only under ``cv_scheme="row_wise"``, removed in
    the same release, so it went too: an unknown keyword is now refused, not swallowed.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((30, 8)))
    # The removed keyword goes in through a dict, so a static analyser does not flag the call.
    with pytest.raises(TypeError, match=next(iter(old_spelling))):
        PCA.select_n_components(X, max_components=4, cv=3, random_state=0, **old_spelling)
    if replacement is not None:
        assert replacement in inspect.signature(PCA.select_n_components).parameters


def test_pca_select_n_components_rule_dispatch() -> None:
    """The 1-SE and Q2-increment rules are exposed under PCA too."""
    rng = np.random.default_rng(3)
    N, K, true_rank = 45, 25, 2
    T = rng.standard_normal((N, true_rank)) * np.array([6.0, 4.0])
    P = rng.standard_normal((true_rank, K))
    X = pd.DataFrame(T @ P + 0.35 * rng.standard_normal((N, K)))

    res_min = PCA.select_n_components(X, max_components=6, cv=5, random_state=0)
    res_1se = PCA.select_n_components(X, max_components=6, cv=5, random_state=0, selection_rule="1se")
    res_q2 = PCA.select_n_components(
        X,
        max_components=6,
        cv=5,
        random_state=0,
        selection_rule="q2_increment",
        min_q2_increase=0.01,
    )
    # 1-SE is never less parsimonious than argmin.
    assert res_1se.n_components <= res_min.n_components
    # All three rules return at least 1.
    assert 1 <= res_q2.n_components <= 6

    with pytest.raises(ValueError, match="Unknown cv_scheme"):
        PCA.select_n_components(X, max_components=3, cv=3, cv_scheme="bogus")  # type: ignore[arg-type]


@pytest.mark.slow
def test_pca_select_n_components_n_repeats_narrows_se() -> None:
    """Repeated ekf gives reproducible answers and a narrower per-component SE."""
    rng = np.random.default_rng(2026)
    N, K, true_rank = 50, 25, 3
    T = rng.standard_normal((N, true_rank)) * np.array([6.0, 4.0, 2.5])
    P = rng.standard_normal((true_rank, K))
    X = pd.DataFrame(T @ P + 0.4 * rng.standard_normal((N, K)))

    one = PCA.select_n_components(X, max_components=8, cv=5, n_repeats=1, random_state=0)
    many = PCA.select_n_components(X, max_components=8, cv=5, n_repeats=8, random_state=0)
    # 1 repeat -> 5 fold columns; 8 repeats -> 40.
    assert one.per_fold_press.shape == (8, 5)
    assert many.per_fold_press.shape == (8, 40)
    # PRESS stays on the per-cell scale (averaged over repeats).
    assert (many.press > 0).all()
    # SE narrows with more repeats: 8 repeats sample fold-PRESS 40 times rather
    # than 5, so the standard error of the mean shrinks by roughly sqrt(8) ~ 2.8x.
    # Checked on the median across components rather than component by component:
    # an SE built from only 5 folds carries a relative sampling error near
    # 1 / sqrt(2 * (5 - 1)) ~ 35%, so a single component can come out lower on 5
    # folds than on 40 by chance. The per-component bound catches a real blow-up
    # without pinning that noise.
    se_ratio = (many.se_press / one.se_press).to_numpy()
    assert np.median(se_ratio) < 0.7
    assert (se_ratio < 1.5).all()
    # The extra repeats sharpen the estimate without moving the recommendation.
    assert one.n_components == many.n_components == true_rank
    # Reproducible given a fixed seed.
    again = PCA.select_n_components(X, max_components=8, cv=5, n_repeats=8, random_state=0)
    np.testing.assert_allclose(many.press.to_numpy(), again.press.to_numpy())
    np.testing.assert_allclose(many.per_fold_press.to_numpy(), again.per_fold_press.to_numpy())

    with pytest.raises(ValueError, match="n_repeats must be >= 1"):
        PCA.select_n_components(X, max_components=3, cv=3, n_repeats=0)


def test_pca_select_n_components_scale_inside_folds_no_leakage() -> None:
    """In-fold scaling lets the caller pass raw (unscaled) X without leakage."""
    rng = np.random.default_rng(0)
    N, K, true_rank = 40, 12, 2
    T = rng.standard_normal((N, true_rank))
    P = rng.standard_normal((true_rank, K))
    # X on an arbitrary, non-zero-mean / non-unit-variance scale per column.
    column_scales = np.array([10.0, 0.1, 100.0, 1.0, 5.0, 0.5, 2.0, 20.0, 0.3, 7.0, 1.5, 0.05])
    X_raw = pd.DataFrame(
        (T @ P + 0.3 * rng.standard_normal((N, K))) * column_scales + 5.0,
    )

    # Default: scale inside folds. Caller passes the raw matrix.
    raw = PCA.select_n_components(X_raw, max_components=6, cv=5, n_repeats=2, random_state=0)
    assert (raw.press > 0).all()
    assert 1 <= raw.n_components <= 6

    # Opt-out path: caller pre-scales (the prior contract), is warned about
    # the leakage, and gets the legacy iterative-recentering EM behaviour.
    # Both should converge to a comparable recommendation on this clean
    # low-rank data.
    X_s = MCUVScaler().fit_transform(X_raw)
    with pytest.warns(SpecificationWarning, match="leaks"):
        legacy = PCA.select_n_components(
            X_s,
            max_components=6,
            cv=5,
            n_repeats=2,
            random_state=0,
            scale_inside_folds=False,
        )
    assert 1 <= legacy.n_components <= 6


def test_pca_select_n_components_ekf_q2_is_scale_invariant() -> None:
    """Re-expressing one column in different units must not move the Q2 curve (#546).

    The element-wise scheme standardises inside every fold, so what it fits does
    not depend on a column's units. Before #546 it then measured PRESS after
    inverting that standardisation, and a column carried weight in proportion to
    its variance: multiplying one column by 1000 handed it the whole curve.
    """
    rng = np.random.default_rng(11)
    N, K, true_rank = 45, 10, 3
    T = rng.standard_normal((N, true_rank)) * np.array([5.0, 3.0, 1.5])
    P = rng.standard_normal((true_rank, K))
    X = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)))
    X_stretched = X.copy()
    X_stretched.iloc[:, 0] *= 1000.0

    kwargs = {"max_components": 6, "cv": 5, "n_repeats": 3, "random_state": 7}
    base = PCA.select_n_components(X, **kwargs)
    stretched = PCA.select_n_components(X_stretched, **kwargs)

    # The folds are the same cells in both runs and the in-fold standardisation
    # removes the factor of 1000 exactly, so this is an identity, not an
    # approximation: only floating point separates the two curves.
    np.testing.assert_allclose(stretched.q2.to_numpy(), base.q2.to_numpy(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(stretched.press.to_numpy(), base.press.to_numpy(), rtol=1e-8)
    assert stretched.n_components == base.n_components == true_rank

    # press_input_units does follow the units, which is the point of having it:
    # the stretched column contributes a million times more squared error.
    assert (stretched.press_input_units > base.press_input_units).all()


def test_pca_select_n_components_ekf_q2_per_variable_reconciles() -> None:
    """The per-variable Q2 splits the pooled figure rather than restating it (#546)."""
    rng = np.random.default_rng(12)
    N, K = 40, 8
    T = rng.standard_normal((N, 2))
    P = rng.standard_normal((2, K))
    # One column is pure noise: it has nothing for a component to predict, so
    # its own Q2 should sit far below the pooled value.
    X = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)), columns=[f"v{j}" for j in range(K)])
    X["noise"] = rng.standard_normal(N) * 4.0

    result = PCA.select_n_components(X, max_components=4, cv=5, n_repeats=3, random_state=3)

    assert list(result.q2_per_variable.columns) == list(X.columns)
    assert result.q2_per_variable.shape == (4, X.shape[1])
    assert np.isfinite(result.q2_per_variable.to_numpy()).all()
    # Every structured column beats the pooled figure at the chosen rank; the
    # noise column drags it down, which is exactly what this table is for.
    at_rank = result.q2_per_variable.loc[result.n_components]
    assert at_rank["noise"] < result.q2[result.n_components] < at_rank.drop("noise").min()


def test_pca_select_n_components_ekf_q2_per_variable_is_nan_for_a_flat_column() -> None:
    """A column with no spread has no Q2, rather than a spurious zero (#546)."""
    rng = np.random.default_rng(21)
    N, K = 40, 6
    T = rng.standard_normal((N, 2))
    P = rng.standard_normal((2, K))
    X = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)), columns=[f"v{j}" for j in range(K)])
    X["flat"] = 7.5

    result = PCA.select_n_components(X, max_components=3, cv=5, n_repeats=2, random_state=0)

    # The null model predicts a constant column exactly, so its reference sum of
    # squares is zero and the ratio is undefined. Reporting 0.0 there would read
    # as "predicted no better than the mean" rather than "nothing to predict".
    assert result.q2_per_variable["flat"].isna().all()
    # It is only that column: the rest are unaffected and the pooled figure,
    # which the flat column contributes nothing to on either side of the ratio,
    # stays finite.
    assert np.isfinite(result.q2_per_variable.drop(columns="flat").to_numpy()).all()
    assert np.isfinite(result.q2.to_numpy()).all()


def test_pca_select_n_components_ekf_degenerate_blocks_stay_quiet() -> None:
    """Degenerate inputs give NaN answers, not exceptions and not numpy warnings.

    Two edges the element-wise loop has to survive: a matrix small enough that a
    column can be left with a single in-fold cell to estimate its centre from,
    and a matrix with no variation at all, where the null model is exact and the
    ratio PRESS / null is 0/0.
    """
    rng = np.random.default_rng(5)
    tiny = pd.DataFrame(rng.standard_normal((4, 3)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        small = PCA.select_n_components(tiny, max_components=2, cv=4, random_state=0)
    assert 1 <= small.n_components <= 2
    assert np.isfinite(small.q2.to_numpy()).all()

    flat = pd.DataFrame(np.full((20, 4), 3.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = PCA.select_n_components(flat, max_components=2, cv=5, random_state=0)
    # Nothing varies, so nothing is predicted and nothing is left over: PRESS is
    # zero and every Q2 is undefined rather than a perfect 1.0.
    assert (result.press.to_numpy() == 0).all()
    assert result.q2.isna().all()
    assert result.q2_per_variable.isna().to_numpy().all()
    assert result.press_ratio.isna().all()


def test_pca_select_n_components_ekf_q2_matches_on_raw_and_prescaled_ldpe() -> None:
    """On LDPE the raw and pre-scaled blocks now give the same curve (#546).

    ``Mw`` holds over 99% of the raw block's sum of squares. Before #546, passing
    the raw block (which the library's own warning recommends) produced the Q2
    curve of ``Mw`` alone, rising monotonically past 0.9 with no interior
    maximum, while the pre-scaled block peaked at two components. The two are the
    same computation and must agree.
    """
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
    ldpe = pd.read_csv(folder / "LDPE" / "LDPE.csv", index_col=0)
    # The premise of the test: one column dominates the raw sum of squares.
    centred = ldpe - ldpe.mean()
    assert (centred**2).sum().max() / (centred**2).to_numpy().sum() > 0.99
    assert (centred**2).sum().idxmax() == "Mw"

    kwargs = {"max_components": 11, "cv": 7, "n_repeats": 3, "random_state": 42}
    raw = PCA.select_n_components(ldpe, **kwargs)
    with pytest.warns(SpecificationWarning, match="already centred and unit-variance"):
        prescaled = PCA.select_n_components(MCUVScaler().fit_transform(ldpe), **kwargs)

    # Not bit-identical: pre-scaling uses each column's whole-sample mean and
    # standard deviation, the raw path uses each fold's own, so the two runs
    # standardise by slightly different constants. The gap has to be negligible
    # against the curve's own standard error, which is about 0.015 here.
    gap = np.abs(raw.q2.to_numpy() - prescaled.q2.to_numpy())
    assert gap.max() < 0.001
    assert gap.max() < 0.1 * float(raw.q2_se.min())
    assert raw.q2.idxmax() == prescaled.q2.idxmax()

    # The curve turns over inside the range rather than climbing to the end,
    # which is the reading these data support and which the Mw-dominated curve
    # this used to produce did not show.
    assert raw.q2[2] > raw.q2[1]
    assert raw.q2[7] < raw.q2[2]
    assert raw.q2.loc[1:8].idxmax() == 2


def test_pca_select_n_components_scale_inside_folds_false_is_unchanged() -> None:
    """The opt-out path keeps its input-units PRESS and centred reference (#546).

    #546 changes what ``scale_inside_folds=True`` measures. ``False`` is the
    legacy contract for callers who scale their own block, and stays as it was:
    PRESS in the units of the matrix passed in, against the centred total sum of
    squares of that same matrix.
    """
    rng = np.random.default_rng(13)
    N, K = 40, 9
    T = rng.standard_normal((N, 2))
    P = rng.standard_normal((2, K))
    X = pd.DataFrame(MCUVScaler().fit_transform(pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)))))

    with pytest.warns(SpecificationWarning, match="leaks"):
        result = PCA.select_n_components(
            X, max_components=5, cv=5, n_repeats=2, random_state=0, scale_inside_folds=False
        )

    # No in-fold scale, so the two PRESS scales coincide.
    np.testing.assert_allclose(result.press_input_units.to_numpy(), result.press.to_numpy(), rtol=1e-12)
    # And Q2 is still normalised by the caller-side centred sum of squares.
    X_arr = np.asarray(X, dtype=float)
    null_ss = float(np.nansum((X_arr - np.nanmean(X_arr, axis=0)) ** 2))
    np.testing.assert_allclose(result.q2.to_numpy(), 1.0 - result.press.to_numpy() / null_ss, rtol=1e-12)


def test_pca_minka_mle_recovers_known_rank() -> None:
    """Minka's MLE picks a plausible rank on clean low-rank data."""
    rng = np.random.default_rng(1)
    N, K, true_rank = 60, 25, 3
    T = rng.standard_normal((N, true_rank)) * np.array([8.0, 5.0, 3.0])
    P = rng.standard_normal((true_rank, K))
    X = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)))
    # MLE should sit close to the true rank; sklearn is conservative on noisy
    # data so allow a 2-component slack on either side.
    n = PCA.minka_mle(X)
    assert isinstance(n, int)
    assert true_rank - 1 <= n <= true_rank + 2


def test_pca_parallel_analysis_recovers_known_rank() -> None:
    """Horn's parallel analysis picks the true rank on a clean dataset."""
    rng = np.random.default_rng(2)
    N, K, true_rank = 60, 20, 3
    T = rng.standard_normal((N, true_rank)) * np.array([7.0, 5.0, 3.0])
    P = rng.standard_normal((true_rank, K))
    X = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)))
    pa = PCA.parallel_analysis(X, n_simulations=100, random_state=0)
    assert pa.n_components == true_rank
    assert pa.observed_eigenvalues.shape == (min(N, K),)
    assert pa.null_threshold.shape == (min(N, K),)
    # The first ``true_rank`` observed eigenvalues exceed the null.
    assert (pa.observed_eigenvalues[:true_rank] > pa.null_threshold[:true_rank]).all()


def test_pca_parallel_analysis_permutation_null_adapts_to_the_data() -> None:
    """#374: the Buja-Eyuboglu surrogate permutes the real columns, so the null follows them.

    Horn's normal surrogate draws standard-normal matrices, so for a given shape it
    produces the same null whatever the data looks like. That is the right null only when
    the columns really are Gaussian. Permuting each column instead breaks the correlation
    between columns, which is what parallel analysis is testing for, while leaving each
    column's own distribution alone.
    """
    rng = np.random.default_rng(3)
    shape = (60, 12)
    gaussian = pd.DataFrame(rng.standard_normal(shape))
    heavy_tailed = pd.DataFrame(rng.standard_t(2.0, size=shape))

    normal_nulls = [
        PCA.parallel_analysis(block, n_simulations=150, surrogate="normal", random_state=0).null_threshold
        for block in (gaussian, heavy_tailed)
    ]
    permuted_nulls = [
        PCA.parallel_analysis(block, n_simulations=150, surrogate="permutation", random_state=0).null_threshold
        for block in (gaussian, heavy_tailed)
    ]

    # The normal null depends on the shape alone, so it is identical for both blocks.
    np.testing.assert_allclose(normal_nulls[0], normal_nulls[1])
    # The permutation null is built from the block itself, so it is not.
    assert not np.allclose(permuted_nulls[0], permuted_nulls[1])

    # Both blocks are independent columns, so the honest answer is no components at all,
    # and both surrogates give it.
    for surrogate in ("normal", "permutation"):
        for block in (gaussian, heavy_tailed):
            result = PCA.parallel_analysis(block, n_simulations=150, surrogate=surrogate, random_state=0)
            assert result.n_components == 0
            assert result.surrogate == surrogate


def test_pca_parallel_analysis_surrogates_agree_on_gaussian_data() -> None:
    """#374: on the data Horn assumed, the two nulls answer the same way."""
    rng = np.random.default_rng(11)
    n_samples, n_features, true_rank = 80, 10, 3
    scores = rng.standard_normal((n_samples, true_rank))
    loadings = rng.standard_normal((true_rank, n_features))
    X = pd.DataFrame(scores @ loadings + 0.6 * rng.standard_normal((n_samples, n_features)))

    by_normal = PCA.parallel_analysis(X, n_simulations=150, surrogate="normal", random_state=0)
    by_permutation = PCA.parallel_analysis(X, n_simulations=150, surrogate="permutation", random_state=0)
    assert by_normal.n_components == true_rank
    assert by_permutation.n_components == true_rank
    # Same observed scree either way: only the null moves.
    np.testing.assert_allclose(by_normal.observed_eigenvalues, by_permutation.observed_eigenvalues)


def test_pca_parallel_analysis_surrogate_is_reproducible_and_validated() -> None:
    """#374: same seed, same null; an unknown surrogate names the two that exist."""
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.standard_normal((40, 6)))
    first = PCA.parallel_analysis(X, n_simulations=50, surrogate="permutation", random_state=5)
    again = PCA.parallel_analysis(X, n_simulations=50, surrogate="permutation", random_state=5)
    np.testing.assert_allclose(first.null_threshold, again.null_threshold)

    with pytest.raises(ValueError, match=r"surrogate must be one of \('normal', 'permutation'\)"):
        PCA.parallel_analysis(X, surrogate="bootstrap")


def test_pca_select_n_components_ckf_recovers_a_known_rank() -> None:
    """#374: column-wise k-fold bottoms out at the true rank on low-rank data."""
    rng = np.random.default_rng(21)
    n_samples, n_features, true_rank = 60, 12, 3
    scores = rng.standard_normal((n_samples, true_rank))
    loadings = rng.standard_normal((true_rank, n_features))
    X = pd.DataFrame(
        scores @ loadings + 0.35 * rng.standard_normal((n_samples, n_features)),
        columns=[f"x{i}" for i in range(n_features)],
    )

    result = PCA.select_n_components(X, max_components=6, cv_scheme="ckf", random_state=0)
    assert result.n_components == true_rank
    # The Q2 curve peaks at the true rank and falls away once extra components fit noise.
    q2 = result.q2.to_numpy()
    assert int(np.argmax(q2)) + 1 == true_rank
    assert q2[true_rank] < q2[true_rank - 1]
    # One PRESS column per column-fold, so the 1-SE rule has a spread to work with.
    assert result.per_fold_press.shape == (6, 5)
    assert result.q2_per_variable.shape == (6, n_features)


def test_pca_select_n_components_ckf_is_more_optimistic_than_ekf() -> None:
    """#374: the documented difference between the two schemes, measured.

    ckf computes the scores from the retained columns only, so no held-out value predicts
    itself; but the loadings still come from an SVD of the whole block, so the held-out
    columns did reach the model. ekf holds out individual cells and imputes them, so
    nothing held out reaches it at all.

    On pure noise the difference is visible rather than theoretical: ekf's Q2 is negative
    at every component count, as it must be when there is nothing to predict, while ckf's
    can come out **positive**. That is the leakage, measured, and it is why ekf remains
    the default.
    """
    rng = np.random.default_rng(22)
    noise = pd.DataFrame(rng.standard_normal((60, 12)), columns=[f"x{i}" for i in range(12)])
    by_ckf = PCA.select_n_components(noise, max_components=5, cv_scheme="ckf", random_state=0)
    by_ekf = PCA.select_n_components(noise, max_components=5, cv_scheme="ekf", random_state=0)

    # ekf is honest about noise: nothing to predict, so nothing is predicted.
    assert (by_ekf.q2.to_numpy() < 0).all()
    # ckf is kinder at every component count, and here it goes above zero outright.
    assert (by_ckf.q2.to_numpy() > by_ekf.q2.to_numpy()).all()
    assert by_ckf.q2.to_numpy().max() > 0


def test_pca_select_n_components_ckf_reproducible_and_guarded() -> None:
    """#374: the same seed gives the same folds, and missing cells are refused by name."""
    rng = np.random.default_rng(23)
    X = pd.DataFrame(rng.standard_normal((40, 8)), columns=[f"x{i}" for i in range(8)])
    first = PCA.select_n_components(X, max_components=4, cv_scheme="ckf", random_state=7)
    again = PCA.select_n_components(X, max_components=4, cv_scheme="ckf", random_state=7)
    np.testing.assert_allclose(first.press.to_numpy(), again.press.to_numpy())

    gappy = X.copy()
    gappy.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"cv_scheme='ckf' cannot take a block with missing cells"):
        PCA.select_n_components(gappy, max_components=3, cv_scheme="ckf", random_state=0)

    with pytest.raises(ValueError, match=r"Unknown cv_scheme 'kfold'.*'ckf'"):
        PCA.select_n_components(X, max_components=3, cv_scheme="kfold")


def test_pca_select_n_components_ckf_does_not_score_an_unevaluated_count_as_perfect() -> None:
    """#675: a component count that no column fold can evaluate must not win with a PRESS of 0.

    Five columns in five folds leave four retained columns per fold, too few to determine
    five scores, so every fold skips A = 5. Those folds are NaN, and the sum over them was
    taken with ``nansum``, which is 0 for all-NaN: pure noise came out with Q2 = 1 at A = 5,
    and that count was recommended. Now the count has no PRESS and no Q2.
    """
    rng = np.random.default_rng(0)
    noise = MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((30, 5))))
    result = PCA.select_n_components(noise, cv_scheme="ckf")
    assert np.isnan(result.press.iloc[-1])
    assert np.isnan(result.q2.iloc[-1])
    assert result.q2_per_variable.iloc[-1].isna().all(), "no column was predicted, so none has a Q2 of 1"
    assert np.isfinite(result.press.iloc[:-1]).all()
    assert result.n_components < 5


def test_pca_select_n_components_ckf_does_not_score_a_count_some_folds_skip() -> None:
    """#675: seven columns in five groups are held out two, two, one, one and one at a time.

    At six components the two groups of two leave five columns in, too few, while the
    three groups of one still predict their column. A total over those three alone would
    count three columns' errors against seven columns' worth of reference, and look like
    a better fit than any count all five groups score.
    """
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((40, 7)), columns=[f"x{i}" for i in range(7)])
    result = PCA.select_n_components(X, max_components=7, cv_scheme="ckf", random_state=0)
    assert np.isfinite(result.press.loc[1:5]).all()
    assert result.press.loc[6:].isna().all()
    assert result.q2.loc[6:].isna().all()
    # The columns of the three single-column groups keep their own Q2 at six components.
    assert sorted(result.q2_per_variable.loc[6].notna().to_numpy().tolist()) == [False] * 4 + [True] * 3
    assert result.per_fold_press.loc[6].isna().sum() == 2
    assert result.n_components <= 5


def test_pca_select_n_components_refuses_data_too_small_for_any_component() -> None:
    """With one row no component count can be cross-validated, and the error says so."""
    with pytest.raises(ValueError, match=r"^No components can be evaluated; the data is too small\.$"):
        PCA.select_n_components(np.ones((1, 3)))


def test_pca_select_n_components_refuses_an_all_nan_press(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scheme that scores no component count says so rather than recommending one."""
    from process_improve.multivariate import _pca

    def scores_nothing(Z: np.ndarray, max_components: int, *, method: str) -> tuple:
        width = Z.shape[1]
        return np.full(max_components, np.nan), np.full((max_components, width), np.nan), 1.0, np.ones(width)

    monkeypatch.setattr(_pca, "_leverage_corrected_press", scores_nothing)
    X = pd.DataFrame(np.random.default_rng(6).standard_normal((30, 5)))
    with pytest.raises(
        RuntimeError,
        match=r"^Cross-validation produced NaN PRESS for every component count; no recommendation can be made\.$",
    ):
        PCA.select_n_components(X, cv_scheme="sacv", max_components=3)


def test_pca_parallel_analysis_without_scaling_compares_in_the_units_of_x() -> None:
    """scale=False leaves X in its own units, so variances far above the unit null keep every component.

    The same noise, standardised first, keeps none: the scaling is what puts the observed
    eigenvalues on the null's footing.
    """
    loud_noise = 100.0 * np.random.default_rng(7).standard_normal((30, 5))
    as_given = PCA.parallel_analysis(loud_noise, scale=False, random_state=0)
    standardised = PCA.parallel_analysis(loud_noise, random_state=0)

    assert as_given.n_components == 5
    assert np.all(as_given.observed_eigenvalues > as_given.null_threshold)
    assert standardised.n_components == 0


def test_pca_consensus_parallel_analysis_is_scaled_whatever_scale_inside_folds_is() -> None:
    """#513: the consensus compares the eigenvalues against the N(0, 1) null on a common footing.

    ``scale_inside_folds`` decides how the cross-validation folds are scaled. It was also
    passed on as ``parallel_analysis(scale=...)``, so with ``scale_inside_folds=False`` the
    unscaled eigenvalues of loud noise were compared against a unit-variance null and every
    component was retained. Scaling an already-scaled block changes nothing, so the
    parallel analysis now always scales, and the noise keeps no component either way.
    """
    loud_noise = pd.DataFrame(100.0 * np.random.default_rng(7).standard_normal((30, 5)))
    expected = PCA.parallel_analysis(loud_noise, random_state=0).n_components
    assert expected == 0

    with warnings.catch_warnings():
        # scale_inside_folds=False warns that it leaks the scaling into the folds.
        warnings.simplefilter("ignore", SpecificationWarning)
        for inside in (True, False):
            result = PCA.select_n_components(
                loud_noise, max_components=3, scale_inside_folds=inside, random_state=0, return_consensus=True
            )
            assert result.parallel_analysis_n_components == expected, f"scale_inside_folds={inside}"


def test_pca_parallel_analysis_pure_noise_returns_zero() -> None:
    """On pure noise PA correctly retains few components (and may return 0)."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((50, 15)))
    pa = PCA.parallel_analysis(X, n_simulations=100, random_state=0)
    # On a Gaussian noise matrix the first observed eigenvalue is at or
    # below the 95th-percentile null - parallel analysis correctly trims to
    # a handful of components at most.
    assert pa.n_components <= 1


def test_pca_select_n_components_consensus_mode() -> None:
    """return_consensus=True surfaces Minka MLE + parallel analysis alongside ekf."""
    rng = np.random.default_rng(4)
    N, K, true_rank = 60, 20, 3
    T = rng.standard_normal((N, true_rank)) * np.array([7.0, 5.0, 3.0])
    P = rng.standard_normal((true_rank, K))
    X = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)))

    result = PCA.select_n_components(
        X,
        max_components=8,
        cv=5,
        n_repeats=2,
        random_state=0,
        return_consensus=True,
    )
    assert {"minka_n_components", "parallel_analysis_n_components", "consensus", "consensus_counts"} <= set(
        result.keys()
    )
    assert result.consensus in {"agree", "disagree"}
    # On clean low-rank data the three rules should agree.
    assert result.consensus == "agree"
    assert result.consensus_counts[0] == result.n_components

    # Backwards compat: without return_consensus the consensus fields are
    # absent (the existing API is unchanged).
    plain = PCA.select_n_components(X, max_components=8, cv=5, random_state=0)
    assert "minka_n_components" not in plain
    assert "consensus" not in plain


def test_pca_score_contributions() -> None:
    """Contributions decompose the score exactly, one term per variable."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((30, 5)), columns=[f"V{i}" for i in range(1, 6)])
    X = MCUVScaler().fit_transform(X)

    pca = PCA(n_components=3).fit(X)

    contrib = pca.score_contributions(X, component=1)
    assert isinstance(contrib, pd.DataFrame)
    assert contrib.shape == (30, 5)
    assert list(contrib.columns) == [f"V{i}" for i in range(1, 6)]
    assert list(contrib.index) == list(X.index)

    # --- The defining property (Miller, Swanson and Heckler, 1994, eq. 3):
    #     the per-variable terms sum to the score they decompose.
    for a in (1, 2, 3):
        terms = pca.score_contributions(X, component=a)
        assert terms.sum(axis=1).to_numpy() == pytest.approx(pca.scores_[a].to_numpy(), abs=1e-12)
        # ... and each term is the data value times its loading.
        expected = X.to_numpy() * pca.loadings_.to_numpy()[:, a - 1]
        assert terms.to_numpy() == pytest.approx(expected, abs=1e-12)

    # --- The diagnosis must depend on the observation, not only on the model.
    # Anything proportional to the loading vector ranks the variables the same
    # way for every observation, carrying no per-observation information.
    rankings = {tuple(np.argsort(-contrib.iloc[i].abs().to_numpy())) for i in range(len(X))}
    assert len(rankings) > 1

    # --- A variable sitting at its mean contributes nothing, whatever its loading.
    at_mean = X.copy()
    at_mean.iloc[0, 2] = 0.0
    assert pca.score_contributions(at_mean, component=1).iloc[0, 2] == pytest.approx(0.0, abs=1e-12)

    # --- Scaling changes the units, not the pattern within a row.
    for scaling in ("maximum", "within"):
        scaled = pca.score_contributions(X, component=1, scaling=scaling)
        ratio = scaled.iloc[0].to_numpy() / contrib.iloc[0].to_numpy()
        assert ratio == pytest.approx(np.full(5, ratio[0]), rel=1e-10)
    assert np.abs(pca.score_contributions(X, component=1, scaling="maximum").to_numpy()).max() == pytest.approx(1.0)

    with pytest.raises(ValueError, match="scaling must be one of"):
        pca.score_contributions(X, scaling="loud")
    with pytest.raises(ValueError, match="1-based index"):
        pca.score_contributions(X, component=4)


def test_score_contributions_rejects_a_score_vector() -> None:
    """A score vector cannot carry the data the calculation needs."""
    rng = np.random.default_rng(3)
    X = MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((25, 4)), columns=list("abcd")))
    pca = PCA(n_components=2).fit(X)

    with pytest.raises(TypeError, match="not a score vector"):
        pca.score_contributions(pca.scores_.iloc[0])
    with pytest.raises(TypeError, match="not a score vector"):
        pca.score_contributions(pca.scores_.iloc[0].to_numpy())
    with pytest.raises(TypeError, match="t2_contributions"):
        pca.score_contributions(X, weighted=True)
    with pytest.raises(TypeError, match="group_contributions"):
        pca.score_contributions(X, t_end=pca.scores_.iloc[1])


def test_score_contributions_selector_and_argument_errors() -> None:
    """The selector and argument guards each report what was actually wrong."""
    rng = np.random.default_rng(5)
    X = MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((12, 3)), columns=list("abc")))
    pca = PCA(n_components=2).fit(X)

    # An unknown keyword is reported as such, rather than being mistaken for a
    # call that passed a score vector.
    with pytest.raises(TypeError, match="unexpected keyword argument: colour"):
        pca.score_contributions(X, colour="red")

    # An empty group has no mean to take.
    with pytest.raises(ValueError, match="group selected no observations"):
        pca.group_contributions(X, group=[])
    with pytest.raises(ValueError, match="reference selected no observations"):
        pca.group_contributions(X, group=[0], reference=[])

    # A boolean mask has to line up with the rows it is masking.
    with pytest.raises(ValueError, match="boolean mask of length 3, but X has 12 rows"):
        pca.group_contributions(X, group=[True, False, True])

    # Selection is by label. Positions go through X.index[...], and an entry
    # that is not a label is refused rather than quietly read as a position:
    # on a frame indexed 1..N that fallback would make [0, 1, 2] mean positions
    # and [10, 11, 12] mean labels.
    assert pca.group_contributions(X, group=X.index[:2]).to_numpy() == pytest.approx(
        pca.score_contributions(X, component=1).iloc[:2].mean(axis=0).to_numpy(), abs=1e-12
    )
    with pytest.raises(ValueError, match="not index labels of X"):
        pca.group_contributions(X, group=[999])

    shifted = X.copy()
    shifted.index = range(1, len(X) + 1)
    pca_shifted = PCA(n_components=2).fit(shifted)
    with pytest.raises(ValueError, match=r"not index labels of X: \[0\]"):
        pca_shifted.group_contributions(shifted, group=[0, 1, 2])


def test_pca_group_contributions() -> None:
    """Group contributions sum to the average score, or to the shift between groups."""
    rng = np.random.default_rng(11)
    X = MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"V{i}" for i in range(1, 6)]))
    pca = PCA(n_components=2).fit(X)
    early, late = list(X.index[:8]), list(X.index[20:30])

    against_centre = pca.group_contributions(X, group=early)
    assert isinstance(against_centre, pd.Series)
    assert list(against_centre.index) == list(X.columns)
    assert against_centre.sum() == pytest.approx(pca.scores_.loc[early, 1].mean(), abs=1e-12)

    shift = pca.group_contributions(X, group=early, reference=late, component=2)
    expected = pca.scores_.loc[early, 2].mean() - pca.scores_.loc[late, 2].mean()
    assert shift.sum() == pytest.approx(expected, abs=1e-12)

    # A single-observation group is the corresponding row of score_contributions.
    one = pca.group_contributions(X, group=[X.index[3]])
    assert one.to_numpy() == pytest.approx(pca.score_contributions(X, component=1).iloc[3].to_numpy(), abs=1e-12)

    # A boolean mask is accepted alongside labels.
    mask = [i < 8 for i in range(len(X))]
    assert pca.group_contributions(X, group=mask).to_numpy() == pytest.approx(against_centre.to_numpy(), abs=1e-12)
    with pytest.raises(ValueError, match="not index labels of X"):
        pca.group_contributions(X, group=["not-a-label"])


def test_pca_t2_spe_contributions() -> None:
    """t2_contributions sum to T2; spe_contributions squares sum to SPE (PCA)."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.standard_normal((35, 6)), columns=[f"V{i}" for i in range(1, 7)])
    Xs = MCUVScaler().fit_transform(X)

    for algorithm in ("svd", "nipals"):
        pca = PCA(n_components=3, algorithm=algorithm).fit(Xs)

        t2c = pca.t2_contributions(Xs)
        spec = pca.spe_contributions(Xs)

        # Shape / labels mirror the input block.
        assert t2c.shape == (35, 6)
        assert spec.shape == (35, 6)
        assert list(t2c.columns) == list(Xs.columns)
        assert list(t2c.index) == list(Xs.index)

        # Invariant 1: per-observation T2 contributions sum to Hotelling's T2.
        assert t2c.sum(axis=1).to_numpy() == pytest.approx(pca.hotellings_t2_.iloc[:, -1].to_numpy(), abs=1e-9)
        # Invariant 2: squared SPE contributions sum to SPE**2.
        assert (spec**2).sum(axis=1).to_numpy() == pytest.approx(pca.spe_.iloc[:, -1].to_numpy() ** 2, abs=1e-9)

        # A component subset sums to the cumulative T2 of those components.
        sub = pca.t2_contributions(Xs, components=[1, 2]).sum(axis=1).to_numpy()
        assert sub == pytest.approx(pca.hotellings_t2_.iloc[:, 1].to_numpy(), abs=1e-9)

        # The standalone functions match the bound methods.
        assert t2_contributions(pca, Xs).to_numpy() == pytest.approx(t2c.to_numpy())
        assert spe_contributions(pca, Xs).to_numpy() == pytest.approx(spec.to_numpy())

        # Array-like input is accepted and gives the same result as the frame.
        assert pca.t2_contributions(Xs.to_numpy()).to_numpy() == pytest.approx(t2c.to_numpy())
        assert pca.spe_contributions(Xs.to_numpy()).to_numpy() == pytest.approx(spec.to_numpy())

    # --- Error handling ---
    with pytest.raises(ValueError, match="not fitted"):
        PCA(n_components=2).t2_contributions(Xs)
    with pytest.raises(ValueError, match="components"):
        PCA(n_components=3).fit(Xs).t2_contributions(Xs, components=[1, 9])


def test_pls_t2_spe_contributions(
    fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """t2/spe contributions hold for PLS, on synthetic and the real LDPE data."""
    # Synthetic PLS data with a known coefficient matrix.
    rng = np.random.default_rng(21)
    X = pd.DataFrame(rng.standard_normal((40, 6)), columns=[f"x{i}" for i in range(6)])
    beta = rng.standard_normal((6, 2))
    Y = pd.DataFrame(X.values @ beta + 0.3 * rng.standard_normal((40, 2)), columns=["y0", "y1"])
    Xs = MCUVScaler().fit_transform(X)
    Ys = MCUVScaler().fit_transform(Y)
    pls = PLS(n_components=3).fit(Xs, Ys)

    t2c = pls.t2_contributions(Xs)
    spec = pls.spe_contributions(Xs)
    assert t2c.sum(axis=1).to_numpy() == pytest.approx(pls.hotellings_t2_.iloc[:, -1].to_numpy(), abs=1e-9)
    assert (spec**2).sum(axis=1).to_numpy() == pytest.approx(pls.spe_.iloc[:, -1].to_numpy() ** 2, abs=1e-9)
    # SPE contributions carry signs (they are residuals, not magnitudes).
    assert (spec.to_numpy() < 0).any()

    # Real LDPE dataset (SIMCA reference data) - exercise the same invariants.
    data = fixture_pls_ldpe_example
    X_ldpe = MCUVScaler().fit_transform(data["X"])
    Y_ldpe = MCUVScaler().fit_transform(data["Y"])
    pls_ldpe = PLS(n_components=data["A"]).fit(X_ldpe, Y_ldpe)
    assert pls_ldpe.t2_contributions(X_ldpe).sum(axis=1).to_numpy() == pytest.approx(
        pls_ldpe.hotellings_t2_.iloc[:, -1].to_numpy(), abs=1e-6
    )
    assert (pls_ldpe.spe_contributions(X_ldpe) ** 2).sum(axis=1).to_numpy() == pytest.approx(
        pls_ldpe.spe_.iloc[:, -1].to_numpy() ** 2, abs=1e-6
    )


def test_t2_contributions_count_the_components_that_fit_counts_in_t2() -> None:
    """A small score sd still counts, as in ``hotellings_t2_``; a ~0 one adds nothing, even off the plane.

    Two latent factors drive four columns. With noise of 1e-6 on one column a third
    component has a score sd near 5e-7: small, but fit() counts it in hotellings_t2_, so
    the row sums are ``sum_a t_a^2 / s_a^2`` over all three. Without the noise the data
    span a plane, the third sd is ~1e-16 and fit() leaves that component out; a new row
    off the plane scores on it all the same, and must not gain ``t_3^2`` from it.
    """
    rng = np.random.default_rng(0)
    plane = rng.standard_normal((30, 2)) @ rng.standard_normal((2, 4))

    noisy = plane.copy()
    noisy[:, 3] += 1e-6 * rng.standard_normal(30)
    noisy = pd.DataFrame(noisy - noisy.mean(axis=0), columns=list("abcd"))
    pca = PCA(n_components=3).fit(noisy)
    s = pca.scaling_factor_for_scores_.to_numpy()
    assert 1e-8 < s[2] < 1e-5
    by_hand = ((pca.scores_.to_numpy() / s) ** 2).sum(axis=1)
    np.testing.assert_allclose(pca.t2_contributions(noisy).sum(axis=1), by_hand, rtol=1e-6)

    flat = pd.DataFrame(plane - plane.mean(axis=0), columns=list("abcd"))
    pca = PCA(n_components=3).fit(flat)
    s = pca.scaling_factor_for_scores_.to_numpy()
    assert s[2] < 1e-12
    new = pd.DataFrame([[1.0, -2.0, 0.5, 3.0]], columns=list("abcd"))
    t = new.to_numpy() @ pca.loadings_.to_numpy()
    assert abs(t[0, 2]) > 1.0  # the new row is well off the plane
    np.testing.assert_allclose(pca.t2_contributions(new).sum(axis=1), [np.sum(t[0, :2] ** 2 / s[:2] ** 2)], rtol=1e-12)


def test_t2_contributions_leave_out_a_component_by_the_bound_fit_uses() -> None:
    """A score sd counts when it exceeds ``epsqrt * max(1, largest sd)``; at the bound it does not.

    A stand-in model whose loadings are the identity, so each score is its own variable
    and each contribution is ``x_k^2 / s_k^2``, or 0 for a component left out. The largest
    sd is 1.5, so the bound is ``1.5 * 2**-26`` (epsqrt is exactly ``2**-26``): an sd of
    exactly the bound is left out, ``1.75 * 2**-26`` is kept, and ``2**-26`` is left out.
    """
    assert epsqrt == 2.0**-26
    s = np.array([1.5, 1.5 * 2.0**-26, 1.75 * 2.0**-26, 2.0**-26])
    model = SimpleNamespace(
        scores_=None, loadings_=pd.DataFrame(np.eye(4), index=list("abcd")), scaling_factor_for_scores_=pd.Series(s)
    )
    contributions = t2_contributions(model, pd.DataFrame([[3.0, 1.0, 1.0, 1.0]], columns=list("abcd")))
    np.testing.assert_allclose(contributions.to_numpy(), [[4.0, 0.0, 2.0**52 / 3.0625, 0.0]], rtol=1e-15)


def _rows_on_a_plane(units: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return 30 centred rows that span a plane in four columns, and one row well off that plane."""
    rng = np.random.default_rng(0)
    plane = rng.standard_normal((30, 2)) @ rng.standard_normal((2, 4))
    flat = pd.DataFrame((plane - plane.mean(axis=0)) * units, columns=list("abcd"))
    return flat, pd.DataFrame([[1.0, -2.0, 0.5, 3.0]], columns=list("abcd")) * units


def test_pca_diagnose_and_project_leave_out_of_t2_the_component_fit_leaves_out() -> None:
    """A new row's T² counts only the components with spread, as ``hotellings_t2_`` does (#1026).

    Thirty rows on a plane: a third component has a score sd near 1e-16, which fit() leaves
    out of T². A row off the plane scores well away from 0 on it, and diagnose and project
    gave it a T² of about 3.5e32. Its T² is ``t_1^2 / s_1^2 + t_2^2 / s_2^2``, from its scores
    by hand, and a training row's is the same sum over its fitted scores, not that sum plus
    noise over noise from the third component.
    """
    flat, new = _rows_on_a_plane()
    pca = PCA(n_components=3).fit(flat)
    s = pca.scaling_factor_for_scores_.to_numpy()
    assert s[2] < 1e-12
    t = new.to_numpy() @ pca.loadings_.to_numpy()
    assert abs(t[0, 2]) > 1.0  # the new row is well off the plane
    by_hand = np.cumsum(t[0, :2] ** 2 / s[:2] ** 2)
    np.testing.assert_allclose(pca.diagnose(new).hotellings_t2.to_numpy(), [[*by_hand, by_hand[1]]], rtol=1e-12)
    np.testing.assert_allclose(pca.project(new).hotellings_t2.to_numpy(), [by_hand[1]], rtol=1e-12)

    training = np.sum((pca.scores_.to_numpy()[:, :2] / s[:2]) ** 2, axis=1)
    np.testing.assert_allclose(pca.hotellings_t2_.iloc[:, -1], training, rtol=1e-12)
    np.testing.assert_allclose(pca.diagnose(flat).hotellings_t2.iloc[:, -1], training, rtol=1e-10)
    np.testing.assert_allclose(pca.project(flat).hotellings_t2, training, rtol=1e-10)


@pytest.mark.dataset
def test_pca_t2_of_a_held_out_ldpe_row_with_as_many_components_as_rows() -> None:
    """``n_components=None`` on ten LDPE rows fits ten components, the last one with no spread (#1026).

    Ten centred rows span nine dimensions, so the tenth component's score sd is ~1e-16. The
    eleventh row, held out, gets the T² of the first nine components, worked from its scores.
    """
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
    ldpe = pd.read_csv(folder / "LDPE" / "LDPE.csv", index_col=0)
    scaler = MCUVScaler().fit(ldpe.iloc[:10])
    pca = PCA(n_components=None).fit(scaler.transform(ldpe.iloc[:10]))
    s = pca.scaling_factor_for_scores_.to_numpy()
    assert pca.n_components_ == 10
    assert s[9] < 1e-12 < s[8]
    held_out = scaler.transform(ldpe.iloc[[10]])
    t = held_out.to_numpy() @ pca.loadings_.to_numpy()
    by_hand = np.sum(t[0, :9] ** 2 / s[:9] ** 2)
    np.testing.assert_allclose(pca.diagnose(held_out).hotellings_t2.iloc[0, -1], by_hand, rtol=1e-10)
    np.testing.assert_allclose(pca.project(held_out).hotellings_t2.iloc[0], by_hand, rtol=1e-10)


def test_pls_diagnose_project_and_invert_leave_out_of_t2_the_component_fit_leaves_out() -> None:
    """PLS's T² of a new row, and of an inverted target, counts only the components with spread (#1026).

    In units of 1e12, unscaled, the noise left after two components still has a sum of squares
    above PLS's absolute floor, so a third component is fitted to it, with a score sd below 1e-3
    against about 3e12: fit() leaves it out of T². A row off the plane scores on it all the same.
    Its T² is ``t_1^2 / s_1^2 + t_2^2 / s_2^2``, by hand, and so is the T² of invert's solution
    from its scores, which diagnose gives ``x_new`` too.
    """
    flat, new = _rows_on_a_plane(units=1e12)
    y = pd.DataFrame({"y": flat.to_numpy() @ [1e-12, 2e-12, 0.0, 0.0]})
    y += 0.1 * np.random.default_rng(1).standard_normal((30, 1))
    with pytest.warns(SpecificationWarning, match="nothing left in Y that X can explain"):
        pls = PLS(n_components=3, scale=False, warn_on_uncentred=False).fit(flat, y - y.mean())
    s = pls.scaling_factor_for_scores_.to_numpy()
    assert s[2] < 1e-6 * s[0]
    t = new.to_numpy() @ pls.direct_weights_.to_numpy()
    assert abs(t[0, 2]) > 1e6 * s[2]  # the new row is well off the plane
    by_hand = np.sum(t[0, :2] ** 2 / s[:2] ** 2)
    np.testing.assert_allclose(pls.diagnose(new).hotellings_t2, [by_hand], rtol=1e-10)
    np.testing.assert_allclose(pls.project(new).hotellings_t2, [by_hand], rtol=1e-10)

    training = np.sum((pls.scores_.to_numpy()[:, :2] / s[:2]) ** 2, axis=1)
    np.testing.assert_allclose(pls.hotellings_t2_.iloc[:, -1], training, rtol=1e-12)
    np.testing.assert_allclose(pls.diagnose(flat).hotellings_t2, training, rtol=1e-8)
    np.testing.assert_allclose(pls.project(flat).hotellings_t2, training, rtol=1e-8)

    inverted = pls.invert(y_desired=1.0)
    tau = inverted.scores.to_numpy()
    assert (tau[2] / s[2]) ** 2 > 100.0  # what the third component added
    assert inverted.hotellings_t2 == pytest.approx(np.sum(tau[:2] ** 2 / s[:2] ** 2), rel=1e-12, abs=1e-30)
    np.testing.assert_allclose(
        pls.diagnose(inverted.x_new.to_frame().T).hotellings_t2, [inverted.hotellings_t2], rtol=1e-6, atol=1e-20
    )


def _multiblock_on_a_plane(estimator: type, flat: pd.DataFrame) -> MBPCA | MBPLS:
    """Fit three components of ``estimator`` to two blocks of rows on a plane, columns a, b and c, d."""
    blocks = {"A": flat[["a", "b"]], "B": flat[["c", "d"]]}
    model = estimator(n_components=3)
    if estimator is MBPLS:
        with pytest.warns(SpecificationWarning, match="nothing left in Y that X can explain"):
            return model.fit(blocks, pd.DataFrame({"y": flat["a"] + 2.0 * flat["b"]}))
    return model.fit(blocks)


@pytest.mark.parametrize("estimator", [MBPCA, MBPLS])
def test_multiblock_super_t2_leaves_out_a_super_score_with_no_spread(estimator: type) -> None:
    """MBPCA's and MBPLS's super T², fitted and of a new row, count only the components with spread (#1026).

    Two blocks of rows on a plane: a third super score has a variance near 1e-31. It added
    noise over noise to every fitted row's T², and about 3e28 to a new row's off the plane.
    Each T² is now ``sum_a t_a^2 / var_a`` over the first two components, by hand.
    """
    flat, new = _rows_on_a_plane()
    model = _multiblock_on_a_plane(estimator, flat)
    variance = model.explained_variance_
    assert variance[2] < 1e-24
    fitted = model.super_scores_.to_numpy()
    by_hand = np.cumsum(np.column_stack([fitted[:, :2] ** 2 / variance[:2], np.zeros(30)]), axis=1)
    np.testing.assert_allclose(model.super_hotellings_t2_.to_numpy(), by_hand, rtol=1e-10)

    result = model.diagnose({"A": new[["a", "b"]], "B": new[["c", "d"]]})
    scores = result.super_scores.to_numpy()
    assert scores[0, 2] ** 2 / variance[2] > 1e20  # the new row is well off the plane
    np.testing.assert_allclose(result.hotellings_t2, [np.sum(scores[0, :2] ** 2 / variance[:2])], rtol=1e-10)


@pytest.mark.parametrize("estimator", [MBPCA, MBPLS])
def test_multiblock_block_t2_leaves_out_a_block_score_with_no_spread(estimator: type) -> None:
    """A block's T² counts only the block components with spread, as the super T² does (#1028).

    Each block of rows on a plane holds two directions, so its third block score has a second
    moment near 1e-32. Divided by it, that score added noise over noise, of order 1, to every
    row's block T². Each block T² is now ``sum_a t_a^2 / m_a`` over the first two components,
    ``m_a = sum_i t_ia^2 / (N - 1)``, by hand.
    """
    flat, _ = _rows_on_a_plane()
    model = _multiblock_on_a_plane(estimator, flat)
    for name in ("A", "B"):
        scores = model.block_scores_[name].to_numpy()
        second_moment = np.sum(scores**2, axis=0) / 29
        assert second_moment[2] < 1e-24 * second_moment[0]
        by_hand = np.cumsum(np.column_stack([scores[:, :2] ** 2 / second_moment[:2], np.zeros(30)]), axis=1)
        np.testing.assert_allclose(model.block_hotellings_t2_[name].to_numpy(), by_hand, rtol=1e-12)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        # Three rows with a score: each column's second moment divides by 3 - 1. The third
        # column, of sd near 1e-17 against about 2, is left out.
        (
            [[1.0, 2.0, 1e-17], [np.nan] * 3, [2.0, -1.0, 0.0], [-2.0, 1.0, -1e-17]],
            [[1 / 4.5, 1 / 4.5 + 4 / 3], [np.nan] * 2, [4 / 4.5, 4 / 4.5 + 1 / 3], [4 / 4.5, 4 / 4.5 + 1 / 3]],
        ),
        # Two rows with a score divide by 1.
        (
            [[1.0, 3.0], [np.nan] * 2, [2.0, -1.0], [np.nan] * 2],
            [[1 / 5, 1 / 5 + 9 / 10], [np.nan] * 2, [4 / 5, 4 / 5 + 1 / 10], [np.nan] * 2],
        ),
        # One row with a score divides by 1 too, not by 0.
        ([[np.nan] * 2, [3.0, -0.5]], [[np.nan] * 2, [1.0, 2.0]]),
    ],
)
def test_cumulative_block_t2_by_hand(scores: list, expected: list) -> None:
    """Each block score over its second moment about zero, ``sum t^2 / max(1, n - 1)`` over its n scores (#1028)."""
    expected = np.array(expected)
    t2 = cumulative_block_t2(np.array(scores))
    np.testing.assert_allclose(t2[:, : expected.shape[1]], expected, rtol=1e-14)
    np.testing.assert_array_equal(t2[:, -1], t2[:, expected.shape[1] - 1])


def test_pca_detect_outliers() -> None:
    """Test detect_outliers method with a synthetic dataset containing known outliers."""
    rng = np.random.default_rng(123)
    # Normal data: 40 observations, 4 variables, mild correlation
    X_normal = rng.standard_normal((40, 4))
    X_normal[:, 1] = X_normal[:, 0] * 0.8 + rng.standard_normal(40) * 0.3
    X_normal[:, 3] = X_normal[:, 2] * 0.6 + rng.standard_normal(40) * 0.5

    # Inject 2 obvious outliers
    X = np.vstack([X_normal, [[15, 15, 15, 15]], [[0.1, 0.1, 12, 12]]])
    X = pd.DataFrame(X, columns=["A", "B", "C", "D"])
    X = MCUVScaler().fit_transform(X)

    pca = PCA(n_components=2).fit(X)

    # --- Basic contract ---
    outliers = pca.detect_outliers(conf_level=0.95)
    assert isinstance(outliers, list)
    assert all(isinstance(o, dict) for o in outliers)

    # Each dict has the required keys
    required_keys = {
        "observation",
        "outlier_types",
        "spe",
        "hotellings_t2",
        "spe_limit",
        "hotellings_t2_limit",
        "severity",
    }
    for o in outliers:
        assert set(o.keys()) == required_keys
        assert isinstance(o["outlier_types"], list)
        assert all(t in ("spe", "hotellings_t2") for t in o["outlier_types"])
        assert len(o["outlier_types"]) >= 1

    # --- Sorted by severity descending ---
    severities = [o["severity"] for o in outliers]
    assert severities == sorted(severities, reverse=True)

    # --- The injected outliers (indices 40, 41) should be detected ---
    detected_obs = {o["observation"] for o in outliers}
    assert 40 in detected_obs, "The extreme outlier at index 40 should be detected"
    assert 41 in detected_obs, "The outlier at index 41 should be detected"

    # --- conf_level validation ---
    with pytest.raises(ValueError, match="conf_level must be between"):
        pca.detect_outliers(conf_level=0.5)

    with pytest.raises(ValueError, match="conf_level must be between"):
        pca.detect_outliers(conf_level=1.0)


def _pls_identity_residuals(model: PLS) -> dict[str, float]:
    """
    Return the worst-case departure from each of the five structural PLS identities.

    Keys, with ``W`` = ``x_weights_``, ``P`` = ``x_loadings_``, ``T`` = ``scores_``,
    ``R`` = ``direct_weights_`` and ``S`` = ``scaling_factor_for_scores_``:

    ``s_definition``
        ``S`` against ``sqrt(diag(T'T) / (N - 1))``, i.e. the score standard
        deviation at ddof=1.
    ``w_orthonormal``
        ``W'W`` against the identity. This one is specific to PLS; PCA's ``W`` is
        not separate from its loadings.
    ``pw_lower`` / ``pw_diag``
        ``P'W`` must be upper triangular with a unit diagonal.
    ``wr_lower`` / ``wr_diag``
        ``W'R`` likewise. When ``W'W = I`` it equals ``inv(P'W)``, and the inverse
        of a unit-diagonal upper-triangular matrix has the same shape.
    ``rp_identity``
        ``R'P`` against the identity, which follows from ``R = W inv(P'W)``.
    """
    weights = model.x_weights_.to_numpy()
    loadings = model.x_loadings_.to_numpy()
    scores = model.scores_.to_numpy()
    rotations = model.direct_weights_.to_numpy()
    n_samples, n_components = scores.shape
    identity = np.eye(n_components)

    score_sd = np.asarray(model.scaling_factor_for_scores_).ravel()
    p_w = loadings.T @ weights
    w_r = weights.T @ rotations

    return {
        "s_definition": np.abs(score_sd - np.sqrt(np.diag(scores.T @ scores) / (n_samples - 1))).max(),
        "w_orthonormal": np.abs(weights.T @ weights - identity).max(),
        "pw_lower": np.abs(np.tril(p_w, -1)).max(),
        "pw_diag": np.abs(np.diag(p_w) - 1).max(),
        "wr_lower": np.abs(np.tril(w_r, -1)).max(),
        "wr_diag": np.abs(np.diag(w_r) - 1).max(),
        "rp_identity": np.abs(rotations.T @ loadings - identity).max(),
    }


class TestLoadingPlotResolvesEveryModel:
    """`loading_plot` must resolve a matrix for each model and each documented type.

    Regression tests for #568 and #564. The previous version computed a "PCA default"
    eagerly, on the line before the branch meant to override it, so every model without
    `loadings_` raised there whatever `loadings_type` asked for. `PLS.loading_plot()`
    was broken for all five documented values, including the two its own docstring gives
    as examples, and nothing in the suite called it.
    """

    @staticmethod
    def _pls() -> PLS:
        rng = np.random.default_rng(0)
        x_data = pd.DataFrame(rng.normal(size=(40, 6)))
        y_data = pd.DataFrame({"y": x_data.to_numpy() @ rng.normal(size=6)})
        return PLS(n_components=2).fit(MCUVScaler().fit_transform(x_data), MCUVScaler().fit_transform(y_data))

    @pytest.mark.parametrize("loadings_type", ["p", "w", "w*", "w*c", "c"])
    def test_pls_supports_every_documented_loadings_type(self, loadings_type: str) -> None:
        assert isinstance(self._pls().loading_plot(loadings_type=loadings_type), go.Figure)

    def test_pls_p_loadings_are_the_x_loadings(self) -> None:
        """For PLS the P loadings are `x_loadings_`, not a PCA-style `loadings_`."""
        model = self._pls()
        trace = model.loading_plot(loadings_type="p").data[0]
        assert trace.x == pytest.approx(model.x_loadings_.loc[:, 1].to_numpy())
        assert trace.y == pytest.approx(model.x_loadings_.loc[:, 2].to_numpy())

    def test_pca_p_loadings_are_unchanged(self) -> None:
        rng = np.random.default_rng(1)
        x_data = MCUVScaler().fit_transform(pd.DataFrame(rng.normal(size=(40, 5))))
        model = PCA(n_components=2).fit(x_data)
        trace = model.loading_plot().data[0]
        assert trace.x == pytest.approx(model.loadings_.loc[:, 1].to_numpy())

    def test_unrecognised_loadings_type_is_rejected(self) -> None:
        """Previously this silently plotted whatever the eager default had resolved."""
        with pytest.raises(ValueError, match=r"loadings_type='zzz' is not recognized"):
            self._pls().loading_plot(loadings_type="zzz")

    def test_a_model_with_no_loadings_at_all_is_reported(self) -> None:
        """The resolver's own failure path: nothing to draw, on the model or its parent.

        Exercised directly because `loading_plot` would stop earlier, in
        `plot_pre_checks`, for an object this incomplete.
        """

        class Bare:
            """Exposes none of `loadings_`, `x_loadings_` or `w_loadings_super`."""

        with pytest.raises(AttributeError, match=r"Bare exposes no X-space loadings to plot"):
            _x_space_loadings(Bare())

    def test_the_parent_is_consulted_when_the_accessor_has_nothing(self) -> None:
        """The TPLS route: the accessor carries no loadings, so `_parent` supplies them."""

        class Parent:
            w_loadings_super = pd.DataFrame({1: [0.1, 0.2], 2: [0.3, 0.4]})

        class Accessor:
            _parent = Parent()

        resolved = _x_space_loadings(Accessor())
        assert resolved is Parent.w_loadings_super


def test_pls_structural_identities_synthetic() -> None:
    """The five structural PLS identities hold to machine precision on complete data.

    None of these was asserted anywhere in the suite before (#213), even though the
    whole NIPALS deflation scheme depends on them.
    """
    rng = np.random.default_rng(7)
    n_samples, n_features = 60, 6
    beta = rng.standard_normal((n_features, 1))
    x_data = pd.DataFrame(rng.standard_normal((n_samples, n_features)))
    y_data = pd.DataFrame(x_data.to_numpy() @ beta + 0.1 * rng.standard_normal((n_samples, 1)))
    model = PLS(n_components=3).fit(MCUVScaler().fit_transform(x_data), MCUVScaler().fit_transform(y_data))

    for name, residual in _pls_identity_residuals(model).items():
        assert residual == pytest.approx(0, abs=1e-12), f"identity {name!r} violated by {residual:.3e}"


def test_pls_structural_identities_ldpe(
    fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """The same five identities on the real, multi-response LDPE data (54 x 14, 5 Y, A=6)."""
    data = fixture_pls_ldpe_example
    assert isinstance(data["X"], np.ndarray)
    assert isinstance(data["Y"], np.ndarray)
    assert isinstance(data["A"], int)
    model = PLS(n_components=int(data["A"])).fit(
        MCUVScaler().fit_transform(pd.DataFrame(data["X"])),
        MCUVScaler().fit_transform(pd.DataFrame(data["Y"])),
    )

    for name, residual in _pls_identity_residuals(model).items():
        assert residual == pytest.approx(0, abs=1e-12), f"identity {name!r} violated by {residual:.3e}"


def test_pls_structural_identities_split_under_missing_data(
    fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """Two of the five identities are definitional and survive missing data; three are not.

    ``S`` is defined from ``T``, and ``R`` is defined as ``W inv(P'W)``, so those two
    identities hold to machine precision whatever the data. The orthogonality and
    triangularity of ``W``, ``P'W`` and ``W'R`` are instead *outcomes* of NIPALS
    deflation on a complete matrix. Under single-component projection the weights are
    computed from partial rows, so they degrade: a single NaN in this 54 x 14 block
    moves ``W'W`` off the identity by about 2e-2, not 1e-16.

    This is expected for missing-data NIPALS rather than a defect, but it means code
    must not assume ``W'W = I`` on a model fitted with gaps. The loose bound below
    pins that the degradation stays small without asserting a brittle exact value.
    """
    data = fixture_pls_ldpe_example
    assert isinstance(data["X"], np.ndarray)
    assert isinstance(data["A"], int)
    x_gapped = pd.DataFrame(data["X"]).copy()
    x_gapped.iloc[11, 0] = np.nan
    model = PLS(n_components=int(data["A"]), missing_data_settings=dict(md_method="nipals")).fit(
        MCUVScaler().fit_transform(x_gapped),
        MCUVScaler().fit_transform(pd.DataFrame(data["Y"])),
    )
    residuals = _pls_identity_residuals(model)

    # Definitional: still exact.
    assert residuals["s_definition"] == pytest.approx(0, abs=1e-12)
    assert residuals["rp_identity"] == pytest.approx(0, abs=1e-12)

    # Outcomes of deflation: no longer exact, but must not fall apart.
    for name in ("w_orthonormal", "pw_lower", "pw_diag", "wr_lower", "wr_diag"):
        assert residuals[name] < 0.1, f"identity {name!r} degraded further than expected: {residuals[name]:.3e}"
    assert residuals["w_orthonormal"] > 1e-6, (
        "W'W is exact under missing data; if the missing-data path improved, tighten this test"
    )


@pytest.mark.skip(reason="API still has to be improved to handle this case")
def test_pls_invalid_calls() -> None:
    """Tests various invalid calls, and corresponding error messages."""
    K, N, M, A = 4, 3, 2, 5
    rng = np.random.default_rng(0)
    data_x = pd.DataFrame(rng.uniform(low=-1, high=1, size=(N, K)))
    data_y = pd.DataFrame(rng.uniform(low=-1, high=1, size=(N, M)))
    with pytest.raises(ValueError, match=r"Tolerance `tol`` must be between 1E-16 and 1.0"):
        _ = PLS(n_components=A, tol=0)

    # The `method` input parameter does not exist anymore
    with pytest.raises(ValueError, match=r"Method 'SVDS' is not known."):
        _ = PLS(n_components=A, method="SVDS")

    with pytest.raises(ValueError, match=r"Missing data method 'SCP' is not known."):
        _ = PLS(n_components=A, md_method="SCP")

    model = PLS(n_components=A)
    with pytest.warns(SpecificationWarning, match=r"The requested number of components is (.*)"):
        model.fit(data_x, data_y)

    sparse_data = csr_matrix([[1, 2], [0, 3], [4, 5]])
    with pytest.raises(TypeError, match=r"This PLS class does not support sparse input."):
        PLS(n_components=2).fit(data_x, sparse_data)


def _pls_ten_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a 10 x 4 X and a y that depends on it, small enough for a CV run in a fraction of a second."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((10, 4)))
    y = pd.DataFrame({"y": X.to_numpy() @ rng.normal(size=4) + 0.1 * rng.standard_normal(10)})
    return X, y


#: Cross-validation settings that keep each select_n_components / nested_cv call cheap.
_PLS_QUICK_SELECT = {"cv": 5, "n_repeats": 1, "max_components": 2, "n_permutations": 19, "random_state": 0}
_PLS_QUICK_NESTED = {"inner_cv": 3, "n_inner_repeats": 1, "max_components": 2, "n_permutations": 19, "random_state": 0}


@pytest.mark.parametrize(
    ("action", "error", "message"),
    [
        (
            lambda X, _: PLS(n_components=2).fit(X, pd.DataFrame({"y": np.ones(len(X))})),
            NotEnoughVarianceError,
            (
                r"^There is no variance left in the data array for Y: cannot compute any more components "
                r"beyond component 0\.$"
            ),
        ),
        (
            lambda X, y: PLS(n_components=2).fit(X, y.iloc[:9]),
            ValueError,
            r"^The X and Y arrays must have the same number of rows: X has 10 and Y has 9\.$",
        ),
        (
            lambda X, y: PLS.select_n_components(X.iloc[:2], y.iloc[:2], cv=2),
            ValueError,
            r"^No components can be evaluated; the data or folds are too small\.$",
        ),
        (
            lambda X, y: PLS.select_n_components(X, y, cv=PredefinedSplit(np.full(10, -1))),
            ValueError,
            r"^The cross-validation splitter produced no folds\.$",
        ),
        (
            lambda X, y: PLS.nested_cv(X, y, outer_cv=PredefinedSplit(np.full(10, -1))),
            ValueError,
            r"^The outer cross-validation splitter produced no folds\.$",
        ),
        (
            lambda X, y: PLS(n_components=2).fit(X, y).cross_validate(X, y.iloc[:9]),
            ValueError,
            r"^X and Y must have the same number of rows, got 10 and 9\.$",
        ),
        (
            lambda X, y: PLS(n_components=2).fit(X, y).cross_validate(X, y, conf_level=0.4),
            ValueError,
            r"^conf_level must be between 0\.5 and 1\.0, got 0\.4\.$",
        ),
    ],
    ids=[
        "fit-a-constant-y",
        "fit-a-short-y",
        "select-on-two-rows",
        "select-with-a-splitter-of-no-folds",
        "nested-with-an-outer-splitter-of-no-folds",
        "cross-validate-a-short-y",
        "cross-validate-at-40-percent-confidence",
    ],
)
def test_pls_refuses_input_it_cannot_use(
    action: Callable[[pd.DataFrame, pd.DataFrame], object], error: type[Exception], message: str
) -> None:
    """Each PLS entry point names what it cannot work with, rather than failing further in."""
    X, y = _pls_ten_rows()
    with pytest.raises(error, match=message):
        action(X, y)


@pytest.mark.parametrize(
    ("entry_point", "y_form"),
    [
        ("project", "frame"),
        ("select_n_components", "series"),
        ("select_n_components", "array"),
        ("nested_cv", "series"),
        ("nested_cv", "array"),
    ],
    ids=["project", "select-series-y", "select-array-y", "nested-series-y", "nested-array-y"],
)
def test_pls_entry_points_read_arrays_as_their_dataframes(entry_point: str, y_form: str) -> None:
    """An array X, with Y as a Series or an array, gives the same numbers as the same values in DataFrames."""
    X, y = _pls_ten_rows()
    other_y = {"frame": y, "series": y["y"], "array": y.to_numpy()}[y_form]
    model = PLS(n_components=2).fit(X, y)
    calls = {
        "project": lambda X_, _: model.project(X_).scores.to_numpy(),
        "select_n_components": lambda X_, y_: PLS.select_n_components(X_, y_, **_PLS_QUICK_SELECT).rmsecv.to_numpy(),
        "nested_cv": lambda X_, y_: PLS.nested_cv(X_, y_, **_PLS_QUICK_NESTED).rmsep.to_numpy(),
    }
    np.testing.assert_allclose(calls[entry_point](X.to_numpy(), other_y), calls[entry_point](X, y))


def test_pls_nested_cv_refuses_when_no_held_out_row_is_predicted() -> None:
    """When no held-out row gets a prediction, nested CV says so.

    One outer fold holds out the two rows with no observed X, which no projection can score,
    and the other eight rows are never held out. Before #676 a missing cell in every row did
    this, since a held-out row with a gap was scored NaN; such rows are now projected over
    their observed columns.
    """
    X, y = _pls_ten_rows()
    X.iloc[8:] = np.nan
    one_fold = PredefinedSplit([-1] * 8 + [0, 0])
    with pytest.raises(RuntimeError, match=r"^Nested CV produced no covered observations; check the outer splitter\.$"):
        PLS.nested_cv(X, y, outer_cv=one_fold, **_PLS_QUICK_NESTED)


class _TrainsOnEverythingTestsNothing(BaseCrossValidator):
    """A splitter of two folds that each train on every row and hold out none."""

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:  # noqa: ARG002
        return 2

    def _iter_test_indices(self, X: object = None, y: object = None, groups: object = None):  # noqa: ARG002
        yield from (np.array([], dtype=int), np.array([], dtype=int))


def test_pls_select_n_components_refuses_folds_that_score_no_rows() -> None:
    """Folds that hold out no rows leave no prediction error to judge, so nothing is recommended."""
    X, y = _pls_ten_rows()
    # With no held-out rows the RMSECV is 0 / 0: that NaN is the condition under test.
    with (
        np.errstate(invalid="ignore"),
        pytest.warns(SpecificationWarning, match=r"scale_inside_folds=False leaks"),
        pytest.raises(RuntimeError, match=r"^Cross-validation produced NaN total-RMSECV for every component count"),
    ):
        PLS.select_n_components(
            X, y, cv=_TrainsOnEverythingTestsNothing(), scale_inside_folds=False, max_components=2, n_permutations=19
        )


@pytest.fixture
def fixture_pls_model_simca_1_component() -> dict[str, pd.DataFrame | np.ndarray | float | int]:
    """
    Test simple model against Simca-P, version 14.1.

    Testing on 28 June 2020.

    When X and y are mean centered and scaled, the model should provide the loadings as listed here

    The R output quoted below is the source of `expected_y_predicted`, which
    `test_pls_compare_model_api` asserts.

    X = matrix(c(41.1187, 21.2833, 21.1523,  0.2446, -0.0044, -0.131,  1.12,
                 41.7755, 22.0978, 21.1653,  0.3598,  0.1622, -0.9325, 1.01,
                 41.2568, 21.4873, 20.7407,  0.2536,  0.1635, -0.7467, 0.97,
                 41.5469, 22.2043, 20.4518,  0.6317,  0.1997, -1.7525, 0.83,
                 40.0234, 23.7399, 21.978,  -0.0534, -0.0158, -1.7619, 0.93,
                 39.9203, 21.9997, 21.5859, -0.1811,  0.089,  -0.4138, 1.02,
                 42.1886, 21.4891, 20.4427,  0.686,   0.1124, -1.0464, 0.91,
                 42.1454, 20.3803, 18.2327,  0.6607,  0.1291, -2.1476, 0.70,
                 42.272,  18.9725, 18.3763,  0.561,   0.0453, -0.5962, 1.26,
                 41.49,   18.603,  17.9978,  0.4872,  0.1198, -0.6052, 1.05,
                 41.5306, 19.1558, 18.2172,  0.6233,  0.1789, -0.9386, 0.95), nrow=11, byrow=T)

    data = data.frame(X)
    colnames(data) <- c("A","B","C", "D", "E", "F", "y")

    library(pls)
    model = plsr(y~., 1, data=data, method="simpls")
    plot(model)

    self.expected_y_predicted = [
        1.17475,
        0.930441,
        0.979066,
        0.773083,
        0.945719,
        1.10583,
        0.929441,
        0.796271,
        1.09379,
        1.0635,
        0.958111,
        ]
    """
    data: dict[str, pd.DataFrame | np.ndarray | float | int] = {}
    data["X"] = pd.DataFrame(
        np.array(
            [
                [
                    41.1187,
                    21.2833,
                    21.1523,
                    0.2446,
                    -0.0044,
                    -0.131,
                ],
                [
                    41.7755,
                    22.0978,
                    21.1653,
                    0.3598,
                    0.1622,
                    -0.9325,
                ],
                [
                    41.2568,
                    21.4873,
                    20.7407,
                    0.2536,
                    0.1635,
                    -0.7467,
                ],
                [
                    41.5469,
                    22.2043,
                    20.4518,
                    0.6317,
                    0.1997,
                    -1.7525,
                ],
                [
                    40.0234,
                    23.7399,
                    21.978,
                    -0.0534,
                    -0.0158,
                    -1.7619,
                ],
                [
                    39.9203,
                    21.9997,
                    21.5859,
                    -0.1811,
                    0.089,
                    -0.4138,
                ],
                [
                    42.1886,
                    21.4891,
                    20.4427,
                    0.686,
                    0.1124,
                    -1.0464,
                ],
                [
                    42.1454,
                    20.3803,
                    18.2327,
                    0.6607,
                    0.1291,
                    -2.1476,
                ],
                [
                    42.272,
                    18.9725,
                    18.3763,
                    0.561,
                    0.0453,
                    -0.5962,
                ],
                [
                    41.49,
                    18.603,
                    17.9978,
                    0.4872,
                    0.1198,
                    -0.6052,
                ],
                [
                    41.5306,
                    19.1558,
                    18.2172,
                    0.6233,
                    0.1789,
                    -0.9386,
                ],
            ]
        )
    )

    data["y"] = pd.DataFrame(np.array([1.12, 1.01, 0.97, 0.83, 0.93, 1.02, 0.91, 0.7, 1.26, 1.05, 0.95]))
    data["expected_y_predicted"] = np.array(
        [
            1.17475,
            0.930441,
            0.979066,
            0.773083,
            0.945719,
            1.10583,
            0.929441,
            0.796271,
            1.09379,
            1.0635,
            0.958111,
        ]
    )
    data["loadings_P1"] = np.array(
        [
            -0.2650725,
            -0.2165038,
            0.08547913,
            -0.3954746,
            -0.4935882,
            0.7541404,
        ]
    )
    data["loadings_r1"] = np.array(
        [
            -0.04766187,
            -0.3137862,
            0.004006641,
            -0.238001,
            -0.4430451,
            0.8039384,
        ]
    )
    data["loadings_y_c1"] = 0.713365
    data["SDt"] = 1.19833
    data["R2X"] = 0.261641
    data["R2Y"] = 0.730769
    data["t1"] = np.array(
        [
            1.889566,
            -0.4481195,
            0.0171578,
            -1.953837,
            -0.3019302,
            1.230112,
            -0.4576912,
            -1.731961,
            1.114923,
            0.8251334,
            -0.1833536,
        ]
    )
    data["Tsq"] = np.array(
        [
            2.48638,
            0.1398399,
            0.0002050064,
            2.658398,
            0.0634829,
            1.053738,
            0.1458776,
            2.08891,
            0.8656327,
            0.4741239,
            0.02341113,
        ]
    )
    data["DModX"] = np.array(
        [
            0.367926,
            1.01727,
            0.970395,
            0.635592,
            2.36596,
            0.449567,
            0.645429,
            1.12458,
            0.520623,
            0.384443,
            0.764301,
        ]
    )
    data["Xavg"] = np.array([41.38802, 21.03755, 20.03097, 0.3884909, 0.1072455, -1.006582])
    data["Xws"] = 1 / np.array(
        [
            1.259059,
            0.628138,
            0.6594034,
            3.379028,
            13.8272,
            1.589986,
        ]
    )
    data["Yavg"] = 0.9772727
    data["Yws"] = 1 / 6.826007  # Simca-P uses inverse standard deviation
    data["A"] = 1
    data["conf"] = 0.95
    return data


def test_pls_compare_sklearn_1_component(fixture_pls_model_simca_1_component: dict) -> None:
    """Test PLS with 1 component."""
    data = fixture_pls_model_simca_1_component

    plsmodel = PLSRegression(n_components=data["A"], scale=True)
    plsmodel.fit(data["X"], data["y"])

    # Check the pre-processing: sig figs have been taken as high as possible.
    assert plsmodel._x_mean == pytest.approx(data["Xavg"], abs=1e-5)
    assert plsmodel._x_std == pytest.approx(data["Xws"], abs=1e-6)
    assert plsmodel._y_mean == pytest.approx(data["Yavg"], abs=1e-7)
    assert plsmodel._y_std == pytest.approx(data["Yws"], abs=1e-8)

    # Extract the model parameters
    T = plsmodel.x_scores_
    P = plsmodel.x_loadings_
    assert T.ravel() == pytest.approx(data["t1"], abs=1e-5)
    assert np.std(T, ddof=1) == pytest.approx(data["SDt"], rel=1e-5)
    assert data["loadings_P1"].ravel() == pytest.approx(P.ravel(), rel=1e-5)
    assert data["loadings_r1"] == pytest.approx(plsmodel.x_weights_.ravel(), rel=1e-4)

    # Check the model's predictions
    t1_predict, _y_pp = plsmodel.transform(data["X"], data["y"])
    assert data["t1"] == pytest.approx(t1_predict.ravel(), abs=1e-5)
    # assert y_pp == pytest.approx((data["y"] - data["Yavg"]) / data["Yws"], abs=1e-6)

    # The manual y_hat reconstruction, Simca's C, the deflation step and R2Y are all
    # asserted on this same fixture by `test_pls_compare_model_api`; this test's purpose
    # is the sklearn-versus-SIMCA cross-check above.


def test_pls_compare_model_api(
    fixture_pls_model_simca_1_component: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """Test two variants of the PLS model."""
    data = fixture_pls_model_simca_1_component
    assert isinstance(data["X"], pd.DataFrame)
    assert isinstance(data["y"], pd.DataFrame)
    assert isinstance(data["A"], int)
    plsmodel = PLS(n_components=int(data["A"]))
    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(np.array(data["y"]))

    # Check the pre-processing: sig figs have been taken as high as possible.
    assert X_mcuv.center_.values == pytest.approx(data["Xavg"], abs=1e-5)
    assert X_mcuv.scale_.values == pytest.approx(data["Xws"], abs=1e-6)
    assert Y_mcuv.center_.values == pytest.approx(data["Yavg"], abs=1e-7)
    assert Y_mcuv.scale_.values == pytest.approx(data["Yws"], abs=1e-8)

    # Extract the model parameters
    plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(np.array(data["y"])))
    assert data["SDt"] == pytest.approx(np.std(plsmodel.scores_, ddof=1), abs=1e-5)
    assert data["t1"] == pytest.approx(plsmodel.scores_.values.ravel(), abs=1e-5)
    assert data["loadings_P1"] == pytest.approx(plsmodel.x_loadings_.values.ravel(), abs=1e-5)
    assert data["loadings_r1"] == pytest.approx(plsmodel.x_weights_.values.ravel(), abs=1e-6)

    assert np.array(data["expected_y_predicted"]).ravel() == pytest.approx(
        Y_mcuv.inverse_transform(plsmodel.predictions_).values.ravel(), abs=1e-5
    )
    assert data["R2Y"] == pytest.approx(plsmodel.r2_cumulative_, abs=1e-6)

    # Check the model's predictions (full diagnostics)
    result = plsmodel.diagnose(X_mcuv.transform(data["X"]))
    assert plsmodel.spe_.values.ravel() == pytest.approx(result.spe.values, abs=1e-9)
    assert data["t1"] == pytest.approx(result.scores.values.ravel(), abs=1e-5)
    assert data["Tsq"] == pytest.approx(result.hotellings_t2.values.ravel(), abs=1e-5)
    assert data["expected_y_predicted"] == pytest.approx(
        Y_mcuv.inverse_transform(result.y_hat).values.ravel(), abs=1e-5
    )


@pytest.fixture
def fixture_pls_simca_2_components() -> dict[str, pd.DataFrame | np.ndarray | float | int]:
    """
    Test simple model against Simca-P, version 14.1.

    Testing on 02 July 2020.
    No missing data

    When X and y are mean centered and scaled, the model should provide the loadings listed here.
    """
    out: dict[str, pd.DataFrame | np.ndarray | float | int] = {}
    out["X"] = pd.DataFrame(
        np.array(
            [
                [
                    1.27472,
                    0.897732,
                    -0.193397,
                ],
                [
                    1.27472,
                    -1.04697,
                    0.264243,
                ],
                [
                    0.00166722,
                    1.26739,
                    1.06862,
                ],
                [
                    0.00166722,
                    -0.0826556,
                    -1.45344,
                ],
                [
                    0.00166722,
                    -1.46484,
                    1.91932,
                ],
                [
                    -1.27516,
                    0.849516,
                    -0.326239,
                ],
                [
                    -1.27516,
                    -1.06304,
                    0.317718,
                ],
                [
                    -0.000590006,
                    1.26739,
                    1.06862,
                ],
                [
                    -0.000590006,
                    -0.0826556,
                    -1.45344,
                ],
                [
                    -0.000590006,
                    -1.09519,
                    0.427109,
                ],
                [
                    -1.27516,
                    0.849516,
                    -0.326239,
                ],
                [
                    -1.27516,
                    -1.06304,
                    0.317718,
                ],
                [
                    1.27398,
                    0.897732,
                    -0.193397,
                ],
                [
                    1.27398,
                    -0.130872,
                    -1.4372,
                ],
            ]
        )
    )

    out["y"] = pd.DataFrame(
        np.array(
            [
                -0.0862851,
                -1.60162,
                0.823439,
                0.242033,
                -1.64304,
                1.59583,
                -0.301604,
                0.877623,
                0.274155,
                -0.967692,
                1.47491,
                -0.194163,
                0.097352,
                -0.590925,
            ]
        )
    )
    out["expected_y_predicted"] = np.array(
        [
            0.04587483,
            -1.671657,
            0.8337691,
            0.2326966,
            -1.622544,
            1.520105,
            -0.209406,
            0.8350853,
            0.2340128,
            -1.002176,
            1.520105,
            -0.209406,
            0.04630876,
            -0.552768,
        ]
    )
    out["loadings_P"] = np.array([[-0.3799977, -0.7815778], [0.8737038, -0.2803103], [-0.3314019, 0.55731]])
    out["loadings_W"] = np.array([[-0.4839311, -0.7837874], [0.8361799, -0.2829775], [-0.2580969, 0.5528119]])  # W
    out["loadings_C"] = np.array([1.019404, 0.1058565])
    out["SDt"] = np.array([0.9724739, 1.098932])
    out["R2X"] = np.array(
        [
            0.3207782,
            0.4025633,
        ]
    )  # cumulative: 32%, then 72% for second component
    out["R2Y"] = np.array([0.9827625, 0.01353244])
    out["T"] = np.array(
        [
            [0.1837029, -1.335702],
            [-1.560534, -0.7636986],
            [0.7831483, 0.334647],
            [0.3052059, -0.7409231],
            [-1.721048, 1.246014],
            [1.411638, 0.7658994],
            [-0.3538088, 1.428992],
            [0.7842407, 0.3365611],
            [0.3062983, -0.7390091],
            [-1.025724, 0.4104719],
            [1.411638, 0.7658994],
            [-0.3538088, 1.428992],
            [0.184063, -1.335071],
            [-0.3550123, -1.803074],
        ]
    )
    out["Tsq"] = np.array(
        [
            1.513014,
            3.05803,
            0.7412658,
            0.5530728,
            4.417653,
            2.592866,
            1.823269,
            0.74414,
            0.5514336,
            1.252029,
            2.592866,
            1.823269,
            1.511758,
            2.825334,
        ]
    )

    # SIMCA's "distance to the model in X". Used by `test_pls_compare_api` as the
    # external reference for our SPE; see the derivation there.
    out["DModX"] = np.array(
        [
            0.8796755,
            0.2482767,
            1.641307,
            1.350485,
            0.9410046,
            0.4101084,
            0.856736,
            1.640294,
            1.351499,
            0.2035393,
            0.4101084,
            0.856736,
            0.8793415,
            0.7906867,
        ]
    )
    out["A"] = 2
    return out


def test_pls_sklearn_2_components(fixture_pls_simca_2_components: dict) -> None:
    """Test the Scikit model against the Simca-P model."""
    data = fixture_pls_simca_2_components

    plsmodel = PLSRegression(n_components=data["A"], scale=False)

    X_mcuv = MCUVScaler().fit_transform(data["X"])
    Y_mcuv = MCUVScaler().fit_transform(data["y"])

    plsmodel.fit(X_mcuv, Y_mcuv)

    # Extract the model parameters
    assert np.abs(data["T"]) == pytest.approx(np.abs(plsmodel.x_scores_), abs=1e-5)
    assert np.std(plsmodel.x_scores_, ddof=1, axis=0) == pytest.approx(data["SDt"], abs=1e-6)
    assert np.abs(data["loadings_P"]) == pytest.approx(np.abs(plsmodel.x_loadings_), abs=1e-5)
    assert np.abs(data["loadings_W"]) == pytest.approx(np.abs(plsmodel.x_weights_), abs=1e-5)


def test_pls_compare_api(fixture_pls_simca_2_components: dict) -> None:
    """Test PLS comparison between two different methods."""
    data = fixture_pls_simca_2_components

    plsmodel = PLS(n_components=data["A"])

    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(data["y"])
    plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(data["y"]))

    # Extract the model parameters
    assert data["SDt"] == pytest.approx(np.std(plsmodel.scores_, ddof=1, axis=0), abs=1e-6)
    assert np.abs(data["T"]) == pytest.approx(np.abs(plsmodel.scores_), abs=1e-5)
    assert np.abs(data["loadings_P"]) == pytest.approx(np.abs(plsmodel.x_loadings_), abs=1e-5)
    assert np.abs(data["loadings_W"]) == pytest.approx(np.abs(plsmodel.x_weights_), abs=1e-5)
    assert Y_mcuv.inverse_transform(plsmodel.predictions_).values == pytest.approx(
        data["expected_y_predicted"].reshape(-1, 1), abs=1e-5
    )
    assert sum(data["R2Y"]) == pytest.approx(plsmodel.r2_cumulative_.values[-1], abs=1e-7)

    # Check the model's predictions (full diagnostics)
    result = plsmodel.diagnose(X_mcuv.transform(data["X"]))
    # Model-building SPE against model-using SPE.
    assert plsmodel.spe_.iloc[:, -1].values == pytest.approx(result.spe, abs=1e-10)

    # SPE against an external reference (Simca-P), via its DModX column. Simca reports
    # the distance to the model in X as a residual standard deviation normalised by the
    # model's pooled one:
    #
    #     DModX_i = s_i / s_0,   s_i = spe_i / sqrt(K - A),
    #                            s_0 = sqrt(sum_ik e_ik^2 / (N * (K - A)))
    #
    # and `spe_` here is sqrt(sum_k e_ik^2) per row, so sum_i spe_i^2 is that total.
    # The two (K - A) factors cancel in the ratio, which is what makes this a check on
    # SPE itself rather than on Simca's normalisation convention. Note K - A = 1 for
    # this 14 x 3 two-component fixture, so it cannot on its own discriminate the
    # (K - A) placement; the ratio is what is being pinned.
    spe_model = plsmodel.spe_.iloc[:, -1].to_numpy()
    n_samples = data["X"].shape[0]
    n_features, n_components = data["X"].shape[1], int(data["A"])
    pooled_residual_sd = np.sqrt((spe_model**2).sum() / (n_samples * (n_features - n_components)))
    assert spe_model / pooled_residual_sd == pytest.approx(data["DModX"], abs=1e-5)
    assert data["Tsq"] == pytest.approx(result.hotellings_t2, abs=1e-5)
    assert data["expected_y_predicted"] == pytest.approx(
        Y_mcuv.inverse_transform(result.y_hat).values.ravel(), abs=1e-5
    )
    assert np.abs(data["T"]) == pytest.approx(np.abs(result.scores), abs=1e-5)


def test_score_limit_pca() -> None:
    """score_limit returns per-component t_(N-1) * std limits for the PCA scores."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 5)))
    X_scaled = MCUVScaler().fit_transform(X)
    model = PCA(n_components=3).fit(X_scaled)

    limits = model.score_limit(conf_level=0.95)
    assert limits.shape == (3,)
    assert np.all(limits > 0)

    # The score standard deviation is estimated from the same N observations,
    # so the quantile is Student-t on N-1 degrees of freedom, not standard
    # normal. At N=60 that is 2.0010 rather than 1.9600.
    t_95 = 2.0009953770482318
    expected = t_95 * np.asarray(model.scores_).std(axis=0, ddof=1)
    np.testing.assert_allclose(limits, expected, rtol=1e-9)

    # The docstring's equivalence: sqrt(F(1, N-1)) IS the two-sided t_(N-1)
    # quantile. Checked against scipy directly so it validates the identity
    # rather than the precision of the literal above.
    np.testing.assert_allclose(t_dist.ppf(0.975, 59), np.sqrt(f_dist.isf(0.05, 1, 59)), rtol=1e-12)

    # A higher confidence level widens the limits.
    assert np.all(model.score_limit(conf_level=0.99) > limits)

    # The bound method and the free function agree.
    np.testing.assert_allclose(score_limit(model, 0.95), limits, rtol=1e-12)


def test_score_limit_pls() -> None:
    """score_limit also works for a fitted PLS model."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(50, 4)))
    beta = rng.normal(size=(4, 1))
    Y = pd.DataFrame(X.values @ beta + rng.normal(scale=0.1, size=(50, 1)))
    model = PLS(n_components=2).fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))

    limits = model.score_limit()
    assert limits.shape == (2,)
    assert np.all(limits > 0)


@pytest.mark.parametrize("conf_level", [1.0, 0.0])
def test_score_limit_refuses_a_confidence_outside_0_1(fixture_pca_for_plots: PCA, conf_level: float) -> None:
    """A confidence level outside (0, 1) raises ValueError, also under `python -O`.

    It was an `assert`, so `-O` removed the check and the limit came back as inf or NaN.
    """
    with pytest.raises(ValueError, match=rf"^conf_level must lie in \(0, 1\); got {conf_level}\.$"):
        score_limit(fixture_pca_for_plots, conf_level=conf_level)


def test_pls_prediction_interval() -> None:
    """PLS.prediction_interval brackets the predictions and widens with confidence."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(45, 4)))
    beta = rng.normal(size=(4, 1))
    Y = pd.DataFrame(X.values @ beta + rng.normal(scale=0.3, size=(45, 1)))
    X_s = MCUVScaler().fit_transform(X)
    Y_s = MCUVScaler().fit_transform(Y)
    model = PLS(n_components=2).fit(X_s, Y_s)

    pi = model.prediction_interval(X_s, conf_level=0.95)
    assert pi.y_hat.shape == (45, 1)
    assert (pi.lower.values < pi.y_hat.values).all()
    assert (pi.y_hat.values < pi.upper.values).all()

    # A higher confidence level widens the interval.
    pi_99 = model.prediction_interval(X_s, conf_level=0.99)
    assert ((pi_99.upper - pi_99.lower).values > (pi.upper - pi.lower).values).all()

    # A cross_validate() result can supply the error variance.
    cv = model.cross_validate(X_s, Y_s, cv="loo", show_progress=False)
    pi_cv = model.prediction_interval(X_s, cv_result=cv)
    assert pi_cv.y_hat.shape == (45, 1)
    assert (pi_cv.lower.values < pi_cv.upper.values).all()

    with pytest.raises(ValueError, match="conf_level"):
        model.prediction_interval(X_s, conf_level=1.5)


def test_pls_get_set_params_and_clone() -> None:
    """PLS follows the sklearn estimator protocol: get_params/set_params/clone."""
    pls = PLS(n_components=2, scale=False, max_iter=321)

    params = pls.get_params()
    assert params["n_components"] == 2
    assert params["scale"] is False
    assert params["max_iter"] == 321

    pls.set_params(n_components=4, max_iter=500)
    assert pls.n_components == 4
    assert pls.max_iter == 500

    fresh = clone(pls)
    assert fresh is not pls
    assert fresh.get_params() == pls.get_params()


def test_pls_in_pipeline_and_clone_matches_plsregression() -> None:
    """PLS runs inside an sklearn Pipeline, clones, and matches PLSRegression."""
    rng = np.random.default_rng(42)
    n_samples, n_features, n_components = 50, 6, 3
    feature_names = [f"x{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=feature_names)
    beta = rng.normal(size=(n_features, 1))
    Y = pd.DataFrame(X.values @ beta + rng.normal(scale=0.2, size=(n_samples, 1)), columns=["y"])
    # PLSRegression centres Y internally; our NIPALS does not, so centre Y up front.
    Y_scaled = pd.DataFrame(StandardScaler().fit_transform(Y.values), columns=["y"])

    with config_context(transform_output="pandas"):
        pipe = Pipeline([("scaler", StandardScaler()), ("pls", PLS(n_components=n_components, scale=False))])
        pipe.fit(X, Y_scaled)

        # Cloning yields an unfitted but identically-parametrised pipeline.
        fresh = clone(pipe)
        fresh.fit(X, Y_scaled)

    ours = pipe.named_steps["pls"]
    refit = fresh.named_steps["pls"]
    # Fitting is deterministic, so clone + refit reproduces the scores exactly.
    assert ours.scores_.values == pytest.approx(refit.scores_.values, abs=1e-9)

    # Numerical equivalence against sklearn PLSRegression on the same scaled X.
    X_scaled = StandardScaler().fit_transform(X)
    ref = PLSRegression(n_components=n_components, scale=False).fit(X_scaled, Y_scaled)

    assert np.abs(ours.scores_.values) == pytest.approx(np.abs(ref.x_scores_), abs=1e-6)
    assert np.abs(ours.x_loadings_.values) == pytest.approx(np.abs(ref.x_loadings_), abs=1e-6)
    assert np.abs(ours.x_weights_.values) == pytest.approx(np.abs(ref.x_weights_), abs=1e-6)
    assert np.abs(ours.beta_coefficients_.values.ravel()) == pytest.approx(np.abs(ref.coef_.ravel()), abs=1e-6)


def test_pls_scale_true_scales_x_and_y_blocks() -> None:
    """PLS(scale=True) mean-centers and unit-variance-scales both X and Y.

    Regression guard for the ``scale`` flag, which was previously inert (X and Y
    were fed to NIPALS unscaled). On a multi-target Y whose columns have very
    different variances, an estimator that fails to scale Y lets the
    high-variance columns dominate the latent extraction and fits poorly. The
    fixed ``scale=True`` must:

    (a) reproduce the manual ``MCUVScaler`` + ``scale=False`` path exactly (same
        NIPALS, scaling only relocated), with predictions on the original scale;
    (b) match ``sklearn.cross_decomposition.PLSRegression(scale=True)``; and
    (c) leave the fit genuinely good (R2 was ~0.1x when Y went unscaled).
    """
    rng = np.random.default_rng(7)
    n, n_comp = 60, 4
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    beta = rng.normal(size=(5, 4))
    core = X.values @ beta
    # Deliberately heterogeneous Y-column variances and non-zero offsets.
    Y = pd.DataFrame(
        core * np.array([0.05, 1.0, 5.0, 20.0]) + np.array([10.0, -3.0, 100.0, 0.2]) + rng.normal(0, 0.02, (n, 4)),
        columns=[f"y{i}" for i in range(4)],
    )

    model = PLS(n_components=n_comp, scale=True).fit(X, Y)

    # (a) Exact equivalence to explicit external scaling, inverse-transformed.
    x_scaler = MCUVScaler().fit(X)
    y_scaler = MCUVScaler().fit(Y)
    manual = PLS(n_components=n_comp, scale=False).fit(x_scaler.transform(X), y_scaler.transform(Y))
    manual_pred = y_scaler.inverse_transform(manual.predict(x_scaler.transform(X)))
    assert model.predict(X).values == pytest.approx(manual_pred.values, abs=1e-9)
    assert model.predictions_.values == pytest.approx(manual_pred.values, abs=1e-9)
    # transform() reproduces the fitted scores (projection uses the same scaling).
    assert model.transform(X).values == pytest.approx(model.scores_.values, abs=1e-9)

    # (c) The fit is genuinely good; the inert flag scored ~0.1 here.
    r2 = r2_score(Y, model.predict(X), multioutput="variance_weighted")
    assert r2 > 0.99

    # (b) Parity with sklearn PLSRegression(scale=True) on the raw data. Its
    # coef_ is (n_targets, n_features); ours is (n_features, n_targets).
    ref = PLSRegression(n_components=n_comp, scale=True).fit(X.values, Y.values)
    assert model.predict(X).values == pytest.approx(ref.predict(X.values), rel=1e-3, abs=1e-3)

    # scale=False stays a pass-through: feeding pre-scaled data is not re-scaled.
    # predict() exercises the diagnose() pass-through; transform() the other one.
    passthrough = PLS(n_components=n_comp, scale=False).fit(x_scaler.transform(X), y_scaler.transform(Y))
    assert passthrough.predict(x_scaler.transform(X)).values == pytest.approx(
        manual.predict(x_scaler.transform(X)).values, abs=1e-12
    )
    assert passthrough.transform(x_scaler.transform(X)).values == pytest.approx(
        manual.transform(x_scaler.transform(X)).values, abs=1e-12
    )


def test_pls_rmse_and_interval_on_original_scale() -> None:
    """rmse_ and a no-cv prediction_interval are on the ORIGINAL Y scale under scale=True.

    Regression guard: internal scaling (scale=True) runs NIPALS in standardized
    space, but rmse_ must be rescaled to original Y units so it stays consistent
    with predictions_ / diagnose().y_hat, and with prediction_interval(), which
    uses rmse_ as its residual error term. A multi-target Y with very different
    column magnitudes makes the scale error obvious.
    """
    rng = np.random.default_rng(3)
    n = 60
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    beta = rng.normal(size=(5, 3))
    Y = pd.DataFrame(
        X.values @ beta * np.array([0.1, 3.0, 40.0]) + np.array([5.0, -2.0, 100.0]) + rng.normal(0, 0.05, (n, 3)),
        columns=["ya", "yb", "yc"],
    )
    model = PLS(n_components=5, scale=True).fit(X, Y)

    # rmse_ at the last component equals the original-unit RMSE of predict().
    y_hat = model.predict(X)
    rmse_original = np.sqrt(((Y.values - y_hat.values) ** 2).mean(axis=0))
    assert model.rmse_.iloc[:, -1].to_numpy() == pytest.approx(rmse_original, rel=1e-6, abs=1e-9)
    # ... and is NOT the standardized-unit RMSE (the pre-fix value), which would
    # differ by roughly the per-column Y standard deviation.
    rmse_scaled = rmse_original / Y.std(ddof=1).to_numpy()
    assert not np.allclose(model.rmse_.iloc[:, -1].to_numpy(), rmse_scaled, rtol=1e-2)

    # The no-cv prediction interval uses original-unit rmse_, so its half-width
    # is on the Y scale (here ~ the 0.05 noise), not shrunk by standardization.
    pi = model.prediction_interval(X.iloc[:5], conf_level=0.95)
    half_width = (pi.upper.to_numpy() - pi.lower.to_numpy()) / 2.0
    assert np.all(half_width > 0)
    # per-target half-width is close to t*sqrt(1+leverage)*rmse_ (original units)
    assert half_width.mean() == pytest.approx(0.1, abs=0.06)


@pytest.fixture
def fixture_pls_ldpe_example() -> dict[str, pd.DataFrame | np.ndarray | float | int]:
    """
    Test PLS example with no missing data.

    Source: https://openmv.net/info/ldpe

    Data from a low-density polyethylene production process.
    There are 14 process variables and 5 quality variables (last 5 columns).
    More details: http://dx.doi.org/10.1002/aic.690400509
    The first 50 observations are from common-cause (normal) operation, while the last 4 show a
    process fault developing: the impurity level in the ethylene feed in both zones is increasing.

    Tin: inlet temperature to zone 1 of the reactor [K]
    Tmax1: maximum temperature along zone 1 [K]
    Tout1: outlet temperature from zone 1 [K]
    Tmax2: maximum temperature along zone 2 [K]
    Tout2: outlet temperature from zone 2 [K]
    Tcin1: temperature of inlet coolant to zone [K]
    Tcin2: temperature of inlet coolant to zone 2 [K]
    z1: percentage along zone 1 where Tmax1 occurs [%]
    z2: percentage along zone 2 where Tmax2 occurs [%]
    Fi1: flow rate of initiators to zone 1 [g/s]
    Fi2: flow rate of initiators to zone 2 [g/s]np.abs(state.scores)
    Fs1: flow rate of solvent to zone 1 [% of ethylene]
    Fs2: flow rate of solvent to zone 2 [% of ethylene]
    Press: pressure in the reactor [atm]
    ------
    Conv: quality variable: cumulative conversion
    Mn: quality variable: number average molecular weight
    Mw: quality variable: weight average molecular weight
    LCB: quality variable: long chain branching per 1000 C atoms
    SCB: quality variable: short chain branching per 1000 C atoms

    N = 54
    K = 14
    M = 5
    A = 6
    """
    out: dict[str, pd.DataFrame | np.ndarray | float | int] = {}
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
    values = pd.read_csv(
        folder / "LDPE" / "LDPE.csv",
        index_col=0,
    )
    out["expected_T"] = pd.read_csv(folder / "LDPE" / "T.csv", header=None)
    out["expected_W"] = pd.read_csv(folder / "LDPE" / "W.csv", header=None)
    out["expected_P"] = pd.read_csv(folder / "LDPE" / "P.csv", header=None)
    out["expected_C"] = pd.read_csv(folder / "LDPE" / "C.csv", header=None)
    out["expected_U"] = pd.read_csv(folder / "LDPE" / "U.csv", header=None)
    out["expected_hotellings_t2_a3"] = pd.read_csv(
        folder / "LDPE" / "Hotellings_T2_A3.csv",
        header=None,
    )
    out["expected_hotellings_t2_a6"] = pd.read_csv(
        folder / "LDPE" / "Hotellings_T2_A6.csv",
        header=None,
    )
    out["expected_yhat_a6"] = pd.read_csv(
        folder / "LDPE" / "Yhat_A6.csv",
        header=None,
    )
    out["expected_sd_t"] = np.array([1.872539, 1.440642, 1.216218, 1.141096, 1.059435, 0.9459715])
    out["expected_t2_lim_95_a6"] = 15.2017
    out["expected_t2_lim_99_a6"] = 21.2239
    out["X"] = np.array(values.iloc[:, :14])
    out["Y"] = np.array(values.iloc[:, 14:])
    out["A"] = 6
    assert isinstance(out["X"], np.ndarray | pd.DataFrame)
    assert out["X"].shape == (54, 14)
    assert isinstance(out["Y"], np.ndarray | pd.DataFrame)
    assert out["Y"].shape == (54, 5)

    return out


def test_pls_simca_ldpe(fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int]) -> None:
    """Unit test for LDPE case study.

    Parameters
    ----------
    PLS_model_SIMCA_LDPE_example : dict
        Dictionary of raw data and expected outputs from the PLS model.
    """
    data = fixture_pls_ldpe_example
    assert isinstance(data["X"], np.ndarray)
    assert isinstance(data["Y"], np.ndarray)
    assert isinstance(data["A"], int)
    plsmodel = PLS(n_components=data["A"])
    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(data["Y"])
    plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(data["Y"]))

    # Can only get these to very loosely match
    assert data["expected_t2_lim_95_a6"] == pytest.approx(plsmodel.hotellings_t2_limit(0.95), rel=1e-1)
    assert data["expected_t2_lim_99_a6"] == pytest.approx(plsmodel.hotellings_t2_limit(0.99), rel=1e-1)

    assert np.mean(np.abs(np.array(data["expected_T"])) - np.abs(plsmodel.scores_.values)) == pytest.approx(0, abs=1e-4)
    assert np.mean(np.abs(np.array(data["expected_P"])) - np.abs(plsmodel.x_loadings_.values)) == pytest.approx(
        0, abs=1e-5
    )
    assert np.mean(np.abs(np.array(data["expected_W"])) - np.abs(plsmodel.x_weights_.values)) == pytest.approx(
        0, abs=1e-6
    )
    assert np.mean(np.abs(np.array(data["expected_C"])) - np.abs(plsmodel.y_loadings_.values)) == pytest.approx(
        0, abs=1e-6
    )
    assert np.mean(np.abs(np.array(data["expected_U"])) - np.abs(plsmodel.y_scores_.values)) == pytest.approx(
        0, abs=1e-5
    )
    assert np.mean(
        np.array(data["expected_hotellings_t2_a3"]).ravel() - plsmodel.hotellings_t2_.iloc[:, 2].values.ravel()
    ) == pytest.approx(0, abs=1e-6)
    assert np.mean(
        np.array(data["expected_hotellings_t2_a6"]).ravel() - plsmodel.hotellings_t2_.iloc[:, 5].values.ravel()
    ) == pytest.approx(0, abs=1e-6)
    assert np.mean(
        np.array(data["expected_sd_t"]).ravel() - plsmodel.scaling_factor_for_scores_.values.ravel()
    ) == pytest.approx(0, abs=1e-5)

    # Absolute sum of the deviations, accounting for the fact that each column in Y has quite
    # different range/scaling.
    assert np.sum(
        np.abs(
            np.sum(np.abs(Y_mcuv.inverse_transform(plsmodel.predictions_) - np.array(data["expected_yhat_a6"])), axis=0)
            / Y_mcuv.center_
        )
    ) == pytest.approx(0, abs=1e-2)


def test_pls_simca_ldpe_missing_data(
    fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """Unit test for LDPE case study.
    From visual inspection, observation 12 has low influence in the model.
    Set 1 value in this observation to missing and check that the results are similar to the
    full-data case: "test_PLS_SIMCA_LDPE",
    the only differences are that the tolerances are slightly relaxed.

    """
    data = fixture_pls_ldpe_example
    assert isinstance(data["X"], np.ndarray)
    assert isinstance(data["Y"], np.ndarray)
    assert isinstance(data["A"], int)
    data["X"][11, 0] = float("nan")
    plsmodel = PLS(n_components=int(data["A"]), missing_data_settings=dict(md_method="nipals"))
    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(data["Y"])
    plsmodel = plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(pd.DataFrame(data["Y"])))
    # Can only get these to very loosely match
    assert data["expected_t2_lim_95_a6"] == pytest.approx(plsmodel.hotellings_t2_limit(0.95), rel=1e-1)
    assert data["expected_t2_lim_99_a6"] == pytest.approx(plsmodel.hotellings_t2_limit(0.99), rel=1e-1)

    assert np.mean(np.abs(np.array(data["expected_T"])) - np.abs(plsmodel.scores_.values)) == pytest.approx(0, abs=1e-2)
    assert np.mean(np.abs(np.array(data["expected_P"])) - np.abs(plsmodel.x_loadings_.values)) == pytest.approx(
        0, abs=1e-3
    )
    assert np.mean(np.abs(np.array(data["expected_W"])) - np.abs(plsmodel.x_weights_.values)) == pytest.approx(
        0, abs=1e-3
    )
    assert np.mean(np.abs(np.array(data["expected_C"])) - np.abs(plsmodel.y_loadings_.values)) == pytest.approx(
        0, abs=1e-3
    )
    assert np.mean(np.abs(np.array(data["expected_U"])) - np.abs(plsmodel.y_scores_.values)) == pytest.approx(
        0, abs=5e-1
    )
    assert np.mean(
        np.array(data["expected_hotellings_t2_a3"]).ravel() - plsmodel.hotellings_t2_.iloc[:, 2].values.ravel()
    ) == pytest.approx(0, abs=1e-6)
    assert np.mean(
        np.array(data["expected_hotellings_t2_a6"]).ravel() - plsmodel.hotellings_t2_.iloc[:, 5].values.ravel()
    ) == pytest.approx(0, abs=1e-6)
    assert np.mean(
        np.array(data["expected_sd_t"]).ravel() - plsmodel.scaling_factor_for_scores_.values.ravel()
    ) == pytest.approx(0, abs=1e-2)

    # Absolute sum of the deviations, accounting for the fact that each column in Y has quite
    # different range/scaling.
    assert np.sum(
        np.abs(
            np.sum(np.abs(Y_mcuv.inverse_transform(plsmodel.predictions_) - np.array(data["expected_yhat_a6"])), axis=0)
            / Y_mcuv.center_
        )
    ) == pytest.approx(0, abs=0.5)


@pytest.fixture
def fixture_pls_vip_calculation() -> dict:
    """
    Test PLS VIP calculation.

    Data from: Simca-P, version 14.1, fake data created for testing from 4 variables: A, B, C, D
    Target = y = 0.4*A + 0.005*B - 0.2*C + 4*D
    """
    out: dict = {}
    out["X"] = pd.DataFrame(
        np.array(
            [
                [1, 87, 6, 8],
                [2, 23, 3, 4],
                [3, 23, 5, 2],
                [4, 41, 1, 6],
                [5, 16, 4, 3],
                [6, 16, 1, 3],
                [7, 74, 5, 5],
                [8, 23, 6, 2],
                [9, 24, 6, 6],
            ]
        ),
        columns=["A", "B", "C", "D"],
    )
    out["X"]["y"] = pd.Series([0.4 * a + 0.005 * b - 0.2 * c + 4 * d for a, b, c, d in out["X"].values])
    out["expected_vip"] = np.array(
        [
            [0.207898, 1.15344, 0.153091, 1.61335],  # after 1 PC; each entry is for a column in X
            [0.519153, 1.11958, 0.361539, 1.53177],  # after 2 PC
            [0.533829, 1.12568, 0.372785, 1.51951],  # after 3 PC
        ]
    )
    out["expected_w*c"] = [
        [-0.103949, 0.576722, 0.0765457, 0.806677],
        [0.618124, -0.306383, -0.425238, 0.631601],
        [-0.33801, -0.764933, 0.249265, 0.554374],
    ]
    out["expected_loadings_p"] = [
        [-0.255614, 0.681147, 0.181164, 0.702548],
        [0.769273, -0.267136, -0.534682, 0.34085],
        [-0.74773, -0.689608, -0.000758336, 0.396744],
    ]
    out["expected_coefficiencts_unscaled"] = [
        [-0.209876, 0.121399, 0.208745, 2.16361],
        [0.655745, 0.0766668, -0.595588, 3.33859],
        [0.364184, 0.00787622, -0.305179, 3.97383],
    ]
    out["expected_coefficiencts_scaled_and_centerd"] = [
        [-0.070996, 0.393894, 0.0522798, 0.550951],
        [0.221822, 0.248755, -0.149164, 0.850153],
        [0.123194, 0.0255553, -0.0764317, 1.01191],
    ]
    out["r2_per_variable"] = [
        [0.117652, 0.835439, 0.0590982, 0.888761],
        [0.480221, 0.87916, 0.234252, 0.95994],
        [0.62223, 0.99995, 0.234252, 0.999921],
    ]
    out["r2_per_component"] = [0.839961, 0.977452, 0.999078]
    out["q2_per_component"] = [0.635802, 0.731724, 0.779401]
    out["q2_group_per_observation"] = [2, 3, 4, 5, 6, 7, 1, 2, 3]  # 9 observations; 7 groups

    return out


def test_pls_variable_importance(fixture_pls_vip_calculation: dict) -> None:
    """Unit tests for PLS VIP calculation."""
    data = fixture_pls_vip_calculation
    x_features = data["X"][["A", "B", "C", "D"]]
    y_target = data["X"]["y"]

    plsmodel = PLS(n_components=3)
    x_mcuv = MCUVScaler().fit(x_features)
    y_mcuv = MCUVScaler().fit(y_target)
    plsmodel.fit(x_mcuv.transform(x_features), y_mcuv.transform(pd.DataFrame(y_target)))

    # Compare the loadings first. Only check absolute values, since sign flips can occur
    expected_loadings_p = pd.DataFrame(data["expected_loadings_p"]).T.abs()
    plsmodel_x_loadings = plsmodel.x_loadings_.abs()
    assert expected_loadings_p.values == pytest.approx(plsmodel_x_loadings.values, abs=1e-5)

    # Compare the w*c values
    expected_w_c = pd.DataFrame(data["expected_w*c"]).T.abs()
    plsmodel_w_c = plsmodel.direct_weights_.abs()
    assert expected_w_c.values == pytest.approx(plsmodel_w_c.values, abs=1e-5)

    # Compare the coefficients
    expected_coefficiencts_scaled_and_centerd = pd.DataFrame(data["expected_coefficiencts_scaled_and_centerd"]).T.iloc[
        :, -1
    ]
    plsmodel_coeff_centered_scaled = plsmodel.beta_coefficients_
    assert expected_coefficiencts_scaled_and_centerd.values.reshape(-1, 1) == pytest.approx(
        plsmodel_coeff_centered_scaled.values, abs=1e-5
    )


def test_pls_vip_basic(fixture_pls_vip_calculation: dict) -> None:
    """VIP for a fitted PLS model: shape, index, positivity, and bound method."""
    data = fixture_pls_vip_calculation
    x_features = data["X"][["A", "B", "C", "D"]]
    y_target = data["X"]["y"]

    plsmodel = PLS(n_components=3)
    x_mcuv = MCUVScaler().fit(x_features)
    y_mcuv = MCUVScaler().fit(y_target)
    plsmodel.fit(x_mcuv.transform(x_features), y_mcuv.transform(pd.DataFrame(y_target)))

    # Standalone function and bound method must agree
    vip_standalone = vip(plsmodel)
    vip_bound = plsmodel.vip()
    assert vip_standalone.equals(vip_bound)

    # Structural checks
    assert isinstance(vip_standalone, pd.Series)
    assert vip_standalone.name == "VIP"
    assert len(vip_standalone) == x_features.shape[1]
    assert vip_standalone.index.tolist() == list(x_features.columns)
    assert (vip_standalone > 0).all()

    # Features with VIP > 1 are considered important
    # With 4 features and cumulative R² close to 1, at least one should be above 1
    assert (vip_standalone > 1).any()


def test_pls_vip_n_components(fixture_pls_vip_calculation: dict) -> None:
    """VIP changes when fewer components are used."""
    data = fixture_pls_vip_calculation
    x_features = data["X"][["A", "B", "C", "D"]]
    y_target = data["X"]["y"]

    plsmodel = PLS(n_components=3)
    x_mcuv = MCUVScaler().fit(x_features)
    y_mcuv = MCUVScaler().fit(y_target)
    plsmodel.fit(x_mcuv.transform(x_features), y_mcuv.transform(pd.DataFrame(y_target)))

    vip_all = plsmodel.vip()
    vip_1 = plsmodel.vip(n_components=1)
    vip_2 = plsmodel.vip(n_components=2)

    assert len(vip_1) == len(vip_all)  # same number of features
    assert not vip_all.equals(vip_1)  # but different values
    assert not vip_all.equals(vip_2)

    # Invalid n_components raises ValueError
    with pytest.raises(ValueError, match="n_components"):
        plsmodel.vip(n_components=0)
    with pytest.raises(ValueError, match="n_components"):
        plsmodel.vip(n_components=99)


def test_pca_vip_basic() -> None:
    """VIP for a fitted PCA model: shape, index, positivity, bound method."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 5)), columns=[f"X{i}" for i in range(1, 6)])
    X_scaled = MCUVScaler().fit_transform(X)

    pcamodel = PCA(n_components=3)
    pcamodel.fit(X_scaled)

    vip_standalone = vip(pcamodel)
    vip_bound = pcamodel.vip()
    assert vip_standalone.equals(vip_bound)

    assert isinstance(vip_standalone, pd.Series)
    assert vip_standalone.name == "VIP"
    assert len(vip_standalone) == X.shape[1]
    assert vip_standalone.index.tolist() == list(X.columns)
    assert (vip_standalone > 0).all()


def test_pca_vip_n_components() -> None:
    """VIP with n_components subset differs from full-model VIP for PCA."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((50, 4)), columns=["A", "B", "C", "D"])
    X_scaled = MCUVScaler().fit_transform(X)

    pcamodel = PCA(n_components=3)
    pcamodel.fit(X_scaled)

    vip_all = pcamodel.vip()
    vip_1 = pcamodel.vip(n_components=1)
    assert not vip_all.equals(vip_1)

    with pytest.raises(ValueError, match="n_components"):
        pcamodel.vip(n_components=0)
    with pytest.raises(ValueError, match="n_components"):
        pcamodel.vip(n_components=10)


def test_vip_unfitted_raises() -> None:
    """vip() raises ValueError when called on an unfitted model."""
    pls_unfitted = PLS(n_components=2)
    with pytest.raises(ValueError, match="not fitted"):
        vip(pls_unfitted)

    pca_unfitted = PCA(n_components=2)
    with pytest.raises(ValueError, match="not fitted"):
        vip(pca_unfitted)


def _driver_proxy_noise(seed: int = 0, n: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic block: a driver of y, a collinear proxy of the driver, and pure noise."""
    rng = np.random.default_rng(seed)
    driver = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "driver": driver,
            "proxy": driver + 0.02 * rng.standard_normal(n),  # collinear with driver
            "noise1": rng.standard_normal(n),
            "noise2": rng.standard_normal(n),
        }
    )
    y = pd.Series(2.0 * driver + 0.3 * rng.standard_normal(n), name="y")
    return X, y


def test_selectivity_ratio_ranks_driver_and_proxy_above_noise() -> None:
    """SR is high for the predictive direction (driver and its collinear proxy), low for noise."""
    X, y = _driver_proxy_noise()
    Xs = MCUVScaler().fit_transform(X)
    ys = MCUVScaler().fit_transform(y.to_frame())
    pls = PLS(n_components=2).fit(Xs, ys)

    sr_standalone = selectivity_ratio(pls, Xs)
    sr_bound = pls.selectivity_ratio(Xs)
    assert sr_standalone.equals(sr_bound)

    assert isinstance(sr_standalone, pd.Series)
    assert sr_standalone.name == "selectivity_ratio"
    assert list(sr_standalone.index) == list(X.columns)
    assert (sr_standalone >= 0).all()
    assert np.isfinite(sr_standalone).all()

    # Driver and its collinear proxy carry the predictive signal; noise does not.
    assert sr_standalone["driver"] > 1.0
    assert sr_standalone["proxy"] > 1.0
    assert sr_standalone["noise1"] < 0.5
    assert sr_standalone["noise2"] < 0.5

    # The selectivity ratio does NOT separate two collinear features (the key
    # limitation that the discriminator narrative rests on).
    assert sr_standalone["driver"] == pytest.approx(sr_standalone["proxy"], rel=0.5)

    # The advisory F-based critical value is attached and exceeded by the signal.
    assert sr_standalone.attrs["conf_level"] == pytest.approx(0.95)
    assert sr_standalone["driver"] > sr_standalone.attrs["f_critical"]
    assert sr_standalone["noise1"] < sr_standalone.attrs["f_critical"]


def test_target_projection_aligns_with_response() -> None:
    """The TP scores reproduce the predictive direction (correlate ~1 with y)."""
    X, y = _driver_proxy_noise()
    Xs = MCUVScaler().fit_transform(X)
    ys = MCUVScaler().fit_transform(y.to_frame())
    pls = PLS(n_components=2).fit(Xs, ys)

    tp = pls.target_projection(Xs)
    assert tp.scores.equals(target_projection(pls, Xs).scores)  # bound and standalone agree
    assert set(tp) >= {"scores", "loadings", "weights", "response"}
    assert list(tp.loadings.index) == list(X.columns)
    assert tp.weights.to_numpy() @ tp.weights.to_numpy() == pytest.approx(1.0)  # unit TP weight

    corr = np.corrcoef(tp.scores.to_numpy(), ys.to_numpy().ravel())[0, 1]
    assert abs(corr) > 0.95


def test_target_projection_refuses_a_response_x_does_not_predict() -> None:
    """A response orthogonal to every X column has a zero regression vector, so it has no TP direction."""
    x1 = np.array([1.0, -1.0, 1.0, -1.0, 0.0, 0.0])
    x2 = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 0.0])
    X = pd.DataFrame({"x1": x1, "x2": x2})
    Y = pd.DataFrame({"y1": x1 + 2.0 * x2, "y2": [0.0, 0.0, 0.0, 0.0, 1.0, -1.0]})
    model = PLS(n_components=1).fit(X, Y)
    with pytest.raises(
        ValueError, match=r"^The regression vector for response 'y2' is ~0; it is not predicted by X\.$"
    ):
        target_projection(model, X, response="y2")


def test_target_projection_refuses_rows_with_no_spread_along_its_direction() -> None:
    """Rows all at the training mean score 0 on the TP direction, which leaves no loading to form."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((30, 4)) * [1.0, 5.0, 10.0, 2.0] + 7.0, columns=list("abcd"))
    y = pd.DataFrame({"y": X.to_numpy() @ [1.0, 0.2, 0.1, -0.5] + rng.standard_normal(30)})
    model = PLS(n_components=2).fit(X, y)
    at_the_mean = pd.DataFrame(np.tile(X.mean().to_numpy(), (5, 1)), columns=X.columns)
    with pytest.raises(
        ValueError, match=r"^The target-projected scores have ~0 variance; cannot form the TP loading\.$"
    ):
        target_projection(model, at_the_mean)


def test_target_projection_without_scaling_uses_x_as_given() -> None:
    """With scale=False the TP scores are X @ w on X as passed, and an array X reads as its DataFrame."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((30, 4)) * [1.0, 5.0, 10.0, 2.0], columns=list("abcd"))
    X = X - X.mean()
    y = pd.DataFrame({"y": X.to_numpy() @ [1.0, 0.2, 0.1, -0.5] + rng.standard_normal(30)})
    model = PLS(n_components=2, scale=False).fit(X, y - y.mean())

    tp = target_projection(model, X)
    np.testing.assert_allclose(tp.scores.to_numpy(), X.to_numpy() @ tp.weights.to_numpy())
    from_array = target_projection(model, X.to_numpy())
    np.testing.assert_allclose(from_array.scores.to_numpy(), tp.scores.to_numpy())
    np.testing.assert_allclose(from_array.loadings.to_numpy(), tp.loadings.to_numpy())


def test_selectivity_ratio_multi_response_and_errors(
    fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """Multi-response SR returns a feature-by-response frame; errors are guarded; LDPE smoke."""
    # Multi-response: a feature-by-response DataFrame, with each column finite.
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"x{i}" for i in range(5)])
    beta = rng.standard_normal((5, 2))
    Y = pd.DataFrame(X.values @ beta + 0.3 * rng.standard_normal((40, 2)), columns=["y0", "y1"])
    Xs = MCUVScaler().fit_transform(X)
    Ys = MCUVScaler().fit_transform(Y)
    pls = PLS(n_components=3).fit(Xs, Ys)

    sr_frame = pls.selectivity_ratio(Xs)
    assert isinstance(sr_frame, pd.DataFrame)
    assert list(sr_frame.columns) == ["y0", "y1"]
    assert list(sr_frame.index) == list(X.columns)
    assert np.isfinite(sr_frame.to_numpy()).all()

    # Selecting one response by name or integer position gives a Series.
    assert isinstance(pls.selectivity_ratio(Xs, response="y0"), pd.Series)
    assert isinstance(pls.selectivity_ratio(Xs, response=1), pd.Series)
    with pytest.raises(ValueError, match="response"):
        pls.selectivity_ratio(Xs, response="not_a_response")
    with pytest.raises(ValueError, match="out of range"):
        pls.selectivity_ratio(Xs, response=9)
    with pytest.raises(ValueError, match=r"not fitted|beta_coefficients_"):
        selectivity_ratio(PLS(n_components=2), Xs)

    # target_projection on a multi-response model needs an explicit response,
    # and refuses an unfitted model.
    with pytest.raises(ValueError, match="several responses"):
        target_projection(pls, Xs)
    with pytest.raises(ValueError, match=r"not fitted|beta_coefficients_"):
        target_projection(PLS(n_components=2), Xs)

    # With only three samples the F-based critical value is undefined (NaN).
    tiny = MCUVScaler().fit_transform(X.iloc[:3])
    tiny_y = MCUVScaler().fit_transform(Y.iloc[:3, [0]])
    sr_tiny = PLS(n_components=1).fit(tiny, tiny_y).selectivity_ratio(tiny)
    assert np.isnan(sr_tiny.attrs["f_critical"])

    # Real LDPE dataset (SIMCA reference data): SR is finite and non-negative.
    data = fixture_pls_ldpe_example
    X_ldpe = MCUVScaler().fit_transform(data["X"])
    Y_ldpe = MCUVScaler().fit_transform(data["Y"])
    pls_ldpe = PLS(n_components=data["A"]).fit(X_ldpe, Y_ldpe)
    sr_values = pls_ldpe.selectivity_ratio(X_ldpe).to_numpy()
    assert np.isfinite(sr_values).all()
    assert (sr_values >= 0).all()


def test_vip_formula_correctness() -> None:
    """VIP formula: verify against manually computed values."""
    # Simple synthetic case with known PLS model internals
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((60, 4)), columns=["A", "B", "C", "D"])
    Y = pd.DataFrame(
        X.values @ np.array([2.0, 0.5, 0.1, 0.0]).reshape(-1, 1) + rng.standard_normal((60, 1)) * 0.05, columns=["y"]
    )
    X_scaled = MCUVScaler().fit_transform(X)
    Y_scaled = MCUVScaler().fit_transform(Y)

    model = PLS(n_components=2)
    model.fit(X_scaled, Y_scaled)

    result = model.vip()

    # Manually replicate the formula: VIP_j = sqrt(K * sum(r2_a * w_ja^2) / sum(r2_a))
    W = model.x_weights_.values  # (K, A)
    r2 = model.r2_per_component_.values  # (A,)
    K = W.shape[0]
    expected = np.sqrt(K * np.sum(r2.reshape(1, -1) * W**2, axis=1) / np.sum(r2))

    assert result.values == pytest.approx(expected, abs=1e-10)

    # Feature "A" (highest coefficient) should have the highest VIP
    assert result.idxmax() == "A"


def test_pls_score_contributions() -> None:
    """PLS contributions use the direct weights, and sum to the X-score."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((40, 6)), columns=[f"X{i}" for i in range(1, 7)])
    Y = pd.DataFrame(X.values @ rng.standard_normal((6, 2)) + rng.standard_normal((40, 2)) * 0.3, columns=["Y1", "Y2"])
    X_scaled = MCUVScaler().fit_transform(X)
    Y_scaled = MCUVScaler().fit_transform(Y)

    plsmodel = PLS(n_components=3)
    plsmodel.fit(X_scaled, Y_scaled)

    contrib = plsmodel.score_contributions(X_scaled, component=1)
    assert isinstance(contrib, pd.DataFrame)
    assert contrib.shape == X_scaled.shape
    assert contrib.columns.tolist() == X_scaled.columns.tolist()

    # For PLS the score-generating matrix is direct_weights_ (T = X R), not the
    # X-loadings, so that the terms sum to the score.
    for a in (1, 2, 3):
        terms = plsmodel.score_contributions(X_scaled, component=a)
        assert terms.sum(axis=1).to_numpy() == pytest.approx(plsmodel.scores_[a].to_numpy(), abs=1e-12)
        expected = X_scaled.to_numpy() * plsmodel.direct_weights_.to_numpy()[:, a - 1]
        assert terms.to_numpy() == pytest.approx(expected, abs=1e-12)

    shift = plsmodel.group_contributions(X_scaled, group=X_scaled.index[:3], reference=X_scaled.index[3:6])
    expected_shift = plsmodel.scores_[1].to_numpy()[:3].mean() - plsmodel.scores_[1].to_numpy()[3:6].mean()
    assert shift.sum() == pytest.approx(expected_shift, abs=1e-12)


def test_pls_detect_outliers() -> None:
    """Test PLS detect_outliers method."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((40, 6)), columns=[f"X{i}" for i in range(1, 7)])
    Y = pd.DataFrame(X.values @ rng.standard_normal((6, 2)) + rng.standard_normal((40, 2)) * 0.3, columns=["Y1", "Y2"])
    X_scaled = MCUVScaler().fit_transform(X)
    Y_scaled = MCUVScaler().fit_transform(Y)

    plsmodel = PLS(n_components=3)
    plsmodel.fit(X_scaled, Y_scaled)

    outliers = plsmodel.detect_outliers(conf_level=0.95)
    assert isinstance(outliers, list)

    # Each outlier should be a dict with expected keys
    if len(outliers) > 0:
        keys = {"observation", "outlier_types", "spe", "hotellings_t2", "spe_limit", "hotellings_t2_limit", "severity"}
        assert set(outliers[0].keys()) == keys
        # Should be sorted by severity descending
        severities = [o["severity"] for o in outliers]
        assert severities == sorted(severities, reverse=True)

    # Invalid confidence level should raise
    with pytest.raises(ValueError, match="conf_level must be between"):
        plsmodel.detect_outliers(conf_level=0.5)


def test_pls_score_contributions_ldpe(
    fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """PLS score contributions on the LDPE data: the identity holds on real data."""
    data = fixture_pls_ldpe_example
    X = pd.DataFrame(data["X"])
    Y = pd.DataFrame(data["Y"])
    X_scaled = MCUVScaler().fit_transform(X)
    Y_scaled = MCUVScaler().fit_transform(Y)

    plsmodel = PLS(n_components=3)
    plsmodel.fit(X_scaled, Y_scaled)

    contrib = plsmodel.score_contributions(X_scaled, component=1)
    assert contrib.shape == X_scaled.shape
    assert contrib.sum(axis=1).to_numpy() == pytest.approx(plsmodel.scores_[1].to_numpy(), abs=1e-10)

    # The known process fault sits in the last observations; the variable blamed
    # there must not simply be the variable with the largest loading, or the
    # diagnostic would carry no per-observation information at all.
    per_observation = {contrib.iloc[i].abs().idxmax() for i in range(len(contrib))}
    assert len(per_observation) > 1


def test_pls_detect_outliers_ldpe(
    fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """Test PLS detect_outliers on LDPE real dataset (known process fault in last 4 obs)."""
    data = fixture_pls_ldpe_example
    X = pd.DataFrame(data["X"])
    Y = pd.DataFrame(data["Y"])
    X_scaled = MCUVScaler().fit_transform(X)
    Y_scaled = MCUVScaler().fit_transform(Y)

    plsmodel = PLS(n_components=3)
    plsmodel.fit(X_scaled, Y_scaled)

    outliers = plsmodel.detect_outliers(conf_level=0.95)
    assert isinstance(outliers, list)

    if len(outliers) > 0:
        keys = {"observation", "outlier_types", "spe", "hotellings_t2", "spe_limit", "hotellings_t2_limit", "severity"}
        assert set(outliers[0].keys()) == keys
        severities = [o["severity"] for o in outliers]
        assert severities == sorted(severities, reverse=True)


def test_pls_select_n_components_q2_se_scales_with_repeats() -> None:
    """The Q2 band shrinks like the square root of the repeats, not like the repeats.

    ``tss_y`` carries a per-row coverage weighting, so it counts every row once
    per repeat that tested it, while the numerator of ``q2_se`` is put on the
    scale of a single pass. Dividing one by the other without matching the two
    made the band a factor of ``n_repeats`` too narrow, which turned
    ``selection_rule="1se"`` into ``"min"`` for any repeated splitter.
    """
    rng = np.random.default_rng(0)
    N, K, M = 60, 8, 3
    T = rng.normal(size=(N, 2))
    X = pd.DataFrame(T @ rng.normal(size=(2, K)) + 0.5 * rng.normal(size=(N, K)))
    Y = pd.DataFrame(T @ rng.normal(size=(2, M)) + 0.3 * rng.normal(size=(N, M)))

    one = PLS.select_n_components(X, Y, max_components=3, cv=5, n_repeats=1, random_state=0)
    many = PLS.select_n_components(X, Y, max_components=3, cv=5, n_repeats=25, random_state=0)

    # Q2 itself is a ratio of two coverage-weighted sums, so it is unmoved by
    # the repeat count; only its band should tighten.
    np.testing.assert_allclose(one.r2y_validated["total"].to_numpy(), many.r2y_validated["total"].to_numpy(), atol=0.02)
    # Twenty-five repeats sample the fold PRESS 125 times rather than 5, so the
    # standard error of the mean shrinks by about sqrt(25) = 5. A band that
    # shrank by the repeat count itself would be 25x narrower, which is the
    # regression this pins: the ratio has to stay well above 1/25.
    ratio = float(np.median((many.q2_se / one.q2_se).to_numpy()))
    assert 1.0 / 12.0 < ratio < 1.0
    assert ((many.q2_se > 0) & np.isfinite(many.q2_se)).all()


def test_pls_select_n_components_scaled_total_is_the_autoscaled_q2() -> None:
    """``scaled_total`` is what ``total`` would have been on an autoscaled Y.

    The ``"total"`` column pools PRESS and TSS on the original Y scale, so a
    target whose spread is a hundred times its neighbour's decides it almost
    alone; here ``y_big`` is pure noise and ``y_small`` is well predicted, and
    the two columns end up on opposite sides of zero. Running the same
    cross-validation on a mean-centred, unit-variance Y is the reference: the
    per-target results are unchanged by that affine map, and its ``"total"``
    has to reproduce ``"scaled_total"`` from the raw-scale run exactly.
    """
    rng = np.random.default_rng(11)
    N, K = 80, 6
    T = rng.normal(size=(N, 2))
    X = pd.DataFrame(T @ rng.normal(size=(2, K)) + 0.3 * rng.normal(size=(N, K)), columns=[f"x{i}" for i in range(K)])
    Y = pd.DataFrame(
        {
            "y_small": T[:, 0] * 0.1 + 0.01 * rng.normal(size=N),
            "y_big": 200.0 * rng.normal(size=N),
        }
    )

    raw = PLS.select_n_components(X, Y, max_components=2, cv=5, n_repeats=2, random_state=0)
    scaled = PLS.select_n_components(
        X, MCUVScaler().fit_transform(Y), max_components=2, cv=5, n_repeats=2, random_state=0
    )

    np.testing.assert_allclose(
        raw.r2y_validated["scaled_total"].to_numpy(), scaled.r2y_validated["total"].to_numpy(), rtol=1e-10
    )
    # Equivalently, the equal-weight pooling is the mean over targets.
    np.testing.assert_allclose(
        raw.r2y_validated["scaled_total"].to_numpy(),
        raw.r2y_validated[["y_small", "y_big"]].mean(axis=1).to_numpy(),
        rtol=1e-12,
    )
    # The point of the column: on this Y the two pooled numbers disagree in sign.
    assert raw.r2y_validated["total"].iloc[0] < 0.0 < raw.r2y_validated["scaled_total"].iloc[0]


def test_pls_select_n_components_scaled_total_ignores_a_constant_target() -> None:
    """A target with no spread is left out of the mean, not allowed to void it."""
    rng = np.random.default_rng(3)
    N, K = 40, 4
    T = rng.normal(size=(N, 1))
    X = pd.DataFrame(T @ rng.normal(size=(1, K)) + 0.2 * rng.normal(size=(N, K)))
    Y = pd.DataFrame({"live": T[:, 0] + 0.1 * rng.normal(size=N), "flat": np.full(N, 7.0)})

    result = PLS.select_n_components(X, Y, max_components=2, cv=4, n_repeats=1, random_state=0)

    assert result.r2y_validated["flat"].isna().all()
    np.testing.assert_allclose(
        result.r2y_validated["scaled_total"].to_numpy(), result.r2y_validated["live"].to_numpy(), rtol=1e-12
    )


@pytest.mark.slow
def test_pls_select_n_components_synthetic() -> None:
    """Cross-validated component selection recovers a known low-rank structure."""
    rng = np.random.default_rng(42)
    N, K = 30, 25

    # X is driven by 2 latent factors; Y depends only on those same 2 factors.
    # With N < K and moderate noise, components beyond 2 overfit, so RMSECV
    # has a clear minimum at the true rank.
    T_true = rng.normal(size=(N, 2))
    P_true = rng.normal(size=(K, 2))
    X = pd.DataFrame(T_true @ P_true.T + 0.8 * rng.normal(size=(N, K)), columns=[f"x{i}" for i in range(K)])
    gamma = rng.normal(size=(2, 1))
    Y = pd.DataFrame(T_true @ gamma + 0.3 * rng.normal(size=(N, 1)), columns=["y"])

    # Raw blocks: the default scale_inside_folds=True fits the scaling on each
    # training fold, which is the usage the library recommends.
    result = PLS.select_n_components(X, Y, max_components=10, random_state=0)

    assert isinstance(result, Bunch)
    assert set(result.keys()) == {
        "n_components",
        "rmsecv",
        "per_fold_rmsecv",
        "se_rmsecv",
        "q2_se",
        "r2y_validated",
        "r2x_validated",
        "press",
        "randomization_pvalues",
        "selection_distribution",
        "selection_mode",
        "selection_is_stable",
        "cv_predictions",
        "selection_rule",
    }
    assert result.selection_rule == "1se"

    # q2_se is the per-component +/-1 SE band on the Q2 scale: a finite,
    # non-negative series aligned to the r2y_validated["total"] curve.
    assert isinstance(result.q2_se, pd.Series)
    assert list(result.q2_se.index) == list(range(1, 11))
    assert ((result.q2_se >= 0) & np.isfinite(result.q2_se)).all()

    # The clear RMSECV minimum is at the true rank of 2.
    assert result.n_components == 2

    # RMSECV is a DataFrame indexed 1..10 with one column per Y variable + total.
    assert isinstance(result.rmsecv, pd.DataFrame)
    assert list(result.rmsecv.index) == list(range(1, 11))
    assert list(result.rmsecv.columns) == ["y", "total"]

    # RMSECV drops from the 1st to the 2nd component, then rises (overfitting).
    assert result.rmsecv["total"][2] < result.rmsecv["total"][1]
    assert result.rmsecv["total"][3] > result.rmsecv["total"][2]

    # Validated explained variance is meaningful at the recommended count and
    # never reaches the (calibration-only) ceiling of 1.0.
    assert 0.5 < result.r2y_validated["total"][2] < 1.0
    assert (result.r2x_validated["total"] < 1.0).all()

    # Out-of-fold predictions cover every observation.
    assert result.cv_predictions.shape == (N, 1)
    assert not result.cv_predictions.isna().any().any()


@pytest.mark.slow
def test_pls_select_n_components_ldpe(
    fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """Smoke test of cross-validated PLS metrics on the real LDPE dataset."""
    data = fixture_pls_ldpe_example
    X = pd.DataFrame(data["X"], columns=[f"x{i}" for i in range(data["X"].shape[1])])
    Y = pd.DataFrame(data["Y"], columns=[f"y{i}" for i in range(data["Y"].shape[1])])
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]

    max_comp = 6
    result = PLS.select_n_components(X, Y, max_components=max_comp, cv=5)

    assert 1 <= result.n_components <= max_comp
    assert result.rmsecv.shape == (max_comp, n_targets + 1)
    assert result.r2y_validated.shape == (max_comp, n_targets + 2)  # targets, "total", "scaled_total"
    assert result.r2x_validated.shape == (max_comp, n_features + 1)
    assert (result.rmsecv.to_numpy() > 0).all()
    assert (result.press > 0).all()
    assert (result.r2y_validated["total"] <= 1.0).all()
    assert result.cv_predictions.shape == (n_samples, n_targets)


@pytest.mark.slow
def test_pls_select_n_components_accepts_splitters() -> None:
    """The cv argument accepts an int and any sklearn splitter object."""
    rng = np.random.default_rng(7)
    N, K = 28, 6
    X = pd.DataFrame(rng.standard_normal((N, K)), columns=[f"x{i}" for i in range(K)])
    beta = rng.standard_normal((K, 2))
    Y = pd.DataFrame(X.values @ beta + 0.2 * rng.standard_normal((N, 2)), columns=["y0", "y1"])

    res_kfold = PLS.select_n_components(X, Y, max_components=4, cv=KFold(5, shuffle=True, random_state=0))
    assert 1 <= res_kfold.n_components <= 4
    assert res_kfold.rmsecv.shape == (4, 3)

    res_loo = PLS.select_n_components(X, Y, max_components=4, cv=LeaveOneOut())
    assert 1 <= res_loo.n_components <= 4
    assert res_loo.cv_predictions.shape == (N, 2)
    assert not res_loo.cv_predictions.isna().any().any()


def test_pls_select_n_components_caps_max_components() -> None:
    """max_components is capped to what every CV training fold can support."""
    rng = np.random.default_rng(11)
    N, K = 30, 25
    X = pd.DataFrame(rng.standard_normal((N, K)), columns=[f"x{i}" for i in range(K)])
    Y = pd.DataFrame(rng.standard_normal((N, 1)), columns=["y"])

    # 5-fold CV leaves a training fold of 24 rows; in-fold mean-centring
    # removes one DoF, so at most 23 components can be evaluated despite the
    # absurdly large request. (n_repeats=1 keeps the cap-vs-singularity test
    # focused on the cap; the rank-boundary fit is fragile by nature.)
    result = PLS.select_n_components(X, Y, max_components=1000, cv=5, n_repeats=1, random_state=0)
    assert len(result.rmsecv) == 23
    assert result.n_components <= 23


def test_recommend_n_components_one_se_rejects_non_1d() -> None:
    """2-D mean_error fails fast - the helper is 1-D only."""
    from process_improve.multivariate._common import _recommend_n_components_one_se

    with pytest.raises(ValueError, match="1-D"):
        _recommend_n_components_one_se(
            np.array([[0.5, 0.4], [0.3, 0.2]]),
            np.array([[0.05, 0.05], [0.05, 0.05]]),
        )


def test_pls_select_n_components_argument_validation() -> None:
    """The new kwargs validate their inputs up front."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((20, 4)), columns=[f"x{i}" for i in range(4)])
    Y = pd.DataFrame(rng.standard_normal((20, 1)), columns=["y"])
    with pytest.raises(ValueError, match=">= 2"):
        PLS.select_n_components(X, Y, cv=1)
    with pytest.raises(ValueError, match="n_repeats must be >= 1"):
        PLS.select_n_components(X, Y, cv=5, n_repeats=0)


@pytest.mark.slow
def test_pls_select_n_components_1se_default_picks_parsimony() -> None:
    """The default (1-SE rule) is more parsimonious than argmin RMSECV on noise-padded data.

    Eight latent factors of geometrically decaying influence drive Y; later
    factors contribute progressively less. Argmin chases noise-level
    improvements; the 1-SE rule should land near the true effective rank.
    """
    rng = np.random.default_rng(0)
    N, K, n_factors = 80, 50, 8
    scales = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.2, 0.8, 0.5])
    T_true = rng.normal(size=(N, n_factors)) * scales
    P_true = rng.normal(size=(K, n_factors))
    X = pd.DataFrame(T_true @ P_true.T + 0.4 * rng.normal(size=(N, K)), columns=[f"x{i}" for i in range(K)])
    gamma = rng.normal(size=(n_factors, 1))
    Y = pd.DataFrame(T_true @ gamma + 0.3 * rng.normal(size=(N, 1)), columns=["y"])

    res_1se = PLS.select_n_components(X, Y, max_components=12, cv=7, n_repeats=5, random_state=0)
    res_min = PLS.select_n_components(X, Y, max_components=12, cv=7, n_repeats=5, random_state=0, selection_rule="min")
    assert res_1se.selection_rule == "1se"
    assert res_min.selection_rule == "min"
    # The 1-SE rule never picks more components than argmin RMSECV, and on this
    # noise-padded design it picks meaningfully fewer.
    assert res_1se.n_components <= res_min.n_components
    assert res_1se.n_components <= 8


@pytest.mark.slow
def test_pls_select_n_components_repeated_kfold_stability() -> None:
    """Repeated K-fold gives reproducible answers given a random_state and
    narrows the per-component standard error as repeats grow.
    """
    rng = np.random.default_rng(13)
    N, K = 50, 12
    X = pd.DataFrame(rng.standard_normal((N, K)), columns=[f"x{i}" for i in range(K)])
    beta = rng.standard_normal((K, 1))
    Y = pd.DataFrame(X.values @ beta + 0.4 * rng.standard_normal((N, 1)), columns=["y"])

    a = PLS.select_n_components(X, Y, max_components=6, cv=5, n_repeats=3, random_state=42)
    b = PLS.select_n_components(X, Y, max_components=6, cv=5, n_repeats=3, random_state=42)
    # Same seed -> identical numerical result.
    np.testing.assert_allclose(a.rmsecv.to_numpy(), b.rmsecv.to_numpy())
    np.testing.assert_allclose(a.se_rmsecv.to_numpy(), b.se_rmsecv.to_numpy())

    few = PLS.select_n_components(X, Y, max_components=6, cv=5, n_repeats=2, random_state=0)
    many = PLS.select_n_components(X, Y, max_components=6, cv=5, n_repeats=20, random_state=0)
    # 2 repeats -> 10 folds; 20 repeats -> 100 folds; SE narrows ~sqrt(10).
    assert few.per_fold_rmsecv.shape == (6, 10)
    assert many.per_fold_rmsecv.shape == (6, 100)
    assert (many.se_rmsecv < few.se_rmsecv).all()


@pytest.mark.slow
def test_pls_select_n_components_in_fold_scaling_no_leakage() -> None:
    """In-fold scaling is the default; raw, unscaled X/Y produce sensible RMSECV."""
    rng = np.random.default_rng(99)
    N, K, n_factors = 40, 8, 3
    # Low-rank X on an arbitrary, non-zero-mean / non-unit-variance scale; Y
    # depends on the latent factors plus noise. Keeping K > n_factors and well
    # under N avoids the rank-boundary fragility this test isn't about.
    T_true = rng.standard_normal((N, n_factors))
    P_true = rng.standard_normal((n_factors, K))
    column_scales = np.array([10.0, 0.1, 100.0, 1.0, 5.0, 0.5, 2.0, 20.0])
    X_raw = pd.DataFrame(
        (T_true @ P_true + 0.4 * rng.standard_normal((N, K))) * column_scales + 7.0,
        columns=[f"x{i}" for i in range(K)],
    )
    Y_raw = pd.DataFrame(
        T_true @ rng.standard_normal((n_factors, 1)) + 0.3 * rng.standard_normal((N, 1)) + 50.0,
        columns=["y"],
    )

    # Default: scale inside folds. Caller passes raw data.
    raw = PLS.select_n_components(X_raw, Y_raw, max_components=5, cv=5, n_repeats=4, random_state=0)
    # RMSECV is reported on the original Y scale.
    assert (raw.rmsecv["total"] > 0).all()
    assert raw.cv_predictions.shape == (N, 1)
    # First-repeat predictions cover every row.
    assert not raw.cv_predictions.isna().any().any()

    # Opt-out: scale outside, warn, and reproduce the leakage-prone path.
    X_s = MCUVScaler().fit_transform(X_raw)
    Y_s = MCUVScaler().fit_transform(Y_raw)
    with pytest.warns(SpecificationWarning, match="leaks"):
        leaky = PLS.select_n_components(
            X_s,
            Y_s,
            max_components=5,
            cv=5,
            n_repeats=4,
            random_state=0,
            scale_inside_folds=False,
        )
    # Both runs choose sensible component counts under the 1-SE default.
    assert 1 <= raw.n_components <= 5
    assert 1 <= leaky.n_components <= 5


@pytest.mark.slow
def test_pls_select_n_components_q2_increment_rule() -> None:
    """The Q2-increment rule (from PR #371) is preserved as an opt-in."""
    rng = np.random.default_rng(2025)
    N, K = 60, 20
    T = rng.standard_normal((N, 3)) * np.array([5.0, 3.0, 1.5])
    P = rng.standard_normal((3, K))
    X = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)), columns=[f"x{i}" for i in range(K)])
    Y = pd.DataFrame(T @ rng.standard_normal((3, 1)) + 0.2 * rng.standard_normal((N, 1)), columns=["y"])
    res = PLS.select_n_components(
        X,
        Y,
        max_components=8,
        cv=5,
        n_repeats=3,
        random_state=0,
        selection_rule="q2_increment",
        min_q2_increase=0.02,
    )
    assert res.selection_rule == "q2_increment"
    assert 1 <= res.n_components <= 4

    # Unknown rule -> ValueError from the dispatcher.
    with pytest.raises(ValueError, match="Unknown selection_rule"):
        PLS.select_n_components(X, Y, max_components=3, cv=3, selection_rule="bogus")  # type: ignore[arg-type]


@pytest.mark.slow
def test_pls_select_n_components_stability_signal() -> None:
    """Per-repeat vote distribution flags confident vs. uncertain recommendations."""
    rng = np.random.default_rng(0)
    N, K, true_rank = 60, 12, 2
    T = rng.standard_normal((N, true_rank)) * np.array([6.0, 3.5])
    P = rng.standard_normal((true_rank, K))
    X = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)), columns=[f"x{i}" for i in range(K)])
    Y = pd.DataFrame(T @ rng.standard_normal((true_rank, 1)) + 0.2 * rng.standard_normal((N, 1)), columns=["y"])

    # Multiple repeats + clean signal -> the distribution concentrates and
    # the recommendation is stable.
    confident = PLS.select_n_components(
        X,
        Y,
        max_components=6,
        cv=5,
        n_repeats=8,
        random_state=0,
    )
    assert confident.selection_distribution is not None
    assert isinstance(confident.selection_distribution, pd.Series)
    assert list(confident.selection_distribution.index) == list(range(1, 7))
    assert np.isclose(confident.selection_distribution.sum(), 1.0)
    assert confident.selection_mode == confident.n_components
    assert confident.selection_is_stable is True

    # A single repeat: no per-repeat distribution to report.
    single = PLS.select_n_components(
        X,
        Y,
        max_components=6,
        cv=5,
        n_repeats=1,
        random_state=0,
    )
    assert single.selection_distribution is None
    assert single.selection_mode is None
    assert single.selection_is_stable is None

    # q2_increment doesn't decompose per repeat in this implementation -
    # stability is None even with many repeats. (Documented in the docstring.)
    q2 = PLS.select_n_components(
        X,
        Y,
        max_components=6,
        cv=5,
        n_repeats=5,
        random_state=0,
        selection_rule="q2_increment",
    )
    assert q2.selection_distribution is None
    assert q2.selection_is_stable is None

    # The threshold knob: a > 1 threshold trips ``is_stable`` to False even
    # on a fully concentrated distribution. (Exercised here because the
    # synthetic data is clean enough that every repeat votes for the same
    # count; ``stability_threshold > 1.0`` is the only way to force False.)
    strict = PLS.select_n_components(
        X,
        Y,
        max_components=6,
        cv=5,
        n_repeats=8,
        random_state=0,
        stability_threshold=1.01,
    )
    assert strict.selection_is_stable is False


@pytest.mark.slow
def test_pls_nested_cv_recovers_known_rank_and_reports_rmsep() -> None:
    """Nested CV converges on the true rank and reports honest RMSEP."""
    rng = np.random.default_rng(7)
    N, K, true_rank = 80, 20, 2
    T = rng.standard_normal((N, true_rank)) * np.array([6.0, 3.5])
    P = rng.standard_normal((true_rank, K))
    X = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)), columns=[f"x{i}" for i in range(K)])
    Y = pd.DataFrame(
        T @ rng.standard_normal((true_rank, 1)) + 0.25 * rng.standard_normal((N, 1)),
        columns=["y"],
    )

    result = PLS.nested_cv(
        X,
        Y,
        max_components=6,
        outer_cv=5,
        inner_cv=5,
        n_inner_repeats=3,
        random_state=0,
    )

    # Bunch shape.
    assert {
        "rmsep",
        "q2y",
        "cv_predictions",
        "selected_components_per_fold",
        "selected_components_distribution",
    } <= set(result.keys())
    assert isinstance(result.rmsep, pd.Series)
    assert list(result.rmsep.index) == ["y", "total"]
    assert (result.rmsep > 0).all()
    assert isinstance(result.q2y, pd.Series)
    # On a clean rank-2 dataset Q2Y should be solidly positive.
    assert result.q2y["total"] > 0.5

    # 5 outer folds -> 5 inner picks, each in [1, max_components].
    assert len(result.selected_components_per_fold) == 5
    assert all(1 <= a <= 6 for a in result.selected_components_per_fold)
    # On clean rank-2 data the inner CV consistently picks around 2.
    assert all(a <= 4 for a in result.selected_components_per_fold)

    # Every row covered exactly once (KFold).
    assert result.cv_predictions.shape == (N, 1)
    assert not result.cv_predictions.isna().any().any()

    # Vote distribution sums to 1.
    assert np.isclose(result.selected_components_distribution.sum(), 1.0)


def test_pls_nested_cv_is_reproducible() -> None:
    """Same random_state -> identical nested-CV result."""
    rng = np.random.default_rng(11)
    N, K = 60, 15
    X = pd.DataFrame(rng.standard_normal((N, K)), columns=[f"x{i}" for i in range(K)])
    Y = pd.DataFrame(rng.standard_normal((N, 1)), columns=["y"])

    a = PLS.nested_cv(
        X,
        Y,
        max_components=4,
        outer_cv=4,
        inner_cv=4,
        n_inner_repeats=2,
        random_state=0,
    )
    b = PLS.nested_cv(
        X,
        Y,
        max_components=4,
        outer_cv=4,
        inner_cv=4,
        n_inner_repeats=2,
        random_state=0,
    )
    np.testing.assert_allclose(a.rmsep.to_numpy(), b.rmsep.to_numpy())
    assert a.selected_components_per_fold == b.selected_components_per_fold


def test_pls_nested_cv_validates_outer_cv() -> None:
    """outer_cv < 2 is rejected."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((20, 4)))
    Y = pd.DataFrame(rng.standard_normal((20, 1)))
    with pytest.raises(ValueError, match=">= 2"):
        PLS.nested_cv(X, Y, max_components=2, outer_cv=1, inner_cv=3, n_inner_repeats=1)


def test_pls_nested_cv_scale_inside_folds_false() -> None:
    """The leakage-prone scale_inside_folds=False path still works on pre-scaled data."""
    rng = np.random.default_rng(13)
    N, K = 40, 8
    T = rng.standard_normal((N, 2))
    P = rng.standard_normal((2, K))
    X = pd.DataFrame(T @ P + 0.3 * rng.standard_normal((N, K)))
    Y = pd.DataFrame(T @ rng.standard_normal((2, 1)) + 0.2 * rng.standard_normal((N, 1)), columns=["y"])
    X_s = MCUVScaler().fit_transform(X)
    Y_s = MCUVScaler().fit_transform(Y)
    with pytest.warns(SpecificationWarning):
        result = PLS.nested_cv(
            X_s,
            Y_s,
            max_components=4,
            outer_cv=3,
            inner_cv=3,
            n_inner_repeats=1,
            random_state=0,
            scale_inside_folds=False,
        )
    assert (result.rmsep > 0).all()
    assert result.cv_predictions.shape == (N, 1)


def test_pls_nested_cv_accepts_custom_outer_splitter() -> None:
    """outer_cv can be any sklearn splitter, not just an int."""
    rng = np.random.default_rng(17)
    N, K = 40, 6
    X = pd.DataFrame(rng.standard_normal((N, K)))
    Y = pd.DataFrame(rng.standard_normal((N, 1)), columns=["y"])
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)
    result = PLS.nested_cv(
        X,
        Y,
        max_components=3,
        outer_cv=splitter,
        inner_cv=3,
        n_inner_repeats=1,
        random_state=0,
    )
    assert len(result.selected_components_per_fold) == 4
    assert result.cv_predictions.shape == (N, 1)


# -----------------------------------------------------------------------------
# sklearn Pipeline / cross_val_score / GridSearchCV interop (#383)
# -----------------------------------------------------------------------------


def _make_pls_pipeline_data(seed: int = 0, *, n_targets: int = 1):
    rng = np.random.default_rng(seed)
    N, K, true_rank = 50, 10, 2
    T = rng.standard_normal((N, true_rank)) * np.array([6.0, 3.5])
    P = rng.standard_normal((true_rank, K))
    X = pd.DataFrame(
        T @ P + 0.3 * rng.standard_normal((N, K)),
        columns=[f"x{i}" for i in range(K)],
    )
    gamma = rng.standard_normal((true_rank, n_targets))
    Y = pd.DataFrame(
        T @ gamma + 0.25 * rng.standard_normal((N, n_targets)),
        columns=[f"y{i}" for i in range(n_targets)],
    )
    return X, Y


def test_pipeline_mcuv_pca_fit_transform() -> None:
    """Pipeline([MCUVScaler, PCA]).fit_transform returns the PCA scores."""
    from sklearn.pipeline import Pipeline

    X, _ = _make_pls_pipeline_data(seed=1)
    pipe = Pipeline([("sc", MCUVScaler()), ("pca", PCA(n_components=2))])
    scores = pipe.fit_transform(X)
    assert scores.shape == (X.shape[0], 2)
    # Roundtrip: transform reproduces fit_transform's result.
    np.testing.assert_allclose(np.asarray(scores), np.asarray(pipe.transform(X)))


def test_pipeline_mcuv_pls_fit_predict() -> None:
    """Pipeline([MCUVScaler, PLS]).fit / predict works end-to-end."""
    from sklearn.pipeline import Pipeline

    X, Y = _make_pls_pipeline_data(seed=2)
    Y_arr = Y.to_numpy()
    pipe = Pipeline([("sc", MCUVScaler()), ("pls", PLS(n_components=2))])
    pipe.fit(X, Y_arr)
    pred = pipe.predict(X)
    assert np.asarray(pred).shape == (X.shape[0], Y_arr.shape[1])


def test_pipeline_clone_round_trip() -> None:
    """clone(pipe) reproduces the same fitted result given the same data."""
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    X, _ = _make_pls_pipeline_data(seed=3)
    pipe = Pipeline([("sc", MCUVScaler()), ("pca", PCA(n_components=3))])
    a = clone(pipe).fit(X)
    b = clone(pipe).fit(X)
    # Loadings agree up to sign; compare absolute values for robustness.
    np.testing.assert_allclose(
        np.abs(a.named_steps["pca"].loadings_.to_numpy()),
        np.abs(b.named_steps["pca"].loadings_.to_numpy()),
    )


def test_pipeline_cross_val_score_pls() -> None:
    """cross_val_score over Pipeline([MCUVScaler, PLS]) avoids leakage by construction."""
    from sklearn.model_selection import RepeatedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    X, Y = _make_pls_pipeline_data(seed=4)
    Y_arr = Y.to_numpy().ravel()
    pipe = Pipeline([("sc", MCUVScaler()), ("pls", PLS(n_components=2))])
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=0)
    scores = cross_val_score(pipe, X, Y_arr, cv=cv, scoring="r2")
    assert scores.shape == (10,)
    # On clean rank-2 data the R^2 over folds should be solidly positive.
    assert scores.mean() > 0.5


def test_pipeline_gridsearchcv_selects_components() -> None:
    """GridSearchCV picks n_components for the PLS step inside a Pipeline."""
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline

    X, Y = _make_pls_pipeline_data(seed=5)
    Y_arr = Y.to_numpy().ravel()
    pipe = Pipeline([("sc", MCUVScaler()), ("pls", PLS(n_components=1))])
    grid = GridSearchCV(
        pipe,
        param_grid={"pls__n_components": [1, 2, 3, 4]},
        cv=KFold(5, shuffle=True, random_state=0),
        scoring="r2",
    )
    grid.fit(X, Y_arr)
    # GridSearchCV ranks by mean CV R^2, which tends to over-select on
    # noise-padded data exactly as documented in the 1-SE rationale; the
    # test is here to demonstrate that Pipeline + GridSearchCV *runs* on
    # MCUVScaler + PLS, not to compete with select_n_components(rule="1se").
    assert grid.best_params_["pls__n_components"] in {1, 2, 3, 4}
    assert grid.best_score_ > 0.5


def test_get_feature_names_out_mcuv_pca_pls() -> None:
    """get_feature_names_out works on each estimator and threads through Pipeline.set_output."""
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((30, 4)), columns=["A", "B", "C", "D"])
    Y = pd.DataFrame(rng.standard_normal((30, 1)), columns=["y"])

    # MCUVScaler is column-preserving: returns the fit-time feature names.
    sc = MCUVScaler().fit(X)
    assert list(sc.get_feature_names_out()) == ["A", "B", "C", "D"]
    # Passing matching input_features explicitly also works (sklearn convention).
    assert list(sc.get_feature_names_out(["A", "B", "C", "D"])) == ["A", "B", "C", "D"]

    # PCA labels its scores PC1..PCn.
    pca = PCA(n_components=3).fit(MCUVScaler().fit_transform(X))
    assert list(pca.get_feature_names_out()) == ["PC1", "PC2", "PC3"]

    # PLS labels its X-scores T1..Tn.
    pls = PLS(n_components=2).fit(
        MCUVScaler().fit_transform(X),
        MCUVScaler().fit_transform(Y),
    )
    assert list(pls.get_feature_names_out()) == ["T1", "T2"]

    # Pipeline introspection: composed get_feature_names_out follows the last
    # transform step's labels.
    pipe = Pipeline([("sc", MCUVScaler()), ("pca", PCA(n_components=2))])
    pipe.fit(X)
    assert list(pipe.get_feature_names_out()) == ["PC1", "PC2"]

    # set_output(transform="pandas") wraps the ndarray transform output in a
    # DataFrame labelled with our get_feature_names_out.
    pipe_pd = Pipeline([("sc", MCUVScaler()), ("pca", PCA(n_components=2))]).set_output(transform="pandas")
    out = pipe_pd.fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["PC1", "PC2"]
    assert out.shape == (30, 2)


@pytest.mark.slow
def test_pls_select_n_components_randomization_rule() -> None:
    """Van der Voet's randomization test picks a parsimonious model."""
    rng = np.random.default_rng(2026)
    N, K, n_factors = 80, 30, 4
    scales = np.array([6.0, 4.0, 2.5, 1.5])
    T = rng.normal(size=(N, n_factors)) * scales
    P = rng.normal(size=(K, n_factors))
    X = pd.DataFrame(T @ P.T + 0.4 * rng.normal(size=(N, K)), columns=[f"x{i}" for i in range(K)])
    gamma = rng.normal(size=(n_factors, 1))
    Y = pd.DataFrame(T @ gamma + 0.3 * rng.normal(size=(N, 1)), columns=["y"])

    res = PLS.select_n_components(
        X,
        Y,
        max_components=10,
        cv=5,
        n_repeats=2,
        random_state=0,
        selection_rule="randomization",
        n_permutations=399,
        alpha=0.01,
    )
    assert res.selection_rule == "randomization"
    assert res.randomization_pvalues is not None
    assert list(res.randomization_pvalues.index) == list(range(1, 11))
    # The reference (argmin-RMSECV) model has p-value exactly 1.0.
    a_ref = int(np.nanargmin(res.rmsecv["total"].to_numpy())) + 1
    assert res.randomization_pvalues.loc[a_ref] == 1.0
    # Recommendation is no larger than the argmin reference and at least 1.
    assert 1 <= res.n_components <= a_ref

    # Reproducible given a fixed seed.
    again = PLS.select_n_components(
        X,
        Y,
        max_components=10,
        cv=5,
        n_repeats=2,
        random_state=0,
        selection_rule="randomization",
        n_permutations=399,
        alpha=0.01,
    )
    np.testing.assert_allclose(
        res.randomization_pvalues.to_numpy(),
        again.randomization_pvalues.to_numpy(),
    )

    # A more permissive alpha (closer to 1) walks further up the curve
    # before declaring a model "equivalent", so it can pick MORE components
    # than a stricter alpha.
    res_strict = PLS.select_n_components(
        X,
        Y,
        max_components=10,
        cv=5,
        n_repeats=2,
        random_state=0,
        selection_rule="randomization",
        n_permutations=399,
        alpha=0.001,
    )
    res_lax = PLS.select_n_components(
        X,
        Y,
        max_components=10,
        cv=5,
        n_repeats=2,
        random_state=0,
        selection_rule="randomization",
        n_permutations=399,
        alpha=0.5,
    )
    assert res_strict.n_components <= res_lax.n_components

    # Without the randomization rule, randomization_pvalues is None.
    plain = PLS.select_n_components(X, Y, max_components=4, cv=5, random_state=0)
    assert plain.randomization_pvalues is None


def test_recommend_n_components_q2_rule() -> None:
    """Q2-increment helper stops at the first non-meaningful component."""
    from process_improve.multivariate._common import (
        Q2_MIN_INCREMENT,
        _recommend_n_components_q2,
    )

    # Two systematic components, then a plateau and a drop: stop at 2.
    assert _recommend_n_components_q2([0.40, 0.62, 0.625, 0.50]) == 2
    # A single strong component then a fractional (noise-level) gain: stop at 1.
    assert _recommend_n_components_q2([0.55, 0.553]) == 1
    # Monotonically improving but each step below the threshold: still stop at 1.
    assert _recommend_n_components_q2([0.30, 0.305, 0.309, 0.312]) == 1
    # NaN stops the search.
    assert _recommend_n_components_q2([0.4, np.nan, 0.9]) == 1
    # min_increment=0 reproduces the permissive "any improvement" behaviour.
    assert _recommend_n_components_q2([0.30, 0.305, 0.309, 0.312], min_increment=0.0) == 4
    # A larger threshold is more conservative.
    assert _recommend_n_components_q2([0.40, 0.62, 0.70], min_increment=0.1) == 2
    assert Q2_MIN_INCREMENT == 0.01


def test_recommend_n_components_one_se_rule() -> None:
    """1-SE helper picks the smallest model within one SE of the best."""
    from process_improve.multivariate._common import _recommend_n_components_one_se

    # Argmin is at 3; SE at the argmin is 0.05; components 2 and 3 both lie at
    # or below 0.40 + 0.05 = 0.45, so the parsimonious 2 wins.
    mean = [0.60, 0.42, 0.40, 0.41, 0.40, 0.405]
    se = [0.04, 0.03, 0.05, 0.05, 0.05, 0.05]
    assert _recommend_n_components_one_se(mean, se) == 2

    # If only the argmin itself is within 1 SE, the 1-SE rule returns argmin.
    mean = [0.80, 0.70, 0.40, 0.55]
    se = [0.02, 0.02, 0.02, 0.02]
    assert _recommend_n_components_one_se(mean, se) == 3

    # A degenerate (zero) SE collapses to argmin.
    assert _recommend_n_components_one_se([0.5, 0.4, 0.3], [0.0, 0.0, 0.0]) == 3

    # NaN entries in mean_error are skipped, not allowed to win.
    assert _recommend_n_components_one_se([0.5, np.nan, 0.4, 0.41], [0.05, np.nan, 0.05, 0.05]) == 3

    # All-NaN input falls back to 1 (callers should have raised already).
    assert _recommend_n_components_one_se([np.nan, np.nan], [np.nan, np.nan]) == 1

    with pytest.raises(ValueError, match="same shape"):
        _recommend_n_components_one_se([0.4, 0.3], [0.05])

    # A negative standard error is floored at zero, so the rule falls back to the argmin.
    assert _recommend_n_components_one_se([0.5, 0.4, 0.3], [0.0, 0.0, -0.1]) == 3


@pytest.mark.parametrize("rule", ["min", "1se", "q2_increment"])
def test_selection_rules_refuse_a_curve_not_indexed_by_component_count(rule: str) -> None:
    """#513: every rule returns a position plus one, which is a component count only for a curve indexed 1..A.

    A Series indexed from 2, as a caller that left out one component would build it, came
    back as a count one too small, with no sign that anything was wrong. It is refused now,
    and the same values indexed 1..A give the answer they always did.
    """
    from process_improve.multivariate._common import _select_n_components

    curves = {
        "mean_error": [0.60, 0.42, 0.40, 0.41],
        "se_error": [0.03, 0.03, 0.05, 0.05],
        "q2_cumulative": [0.30, 0.62, 0.625, 0.50],
    }
    shifted = {name: pd.Series(values, index=[2, 3, 4, 5]) for name, values in curves.items()}
    with pytest.raises(ValueError, match=r"must be indexed by component count 1\.\.4; got \[2, 3, 4, 5\]\.$"):
        _select_n_components(rule, **shifted)

    indexed = {name: pd.Series(values, index=range(1, 5)) for name, values in curves.items()}
    assert _select_n_components(rule, **indexed) == _select_n_components(rule, **curves)


def test_select_n_components_dispatcher() -> None:
    """The selection-rule dispatcher routes to the right helper and validates inputs."""
    from process_improve.multivariate._common import _select_n_components

    mean = [0.60, 0.42, 0.40, 0.41]
    se = [0.03, 0.03, 0.05, 0.05]
    q2 = [0.30, 0.62, 0.625, 0.50]

    assert _select_n_components("min", mean_error=mean) == 3
    assert _select_n_components("1se", mean_error=mean, se_error=se) == 2
    assert _select_n_components("q2_increment", mean_error=mean, q2_cumulative=q2) == 2

    with pytest.raises(ValueError, match="standard errors"):
        _select_n_components("1se", mean_error=mean)
    with pytest.raises(ValueError, match="Q\\^2 curve"):
        _select_n_components("q2_increment", mean_error=mean)
    with pytest.raises(ValueError, match="Unknown selection_rule"):
        _select_n_components("not_a_rule", mean_error=mean)  # type: ignore[arg-type]


def test_min_rule_falls_back_to_one_component_without_a_finite_error() -> None:
    """With no finite error at any count, "min" recommends one component, as "1se" does."""
    from process_improve.multivariate._common import _select_n_components

    nothing = [np.nan, np.nan, np.nan]
    assert _select_n_components("min", mean_error=nothing) == 1
    assert _select_n_components("1se", mean_error=nothing, se_error=nothing) == 1


def test_equal_weight_r2_total_needs_one_row_per_component() -> None:
    """The per-target table is (components x targets); a flat vector is refused, naming its shape."""
    from process_improve.multivariate._common import _equal_weight_r2_total

    np.testing.assert_allclose(_equal_weight_r2_total(np.array([[0.2, 0.4], [0.5, np.nan]])), [0.3, 0.5])
    with pytest.raises(ValueError, match=r"^per_target must be 2-D \(A x M\); got shape \(3,\)\.$"):
        _equal_weight_r2_total(np.ones(3))


def test_not_enough_variance_error_is_typed_and_runtimeerror() -> None:
    """Rank overflow raises NotEnoughVarianceError, still catchable as RuntimeError."""
    # NotEnoughVarianceError subclasses RuntimeError so existing broad
    # ``except RuntimeError`` handlers keep working while callers can catch the
    # narrower type (e.g. to trim the component count during cross-validation).
    assert issubclass(NotEnoughVarianceError, RuntimeError)

    # PLS: rank-1 X (two identical columns) cannot support a 2nd component.
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
    Y = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(NotEnoughVarianceError):
        PLS(n_components=2).fit(X, Y)
    with pytest.raises(RuntimeError):  # the same error, caught broadly
        PLS(n_components=2).fit(X, Y)

    # PCA NIPALS path: rank-2 data (3rd column is a linear combination) has no
    # variance left for a 3rd component even though 3 <= min(N, K).
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 1.0, 4.0, 3.0, 6.0])
    X3 = pd.DataFrame({"a": a, "b": b, "c": a + 2.0 * b})
    with pytest.raises(NotEnoughVarianceError):
        PCA(n_components=3, algorithm="nipals").fit(X3)


def test_pls_cross_validate_jackknife() -> None:
    """Test PLS cross-validation with jackknife (leave-one-out)."""
    rng = np.random.default_rng(42)
    N, K, M = 30, 5, 2
    beta_true = np.array([[3.0, 0.0], [0.0, 2.5], [-2.0, 0.0], [0.0, 0.0], [1.5, -1.0]])
    X = pd.DataFrame(rng.standard_normal((N, K)), columns=[f"X{i}" for i in range(1, K + 1)])
    Y = pd.DataFrame(X.values @ beta_true + rng.standard_normal((N, M)) * 0.5, columns=["Y1", "Y2"])

    scaler_x = MCUVScaler().fit(X)
    scaler_y = MCUVScaler().fit(Y)
    X_s, Y_s = scaler_x.transform(X), scaler_y.transform(Y)

    pls = PLS(n_components=2).fit(X_s, Y_s)
    cv_result = pls.cross_validate(X_s, Y_s, cv="loo", show_progress=False)

    assert cv_result.method == "jackknife"
    assert cv_result.n_resamples == N
    assert cv_result.conf_level == 0.95
    assert cv_result.beta_samples.shape == (N, K, M)
    assert cv_result.beta_mean.shape == (K, M)
    assert cv_result.beta_std.shape == (K, M)
    assert cv_result.beta_ci_lower.shape == (K, M)
    assert cv_result.beta_ci_upper.shape == (K, M)
    assert cv_result.significant.shape == (K, M)

    # CI lower < mean < CI upper everywhere
    assert (cv_result.beta_ci_lower.values <= cv_result.beta_mean.values + 1e-10).all()
    assert (cv_result.beta_ci_upper.values >= cv_result.beta_mean.values - 1e-10).all()

    # Prediction metrics should be present for LOO
    assert cv_result.y_hat_cv is not None
    assert cv_result.y_hat_cv.shape == (N, M)
    assert cv_result.press is not None
    assert cv_result.press > 0
    assert cv_result.rmse_cv is not None
    assert len(cv_result.rmse_cv) == M
    assert cv_result.q_squared is not None
    assert len(cv_result.q_squared) == M
    # Q² should be positive for this well-conditioned problem
    assert (cv_result.q_squared.values > 0).all()


def test_pls_cross_validate_kfold() -> None:
    """Test PLS cross-validation with K-fold."""
    rng = np.random.default_rng(123)
    N, K = 40, 4
    X = pd.DataFrame(rng.standard_normal((N, K)), columns=[f"X{i}" for i in range(1, K + 1)])
    beta_true = np.array([[2.0], [-1.5], [0.0], [1.0]])
    Y = pd.DataFrame(X.values @ beta_true + rng.standard_normal((N, 1)) * 0.3, columns=["Y1"])

    scaler_x = MCUVScaler().fit(X)
    scaler_y = MCUVScaler().fit(Y)
    X_s, Y_s = scaler_x.transform(X), scaler_y.transform(Y)

    pls = PLS(n_components=2).fit(X_s, Y_s)
    cv_result = pls.cross_validate(X_s, Y_s, cv=5, random_state=42, show_progress=False)

    assert cv_result.method == "kfold"
    assert cv_result.n_resamples == 5
    assert cv_result.beta_samples.shape == (5, K, 1)
    assert cv_result.y_hat_cv is not None
    assert cv_result.y_hat_cv.shape == (N, 1)
    assert cv_result.q_squared is not None
    assert cv_result.q_squared.values[0] > 0


def test_pls_cross_validate_bootstrap() -> None:
    """Test PLS cross-validation with bootstrap."""
    rng = np.random.default_rng(99)
    N, K = 30, 3
    X = pd.DataFrame(rng.standard_normal((N, K)), columns=[f"X{i}" for i in range(1, K + 1)])
    Y = pd.DataFrame(X.values @ np.array([[1.0], [0.0], [-1.0]]) + rng.standard_normal((N, 1)) * 0.5, columns=["Y1"])

    scaler_x = MCUVScaler().fit(X)
    scaler_y = MCUVScaler().fit(Y)
    X_s, Y_s = scaler_x.transform(X), scaler_y.transform(Y)

    pls = PLS(n_components=2).fit(X_s, Y_s)
    cv_result = pls.cross_validate(X_s, Y_s, n_bootstrap=50, random_state=42, show_progress=False)

    assert cv_result.method == "bootstrap"
    assert cv_result.n_resamples == 50
    assert cv_result.beta_samples.shape == (50, K, 1)
    # Bootstrap uses percentile CI
    assert (cv_result.beta_ci_lower.values <= cv_result.beta_ci_upper.values).all()
    # No cross-validated predictions for bootstrap
    assert cv_result.y_hat_cv is None
    assert cv_result.press is None
    assert cv_result.q_squared is None


def test_pls_cross_validate_significance() -> None:
    """Test that significant betas are correctly identified with strong signal."""
    rng = np.random.default_rng(7)
    N, K = 50, 4
    # Strong signal for X1 and X3, zero for X2 and X4
    beta_true = np.array([[5.0], [0.0], [-5.0], [0.0]])
    X = pd.DataFrame(rng.standard_normal((N, K)), columns=[f"X{i}" for i in range(1, K + 1)])
    Y = pd.DataFrame(X.values @ beta_true + rng.standard_normal((N, 1)) * 0.1, columns=["Y1"])

    scaler_x = MCUVScaler().fit(X)
    scaler_y = MCUVScaler().fit(Y)
    X_s, Y_s = scaler_x.transform(X), scaler_y.transform(Y)

    pls = PLS(n_components=2).fit(X_s, Y_s)
    cv_result = pls.cross_validate(X_s, Y_s, cv=5, random_state=0, show_progress=False)

    # With very strong signal and low noise, the significant variables (X1, X3) should be flagged
    sig = cv_result.significant
    assert sig.loc["X1", "Y1"]  # strong positive
    assert sig.loc["X3", "Y1"]  # strong negative


def test_pls_cross_validate_not_fitted() -> None:
    """Test that cross_validate raises error on unfitted model."""
    pls = PLS(n_components=2)
    X = pd.DataFrame(np.random.default_rng(0).standard_normal((20, 3)))
    Y = pd.DataFrame(np.random.default_rng(0).standard_normal((20, 1)))
    with pytest.raises(NotFittedError):
        pls.cross_validate(X, Y)


def test_pls_old_attribute_names_raise() -> None:
    """Test that old PLS attribute names raise helpful errors."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((30, 4)), columns=[f"X{i}" for i in range(1, 5)])
    Y = pd.DataFrame(X.values @ rng.standard_normal((4, 1)) + rng.standard_normal((30, 1)) * 0.3, columns=["Y1"])
    plsmodel = PLS(n_components=2)
    plsmodel.fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))

    with pytest.raises(AttributeError, match="renamed"):
        _ = plsmodel.x_scores
    with pytest.raises(AttributeError, match="renamed"):
        _ = plsmodel.squared_prediction_error
    with pytest.raises(AttributeError, match="renamed"):
        _ = plsmodel.R2


# ---- TPLS models ----
@pytest.fixture
def fixture_tpls_example() -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load example data for TPLS model.

    Data from: https://github.com/salvadorgarciamunoz/pyphi/tree/master/examples/JRPLS%20and%20TPLS
    """
    properties = {
        "Group 1": pd.read_csv(  # 162 x 7
            "src/process_improve/datasets/multivariate/tpls-pyphi/properties_Group1.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 2": pd.read_csv(  # 9 x 6
            "src/process_improve/datasets/multivariate/tpls-pyphi/properties_Group2.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 3": pd.read_csv(  # 22 x 6
            "src/process_improve/datasets/multivariate/tpls-pyphi/properties_Group3.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 4": pd.read_csv(  # 19 x 11
            "src/process_improve/datasets/multivariate/tpls-pyphi/properties_Group4.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 5": pd.read_csv(  # 18 x 3
            "src/process_improve/datasets/multivariate/tpls-pyphi/properties_Group5.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
    }
    formulas = {
        "Group 1": pd.read_csv(  # 105 x 162
            "src/process_improve/datasets/multivariate/tpls-pyphi/formulas_Group1.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 2": pd.read_csv(  # 105 x 9
            "src/process_improve/datasets/multivariate/tpls-pyphi/formulas_Group2.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 3": pd.read_csv(  # 105 x 22
            "src/process_improve/datasets/multivariate/tpls-pyphi/formulas_Group3.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 4": pd.read_csv(  # 105 x 19
            "src/process_improve/datasets/multivariate/tpls-pyphi/formulas_Group4.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 5": pd.read_csv(  # 105 x 18
            "src/process_improve/datasets/multivariate/tpls-pyphi/formulas_Group5.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
    }
    process_conditions: dict[str, pd.DataFrame] = {
        "Conditions": pd.read_csv(  # 105 x 10
            "src/process_improve/datasets/multivariate/tpls-pyphi/process_conditions.csv",
            sep=",",
            index_col=0,
            header=0,
        ).astype("float64")
    }

    quality_indicators: dict[str, pd.DataFrame] = {
        "Quality": pd.read_csv(  # 105 x 6
            "src/process_improve/datasets/multivariate/tpls-pyphi/quality_indicators.csv",
            sep=",",
            index_col=0,
            header=0,
        ).astype("float64")
    }
    return {
        "Z": process_conditions,
        "D": properties,
        "F": formulas,
        "Y": quality_indicators,
    }


def test_tpls_pickle_round_trips(fixture_tpls_example: dict) -> None:
    """ENG-05 (#287): a fitted TPLS model pickles and its hotellings_t2_limit method survives."""
    import pickle

    d_matrix = fixture_tpls_example.pop("D")
    tpls_test = TPLS(n_components=3, d_matrix=d_matrix).fit(DataFrameDict(fixture_tpls_example))
    reloaded = pickle.loads(pickle.dumps(tpls_test))  # noqa: S301 - trusted round-trip of our own object
    assert float(reloaded.hotellings_t2_limit()) == float(tpls_test.hotellings_t2_limit())
    # The spe_limit dict-of-partials API also survives the round-trip.
    block, group = "Y", next(iter(reloaded.spe_limit["Y"]))
    assert float(reloaded.spe_limit[block][group](0.95)) == float(tpls_test.spe_limit[block][group](0.95))


def test_tpls_model_fitting(fixture_tpls_example: dict) -> None:
    """
    Test the fitting process of the TPLS model to ensure it functions as expected.

    Tests here include tests for pre-processing and model fitting.
    """

    n_components = 3
    d_matrix = fixture_tpls_example.pop("D")
    tpls_test = TPLS(n_components=n_components, d_matrix=d_matrix)
    tpls_test.fit(DataFrameDict(fixture_tpls_example))

    # Test the coefficients in the centering and scaling self.preproc_ structure. Group 1, for D matrix pre-processing:
    known_truth_d1m = np.array([99.85432099, 73.67901235, 3.07469136, 0.13950617, 2.09876543, 12.53703704, 41.58641975])
    known_truth_d1s = np.array([0.36883209, 1.80631852, 0.69231018, 0.09610943, 0.29927192, 1.99109474, 4.82828759])
    assert pytest.approx(tpls_test.preproc_["D"]["Group 1"]["center"]) == known_truth_d1m
    assert pytest.approx(tpls_test.preproc_["D"]["Group 1"]["scale"]) == known_truth_d1s
    assert pytest.approx(tpls_test.preproc_["D"]["Group 1"]["center"]) == d_matrix["Group 1"].mean()
    assert pytest.approx(tpls_test.preproc_["D"]["Group 1"]["scale"]) == d_matrix["Group 1"].std()

    # Group 4 in D has missing values pre-processing. Test these.
    known_truth_d4m = np.array(
        [
            9.99055556e-01,
            1.15555556e-01,
            2.69444444e-01,
            2.14210526e01,
            1.07000000e01,
            2.04894737e02,
            5.82947368e01,
            9.86000000e01,
            4.88947368e00,
            3.43684211e00,
            7.03684211e00,
        ]
    )
    known_truth_d4s = np.array(
        [
            8.02365783e-04,
            7.04792186e-03,
            1.73110717e-02,
            3.64105890e00,
            2.11213425e00,
            2.18313579e00,
            6.27211409e00,
            1.84149698e00,
            6.57836255e-02,
            1.77045275e-01,
            3.68496233e-01,
        ]
    )
    assert pytest.approx(tpls_test.preproc_["D"]["Group 4"]["center"]) == known_truth_d4m
    assert pytest.approx(tpls_test.preproc_["D"]["Group 4"]["scale"]) == known_truth_d4s
    assert pytest.approx(tpls_test.preproc_["D"]["Group 4"]["center"]) == d_matrix["Group 4"].mean()
    assert pytest.approx(tpls_test.preproc_["D"]["Group 4"]["scale"]) == d_matrix["Group 4"].std()

    # Test the formula block, group 2 pre-processing
    known_truth_f2m = np.array(
        [0.13333333, 0.0020127, 0.01904762, 0.00952381, 0.13282593, 0.1889709, 0.1047619, 0.17035596, 0.23916785]
    )
    known_truth_f2s = np.array(
        [0.34156503, 0.02062402, 0.13734798, 0.09759001, 0.33676188, 0.39173332, 0.3077152, 0.37647422, 0.4274989]
    )
    assert pytest.approx(tpls_test.preproc_["F"]["Group 2"]["center"]) == known_truth_f2m
    assert pytest.approx(tpls_test.preproc_["F"]["Group 2"]["scale"]) == known_truth_f2s
    assert pytest.approx(tpls_test.preproc_["F"]["Group 2"]["center"]) == fixture_tpls_example["F"]["Group 2"].mean()
    assert pytest.approx(tpls_test.preproc_["F"]["Group 2"]["scale"]) == fixture_tpls_example["F"]["Group 2"].std()

    # Test the `Conditions` (Z) block:
    known_truth_zm = np.array(
        [
            8.63809524e00,
            2.27764762e01,
            7.21470000e01,
            2.13978571e01,
            7.17852381e01,
            2.06402381e01,
            7.10553333e01,
            1.35438095e05,
            8.19047619e-01,
            1.80952381e-01,
        ]
    )
    known_truth_zs = np.array(
        [
            3.56835399e00,
            6.95963475e00,
            6.53039712e-01,
            6.27483262e00,
            1.29077224e00,
            3.59658120e00,
            1.45728018e00,
            1.89215897e03,
            3.86825154e-01,
            3.86825154e-01,
        ]
    )
    assert pytest.approx(tpls_test.preproc_["Z"]["Conditions"]["center"]) == known_truth_zm
    assert pytest.approx(tpls_test.preproc_["Z"]["Conditions"]["scale"]) == known_truth_zs
    assert (
        pytest.approx(tpls_test.preproc_["Z"]["Conditions"]["center"]) == fixture_tpls_example["Z"]["Conditions"].mean()
    )
    assert (
        pytest.approx(tpls_test.preproc_["Z"]["Conditions"]["scale"]) == fixture_tpls_example["Z"]["Conditions"].std()
    )

    # Test the `Quality` (Y) block pre-processing
    known_truth_ym = np.array([30.96834605, 3.328312, 57.01620571, 3.6485004, 79.34224822, 3.06157598])
    known_truth_ys = np.array([3.46989807, 0.96879553, 4.9740008, 1.47585478, 3.8676095, 1.18600054])
    assert pytest.approx(tpls_test.preproc_["Y"]["Quality"]["center"]) == known_truth_ym
    assert pytest.approx(tpls_test.preproc_["Y"]["Quality"]["scale"]) == known_truth_ys
    assert pytest.approx(tpls_test.preproc_["Y"]["Quality"]["center"]) == fixture_tpls_example["Y"]["Quality"].mean()
    assert pytest.approx(tpls_test.preproc_["Y"]["Quality"]["scale"]) == fixture_tpls_example["Y"]["Quality"].std()

    # Test various R2 values for columns in blocks
    # For the last component, for the Z block:
    assert pytest.approx(tpls_test.r2_frac[-1]["Z"]["Conditions"].round(3)) == np.array(
        [0.414, 0.001, 0.004, 0.259, 0.05, 0.0, 0.028, 0.001, 0.014, 0.014]
    ).astype(np.float64)
    # For the D-block
    assert pytest.approx(tpls_test.r2_frac[-1]["D"]["Group 1"].round(3)) == np.array(
        [0.296, 0.051, 0.432, 0.0, 0.015, 0.131, 0.128]
    ).astype(np.float64)
    # For the F-block
    assert pytest.approx(tpls_test.r2_frac[-1]["F"]["Group 2"].round(3)) == np.array(
        [0.046, 0.0, 0.01, 0.0, 0.048, 0.026, 0.0, 0.003, 0.052]
    ).astype(np.float64)
    # For the Y-block after 1 component, 2 components, and 3 components:
    assert pytest.approx(tpls_test.r2_frac[1]["Y"]["Quality"].round(3)) == np.array(
        [0.005, 0.078, 0.293, 0.329, 0.337, 0.153]
    ).astype(np.float64)
    assert pytest.approx(tpls_test.r2_frac[2]["Y"]["Quality"].round(3)) == np.array(
        [0.135, 0.0, 0.039, 0.023, 0.007, 0.052]
    ).astype(np.float64)
    assert pytest.approx(tpls_test.r2_frac[-1]["Y"]["Quality"].round(3)) == np.array(
        [0.017, 0.074, 0.018, 0.0, 0.005, 0.0]
    ).astype(np.float64)


def test_tpls_model_plots(fixture_tpls_example: dict) -> None:
    """Test the plotting functionality of the TPLS model."""

    n_components = 3
    tpls_test = TPLS(n_components=n_components, d_matrix=fixture_tpls_example.pop("D"))
    tpls_test.fit(DataFrameDict(fixture_tpls_example))

    fig = tpls_test.plot.scores()
    assert isinstance(fig, go.Figure)

    # One marker trace carrying every observation, plus the 95% Hotelling's T2 ellipse.
    assert [trace.name for trace in fig.data] == ["Scores [T]", "Hotelling's T^2 [95%]"]
    scores_trace, ellipse_trace = fig.data
    assert scores_trace.mode == "markers"
    assert len(scores_trace.x) == tpls_test.n_samples == 105
    assert ellipse_trace.mode == "lines"

    # The ellipse is a closed curve centred on the origin, not a degenerate line.
    ellipse_x, ellipse_y = np.asarray(ellipse_trace.x), np.asarray(ellipse_trace.y)
    assert len(ellipse_x) >= 50
    assert ellipse_x.min() < 0 < ellipse_x.max()
    assert ellipse_y.min() < 0 < ellipse_y.max()
    # Centred on the origin: the extents are symmetric. The x pair is symmetric only to
    # ~0.05% of its range because 100 angular samples do not land exactly on the extremum.
    assert ellipse_x.max() == pytest.approx(-ellipse_x.min(), rel=1e-3)
    assert ellipse_y.max() == pytest.approx(-ellipse_y.min(), rel=1e-3)
    # Closed curve: it returns to its starting point.
    assert (ellipse_x[0], ellipse_y[0]) == pytest.approx((ellipse_x[-1], ellipse_y[-1]))

    assert fig.layout.xaxis.title.text == "PC 1"
    assert fig.layout.yaxis.title.text == "PC 2"
    # Two crosshair lines through the origin.
    assert [shape.type for shape in fig.layout.shapes] == ["line", "line"]

    # The loadings plot reaches the super-level weights through `_parent`, the same way
    # the score plot reaches `t_scores_super`. Before #564 the fallback resolved to
    # `Plot.loadings` itself and raised AttributeError on a bound method.
    loadings_fig = tpls_test.plot.loadings()
    assert isinstance(loadings_fig, go.Figure)
    # One point per super block (Z and F), against components 1 and 2.
    assert tpls_test.w_loadings_super.shape == (2, n_components)
    loadings_trace = loadings_fig.data[0]
    assert len(loadings_trace.x) == 2
    assert loadings_trace.x == pytest.approx(tpls_test.w_loadings_super.loc[:, 1].to_numpy())
    assert loadings_trace.y == pytest.approx(tpls_test.w_loadings_super.loc[:, 2].to_numpy())


def test_tpls_model_predictions(fixture_tpls_example: dict) -> None:  # noqa: PLR0915
    """Test the prediction process of the TPLS model to ensure it functions as expected."""
    n_components = 3
    d_matrix = fixture_tpls_example.pop("D")
    tpls_test = TPLS(n_components=n_components, d_matrix=d_matrix)
    tpls_test.fit(DataFrameDict(fixture_tpls_example))

    # Test the model's predictions. Use the first few samples of the data as a testing data point.
    testing_samples = ["L001", "L002", "L003", "L004"]
    new_observation_raw = {
        "Z": {"Conditions": fixture_tpls_example["Z"]["Conditions"].loc[testing_samples]},
        "F": {key: val.loc[testing_samples] for key, val in fixture_tpls_example["F"].items()},
    }

    # OK, now use these to make predictions
    predictions = tpls_test.diagnose(DataFrameDict(new_observation_raw))
    expected_keys = ["hat", "t_scores_super", "hotellings_t2", "spe"]
    assert set(expected_keys) == set(predictions.keys())

    # Test the predicted values are what were expected:
    assert pytest.approx(predictions["hat"]["Quality"].iloc[0, :]) == np.array(
        [33.08751115, 3.15312493, 58.74578748, 3.2956403, 79.89017742, 2.66773621]
    )
    # Compare the predictions to what is stored in the training data
    assert pytest.approx(predictions["hat"]["Quality"].iloc[0, :]) == np.array(
        tpls_test.hat["Quality"].iloc[0, :].values
    )

    # Test that the SPE_z values:
    assert pytest.approx([predictions["spe"]["Z"][key].values[0, -1] for key in predictions["spe"]["Z"]]) == [
        1.87556907
    ]

    # Test that the SPE_f values are correct:
    assert pytest.approx([predictions["spe"]["F"][key].values[0, -1] for key in predictions["spe"]["F"]]) == [
        12.94216819,
        6.93294692,
        5.56820559,
        5.59365285,
        5.54739917,
    ]
    # Check that the Hotelling's T2 matches what would have been calculated by the model:
    # fit() and diagnose() now share the same cumulative per-component form (#502), so the
    # full (n_obs, n_components) frames agree, not just the last column.
    assert (
        pytest.approx(tpls_test.hotellings_t2.loc[testing_samples].to_numpy())
        == predictions["hotellings_t2"].to_numpy()
    )

    # from sklearn.model_selection import cross_val_score

    # scores = cross_val_score(tpls_test, transformed_data, None, cv=5)
    # Ensure model is fitted appropriately, with the expected number of iterations
    assert tpls_test.fitting_statistics["iterations"] == [11, 9, 27]
    assert all(tol < epsqrt for tol in tpls_test.fitting_statistics["convergance_tolerance"])

    # Model parameters tested. ``hotellings_t2`` is now the cumulative per-component form
    # with the unbiased (N - 1) score-variance divisor (#502), shape (N, n_components); the
    # full-model T2 is the last column. The training super-scores are orthogonal (largest
    # normalized off-diagonal cross-product ~1e-17), so the previous full-Mahalanobis pins
    # [2.49885911, 2.961537, 2.88846366, 4.58151729, 5.06621405] equal the diagonal form
    # with the N divisor: the new values are exactly the old ones times (N-1)/N = 104/105.
    assert pytest.approx(tpls_test.hotellings_t2.iloc[0:5, -1].values.ravel()) == np.array(
        [
            2.47506046,
            2.93333189,
            2.86095448,
            4.53788379,
            5.01796439,
        ]
    )

    # Model limits tested
    assert tpls_test.hotellings_t2_limit(0.95) == pytest.approx(8.318089340, rel=1e-6)
    assert tpls_test.hotellings_t2_limit(0.99) == pytest.approx(12.288844, rel=1e-6)

    assert np.square(tpls_test.spe["Y"]["Quality"].iloc[0:4].values) == pytest.approx(
        [5.60356584, 2.82414766, 1.69358176, 3.4829089], rel=1e-8
    )
    # Test the last 4 observations
    assert np.square(tpls_test.spe["Z"]["Conditions"].iloc[-4:].values) == pytest.approx(
        [2.79253124, 2.00872637, 10.80292179, 3.26432946], rel=1e-8
    )
    assert np.square(tpls_test.spe["F"]["Group 1"].iloc[0:4].values) == pytest.approx(
        [167.49971736, 132.25813164, 201.56460235, 198.02742727], rel=1e-8
    )
    assert np.square(tpls_test.spe["F"]["Group 2"].iloc[0:4].values) == pytest.approx(
        [48.065753, 100.17604529, 47.80010992, 1.71302689], rel=1e-8
    )
    assert np.square(tpls_test.spe["F"]["Group 3"].iloc[0:4].values) == pytest.approx(
        [31.00491349, 31.47060518, 30.82518225, 4.42512332], rel=1e-8
    )
    assert np.square(tpls_test.spe["F"]["Group 4"].iloc[0:4].values) == pytest.approx(
        [31.28895219, 31.85278369, 31.07114402, 28.82708874], rel=1e-8
    )
    assert np.square(tpls_test.spe["F"]["Group 5"].iloc[0:4].values) == pytest.approx(
        [30.77363754, 31.61877076, 30.53696788, 97.88805239], rel=1e-8
    )
    assert np.square(tpls_test.spe["D"]["Group 3"].iloc[0:4].values) == pytest.approx(
        [0.015504765277819974, 0.002021083231104291, 0.04795499280275304, 0.0024131901105140696], rel=1e-7
    )
    # Is this is all zero because there are only 3 columns of data, and we fit 3 components?
    assert np.square(tpls_test.spe["D"]["Group 5"].iloc[0:4].values) == pytest.approx([0, 0, 0, 0], rel=1e-8)

    # Test case uses a different method to calculate the chi2 value, so we use a different tolerance
    assert tpls_test.spe_limit["Y"]["Quality"](0.95) == pytest.approx(3.70131535, rel=1e-8)
    assert tpls_test.spe_limit["Y"]["Quality"](0.99) == pytest.approx(4.62900976, rel=1e-8)
    assert tpls_test.spe_limit["D"]["Group 1"](0.95) == pytest.approx(0.08789854, rel=1e-7)
    assert tpls_test.spe_limit["D"]["Group 2"](0.95) == pytest.approx(0.22412007, rel=1e-7)
    assert tpls_test.spe_limit["D"]["Group 3"](0.95) == pytest.approx(0.19404685, rel=1e-7)
    assert tpls_test.spe_limit["D"]["Group 4"](0.95) == pytest.approx(0.24915518, rel=1e-7)
    assert tpls_test.spe_limit["D"]["Group 5"](0.95) == pytest.approx(0, rel=1e-8)
    assert tpls_test.spe_limit["F"]["Group 1"](0.99) == pytest.approx(16.17958502, rel=1e-8)
    assert tpls_test.spe_limit["F"]["Group 2"](0.99) == pytest.approx(8.47783716, rel=1e-8)
    assert tpls_test.spe_limit["F"]["Group 3"](0.99) == pytest.approx(9.11819153, rel=1e-8)
    assert tpls_test.spe_limit["F"]["Group 4"](0.99) == pytest.approx(8.77664292, rel=1e-8)
    assert tpls_test.spe_limit["F"]["Group 5"](0.99) == pytest.approx(7.84401832, rel=1e-8)

    # Test building the model without the "Z" block.
    fixture_tpls_example.pop("Z")
    n_components = 3
    tpls_test_no_z = TPLS(n_components=n_components, d_matrix=d_matrix).fit(DataFrameDict(fixture_tpls_example))
    # Test the model's predictions. Use the first few samples of the data as a testing data point.
    testing_samples = ["L001", "L002", "L003", "L004"]
    new_observation_raw_no_z = DataFrameDict(
        {
            "F": {key: val.loc[testing_samples] for key, val in fixture_tpls_example["F"].items()},
        }
    )

    # OK, now use these to make predictions
    predictions = tpls_test_no_z.diagnose(new_observation_raw_no_z)
    expected_keys = ["hat", "t_scores_super", "hotellings_t2", "spe"]
    assert set(expected_keys) == set(predictions.keys())
    assert predictions["spe"]["Z"] == {}
    assert predictions["spe"]["F"] is not None
    assert predictions.spe is not None  # test the `Bunch` functionality


def test_tpls_fit_and_diagnose_t2_agree(fixture_tpls_example: dict) -> None:
    """#502: ``fit()`` and ``diagnose()`` compute Hotelling's T2 by the same
    cumulative per-component formula with the unbiased (N - 1) score-variance
    divisor, so the training rows give the same values, with the same
    (n_obs, n_components) shape, from both entry points.
    """
    d_matrix = fixture_tpls_example.pop("D")
    model = TPLS(n_components=3, d_matrix=d_matrix).fit(DataFrameDict(fixture_tpls_example))
    result = model.diagnose(DataFrameDict({"Z": fixture_tpls_example["Z"], "F": fixture_tpls_example["F"]}))
    n_obs = model.t_scores_super.shape[0]
    assert model.hotellings_t2.shape == (n_obs, 3)
    assert result.hotellings_t2.shape == model.hotellings_t2.shape
    np.testing.assert_allclose(result.hotellings_t2.to_numpy(), model.hotellings_t2.to_numpy(), atol=1e-10)

    # The score standard deviations feed ellipse_coordinates; pin the unbiased
    # (N - 1) computation against a direct calculation on the super-scores.
    expected_sd = np.sqrt(np.diag(model.t_scores_super.to_numpy().T @ model.t_scores_super.to_numpy()) / (n_obs - 1))
    assert model.scaling_factor_for_scores.to_numpy() == pytest.approx(expected_sd, rel=1e-12)


@pytest.mark.slow
def test_tpls_cross_validation(fixture_tpls_example: dict) -> None:
    """Test the prediction process of the TPLS model to ensure it functions as expected."""
    n_components = 3
    d_matrix = fixture_tpls_example.pop("D")
    blocks = DataFrameDict(fixture_tpls_example)

    # Default scoring routes through TPLS.score(), which needs no separate `y` because
    # the response block travels inside the DataFrameDict.
    scores = cross_val_score(
        estimator=TPLS(n_components=n_components, d_matrix=d_matrix),
        X=blocks,
        cv=5,
        n_jobs=1,
    )
    scores = np.asarray(scores)
    assert scores.shape == (5,)
    assert np.isfinite(scores).all(), f"cross_val_score produced non-finite folds: {scores}"
    # R2 is bounded above by 1; it may well be negative on held-out folds for this
    # small T-shaped block, and here it is, so do not assert a positive floor.
    assert (scores <= 1.0).all()

    # Deterministic: KFold without shuffling gives the same folds, so the same scores.
    repeat = np.asarray(
        cross_val_score(
            estimator=TPLS(n_components=n_components, d_matrix=d_matrix),
            X=blocks,
            cv=5,
            n_jobs=1,
        )
    )
    assert repeat == pytest.approx(scores, rel=1e-12)

    # #565: a named metric now goes through `make_tpls_scorer`, which sklearn accepts as a
    # callable `scoring=` and calls as `scorer(estimator, X_test)` -- the same two-argument
    # call that breaks the string form. `make_tpls_scorer("r2")` computes exactly what
    # `TPLS.score` computes, so it must reproduce the default-scoring folds bit for bit.
    r2_scores = np.asarray(
        cross_val_score(
            estimator=TPLS(n_components=n_components, d_matrix=d_matrix),
            X=blocks,
            cv=5,
            scoring=make_tpls_scorer("r2"),
            n_jobs=1,
        )
    )
    assert np.isfinite(r2_scores).all(), f"make_tpls_scorer produced non-finite folds: {r2_scores}"
    assert r2_scores == pytest.approx(scores, rel=1e-12)


@pytest.mark.slow
def test_make_tpls_scorer_named_metrics(fixture_tpls_example: dict) -> None:
    """#565: every named metric scores a fitted TPLS model and keeps sklearn's sign convention."""
    d_matrix = fixture_tpls_example.pop("D")
    blocks = DataFrameDict(fixture_tpls_example)
    model = TPLS(n_components=3, d_matrix=d_matrix).fit(blocks)

    # "r2" is the default and reproduces TPLS.score() exactly.
    assert make_tpls_scorer()(model, blocks) == pytest.approx(model.score(blocks), rel=1e-12)

    # The neg_* metrics are losses, so a fitted model scores at most 0.0.
    neg_mse = make_tpls_scorer("neg_mean_squared_error")(model, blocks)
    neg_rmse = make_tpls_scorer("neg_root_mean_squared_error")(model, blocks)
    neg_mae = make_tpls_scorer("neg_mean_absolute_error")(model, blocks)
    assert neg_mse <= 0.0
    assert neg_mae <= 0.0
    assert neg_rmse <= 0.0
    # sklearn averages the six quality columns *after* taking each square root, so the RMSE
    # is a mean of square roots while sqrt(MSE) is the square root of a mean. Jensen's
    # inequality makes the first no larger than the second, and it is strictly smaller here
    # because the columns do not share one error variance.
    assert -neg_rmse < np.sqrt(-neg_mse)


@pytest.mark.slow
def test_make_tpls_scorer_accepts_a_callable_metric(fixture_tpls_example: dict) -> None:
    """#565: a callable metric is honoured, and `greater_is_better=False` flips its sign."""
    d_matrix = fixture_tpls_example.pop("D")
    blocks = DataFrameDict(fixture_tpls_example)
    model = TPLS(n_components=3, d_matrix=d_matrix).fit(blocks)

    def max_abs_error(y_true: pd.DataFrame, y_pred: pd.DataFrame, sample_weight: None = None) -> float:
        return float(np.abs(np.asarray(y_true) - np.asarray(y_pred)).max())

    loss = make_tpls_scorer(max_abs_error, greater_is_better=False)(model, blocks)
    raw = make_tpls_scorer(max_abs_error)(model, blocks)
    assert loss < 0.0
    assert loss == pytest.approx(-raw, rel=1e-12)

    # metric_kwargs reach the metric: r2_score's `force_finite` is forwarded verbatim.
    assert make_tpls_scorer("r2", force_finite=False)(model, blocks) == pytest.approx(model.score(blocks), rel=1e-12)


def test_make_tpls_scorer_rejects_an_unknown_metric() -> None:
    """#565: an unknown metric name fails at build time, naming the alternatives."""
    with pytest.raises(ValueError, match=r"Unknown metric 'accuracy'.*neg_mean_squared_error"):
        make_tpls_scorer("accuracy")


@pytest.mark.slow
def test_tpls_rejects_a_separate_y(fixture_tpls_example: dict) -> None:
    """#565: `y` is no longer silently ignored by fit / score / the scorer."""
    d_matrix = fixture_tpls_example.pop("D")
    blocks = DataFrameDict(fixture_tpls_example)
    y_elsewhere = blocks["Y"]["Quality"]

    expected = r'does not take a separate `y`.*X\["Y"\].*make_tpls_scorer'
    with pytest.raises(ValueError, match=expected):
        TPLS(n_components=2, d_matrix=d_matrix).fit(blocks, y_elsewhere)

    model = TPLS(n_components=2, d_matrix=d_matrix).fit(blocks)
    with pytest.raises(ValueError, match=expected):
        model.score(blocks, y_elsewhere)
    with pytest.raises(ValueError, match=expected):
        make_tpls_scorer()(model, blocks, y_elsewhere)


def test_tpls_constant_column_warns_instead_of_asserting() -> None:
    """#513: a constant column is excluded from the fit, and says so.

    `_learn_center_and_scaling_parameters` gives a no-variance column a NaN scale, which
    makes the whole column NaN and then zero: the column is excluded, which is the right
    thing to do. What was missing was any word of it. The post-preprocessing check on the
    Z and Y blocks did not tolerate the NaN statistic that exclusion produces (the F and D
    checks did), so a single constant column raised a message-less `AssertionError()`
    under normal Python and, because the check was a bare `assert`, fitted silently under
    `python -O`. Two behaviours for one input.
    """
    rng = np.random.default_rng(0)
    n_blends, n_materials = 20, 4
    index = [f"blend{i}" for i in range(n_blends)]
    materials = [f"mat{j}" for j in range(n_materials)]
    d_matrix = {"G": pd.DataFrame(rng.normal(size=(n_materials, 3)), index=materials, columns=list("pqr"))}
    blocks = DataFrameDict(
        {
            "F": {"G": pd.DataFrame(rng.random((n_blends, n_materials)), index=index, columns=materials)},
            "Z": {"Z1": pd.DataFrame({"moves": rng.normal(size=n_blends), "stuck": np.ones(n_blends)}, index=index)},
            "Y": {"Y1": pd.DataFrame({"y": rng.normal(size=n_blends)}, index=index)},
        }
    )

    with pytest.warns(UserWarning, match=r"Block Z\[\"Z1\"\] has columns with no variance.*'stuck'"):
        model = TPLS(n_components=2, d_matrix=d_matrix).fit(blocks)

    # The fit completes, and the excluded column leaves no NaN behind in the scores.
    assert np.isfinite(model.t_scores_super.to_numpy()).all()
    assert np.isfinite(model.z_mats["Z1"]).all()


def test_tpls_preprocessing_invariant_names_what_failed() -> None:
    """#513: the invariant check raises with a message, and survives `python -O`.

    It was a bare `assert`, so a violation gave `AssertionError()` with nothing in it,
    and `-O` removed the check altogether. The repo runs a `-O` CI job precisely to catch
    that, so this is a `RuntimeError` now.
    """
    # Columns 0 and 2 are fine; column 1 is off-centre by 0.5, and must be the one named.
    observed = np.array([0.0, 0.5, 1e-9])
    with pytest.raises(RuntimeError, match=r"block 'Z' group 'Z1' is not centred.*Column 1 has 0.5"):
        TPLS._check_preprocessed(observed, 0.0, "Z", "Z1", "centre")

    # A NaN entry is an excluded column, and a zero standard deviation is the same case
    # seen after scaling; neither is a failure.
    TPLS._check_preprocessed(np.array([0.0, np.nan]), 0.0, "Z", "Z1", "centre")
    TPLS._check_preprocessed(np.array([1.0, np.nan, 0.0]), 1.0, "Y", "Y1", "scale")


def test_tpls_score_single_block_y(fixture_tpls_example: dict) -> None:
    """`TPLS.score` on a single-block `Y` returns a finite float."""
    n_components = 3
    d_matrix = fixture_tpls_example.pop("D")
    tpls_test = TPLS(n_components=n_components, d_matrix=d_matrix)
    tpls_test.fit(DataFrameDict(fixture_tpls_example))

    # The fixture's Y block has a single block ("Quality"); scoring on the training data
    # should return a finite float.
    score = tpls_test.score(DataFrameDict(fixture_tpls_example))
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_tpls_score_empty_y_raises(fixture_tpls_example: dict) -> None:
    """`TPLS.score` on an empty `Y` dict raises a clear `ValueError` (not `NameError`)."""
    n_components = 3
    d_matrix = fixture_tpls_example.pop("D")
    tpls_test = TPLS(n_components=n_components, d_matrix=d_matrix)
    tpls_test.fit(DataFrameDict(fixture_tpls_example))

    # Build a test bundle whose "Y" block is an empty dict; the loop body never runs.
    malformed = DataFrameDict({**fixture_tpls_example, "Y": {}})
    with pytest.raises(ValueError, match="y_actual is empty"):
        tpls_test.score(malformed)


def manual_cross_validation(tpls_model: TPLS, full_datadict: dict, cv: int = 5, scoring: str = "r2") -> np.ndarray:
    """Perform manual cross-validation for a TPLS model."""
    kfold = KFold(n_splits=cv, shuffle=True)
    n_samples = len(full_datadict)
    scores = []

    for train_idx, test_idx in kfold.split(range(n_samples)):
        training_datadict = {}
        testing_datadict = {}

        training_datadict = full_datadict[train_idx]
        testing_datadict = full_datadict[test_idx]

        # Fit and predict
        tpls_model.fit(training_datadict)
        tpls_model.display_results()
        inference = tpls_model.diagnose(testing_datadict)

        # Calculate score
        if scoring == "r2":
            score = r2_score(testing_datadict["Y"]["Quality"], inference.hat["Quality"])
        elif scoring == "mse":
            score = -mean_squared_error(testing_datadict["Y"]["Quality"], inference.hat["Quality"])
        else:
            score = tpls_model.score(testing_datadict, inference.hat["Quality"])

        scores.append(score)

    return np.array(scores)


# ---- Additional coverage tests for multivariate/methods.py ----


def test_pca_diagnose_new_data() -> None:
    """PCA.diagnose() should return scores, T2, and SPE for new observations."""
    rng = np.random.default_rng(42)
    N, K = 100, 5
    X = pd.DataFrame(rng.standard_normal((N, K)), columns=[f"V{i}" for i in range(K)])
    scaler = MCUVScaler()
    X_scaled = scaler.fit_transform(X)

    model = PCA(n_components=2)
    model.fit(X_scaled)

    # Diagnose on a subset of the data
    X_new = scaler.transform(X.iloc[:10])
    result = model.diagnose(X_new)

    assert isinstance(result, Bunch)
    assert result.scores.shape == (10, 2)
    assert result.hotellings_t2.shape == (10, 2)
    assert len(result.spe) == 10
    assert (result.spe >= 0).all()


def test_pls_diagnose_spe_matches_fitted_with_named_columns() -> None:
    """PLS.diagnose(X).spe must equal the fitted spe_ (last component) when X has named columns.

    Regression test: the reconstruction was built as ``scores @ self._x_loadings.T`` on the scores
    DataFrame, which relabelled the result columns 0..K-1. With string feature names the following
    ``X - X_hat`` subtraction then misaligned to all-NaN and every SPE collapsed to 0. Integer
    column labels hid the bug because the relabelled columns happened to line up.
    """
    rng = np.random.default_rng(0)
    n_rows, n_cols = 40, 6
    x_data = pd.DataFrame(rng.standard_normal((n_rows, n_cols)), columns=[f"feat_{i}" for i in range(n_cols)])
    y_data = pd.DataFrame({"y": x_data.to_numpy() @ rng.standard_normal(n_cols) + 0.1 * rng.standard_normal(n_rows)})
    model = PLS(n_components=3, scale=True).fit(x_data, y_data)

    diag = model.diagnose(x_data)
    fitted_spe = model.spe_.iloc[:, -1].to_numpy()
    assert diag.spe.to_numpy() == pytest.approx(fitted_spe, abs=1e-9)
    assert float(diag.spe.to_numpy().max()) > 0.0  # the bug collapsed every SPE to exactly 0


def test_pca_diagnose_numpy_input() -> None:
    """PCA.diagnose() should accept numpy arrays as well as DataFrames."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((50, 4)))
    X_scaled = MCUVScaler().fit_transform(X)

    model = PCA(n_components=2)
    model.fit(X_scaled)

    result = model.diagnose(X_scaled.values[:5])
    assert result.scores.shape == (5, 2)


@pytest.mark.parametrize("estimator", [PCA, MBPCA, TPLS])
def test_the_predict_alias_of_diagnose_is_gone(estimator: type) -> None:
    """The deprecated ``predict`` forwarders to ``diagnose`` were removed in 2.0 (#395, #396)."""
    assert hasattr(estimator, "diagnose")
    assert not hasattr(estimator, "predict")
    # PLS.predict and MBPLS.predict (#781) are the sklearn regressor contract, not the alias, and they stay.
    assert hasattr(PLS, "predict")
    assert hasattr(MBPLS, "predict")


def test_pca_score_method() -> None:
    """PCA.score() should return negative reconstruction error (higher = better)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((100, 5)))
    X_scaled = MCUVScaler().fit_transform(X)

    model_2 = PCA(n_components=2).fit(X_scaled)
    model_4 = PCA(n_components=4).fit(X_scaled)

    score_2 = model_2.score(X_scaled)
    score_4 = model_4.score(X_scaled)

    assert score_2 < 0  # negative MSE
    assert score_4 < 0
    # More components should give better (higher/less negative) score
    assert score_4 > score_2


def test_pca_transform_new_data() -> None:
    """PCA.transform() should project new data onto the loading space."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((80, 6)))
    X_scaled = MCUVScaler().fit_transform(X)

    model = PCA(n_components=3)
    model.fit(X_scaled)

    new_scores = model.transform(X_scaled.iloc[:5])
    assert new_scores.shape == (5, 3)

    # Transform of training data should match fitted scores
    all_scores = model.transform(X_scaled)
    np.testing.assert_allclose(all_scores.values, model.scores_.values, atol=1e-10)


def test_pca_fit_transform() -> None:
    """PCA.fit_transform() should return the same scores as fit() then accessing scores_."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((50, 4)))
    X_scaled = MCUVScaler().fit_transform(X)

    scores = PCA(n_components=2).fit_transform(X_scaled)
    model = PCA(n_components=2).fit(X_scaled)

    np.testing.assert_allclose(scores.values, model.scores_.values, atol=1e-10)


def test_pls_fit_transform_without_y_raises_value_error() -> None:
    """ENG-11 / #293 case 1: the missing-Y check must survive ``python -O``.

    ``Y`` carries a ``None`` default only to match sklearn's
    ``fit_transform(X, y=None)`` signature. The check used to be an ``assert``,
    which ``python -O`` strips: the call then reached ``fit`` and died with
    ``ValueError: at least one array or dtype is required`` from inside
    sklearn, far from the argument the caller got wrong. It is a caller-contract
    violation, so it raises ``ValueError`` up front and names ``Y``.
    """
    rng = np.random.default_rng(42)
    X = MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((20, 4))))

    with pytest.raises(ValueError, match=r"Y is required by PLS\.fit_transform"):
        PLS(n_components=2).fit_transform(X)


def test_pls_fit_transform_with_y_matches_fit() -> None:
    """The happy path is unchanged: fit_transform equals fit() then ``scores_``."""
    rng = np.random.default_rng(42)
    X = MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((20, 4))))
    Y = pd.DataFrame({"y": X.values @ rng.standard_normal(4)})

    scores = PLS(n_components=2).fit_transform(X, Y)
    model = PLS(n_components=2).fit(X, Y)

    np.testing.assert_allclose(scores.values, model.scores_.values, atol=1e-10)


def test_ellipse_coordinates_basic() -> None:
    """ellipse_coordinates should return x, y arrays forming a closed ellipse."""
    scaling = pd.Series([2.0, 1.5, 1.0])
    x, y = ellipse_coordinates(
        score_horiz=1,
        score_vert=2,
        conf_level=0.95,
        n_points=50,
        n_components=3,
        scaling_factor_for_scores=scaling,
        n_rows=100,
    )
    assert len(x) == 50
    assert len(y) == 50
    # Ellipse should be roughly closed (first ~= last point)
    assert x[0] == pytest.approx(x[-1], abs=0.1)
    assert y[0] == pytest.approx(y[-1], abs=0.1)


def test_ellipse_coordinates_symmetry() -> None:
    """Ellipse with equal scaling should be roughly circular."""
    scaling = pd.Series([1.0, 1.0])
    x, y = ellipse_coordinates(
        score_horiz=1,
        score_vert=2,
        conf_level=0.95,
        n_points=100,
        n_components=2,
        scaling_factor_for_scores=scaling,
        n_rows=50,
    )
    # For equal scaling, max |x| and max |y| should be similar
    assert max(abs(x)) == pytest.approx(max(abs(y)), rel=0.05)


#: A valid ellipse_coordinates call, which each row below breaks in one argument.
_ELLIPSE_ARGUMENTS = {
    "score_horiz": 1,
    "score_vert": 2,
    "conf_level": 0.95,
    "n_components": 2,
    "scaling_factor_for_scores": pd.Series([1.0, 1.0]),
    "n_rows": 10,
}


@pytest.mark.parametrize(
    ("limit", "arguments", "error", "message"),
    [
        (
            hotellings_t2_limit,
            {"conf_level": 1.0, "n_components": 2, "n_rows": 10},
            ValueError,
            r"^conf_level must lie in \(0, 1\); got 1\.0\.$",
        ),
        (
            hotellings_t2_limit,
            {"conf_level": 0.95, "n_components": 2, "n_rows": 0},
            ValueError,
            r"^n_rows must be positive; got 0\.$",
        ),
        (
            spe_calculation,
            {"spe_values": np.ones(5), "conf_level": 1.0},
            ValueError,
            r"^conf_level must lie in \(0, 1\); got 1\.0\.$",
        ),
        (
            ellipse_coordinates,
            {**_ELLIPSE_ARGUMENTS, "score_horiz": 3},
            ValueError,
            r"^score_horiz must lie in \[1, 2\]; got 3\.$",
        ),
        (
            ellipse_coordinates,
            {**_ELLIPSE_ARGUMENTS, "score_vert": 0},
            ValueError,
            r"^score_vert must lie in \[1, 2\]; got 0\.$",
        ),
        (
            ellipse_coordinates,
            {**_ELLIPSE_ARGUMENTS, "conf_level": 1.0},
            ValueError,
            r"^conf_level must lie in \(0, 1\); got 1\.0\.$",
        ),
        (
            ellipse_coordinates,
            {**_ELLIPSE_ARGUMENTS, "n_rows": 0},
            ValueError,
            r"^n_rows must be positive; got 0\.$",
        ),
        (
            ellipse_coordinates,
            {**_ELLIPSE_ARGUMENTS, "scaling_factor_for_scores": None},
            TypeError,
            (
                r"^scaling_factor_for_scores is required: pass the per-component score standard deviations "
                r"\(model\.scaling_factor_for_scores_\); got None\.$"
            ),
        ),
    ],
    ids=[
        "t2-limit-at-100-percent",
        "t2-limit-from-no-rows",
        "spe-limit-at-100-percent",
        "ellipse-horizontal-score-beyond-A",
        "ellipse-vertical-score-0",
        "ellipse-at-100-percent",
        "ellipse-from-no-rows",
        "ellipse-without-score-scaling",
    ],
)
def test_limit_functions_refuse_arguments_out_of_range(
    limit: Callable[..., object], arguments: dict, error: type[Exception], message: str
) -> None:
    """Each limit names the argument it cannot use, and the value it was given."""
    with pytest.raises(error, match=message):
        limit(**arguments)


def test_pls_predict_new_data() -> None:
    """PLS.predict() returns y_hat for sklearn compatibility; diagnose() the rich Bunch."""
    rng = np.random.default_rng(42)
    N, K = 80, 5
    X = pd.DataFrame(rng.standard_normal((N, K)), columns=[f"X{i}" for i in range(K)])
    beta = np.array([[2.0], [1.0], [-1.0], [0.5], [0.0]])
    Y = pd.DataFrame(X.values @ beta + rng.standard_normal((N, 1)) * 0.5, columns=["y"])

    scaler = MCUVScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    Y_scaled = pd.DataFrame(MCUVScaler().fit_transform(Y), columns=Y.columns)

    model = PLS(n_components=2)
    model.fit(X_scaled, Y_scaled)

    # sklearn-compatible predict returns just y_hat (as a DataFrame).
    y_pred = model.predict(X_scaled.iloc[:10])
    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape == (10, 1)

    # diagnose() returns the rich Bunch (scores / T² / SPE / y_hat).
    result = model.diagnose(X_scaled.iloc[:10])
    assert isinstance(result, Bunch)
    assert {"scores", "hotellings_t2", "spe", "y_hat"} <= set(result.keys())
    assert result.y_hat.shape[0] == 10
    # predict's output matches diagnose().y_hat.
    np.testing.assert_allclose(y_pred.to_numpy(), result.y_hat.to_numpy())


def test_pls_old_attribute_names_raise_simple() -> None:
    """Access old PLS attribute name to verify helpful AttributeError is raised."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((30, 3)), columns=["A", "B", "C"])
    beta = np.array([[1.0], [0.5], [-0.5]])
    Y = pd.DataFrame(X.values @ beta, columns=["y"])

    model = PLS(n_components=1)
    model.fit(X, Y)

    with pytest.raises(AttributeError, match="scores_"):
        _ = model.x_scores


# ---- Plot tests (improving multivariate/plots.py coverage) ----


@pytest.fixture
def fixture_pca_for_plots() -> PCA:
    """Return a simple PCA model for plot testing."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((50, 5)), columns=[f"V{i}" for i in range(5)])
    X_scaled = MCUVScaler().fit_transform(X)
    model = PCA(n_components=3)
    model.fit(X_scaled)
    return model


@pytest.fixture
def fixture_pls_for_plots() -> PLS:
    """Return a simple PLS model for plot testing."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((50, 5)), columns=[f"X{i}" for i in range(5)])
    beta = np.array([[2.0], [1.0], [-1.0], [0.5], [0.0]])
    Y = pd.DataFrame(X.values @ beta + rng.standard_normal((50, 1)) * 0.3, columns=["y"])
    X_scaled = MCUVScaler().fit_transform(X)
    Y_scaled = MCUVScaler().fit_transform(Y)
    model = PLS(n_components=2)
    model.fit(X_scaled, Y_scaled)
    return model


def test_score_plot_basic(fixture_pca_for_plots: PCA) -> None:
    """score_plot should return a Plotly Figure with scatter trace."""
    fig = fixture_pca_for_plots.score_plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # at least the scores trace


def test_score_plot_sizes_area_encoding(fixture_pca_for_plots: PCA) -> None:
    """`sizes` puts the value on the marker area, on one scale shared by every trace."""
    model = fixture_pca_for_plots
    spe = model.spe_.iloc[:, -1]
    plain = model.score_plot()
    assert plain.data[0].marker.sizemode is None  # the default path is untouched
    assert plain.data[0].marker.size == 7

    fig = model.score_plot(sizes=spe, size_name="SPE")
    marker = fig.data[0].marker
    assert marker.sizemode == "area"
    assert marker.size == pytest.approx(spe.to_numpy())
    # sizeref is what turns a value into an area: the largest value fills `size_max` pixels.
    assert marker.sizeref == pytest.approx(2.0 * spe.max() / 26.0**2)
    assert "SPE" in fig.data[0].hovertemplate

    highlighted = model.score_plot(
        items_to_highlight={'{"color": "red"}': list(spe.index[:2])}, sizes=spe, size_name="SPE"
    )
    refs = {trace.marker.sizeref for trace in highlighted.data if trace.marker.sizeref is not None}
    assert len(refs) == 1, "the highlighted trace must be read on the same scale as the rest"


def test_score_plot_sizes_settings_and_guards(fixture_pca_for_plots: PCA) -> None:
    """`size_max` sets the largest marker, and a value that cannot be an area is refused."""
    model = fixture_pca_for_plots
    spe = model.spe_.iloc[:, -1]
    fig = model.score_plot(sizes=spe, settings={"size_max": 40})
    assert fig.data[0].marker.sizeref == pytest.approx(2.0 * spe.max() / 40.0**2)

    with pytest.raises(ValueError, match="cannot be negative"):
        model.score_plot(sizes=-spe)
    with pytest.raises(ValueError, match="no value for these observations"):
        model.score_plot(sizes=spe.iloc[:3])
    with pytest.raises(ValueError, match="at least one positive value"):
        model.score_plot(sizes=spe * 0.0)


def test_score_plot_with_ellipse(fixture_pca_for_plots: PCA) -> None:
    """score_plot with ellipse should have an extra trace for the ellipse."""
    fig = fixture_pca_for_plots.score_plot(settings={"show_ellipse": True})
    assert isinstance(fig, go.Figure)
    # Should have at least 2 traces: scores + ellipse
    assert len(fig.data) >= 2


def test_score_plot_no_ellipse(fixture_pca_for_plots: PCA) -> None:
    """score_plot with show_ellipse=False should only have the scores trace."""
    fig = fixture_pca_for_plots.score_plot(settings={"show_ellipse": False})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_score_plot_custom_components(fixture_pca_for_plots: PCA) -> None:
    """score_plot should accept custom component selection."""
    fig = fixture_pca_for_plots.score_plot(pc_horiz=1, pc_vert=3)
    assert isinstance(fig, go.Figure)


def test_score_plot_with_highlights(fixture_pca_for_plots: PCA) -> None:
    """score_plot with items_to_highlight should add extra traces."""
    model = fixture_pca_for_plots
    idx = model.scores_.index[:5].tolist()
    highlights = {'{"color": "red", "symbol": "cross"}': idx}
    fig = model.score_plot(items_to_highlight=highlights)
    assert isinstance(fig, go.Figure)
    # Should have at least 3 traces: default scores, highlighted, ellipse
    assert len(fig.data) >= 3


def test_score_plot_3d_with_highlights(fixture_pca_for_plots: PCA) -> None:
    """A 3D score plot must also decode and apply highlight styling."""
    model = fixture_pca_for_plots
    idx = model.scores_.index[:5].tolist()
    highlights = {'{"color": "red", "symbol": "cross"}': idx}
    fig = model.score_plot(pc_horiz=1, pc_vert=2, pc_depth=3, items_to_highlight=highlights)
    assert isinstance(fig, go.Figure)
    # Default scores trace + the highlighted trace (3D path has no ellipse).
    assert len(fig.data) >= 2


@pytest.mark.parametrize("plot_method", ["score_plot", "spe_plot", "t2_plot"])
def test_highlight_key_must_be_json(fixture_pca_for_plots: PCA, plot_method: str) -> None:
    """A non-JSON ``items_to_highlight`` key raises a clear ValueError.

    Mirrors the SEC-32 guard in ``batch.plotting``: a malformed style key must
    surface a clear ``ValueError`` at the API boundary instead of a confusing
    ``json.JSONDecodeError`` deep inside the trace-building loop.
    """
    model = fixture_pca_for_plots
    idx = model.scores_.index[:3].tolist()
    bad_highlights = {"not-json": idx}  # plain string, not a JSON style spec
    with pytest.raises(ValueError, match="JSON-encoded"):
        getattr(model, plot_method)(items_to_highlight=bad_highlights)


def test_loading_plot_pca(fixture_pca_for_plots: PCA) -> None:
    """loading_plot for PCA should return a Figure with P loadings."""
    fig = fixture_pca_for_plots.loading_plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_spe_plot_basic(fixture_pca_for_plots: PCA) -> None:
    """spe_plot should return a Figure with SPE markers and limit line."""
    fig = fixture_pca_for_plots.spe_plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_t2_plot_basic(fixture_pca_for_plots: PCA) -> None:
    """t2_plot should return a Figure with T2 markers and limit line."""
    fig = fixture_pca_for_plots.t2_plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_spe_plot_with_highlights(fixture_pca_for_plots: PCA) -> None:
    """spe_plot with items_to_highlight should add extra highlighted traces."""
    model = fixture_pca_for_plots
    idx = model.scores_.index[:3].tolist()
    highlights = {'{"color": "orange", "symbol": "diamond"}': idx}
    fig = model.spe_plot(items_to_highlight=highlights)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_t2_plot_with_highlights(fixture_pca_for_plots: PCA) -> None:
    """t2_plot with items_to_highlight should add extra highlighted traces."""
    model = fixture_pca_for_plots
    idx = model.scores_.index[:3].tolist()
    highlights = {'{"color": "orange", "symbol": "diamond"}': idx}
    fig = model.t2_plot(items_to_highlight=highlights)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_t2_plot_rejects_zero_components(fixture_pca_for_plots: PCA) -> None:
    """t2_plot should reject `n_components == 0`."""
    with pytest.raises(ValueError, match="n_components"):
        fixture_pca_for_plots.t2_plot(n_components=0)


def test_t2_plot_rejects_more_components_than_fitted(fixture_pca_for_plots: PCA) -> None:
    """t2_plot should reject `n_components` larger than the number of fitted components."""
    model = fixture_pca_for_plots
    with pytest.raises(ValueError, match="n_components"):
        model.t2_plot(n_components=model.n_components + 1)


def test_t2_plot_rejects_invalid_conf_level(fixture_pca_for_plots: PCA) -> None:
    """t2_plot should reject `conf_level` outside the open interval (0, 1)."""
    with pytest.raises(ValueError, match="conf_level"):
        fixture_pca_for_plots.t2_plot(settings={"conf_level": 1.0})
    with pytest.raises(ValueError, match="conf_level"):
        fixture_pca_for_plots.t2_plot(settings={"conf_level": 0.0})


def test_t2_plot_accepts_valid_conf_level(fixture_pca_for_plots: PCA) -> None:
    """t2_plot should accept `conf_level` strictly inside (0, 1)."""
    fig = fixture_pca_for_plots.t2_plot(settings={"conf_level": 0.99})
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    ("plot", "arguments", "message"),
    [
        ("score_plot", {"pc_horiz": 4}, r"^The model has 3 components\. Ensure that 1 <= pc_horiz <= 3\.$"),
        ("score_plot", {"pc_vert": 4}, r"^The model has 3 components\. Ensure that 1 <= pc_vert <= 3\.$"),
        (
            "score_plot",
            {"pc_depth": 4},
            r"^The model has 3 components\. Ensure that pc_depth is -1 \(no depth axis\) or 1 <= pc_depth <= 3\.$",
        ),
        ("score_plot", {"pc_horiz": 1, "pc_vert": 1}, r"^Specify distinct components for each axis\.$"),
        ("score_plot", {"settings": {"ellipse_conf_level": 1.0}}, r"0\.0 < `ellipse_conf_level` < 1\.0"),
        ("score_plot", {"settings": {"ellipse_conf_level": 0.0}}, r"0\.0 < `ellipse_conf_level` < 1\.0"),
        ("spe_plot", {"n_components": 0}, r"^`n_components` must be >= 1, or specified with negative indexing\.$"),
        (
            "spe_plot",
            {"n_components": 4},
            r"^`n_components` must be <= the number of components fitted \(3\); got 4\.$",
        ),
        ("spe_plot", {"settings": {"conf_level": 1.0}}, r"0\.0 < `conf_level` < 1\.0"),
        ("spe_plot", {"settings": {"conf_level": 0.0}}, r"0\.0 < `conf_level` < 1\.0"),
    ],
    ids=[
        "score-horizontal-axis-beyond-A",
        "score-vertical-axis-beyond-A",
        "score-depth-axis-beyond-A",
        "score-same-component-twice",
        "score-ellipse-at-100-percent",
        "score-ellipse-at-0-percent",
        "spe-after-0-components",
        "spe-after-more-components-than-fitted",
        "spe-limit-at-100-percent",
        "spe-limit-at-0-percent",
    ],
)
def test_score_and_spe_plots_refuse_arguments_out_of_range(
    fixture_pca_for_plots: PCA, plot: str, arguments: dict, message: str
) -> None:
    """Axes must name distinct fitted components, and confidence levels must lie strictly inside (0, 1)."""
    with pytest.raises(ValueError, match=message):
        getattr(fixture_pca_for_plots, plot)(**arguments)


def test_score_plot_draws_the_ellipse_at_the_confidence_level_asked_for(fixture_pca_for_plots: PCA) -> None:
    """A 99% ellipse is named as such and drawn outside the default 95% one."""
    model = fixture_pca_for_plots
    default = model.score_plot().data[1]
    wider = model.score_plot(settings={"ellipse_conf_level": 0.99}).data[1]
    assert (default.name, wider.name) == ("Hotelling's T^2 [95%]", "Hotelling's T^2 [99%]")
    assert np.max(np.abs(wider.x)) > np.max(np.abs(default.x))


def test_spe_plot_draws_the_limit_at_the_confidence_level_asked_for(fixture_pca_for_plots: PCA) -> None:
    """The limit line sits at the model's SPE limit for the level asked for."""
    model = fixture_pca_for_plots
    drawn = max(shape.y0 for shape in model.spe_plot(settings={"conf_level": 0.99}).layout.shapes)
    assert drawn == pytest.approx(model.spe_limit(conf_level=0.99))
    assert drawn > model.spe_limit(conf_level=0.95)


@pytest.mark.parametrize("plot", ["score_plot", "loading_plot", "spe_plot", "t2_plot"])
def test_pca_plots_draw_onto_a_figure_they_are_given(fixture_pca_for_plots: PCA, plot: str) -> None:
    """Passing `fig` adds the traces to that figure, so plots can be layered, and returns it."""
    canvas = go.Figure()
    returned = getattr(fixture_pca_for_plots, plot)(fig=canvas)
    assert returned is canvas
    assert len(canvas.data) >= 1


def test_explained_variance_plot_pca(fixture_pca_for_plots: PCA) -> None:
    """explained_variance_plot returns per-component bars and a cumulative line."""
    model = fixture_pca_for_plots
    fig = model.explained_variance_plot()
    assert isinstance(fig, go.Figure)
    assert [trace.type for trace in fig.data] == ["bar", "scatter"]

    # Bars are the per-component R2; the line is the cumulative R2 (as percentages).
    assert np.allclose(fig.data[0].y, model.r2_per_component_.to_numpy() * 100.0)
    assert np.allclose(fig.data[1].y, model.r2_cumulative_.to_numpy() * 100.0)

    # The cumulative curve never decreases.
    cumulative = list(fig.data[1].y)
    assert cumulative == sorted(cumulative)
    assert "X-variance" in fig.layout.title.text

    # The standalone function agrees with the bound method.
    assert np.array_equal(explained_variance_plot(model).data[0].y, fig.data[0].y)


def test_explained_variance_plot_pls(fixture_pls_for_plots: PLS) -> None:
    """explained_variance_plot works for PLS and labels the Y-block."""
    fig = fixture_pls_for_plots.explained_variance_plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert "Y-variance" in fig.layout.title.text


def test_explained_variance_plot_settings(fixture_pca_for_plots: PCA) -> None:
    """The as_percentage and title settings are honoured."""
    fig = fixture_pca_for_plots.explained_variance_plot(
        settings={"as_percentage": False, "title": "Custom title"},
    )
    # Fractions, not percentages: every bar is at most 1.0.
    assert max(fig.data[0].y) <= 1.0
    assert fig.layout.title.text == "Custom title"
    assert "fraction" in fig.layout.yaxis.title.text


def test_explained_variance_plot_unfitted_raises() -> None:
    """An unfitted model raises a clear error."""
    for unfitted in (PCA(n_components=2), PLS(n_components=2)):
        with pytest.raises(ValueError, match="not fitted"):
            explained_variance_plot(unfitted)


def test_explained_variance_plot_accepts_existing_figure(fixture_pca_for_plots: PCA) -> None:
    """Passing a figure draws onto it rather than creating a new one."""
    base = go.Figure()
    returned = fixture_pca_for_plots.explained_variance_plot(fig=base)
    assert returned is base
    assert len(returned.data) == 2


def test_correlation_loadings_plot_pca() -> None:
    """Correlation loadings: points inside the unit circle, matching direct correlations."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((60, 6)), columns=[f"V{i}" for i in range(6)])
    X_scaled = MCUVScaler().fit_transform(X)
    model = PCA(n_components=3).fit(X_scaled)

    fig = model.correlation_loadings_plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # X-variables only, for PCA
    assert fig.data[0].name == "X-variables"

    xs = np.asarray(fig.data[0].x, dtype=float)
    ys = np.asarray(fig.data[0].y, dtype=float)
    # Every variable lies within the unit circle.
    assert np.all(np.hypot(xs, ys) <= 1.0 + 1e-9)

    # The plotted coordinates are the correlations of each variable with the scores.
    t1 = model.scores_.iloc[:, 0].to_numpy()
    direct = np.array([np.corrcoef(X_scaled.iloc[:, k], t1)[0, 1] for k in range(6)])
    assert np.allclose(xs, direct, atol=1e-6)

    # Two variance ellipses are drawn by default.
    circles = [shape for shape in fig.layout.shapes if shape.type == "circle"]
    assert len(circles) == 2


def test_correlation_loadings_plot_pls() -> None:
    """For PLS both the X- and Y-variables are overlaid."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((50, 5)), columns=[f"X{i}" for i in range(5)])
    beta = rng.standard_normal((5, 2))
    Y = pd.DataFrame(X.values @ beta + 0.3 * rng.standard_normal((50, 2)), columns=["y0", "y1"])
    model = PLS(n_components=2).fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))

    fig = model.correlation_loadings_plot()
    assert [trace.name for trace in fig.data] == ["X-variables", "Y-variables"]
    assert len(fig.data[1].x) == 2  # two Y-variables


def test_correlation_loadings_plot_configurable_ellipses() -> None:
    """The variance_ellipses argument controls which ellipses are drawn."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.standard_normal((40, 5)))
    model = PCA(n_components=2).fit(MCUVScaler().fit_transform(X))

    default_fig = model.correlation_loadings_plot()
    assert {ann.text for ann in default_fig.layout.annotations} == {"50%", "100%"}

    custom_fig = model.correlation_loadings_plot(variance_ellipses=(0.75, 0.95))
    assert {ann.text for ann in custom_fig.layout.annotations} == {"75%", "95%"}
    assert len([shape for shape in custom_fig.layout.shapes if shape.type == "circle"]) == 2


def test_correlation_loadings_plot_settings_and_figure() -> None:
    """correlation_loadings_plot honours settings and draws onto a given figure."""
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.standard_normal((40, 5)))
    model = PCA(n_components=2).fit(MCUVScaler().fit_transform(X))

    base = go.Figure()
    fig = model.correlation_loadings_plot(settings={"show_labels": False, "title": "Custom"}, fig=base)
    assert fig is base
    assert fig.layout.title.text == "Custom"
    assert fig.data[0].mode == "markers"


def test_correlation_loadings_plot_errors() -> None:
    """correlation_loadings_plot validates the model and its arguments."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((30, 4)))
    model = PCA(n_components=3).fit(MCUVScaler().fit_transform(X))

    with pytest.raises(ValueError, match="not fitted"):
        correlation_loadings_plot(PCA(n_components=2))
    with pytest.raises(ValueError, match="not a fitted component"):
        model.correlation_loadings_plot(pc_vert=9)
    with pytest.raises(ValueError, match="must be different"):
        model.correlation_loadings_plot(pc_horiz=2, pc_vert=2)
    with pytest.raises(ValueError, match="fraction in"):
        model.correlation_loadings_plot(variance_ellipses=(1.5,))


def test_predictions_vs_observed_plot() -> None:
    """The parity plot pairs observed and predicted Y, with a y=x line and RMSE."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f"X{i}" for i in range(6)])
    beta = rng.standard_normal((6, 2))
    Y = pd.DataFrame(X.values @ beta + 0.3 * rng.standard_normal((50, 2)), columns=["conv", "quality"])
    X_s, Y_s = MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y)
    model = PLS(n_components=3).fit(X_s, Y_s)

    fig = model.predictions_vs_observed_plot(y_observed=Y_s)
    assert isinstance(fig, go.Figure)
    assert {trace.name for trace in fig.data} == {"y = x", "Observations"}

    # The observations trace carries the observed/predicted pairs of the first Y.
    obs_trace = next(trace for trace in fig.data if trace.name == "Observations")
    assert np.allclose(obs_trace.x, Y_s["conv"].to_numpy())
    assert np.allclose(obs_trace.y, model.predictions_["conv"].to_numpy())

    # The RMSE annotation matches a direct calculation.
    rmse = np.sqrt(np.mean((Y_s["conv"].to_numpy() - model.predictions_["conv"].to_numpy()) ** 2))
    assert fig.layout.annotations[0].text == f"RMSE = {rmse:.4g}"

    # A specific Y-variable can be selected.
    fig_q = model.predictions_vs_observed_plot(y_observed=Y_s, variable="quality")
    assert "quality" in fig_q.layout.title.text


def test_coefficient_plot() -> None:
    """The coefficient plot shows one bar per X-variable, matching beta_coefficients_."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"X{i}" for i in range(5)])
    beta = rng.standard_normal((5, 2))
    Y = pd.DataFrame(X.values @ beta + 0.3 * rng.standard_normal((40, 2)), columns=["y0", "y1"])
    model = PLS(n_components=2).fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))

    fig = model.coefficient_plot()
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"
    assert np.allclose(fig.data[0].y, model.beta_coefficients_["y0"].to_numpy())
    assert list(fig.data[0].x) == [str(name) for name in model.beta_coefficients_.index]

    fig_y1 = model.coefficient_plot(variable="y1")
    assert np.allclose(fig_y1.data[0].y, model.beta_coefficients_["y1"].to_numpy())


def test_pls_prediction_plots_errors() -> None:
    """The PLS prediction plots validate the model and their arguments."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.standard_normal((30, 4)))
    Y = pd.DataFrame(rng.standard_normal((30, 1)), columns=["y"])
    model = PLS(n_components=2).fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))
    Y_s = MCUVScaler().fit_transform(Y)

    with pytest.raises(ValueError, match="not fitted"):
        coefficient_plot(PLS(n_components=2))
    with pytest.raises(ValueError, match="not fitted"):
        predictions_vs_observed_plot(PLS(n_components=2), y_observed=Y_s)
    with pytest.raises(ValueError, match="Unknown Y-variable"):
        model.coefficient_plot(variable="missing")
    with pytest.raises(ValueError, match="Unknown Y-variable"):
        model.predictions_vs_observed_plot(y_observed=Y_s, variable="missing")
    with pytest.raises(ValueError, match="must have 30 rows"):
        model.predictions_vs_observed_plot(y_observed=Y_s.iloc[:5])
    # A bare ndarray is coerced to a DataFrame; its columns then will not match.
    with pytest.raises(ValueError, match="has no column"):
        model.predictions_vs_observed_plot(y_observed=Y_s.to_numpy())


def test_pls_prediction_plots_settings_and_figure() -> None:
    """Both PLS prediction plots honour settings and draw onto a given figure."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((35, 4)))
    Y = pd.DataFrame(rng.standard_normal((35, 1)), columns=["y"])
    model = PLS(n_components=2).fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))
    Y_s = MCUVScaler().fit_transform(Y)

    base = go.Figure()
    parity = model.predictions_vs_observed_plot(y_observed=Y_s, settings={"title": "P"}, fig=base)
    assert parity is base
    assert parity.layout.title.text == "P"

    base2 = go.Figure()
    coeffs = model.coefficient_plot(settings={"title": "C", "bar_color": "green"}, fig=base2)
    assert coeffs is base2
    assert coeffs.layout.title.text == "C"


# ---- Per-observation diagnostics: cos2, contributions, eigenvalue summary, supplementary variables ----


def test_squared_cosine_pca() -> None:
    """cos2 for a fitted PCA model: shape, range, and agreement with the bound method."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((40, 6)), columns=[f"V{i}" for i in range(6)])
    model = PCA(n_components=3).fit(MCUVScaler().fit_transform(X))

    standalone = squared_cosine(model)
    bound = model.squared_cosine()
    assert standalone.equals(bound)

    assert isinstance(standalone, pd.DataFrame)
    assert standalone.shape == (40, 3)
    assert standalone.index.equals(model.scores_.index)
    assert list(standalone.columns) == list(model.scores_.columns)
    assert ((standalone.to_numpy() >= 0) & (standalone.to_numpy() <= 1 + 1e-9)).all()

    # cos2 over all components plus the residual fraction must sum to 1 per row.
    residual_ss = model.spe_.iloc[:, -1].to_numpy() ** 2
    score_ss = (model.scores_.to_numpy() ** 2).sum(axis=1)
    residual_fraction = residual_ss / (score_ss + residual_ss)
    assert standalone.sum(axis=1).to_numpy() + residual_fraction == pytest.approx(1.0, abs=1e-9)


def test_squared_cosine_pls() -> None:
    """cos2 works for PLS and matches the bound method."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"X{i}" for i in range(5)])
    beta = np.array([[2.0], [1.0], [-1.0], [0.5], [0.0]])
    Y = pd.DataFrame(X.values @ beta + rng.standard_normal((40, 1)) * 0.3, columns=["y"])
    model = PLS(n_components=2).fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))

    bound = model.squared_cosine()
    assert squared_cosine(model).equals(bound)
    assert bound.shape == (40, 2)
    assert ((bound.to_numpy() >= 0) & (bound.to_numpy() <= 1 + 1e-9)).all()


def test_squared_cosine_n_components() -> None:
    """cos2 honours n_components and rejects out-of-range values."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.standard_normal((30, 5)), columns=list("ABCDE"))
    model = PCA(n_components=3).fit(MCUVScaler().fit_transform(X))

    assert model.squared_cosine(n_components=2).shape == (30, 2)
    # Restricting the number of components does not change the values that remain.
    assert model.squared_cosine(n_components=2).equals(model.squared_cosine().iloc[:, :2])

    with pytest.raises(ValueError, match="n_components"):
        model.squared_cosine(n_components=0)
    with pytest.raises(ValueError, match="n_components"):
        model.squared_cosine(n_components=99)


def test_observation_contributions_pca() -> None:
    """Observation contributions for PCA: each component column sums to 1."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((35, 6)), columns=[f"V{i}" for i in range(6)])
    model = PCA(n_components=3).fit(MCUVScaler().fit_transform(X))

    standalone = observation_contributions(model)
    bound = model.observation_contributions()
    assert standalone.equals(bound)

    assert standalone.shape == (35, 3)
    assert (standalone.to_numpy() >= 0).all()
    assert standalone.sum(axis=0).to_numpy() == pytest.approx(1.0, abs=1e-9)


def test_observation_contributions_pls() -> None:
    """Observation contributions for PLS: each component column sums to 1."""
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.standard_normal((35, 5)), columns=[f"X{i}" for i in range(5)])
    beta = np.array([[1.5], [-0.5], [1.0], [0.0], [0.7]])
    Y = pd.DataFrame(X.values @ beta + rng.standard_normal((35, 1)) * 0.2, columns=["y"])
    model = PLS(n_components=2).fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))

    contributions = model.observation_contributions()
    assert contributions.shape == (35, 2)
    assert contributions.sum(axis=0).to_numpy() == pytest.approx(1.0, abs=1e-9)

    with pytest.raises(ValueError, match="n_components"):
        model.observation_contributions(n_components=5)


def test_observation_contributions_for_fewer_components_are_the_leading_columns() -> None:
    """Asking for the first two components gives the first two columns of the full table."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((35, 6)), columns=[f"V{i}" for i in range(6)])
    model = PCA(n_components=3).fit(MCUVScaler().fit_transform(X))
    pd.testing.assert_frame_equal(
        observation_contributions(model, n_components=2), model.observation_contributions().iloc[:, :2]
    )


@pytest.mark.slow
def test_eigenvalue_summary_pca(fixture_tablet_spectra_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Eigenvalue summary for PCA on a real dataset: tidy table, monotone cumulative."""
    spectra, _ = fixture_tablet_spectra_data
    model = PCA(n_components=4).fit(MCUVScaler().fit_transform(spectra))

    summary = eigenvalue_summary(model)
    assert model.eigenvalue_summary().equals(summary)
    assert list(summary.columns) == ["eigenvalue", "percent_variance", "cumulative_percent"]
    assert summary.index.name == "component"
    assert len(summary) == 4

    # Cumulative percentage is non-decreasing and matches r2_cumulative_.
    cumulative = summary["cumulative_percent"].to_numpy()
    assert np.all(np.diff(cumulative) >= -1e-9)
    assert cumulative == pytest.approx(model.r2_cumulative_.to_numpy() * 100.0)
    # The R reference (prcomp) reports ~95.85% cumulative variance for 4 components.
    assert cumulative[-1] == pytest.approx(95.846, abs=0.5)


def test_eigenvalue_summary_pls() -> None:
    """Eigenvalue summary works for PLS."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"X{i}" for i in range(5)])
    beta = np.array([[2.0], [1.0], [-1.0], [0.5], [0.0]])
    Y = pd.DataFrame(X.values @ beta + rng.standard_normal((40, 1)) * 0.3, columns=["y"])
    model = PLS(n_components=3).fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))

    summary = model.eigenvalue_summary()
    assert list(summary.columns) == ["eigenvalue", "percent_variance", "cumulative_percent"]
    assert len(summary) == 3
    assert summary["cumulative_percent"].iloc[-1] == pytest.approx(model.r2_cumulative_.iloc[-1] * 100.0)


def test_project_variables_recovers_scores() -> None:
    """Projecting the score matrix as supplementary variables yields the identity."""
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f"V{i}" for i in range(6)])
    model = PCA(n_components=3).fit(MCUVScaler().fit_transform(X))

    # PCA scores are mutually orthogonal, so projecting them as supplementary
    # variables must give an (almost) identity correlation matrix.
    projected = model.project_variables(model.scores_)
    assert projected.shape == (3, 3)
    assert projected.to_numpy() == pytest.approx(np.eye(3), abs=1e-9)


def test_project_variables_pls_and_errors() -> None:
    """project_variables works for PLS and validates the row count."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"X{i}" for i in range(5)])
    beta = np.array([[2.0], [1.0], [-1.0], [0.5], [0.0]])
    Y = pd.DataFrame(X.values @ beta + rng.standard_normal((40, 1)) * 0.3, columns=["y"])
    model = PLS(n_components=2).fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))

    supplementary = pd.DataFrame(rng.standard_normal((40, 3)), columns=["s1", "s2", "s3"])
    projected = model.project_variables(supplementary)
    assert projected.shape == (3, 2)
    assert list(projected.index) == ["s1", "s2", "s3"]
    assert ((projected.to_numpy() >= -1 - 1e-9) & (projected.to_numpy() <= 1 + 1e-9)).all()

    # The standalone function agrees with the bound method.
    assert project_variables(model, supplementary).equals(projected)

    # A supplementary block with the wrong number of rows must raise.
    with pytest.raises(ValueError, match="rows"):
        model.project_variables(pd.DataFrame(rng.standard_normal((10, 2))))


def test_project_variables_reads_an_array_as_its_dataframe() -> None:
    """Supplementary variables passed as an array give the same correlations as the same DataFrame."""
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f"V{i}" for i in range(6)])
    model = PCA(n_components=3).fit(MCUVScaler().fit_transform(X))
    supplementary = pd.DataFrame(rng.standard_normal((50, 2)))
    np.testing.assert_allclose(
        project_variables(model, supplementary.to_numpy()).to_numpy(),
        project_variables(model, supplementary).to_numpy(),
    )


def test_diagnostics_unfitted_raise() -> None:
    """The new diagnostics raise a clear error on an unfitted model."""
    for unfitted in (PCA(n_components=2), PLS(n_components=2)):
        with pytest.raises(ValueError, match="not fitted"):
            squared_cosine(unfitted)
        with pytest.raises(ValueError, match="not fitted"):
            observation_contributions(unfitted)
        with pytest.raises(ValueError, match="not fitted"):
            eigenvalue_summary(unfitted)
        with pytest.raises(ValueError, match="not fitted"):
            project_variables(unfitted, pd.DataFrame(np.zeros((3, 2))))


def test_rv_coefficient_basic() -> None:
    """RV coefficient: identity, rotation/scale invariance, range, symmetry."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))

    # A block is perfectly correlated with itself.
    assert rv_coefficient(X, X) == pytest.approx(1.0)

    # RV is invariant to an orthogonal rotation and an overall rescaling.
    rotation, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    assert rv_coefficient(X, 3.0 * X @ rotation) == pytest.approx(1.0)

    # An unrelated block gives a value strictly inside [0, 1].
    Y = rng.standard_normal((40, 4))
    rv = rv_coefficient(X, Y)
    assert 0.0 <= rv <= 1.0

    # The coefficient is symmetric in its two arguments.
    assert rv_coefficient(X, Y) == pytest.approx(rv_coefficient(Y, X))


def test_rv2_coefficient_basic() -> None:
    """Modified RV coefficient: identity, rotation invariance, symmetry."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((35, 5))

    assert rv2_coefficient(X, X) == pytest.approx(1.0)

    rotation, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    assert rv2_coefficient(X, 2.5 * X @ rotation) == pytest.approx(1.0)

    Y = rng.standard_normal((35, 7))
    assert rv2_coefficient(X, Y) == pytest.approx(rv2_coefficient(Y, X))
    assert -1.0 <= rv2_coefficient(X, Y) <= 1.0


def test_rv2_unbiased_for_high_dimensional_data() -> None:
    """RV is inflated for many-variable data; RV2 corrects that bias."""
    rng = np.random.default_rng(2)
    # Two independent blocks with far more variables than observations.
    A = rng.standard_normal((80, 400))
    B = rng.standard_normal((80, 400))

    rv = rv_coefficient(A, B)
    rv2 = rv2_coefficient(A, B)

    # The ordinary RV coefficient is inflated towards 1 here.
    assert rv > 0.7
    # The modified coefficient stays close to 0 for genuinely unrelated blocks.
    assert abs(rv2) < 0.3
    assert rv2 < rv


def test_rv_coefficient_inputs_and_errors() -> None:
    """RV accepts DataFrames and ndarrays, and validates its inputs."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((25, 4))
    Y = rng.standard_normal((25, 6))

    # DataFrame and ndarray inputs give the same answer.
    from_arrays = rv_coefficient(X, Y)
    from_frames = rv_coefficient(pd.DataFrame(X), pd.DataFrame(Y))
    assert from_arrays == pytest.approx(from_frames)

    # A 1-D block (a single variable) is treated as one column.
    one_d = rng.standard_normal(25)
    assert np.isfinite(rv_coefficient(one_d, Y))
    assert rv2_coefficient(one_d, one_d) == pytest.approx(1.0)

    # Mismatched row counts raise a clear error.
    with pytest.raises(ValueError, match="same number of rows"):
        rv_coefficient(X, rng.standard_normal((10, 3)))

    # A 3-D array is not a valid data block.
    with pytest.raises(ValueError, match="1- or 2-dimensional"):
        rv_coefficient(rng.standard_normal((25, 2, 3)), Y)

    # A block with no variance has an undefined coefficient.
    assert np.isnan(rv_coefficient(np.ones((25, 3)), Y))
    assert np.isnan(rv2_coefficient(X, np.full((25, 3), 7.0)))


def test_rv_coefficient_ldpe(
    fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """RV / RV2 give sensible values on the real LDPE process and quality blocks."""
    data = fixture_pls_ldpe_example
    X = MCUVScaler().fit_transform(pd.DataFrame(data["X"]))
    Y = MCUVScaler().fit_transform(pd.DataFrame(data["Y"]))

    rv = rv_coefficient(X, Y)
    rv2 = rv2_coefficient(X, Y)

    # Process and quality blocks share real structure: a positive coefficient.
    assert 0.0 < rv <= 1.0
    assert np.isfinite(rv2)
    assert rv2 < rv


# n_components = 3
# data = fixture_tpls_example()
# full_model = TPLS(n_components=n_components, d_matrix=data.pop("D"))
# full_model.fit(DataFrameDict(data))

# data = fixture_tpls_example()
# cv_results = cross_val_score(
#     estimator=TPLS(n_components=n_components, d_matrix=data.pop("D")),
#     X=DataFrameDict(data),
#     cv=10,
#     # scoring={"score": scorer},
#     n_jobs=-1,
#     verbose=1,
# )
# print(cv_results)
# fixture_tpls_example = fixture_tpls_example()
# test_tpls_model_predictions(fixture_tpls_example)


def test_pca_renamed_attribute_raises_helpful_error() -> None:
    """Old PCA attribute names should raise an AttributeError pointing to the new name."""
    pca = PCA(n_components=2)
    with pytest.raises(AttributeError, match="renamed to 'scores_'"):
        _ = pca.x_scores
    with pytest.raises(AttributeError, match="renamed to 'spe_'"):
        _ = pca.squared_prediction_error
    with pytest.raises(AttributeError, match="no attribute"):
        _ = pca.totally_made_up_name


def test_pls_renamed_attribute_raises_helpful_error() -> None:
    """Old PLS attribute names should raise an AttributeError pointing to the new name."""
    pls = PLS(n_components=2)
    with pytest.raises(AttributeError):
        _ = pls.x_scores
    with pytest.raises(AttributeError, match="no attribute"):
        _ = pls.totally_made_up_name


def test_vip_raises_on_unfitted_model() -> None:
    """vip() requires a fitted model with r2_per_component_."""
    with pytest.raises(ValueError, match="not fitted"):
        vip(object())


def test_vip_raises_without_weights_or_loadings() -> None:
    """vip() needs x_weights_ (PLS) or loadings_ (PCA) to compute scores."""

    class _FakeFitted:
        r2_per_component_ = pd.DataFrame([[0.5, 0.3]])

    message = "Model must have 'x_weights_' (PLS) or 'loadings_' (PCA) to compute VIP."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        vip(_FakeFitted())


# ---- The diagnostics worked by hand on stand-in models: each reads a fitted model's attributes only ----


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (vip, "Model is not fitted. Call fit() before computing VIP."),
        (squared_cosine, "Model is not fitted. Call fit() before computing the squared cosine."),
        (observation_contributions, "Model is not fitted. Call fit() before computing observation contributions."),
        (eigenvalue_summary, "Model is not fitted. Call fit() before computing the eigenvalue summary."),
        (
            lambda model: t2_contributions(model, np.zeros((2, 2))),
            "Model is not fitted. Call fit() before computing contributions.",
        ),
        (
            lambda model: project_variables(model, np.zeros((2, 2))),
            "Model is not fitted. Call fit() before projecting variables.",
        ),
        (
            lambda model: target_projection(model, np.zeros((2, 2))),
            "Model is not fitted, or is not a PLS model with 'beta_coefficients_'.",
        ),
    ],
)
def test_diagnostics_refuse_an_unfitted_model_in_full(call: Callable, message: str) -> None:
    """Each diagnostic names what it could not compute, word for word."""
    for unfitted in (PCA(n_components=2), PLS(n_components=2)):
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            call(unfitted)


def test_squared_cosine_needs_the_spe_as_well_as_the_scores() -> None:
    """A model with scores but no SPE has no residual to put in the denominator."""
    message = "Model is not fitted. Call fit() before computing the squared cosine."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        squared_cosine(SimpleNamespace(scores_=pd.DataFrame([[1.0]])))


def test_vip_by_hand_on_a_stand_in_model() -> None:
    """``VIP_j^2 = K sum_a r2_a w_ja^2 / sum_a r2_a``, for three features and two components.

    With r2 = (0.5, 0.25) and K = 3 this is ``2 w_j1^2 + w_j2^2``: 0.72, 1.64 and 0.64 for the
    weights below, which sum to K. On the first component alone it is ``3 w_j1^2``: 1.08, 1.92
    and 0. A PLS model's weights are read before its loadings, set to zero here to show it.
    """
    weights = pd.DataFrame([[0.6, 0.0], [0.8, 0.6], [0.0, 0.8]], index=["a", "b", "c"], columns=[1, 2])
    r2 = pd.Series([0.5, 0.25], index=[1, 2])
    expected = pd.Series(np.sqrt([0.72, 1.64, 0.64]), index=["a", "b", "c"], name="VIP")
    pca_like = SimpleNamespace(r2_per_component_=r2, loadings_=weights)
    pls_like = SimpleNamespace(r2_per_component_=r2, x_weights_=weights, loadings_=0.0 * weights)
    for model in (pca_like, pls_like):
        pd.testing.assert_series_equal(vip(model), expected, rtol=1e-14)
        pd.testing.assert_series_equal(vip(model, n_components=2), expected, rtol=1e-14)
        np.testing.assert_allclose(vip(model, n_components=1), np.sqrt([1.08, 1.92, 0.0]), rtol=1e-14)
        with pytest.raises(ValueError, match=r"^n_components must be between 1 and 2, got 3\.$"):
            vip(model, n_components=3)


def test_response_selector_counts_positions_from_0_and_refuses_in_full() -> None:
    """An integer picks a response by position, 0 to M - 1; anything else is refused with the reason."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.standard_normal((20, 3)), columns=["x0", "x1", "x2"])
    Y = pd.DataFrame({"y0": X["x0"] + 0.1 * rng.standard_normal(20), "y1": X["x1"] - X["x2"]})
    pls = PLS(n_components=2).fit(X, Y)
    assert target_projection(pls, X, response=0).response == "y0"
    assert target_projection(pls, X, response=np.int64(1)).response == "y1"
    for position in (2, -1):
        with pytest.raises(ValueError, match=rf"^response index {position} is out of range for 2 responses\.$"):
            target_projection(pls, X, response=position)
    with pytest.raises(ValueError, match=r"^response 'y9' is not one of the model responses \['y0', 'y1'\]\.$"):
        target_projection(pls, X, response="y9")


def _two_feature_tp_model(direct_weights: list[float]) -> SimpleNamespace:
    """Return a one-response PLS stand-in whose regression vector is ``direct_weights``; it has no X scaler."""
    return SimpleNamespace(
        beta_coefficients_=pd.DataFrame({"y": [0.0, 0.0]}, index=["a", "b"]),
        direct_weights_=pd.DataFrame([[weight] for weight in direct_weights], index=["a", "b"]),
        y_loadings_=pd.DataFrame([[1.0]], index=["y"]),
    )


_TP_X = pd.DataFrame(
    [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0], [2.0, -1.0]], index=["r1", "r2", "r3", "r4"], columns=["a", "b"]
)


def test_target_projection_and_selectivity_ratio_by_hand() -> None:
    """``w = b / |b|``, ``t = X w``, ``p = X't / t't`` and ``SR_j = p_j^2 t't / |x_j - t p_j|^2``.

    The regression vector ``W* q'`` is (3, 4), so ``w = (0.6, 0.8)``. The four rows give
    ``t = (0.6, 0.8, -1.4, 0.4)``, ``t't = 3.12`` and ``p = (2.8, 1.8) / 3.12``. The columns'
    sums of squares are 6 and 3, so the residual ones are ``6 - 2.8^2 / 3.12`` and
    ``3 - 1.8^2 / 3.12``, and the ratios 49/68 and 9/17. With N = 4 the F critical value has 2
    and 1 degrees of freedom, where ``1 - (1 + 2F)^(-1/2) = 0.9`` gives F = 49.5.
    """
    model = _two_feature_tp_model([3.0, 4.0])
    tp = target_projection(model, _TP_X)
    pd.testing.assert_series_equal(tp.weights, pd.Series([0.6, 0.8], index=["a", "b"], name="TP weight"), rtol=1e-15)
    expected_scores = pd.Series([0.6, 0.8, -1.4, 0.4], index=_TP_X.index, name="TP score")
    pd.testing.assert_series_equal(tp.scores, expected_scores, rtol=1e-14)
    expected_loadings = pd.Series([2.8 / 3.12, 1.8 / 3.12], index=["a", "b"], name="TP loading")
    pd.testing.assert_series_equal(tp.loadings, expected_loadings, rtol=1e-14)
    assert tp.response == "y"

    ratio = selectivity_ratio(model, _TP_X, conf_level=0.9)
    pd.testing.assert_series_equal(
        ratio, pd.Series([49 / 68, 9 / 17], index=["a", "b"], name="selectivity_ratio"), rtol=1e-13
    )
    assert ratio.attrs["f_critical"] == pytest.approx(49.5, rel=1e-12)
    assert ratio.attrs["conf_level"] == 0.9
    assert ratio.attrs["response"] == "y"


def test_target_projection_reads_object_data_as_numbers() -> None:
    """X held as Python objects gives the float results, with no scaler to convert it first."""
    model = _two_feature_tp_model([3.0, 4.0])
    as_objects = _TP_X.astype(object)
    pd.testing.assert_series_equal(
        target_projection(model, as_objects).loadings, target_projection(model, _TP_X).loadings
    )
    pd.testing.assert_series_equal(selectivity_ratio(model, as_objects), selectivity_ratio(model, _TP_X))


def test_target_projection_counts_a_norm_of_exactly_epsqrt_as_0() -> None:
    """A regression vector, or TP scores, of norm exactly epsqrt (``2**-26``) count as ~0."""
    assert epsqrt == 2.0**-26
    message = "The regression vector for response 'y' is ~0; it is not predicted by X."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        target_projection(_two_feature_tp_model([2.0**-26, 0.0]), _TP_X)
    flat = pd.DataFrame([[2.0**-13, 5.0], [0.0, 3.0]], columns=["a", "b"])  # scores (2**-13, 0)
    message = "The target-projected scores have ~0 variance; cannot form the TP loading."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        target_projection(_two_feature_tp_model([1.0, 0.0]), flat)


def test_selectivity_ratio_is_0_without_a_warning_where_it_cannot_be_formed() -> None:
    """A feature the TP component reproduces exactly (x / 0) and an all-zero one (0 / 0) both get 0, quietly.

    With ``w = (1, 0)`` the scores are column a itself, so its loading is 1 and its residual 0.
    Three rows leave the F distribution 0 denominator degrees of freedom: its critical value is NaN.
    """
    X = pd.DataFrame({"a": [1.0, -1.0, 2.0], "b": [0.0, 0.0, 0.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ratio = selectivity_ratio(_two_feature_tp_model([1.0, 0.0]), X)
    np.testing.assert_array_equal(ratio.to_numpy(), [0.0, 0.0])
    assert np.isnan(ratio.attrs["f_critical"])


def test_squared_cosine_and_observation_contributions_by_hand() -> None:
    """Squared scores over each row's total (scores and SPE squared), and over each column's total.

    Row p has scores (3, 4, 0) and an SPE of 5, a total of 50; row q sits at the origin with no
    residual (0 / 0, read as 0); row r has scores (1, 0, 0) and no residual. Only the last SPE
    column, the full model's, counts. The columns' sums of squares are 10, 16 and 0 (0 / 0 again).
    """
    index, columns = ["p", "q", "r"], [1, 2, 3]
    scores = pd.DataFrame([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], index=index, columns=columns)
    spe = pd.DataFrame([[9.0, 9.0, 5.0], [9.0, 9.0, 0.0], [9.0, 9.0, 0.0]], index=index, columns=columns)
    model = SimpleNamespace(scores_=scores, spe_=spe)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cos2, shares = squared_cosine(model), observation_contributions(model)
    expected_cos2 = pd.DataFrame([[0.18, 0.32, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], index=index, columns=columns)
    expected_shares = pd.DataFrame([[0.9, 1.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], index=index, columns=columns)
    for function, result, expected in (
        (squared_cosine, cos2, expected_cos2),
        (observation_contributions, shares, expected_shares),
    ):
        pd.testing.assert_frame_equal(result, expected)
        pd.testing.assert_frame_equal(function(model, n_components=1), expected[[1]])
        pd.testing.assert_frame_equal(function(model, n_components=3), expected)
        for wrong in (0, 4):
            with pytest.raises(ValueError, match=rf"^n_components must be between 1 and 3, got {wrong}\.$"):
                function(model, n_components=wrong)


def test_eigenvalue_summary_by_hand() -> None:
    """The score variances, and r2 per component and cumulated, in percent, indexed by component."""
    model = SimpleNamespace(
        explained_variance_=np.array([4.0, 1.0]),
        r2_per_component_=pd.Series([0.5, 0.25], index=[1, 2]),
        r2_cumulative_=pd.Series([0.5, 0.75], index=[1, 2]),
    )
    expected = pd.DataFrame(
        {"eigenvalue": [4.0, 1.0], "percent_variance": [50.0, 25.0], "cumulative_percent": [50.0, 75.0]},
        index=pd.Index([1, 2], name="component"),
    )
    pd.testing.assert_frame_equal(eigenvalue_summary(model), expected)


def test_project_variables_are_the_correlations_np_corrcoef_gives() -> None:
    """Each entry is the Pearson correlation of a supplementary column with a score column.

    Neither block is centred (the stand-in's scores have means 3 and 1, the columns 12 and
    1.25), so the centring counts. A constant column (0 / 0) and one too small to square (its
    sum of squares underflows to 0, so x / 0) both read as uncorrelated, without a warning.
    """
    scores = pd.DataFrame([[1.0, 2.0], [2.0, 0.0], [3.0, 1.0], [6.0, 1.0]], columns=[1, 2])
    model = SimpleNamespace(scores_=scores)
    supplementary = pd.DataFrame(
        {"u": [10.0, 12.0, 11.0, 15.0], "v": [-1.0, 0.0, 4.0, 2.0], "flat": 7.0, "tiny": [1e-170, -1e-170] * 2}
    )
    expected = pd.DataFrame(0.0, index=supplementary.columns, columns=scores.columns)
    for name in ("u", "v"):
        for component in (1, 2):
            expected.loc[name, component] = np.corrcoef(supplementary[name], scores[component])[0, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pd.testing.assert_frame_equal(project_variables(model, supplementary), expected, rtol=1e-13)
    pd.testing.assert_frame_equal(project_variables(model, supplementary.astype(object)), expected, rtol=1e-13)
    message = "Supplementary data must have 4 rows (the number of observations used to fit the model), got 3."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        project_variables(model, supplementary.iloc[:3])


def test_rv_and_rv2_by_hand() -> None:
    """Cross-product matrices worked by hand.

    For x = (1, 0, -1) and y = (1, -1, 0), XX' and YY' share only their (1, 1) entry, 1, and
    each has a sum of squares of 4: RV = 1/4. Off their diagonals they share nothing: RV2 = 0.
    For x = (0.5, -0.5) and y = (1, -1) the denominator is exactly 1, and RV is 1.
    """
    assert rv_coefficient([[1.0], [0.0], [-1.0]], [[1.0], [-1.0], [0.0]]) == 0.25
    assert rv2_coefficient([[1.0], [0.0], [-1.0]], [[1.0], [-1.0], [0.0]]) == 0.0
    assert rv_coefficient([[0.5], [-0.5]], [[1.0], [-1.0]]) == 1.0


def test_rv_computes_float32_data_in_double_precision() -> None:
    """Single-precision blocks give the double-precision coefficient, not one rounded at 1e-7."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((30, 5)).astype(np.float32)
    Y = rng.standard_normal((30, 3)).astype(np.float32)
    for coefficient in (rv_coefficient, rv2_coefficient):
        assert coefficient(X, Y) == pytest.approx(coefficient(X.astype(float), Y.astype(float)), rel=1e-14)


def test_rv_refuses_its_inputs_in_full() -> None:
    """The messages name the block and both row counts."""
    with pytest.raises(ValueError, match=r"^X must be 1- or 2-dimensional, got 3 dimensions\.$"):
        rv_coefficient(np.zeros((2, 2, 2)), np.zeros((2, 1)))
    with pytest.raises(ValueError, match=r"^Y must be 1- or 2-dimensional, got 3 dimensions\.$"):
        rv2_coefficient(np.zeros((2, 1)), np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match=r"^X and Y must have the same number of rows; got 4 and 3\.$"):
        rv_coefficient(np.ones((4, 2)), np.ones((3, 5)))


def _identity_model(n_features: int, **attributes: object) -> SimpleNamespace:
    """Return a PCA stand-in whose loadings are the identity, so each score is its own variable."""
    names = list("abcdefgh"[:n_features])
    return SimpleNamespace(scores_=None, loadings_=pd.DataFrame(np.eye(n_features), index=names), **attributes)


def test_score_and_group_contributions_by_hand() -> None:
    """With the identity as loadings, a contribution to score a is the datum on variable a, and 0 elsewhere.

    The default component is 1, and a 2-D array is data, not a score vector. Rows r1 and r3
    average (2, 3), and against r2 = (0, 1) shift by (2, 2), which component 2 reads as (0, 2).
    Data and weights held as Python objects give the float results.
    """
    model = _identity_model(2)
    X = pd.DataFrame([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0]], index=["r1", "r2", "r3"], columns=["a", "b"])
    expected = pd.DataFrame([[1.0, 0.0], [0.0, 0.0], [3.0, 0.0]], index=X.index, columns=X.columns)
    pd.testing.assert_frame_equal(score_contributions(model, X), expected)
    pd.testing.assert_frame_equal(score_contributions(model, X.to_numpy()), expected.reset_index(drop=True))
    pd.testing.assert_frame_equal(score_contributions(model, X.astype(object)), expected)

    shift = group_contributions(model, X, group=["r1", "r3"], reference=["r2"], component=2)
    pd.testing.assert_series_equal(shift, pd.Series([0.0, 2.0], index=["a", "b"], name="group_contributions"))
    weights = pd.Series([0.5, -1.0, 0.5], dtype=object)
    pd.testing.assert_series_equal(group_contributions(model, X.astype(object), weights=weights, component=2), shift)


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda model, X: group_contributions(model, X, group=["r1"], weights=[1.0, 0.0, 0.0]),
            ValueError,
            "Pass either weights, or group/reference, not both.",
        ),
        (group_contributions, ValueError, "Pass group (optionally with reference), or weights."),
        (
            lambda model, X: group_contributions(model, X, group=[]),
            ValueError,
            "group selected no observations from X.",
        ),
        (
            lambda model, X: group_contributions(model, X, group=["r1"], reference=[]),
            ValueError,
            "reference selected no observations from X.",
        ),
        (
            lambda model, X: group_contributions(model, X, group=[True, False]),
            ValueError,
            "group is a boolean mask of length 2, but X has 3 rows.",
        ),
        (
            lambda model, X: group_contributions(model, X, group=["r1"], reference=["r9", "r8"]),
            ValueError,
            (
                "reference contains entries that are not index labels of X: ['r9', 'r8']. Selection is by index "
                "label or boolean mask; to select by position, pass X.index[...] instead."
            ),
        ),
        (
            lambda model, X: score_contributions(model, X, colour="red", size=3),
            TypeError,
            "score_contributions() got an unexpected keyword argument: colour, size.",
        ),
    ],
)
def test_contribution_arguments_are_refused_in_full(call: Callable, error: type, message: str) -> None:
    """Each refusal says exactly what was wrong."""
    X = pd.DataFrame([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0]], index=["r1", "r2", "r3"], columns=["a", "b"])
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        call(_identity_model(2), X)


def test_t2_contributions_by_hand_over_the_components_selected() -> None:
    """With the identity as loadings, the contribution of variable k is ``x_k^2 / s_k^2`` on a selected component.

    For x = (1, 2, 3) and score sds (1, 2, 0.5) that is (1, 1, 36); components are numbered from
    1, and whole floats count as their integers. Data held as Python objects give the same numbers.
    """
    model = _identity_model(3, scaling_factor_for_scores_=pd.Series([1.0, 2.0, 0.5]))
    X = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["a", "b", "c"])
    every = pd.DataFrame([[1.0, 1.0, 36.0]], columns=X.columns)
    pd.testing.assert_frame_equal(t2_contributions(model, X), every)
    pd.testing.assert_frame_equal(t2_contributions(model, X.astype(object)), every)
    pd.testing.assert_frame_equal(
        t2_contributions(model, X, components=[2]), pd.DataFrame([[0.0, 1.0, 0.0]], columns=X.columns)
    )
    pd.testing.assert_frame_equal(
        t2_contributions(model, X, components=np.array([1.0, 3.0])), pd.DataFrame([[1.0, 0.0, 36.0]], columns=X.columns)
    )
    pd.testing.assert_frame_equal(spe_contributions(model, X.astype(object)), 0.0 * X)
    for wrong in ([], [0, 1], [4]):
        message = f"components must be 1-based indices within 1..3, got {wrong}."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            t2_contributions(model, X, components=wrong)


def test_contribution_scalings_leave_a_largest_or_total_of_at_most_epsqrt_alone() -> None:
    """``"maximum"`` and ``"within"`` divide by 1 when the largest contribution, or a row's total, is at most epsqrt.

    epsqrt is exactly ``2**-26``. Such contributions keep their size, rather than being blown up to 1.
    """
    model, tiny = _identity_model(2), 2.0**-26
    X = pd.DataFrame([[tiny, 5.0], [-tiny / 2, 7.0]], columns=["a", "b"])
    expected = pd.DataFrame([[tiny, 0.0], [-tiny / 2, 0.0]], columns=["a", "b"])
    pd.testing.assert_frame_equal(score_contributions(model, X, scaling="maximum"), expected, check_exact=True)
    rows = pd.DataFrame([[tiny, 5.0], [tiny / 2, 7.0], [0.0, 3.0], [3.0, 1.0]], columns=["a", "b"])
    expected = pd.DataFrame([[tiny, 0.0], [tiny / 2, 0.0], [0.0, 0.0], [1.0, 0.0]], columns=["a", "b"])
    pd.testing.assert_frame_equal(score_contributions(model, rows, scaling="within"), expected, check_exact=True)


# --------------------------------------------------------------------------
# The cell-wise cross-validation schemes added in 1.84: "ek", "sacv", "gcv".
# --------------------------------------------------------------------------


def _known_rank_block(seed: int = 0, n: int = 100, k: int = 12, rank: int = 3) -> pd.DataFrame:
    """Build a block whose true rank is known, so a criterion can be judged right or wrong."""
    rng = np.random.default_rng(seed)
    # A fixed, decreasing set of component sizes, sliced to the rank asked for, so
    # that the default rank of three keeps the values the other tests are pinned to.
    spread = np.array([6.0, 4.0, 2.5, 1.8, 1.2])[:rank]
    scores = rng.standard_normal((n, rank)) * spread
    return pd.DataFrame(scores @ rng.standard_normal((rank, k)) + rng.standard_normal((n, k)))


@pytest.mark.parametrize("scheme", ["ek", "sacv", "gcv"])
def test_cell_schemes_return_the_documented_bunch(scheme: str) -> None:
    """Each new scheme fills the same Bunch, and reports no fabricated fold spread."""
    X = _known_rank_block()
    result = PCA.select_n_components(X, max_components=6, cv=5, cv_scheme=scheme, random_state=0)

    assert result.cv_scheme == scheme
    assert list(result.q2.index) == list(range(1, 7))
    assert result.q2_per_variable.shape == (6, X.shape[1])
    # Nothing is held out in folds that could disagree, so the spread is absent
    # rather than zero: a zero would let the 1-SE rule read a false certainty.
    assert result.per_fold_press.isna().to_numpy().all()
    assert np.isnan(result.se_press.to_numpy()).all()


def test_sacv_tracks_the_element_wise_scheme_it_approximates() -> None:
    """The leverage correction stands in for holding cells out, over the range that is read."""
    X = _known_rank_block()
    ekf = PCA.select_n_components(X, max_components=8, cv=7, n_repeats=10, random_state=0)
    sacv = PCA.select_n_components(X, max_components=8, cv_scheme="sacv", random_state=0)

    # Same answer, for one decomposition instead of 560.
    assert int(np.nanargmax(sacv.q2.to_numpy())) == int(np.nanargmax(ekf.q2.to_numpy())) == 2
    # And the same curve, within five points, through the optimum and past it.
    assert np.nanmax(np.abs(sacv.q2.to_numpy()[:5] - ekf.q2.to_numpy()[:5])) < 0.05


def test_sacv_degrades_as_the_components_approach_the_variables() -> None:
    """The trap in the leverage correction, pinned so nobody meets it unawares.

    A column's leverage is the share of its variance inside the retained
    subspace, and it reaches one when the components reach the variables. The
    divisor goes to zero with it, so the corrected residual blows up. On a
    12-variable block the approximation is faithful to about half that many
    components and unusable near the top.
    """
    X = _known_rank_block(n=100, k=12)
    ekf = PCA.select_n_components(X, max_components=8, cv=7, n_repeats=10, random_state=0).q2.to_numpy()
    sacv = PCA.select_n_components(X, max_components=8, cv_scheme="sacv", random_state=0).q2.to_numpy()

    assert np.max(np.abs(sacv[:6] - ekf[:6])) < 0.05
    assert np.abs(sacv[7] - ekf[7]) > 0.5


def test_cell_schemes_find_the_true_rank() -> None:
    """The point of all of this: an interior optimum exists, and the cell schemes find it."""
    X = _known_rank_block()
    for scheme in ("ekf", "sacv"):
        kwargs = {"n_repeats": 10} if scheme == "ekf" else {}
        result = PCA.select_n_components(X, max_components=11, cv=7, cv_scheme=scheme, random_state=0, **kwargs)
        assert int(np.nanargmax(result.q2.to_numpy())) + 1 == 3, scheme


def test_ek_never_lets_a_cell_reach_the_model_that_predicts_it() -> None:
    """Eastment-Krzanowski's defining property, checked rather than assumed.

    A cell sits in one row group and one column group. The scores come from a
    decomposition that dropped its column, so perturbing the cell cannot move
    them. Preprocessing is fitted on the whole matrix, as in the classical
    scheme, so the loadings shift only through its column's centre and spread.
    """
    from process_improve.multivariate._pca import _preprocess_for_cell_schemes

    rng = np.random.default_rng(1)
    n, k = 40, 8
    X = (rng.standard_normal((n, 3)) * np.array([5.0, 3.0, 1.5])) @ rng.standard_normal((3, k))
    X += rng.standard_normal((n, k))

    kept_columns = np.setdiff1d(np.arange(k), np.arange(0, k, 2))
    scores_before, _, _ = np.linalg.svd(_preprocess_for_cell_schemes(X, "ek")[:, kept_columns], full_matrices=False)

    moved = X.copy()
    moved[0, 0] += 1000.0  # a cell inside the dropped column group
    scores_after, _, _ = np.linalg.svd(_preprocess_for_cell_schemes(moved, "ek")[:, kept_columns], full_matrices=False)
    assert np.abs(np.abs(scores_before) - np.abs(scores_after)).max() < 1e-12


def test_cell_schemes_refuse_a_block_with_missing_cells() -> None:
    """They factorise X directly, so they must say so rather than fail inside an SVD."""
    X = _known_rank_block().copy()
    X.iloc[0, 0] = np.nan
    for scheme in ("ek", "sacv", "gcv"):
        with pytest.raises(ValueError, match="missing cells"):
            PCA.select_n_components(X, max_components=3, cv=5, cv_scheme=scheme)
    # The default scheme does take it, which is the reason to reach for it.
    assert PCA.select_n_components(X, max_components=3, cv=5, random_state=0).q2.notna().all()


def test_the_missing_cell_refusal_says_how_many_and_where() -> None:
    """Where the gaps sit decides what to do about them, so the message says.

    A block whose misses are all in one column is repaired by dropping that
    column, keeping every row; one whose misses are scattered is not. Both were
    the choice faced on a real quality block, so the error names which it is.
    """
    X = _known_rank_block(n=40, k=6).copy()
    X.iloc[:9, 2] = np.nan
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"10 of 240, 9 of them in column 2"):
        PCA.select_n_components(X, max_components=3, cv=5, cv_scheme="sacv")

    scattered = _known_rank_block(n=40, k=6).copy()
    scattered.iloc[0, 0] = scattered.iloc[1, 3] = scattered.iloc[2, 5] = np.nan
    with pytest.raises(ValueError, match=r"3 of 240, spread over 3 columns"):
        PCA.select_n_components(scattered, max_components=3, cv=5, cv_scheme="ek")


def test_ek_warns_when_the_folds_cannot_reach_the_requested_components() -> None:
    """Each fold model is fitted on fewer rows or columns, so it saturates early."""
    X = _known_rank_block(n=30, k=8)
    with pytest.warns(SpecificationWarning, match="can evaluate at most"):
        result = PCA.select_n_components(X, max_components=7, cv=7, cv_scheme="ek", random_state=0)
    assert result.q2.isna().any()


def test_row_wise_was_removed() -> None:
    """Deprecated since 1.84 and removed in 2.0: the error lists the schemes that remain."""
    X = _known_rank_block(n=40, k=6)
    listed = r"expected one of 'ekf', 'ckf', 'ek', 'sacv' or 'gcv'\. 'row_wise' was removed in 2\.0"
    with pytest.raises(ValueError, match=listed):
        PCA.select_n_components(X, max_components=4, cv=5, cv_scheme="row_wise", random_state=0)  # type: ignore[arg-type]


def test_sacv_reports_nothing_rather_than_a_perfect_score_at_the_top() -> None:
    """The counterpart of the generalised criterion's guard, on the smoothing one.

    Once the components reach the variables every column leverage is one, so
    every cell is dropped as undefined. Totalling an empty set gives zero error,
    which reads as a flawless model and wins the selection outright. It must
    come back as NaN, the way the generalised criterion already does.
    """
    n, p = 40, 5
    X = _known_rank_block(n=n, k=p, rank=2)
    result = PCA.select_n_components(X, max_components=p, cv_scheme="sacv", random_state=0)
    q2 = result.q2.to_numpy()

    assert np.isnan(q2[p - 1]), "the component count that equals the variable count means nothing"
    assert np.isfinite(q2[: p - 1]).all()
    assert int(result.n_components) < p


def test_leverage_criterion_scores_the_same_cells_it_divides_by() -> None:
    """A dropped cell leaves both totals, not just the error.

    A cell whose leverage reaches one is not scored, so its share of the block's
    variance must leave the reference too. Otherwise the criterion is an error
    over part of the block against a null over all of it, which flatters the
    model in proportion to how many cells were dropped.
    """
    n, p, components = 30, 8, 4
    X = _known_rank_block(n=n, k=p, rank=3)
    Z = MCUVScaler().fit_transform(X).to_numpy()
    press, per_column, null_ss, per_column_null = _leverage_corrected_press(Z, components, method="sacv")

    # Nothing is dropped at this shape, so the rescaling must be the identity and
    # the totals must agree with a direct calculation of the same quantity.
    assert np.isclose(null_ss, float(np.sum(Z**2)))
    assert np.allclose(per_column_null, np.sum(Z**2, axis=0))
    assert np.allclose(press, per_column.sum(axis=1))
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    for a in range(1, components + 1):
        residual = Z - (U[:, :a] * S[:a]) @ Vt[:a]
        denominator = np.outer(1.0 - 1.0 / n - np.sum(U[:, :a] ** 2, axis=1), 1.0 - np.sum(Vt[:a] ** 2, axis=0))
        assert (denominator > 0).all(), "pick a shape where no cell is dropped"
        assert np.isclose(press[a - 1], float(np.sum((residual / denominator) ** 2)))


def test_the_no_optimum_warning_names_a_scheme_other_than_the_one_in_use() -> None:
    """Told that a criterion never turned over, the reader needs somewhere to go.

    The remedy used to name both element-wise schemes whatever was running, so a
    caller already using one was pointed back at it.
    """
    X = _known_rank_block(n=60, k=8)
    for scheme, expected, forbidden in (("ekf", "'ek'", "cv_scheme='ekf' or"), ("ek", "'ekf'", "or cv_scheme='ek'")):
        with pytest.warns(SpecificationWarning, match="did not turn over") as caught:
            PCA.select_n_components(X, max_components=3, cv=5, cv_scheme=scheme, random_state=0)
        message = next(str(w.message) for w in caught if "did not turn over" in str(w.message))
        assert expected in message
        assert forbidden not in message


def test_gcv_reports_nothing_once_the_parameters_outnumber_the_data() -> None:
    """The guard on the generalised criterion, exercised rather than assumed.

    A rank-``a`` bilinear model spends ``a(n + p - a - 1)`` free parameters. Once
    that reaches the ``(n - 1)p`` degrees of freedom available, the inflation
    factor has a non-positive denominator and the criterion means nothing. It
    must come back as NaN rather than as a number with the wrong sign.
    """
    n, p = 20, 6
    X = _known_rank_block(n=n, k=p, rank=2)
    result = PCA.select_n_components(X, max_components=p, cv_scheme="gcv", random_state=0)

    spent = np.array([a * (n + p - a - 1) for a in range(1, p + 1)])
    available = (n - 1) * p
    exhausted = spent >= available
    assert exhausted.any(), "pick a shape where the guard actually fires"
    assert np.isnan(result.q2.to_numpy()[exhausted]).all()
    assert np.isfinite(result.q2.to_numpy()[~exhausted]).all()


def test_a_criterion_that_never_turns_over_says_so() -> None:
    """A count that is just the largest one tried is not a recommendation.

    The leverage approximations inflate a residual, and on data whose fit
    approaches one there is almost nothing left in that residual to inflate, so
    the criterion can rise at every component count. That is the same failure
    that makes the row-wise scheme useless, and it has to be visible.
    """
    rng = np.random.default_rng(3)
    n, k = 54, 12
    # Structure at many components and very little noise: R2 reaches ~1 early.
    X = pd.DataFrame(rng.standard_normal((n, 8)) @ rng.standard_normal((8, k)) + 0.01 * rng.standard_normal((n, k)))

    with pytest.warns(SpecificationWarning, match="did not turn over"):
        result = PCA.select_n_components(X, max_components=6, cv_scheme="gcv", random_state=0)
    assert result.n_components == 6

    # The schemes that hold data out are not warned about on data where they do
    # turn over, so the warning is a signal and not noise.
    with warnings.catch_warnings():
        warnings.simplefilter("error", SpecificationWarning)
        PCA.select_n_components(_known_rank_block(), max_components=8, cv=7, n_repeats=5, random_state=0)


class TestPlsMissingDataSettingsResolution:
    """The fit-time resolution of ``missing_data_settings`` (#588).

    Each defect is verified by a test that fails against the previous code.
    """

    @staticmethod
    def _xy(*, gapped: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(0)
        x = pd.DataFrame(rng.standard_normal((30, 5)))
        y = pd.DataFrame(x.to_numpy() @ rng.standard_normal((5, 1)) + 0.1 * rng.standard_normal((30, 1)))
        if gapped:
            x = x.copy()
            x.iloc[0, 0] = np.nan
        return x, y

    def test_tol_reaches_the_nipals_loop_when_data_has_gaps(self) -> None:
        """``PLS(tol=...)`` governs convergence whether or not a cell is missing.

        The missing-data branch used to hard-code ``md_tol=epsqrt`` while taking
        ``md_max_iter`` from the constructor, so ``tol`` applied to complete data and was
        dropped the moment a cell went missing. A loose tolerance must now stop sooner
        than a tight one.
        """
        x, y = self._xy(gapped=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tight = PLS(n_components=2, tol=1e-14, max_iter=500).fit(x, y)
            loose = PLS(n_components=2, tol=5e-1, max_iter=500).fit(x, y)
        assert loose.fitting_info_["iterations"][0] < tight.fitting_info_["iterations"][0]

    @pytest.mark.parametrize("bad", ["scp", "rubbish", "NIPALs "])
    def test_unknown_md_method_is_refused(self, bad: str) -> None:
        """An unrecognised value used to fall through to NIPALS without a word.

        ``"scp"`` is the trap: it is a real method name for ``project()`` and the
        contribution helpers, so asking for it here looked reasonable and quietly ran a
        different algorithm.
        """
        x, y = self._xy(gapped=True)
        with pytest.raises(ValueError, match="md_method must be one of"):
            PLS(n_components=2, missing_data_settings={"md_method": bad}).fit(x, y)

    @pytest.mark.parametrize("method", ["tsr", "pmp"])
    def test_imputing_methods_run_themselves_not_nipals(self, method: str) -> None:
        """``tsr`` and ``pmp`` are built now (#189), so they must not be NIPALS under another name.

        This used to pin the ``NotImplementedError`` they raised while stubbed. What it
        guarded then still matters: asking for a method must run that method. On gapped
        data an imputed fit and NIPALS reach different models, and the fit says which ran.
        """
        x, y = self._xy(gapped=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputed = PLS(n_components=2, missing_data_settings={"md_method": method}).fit(x, y)
            nipals = PLS(n_components=2).fit(x, y)
        assert imputed.fitting_info_["md_method"] == method
        assert imputed.fitting_info_["md_rounds"] >= 1
        assert not np.allclose(imputed.beta_coefficients_.to_numpy(), nipals.beta_coefficients_.to_numpy())

    def test_default_construction_is_unchanged(self) -> None:
        """The defaults resolve to what the old code resolved to.

        ``PLS`` defaults ``tol`` to ``epsqrt``, which is the value the missing-data branch
        used to hard-code, so a model built without arguments fits exactly as before.
        """
        x, y = self._xy(gapped=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PLS(n_components=2).fit(x, y)
            pinned = PLS(n_components=2, tol=epsqrt).fit(x, y)
        np.testing.assert_array_equal(model.fitting_info_["iterations"], pinned.fitting_info_["iterations"])
        np.testing.assert_allclose(model.predictions_.to_numpy(), pinned.predictions_.to_numpy(), rtol=0, atol=0)


class TestPcaIterationParameters:
    """`PCA(tol=..., max_iter=...)`: the loop settings are constructor parameters (#588).

    They used to be reachable only through ``missing_data_settings``, so a caller had to
    describe a convergence setting as a missing-data setting even on complete data. The
    literals that dict was seeded with, ``epsqrt`` and 1000, are now the defaults, so the
    change is additive: these tests pin both halves of that claim.
    """

    @staticmethod
    def _x(gapped: bool = False) -> pd.DataFrame:
        """Build a small, well-conditioned block, optionally with one cell removed."""
        rng = np.random.default_rng(seed=13)
        x = pd.DataFrame(rng.normal(size=(40, 6)), columns=[f"x{i}" for i in range(6)])
        x = MCUVScaler().fit_transform(x)
        if gapped:
            x.iloc[3, 2] = np.nan
        return x

    def test_the_default_fit_is_unchanged(self) -> None:
        """A model built with no arguments must fit exactly as it did before the parameters existed."""
        x = self._x()
        model = PCA(n_components=3, algorithm="nipals").fit(x)
        pinned = PCA(n_components=3, algorithm="nipals", tol=epsqrt, max_iter=1000).fit(x)
        np.testing.assert_array_equal(model.fitting_info_["iterations"], pinned.fitting_info_["iterations"])
        np.testing.assert_array_equal(model.scores_.to_numpy(), pinned.scores_.to_numpy())

    @pytest.mark.parametrize("gapped", [False, True], ids=["complete", "missing"])
    def test_tol_reaches_the_loop(self, gapped: bool) -> None:
        """A looser tolerance must stop the loop sooner, whether or not the data has gaps.

        Both cases are checked because this is exactly where ``PLS`` was broken: it honoured
        ``tol`` on complete data and dropped it as soon as a cell went missing.
        """
        x = self._x(gapped=gapped)
        tight = PCA(n_components=3, algorithm="nipals", tol=epsqrt).fit(x)
        loose = PCA(n_components=3, algorithm="nipals", tol=1e-1).fit(x)
        assert np.sum(loose.fitting_info_["iterations"]) < np.sum(tight.fitting_info_["iterations"])

    def test_max_iter_caps_the_loop_and_says_so(self) -> None:
        """A cap low enough to bite must stop the loop there and warn, not fail silently."""
        x = self._x()
        with pytest.warns(SpecificationWarning, match="without converging"):
            model = PCA(n_components=2, algorithm="nipals", max_iter=2).fit(x)
        np.testing.assert_array_equal(model.fitting_info_["iterations"], [2, 2])

    def test_clone_carries_both_parameters(self) -> None:
        """Sklearn's contract: the constructor parameters survive get_params / clone."""
        model = PCA(n_components=2, tol=1e-3, max_iter=42)
        assert clone(model).get_params()["tol"] == 1e-3
        assert clone(model).get_params()["max_iter"] == 42

    def test_svd_ignores_them_rather_than_refusing(self) -> None:
        """SVD is direct, not iterative, so the two parameters have no loop to govern."""
        x = self._x()
        direct = PCA(n_components=3, algorithm="svd", tol=1e-1, max_iter=3).fit(x)
        plain = PCA(n_components=3, algorithm="svd").fit(x)
        np.testing.assert_array_equal(direct.scores_.to_numpy(), plain.scores_.to_numpy())


class TestTplsToleranceParameter:
    """`TPLS(tol=...)`: the convergence tolerance is a constructor parameter (#588).

    It was a hard-coded ``tolerance_`` attribute assigned in ``__init__`` and read at two
    unrelated sites: the convergence test, and a test for columns with no variance. Those
    are different quantities, so ``tol`` now owns the first and a fixed floor owns the
    second. These tests pin that separation as much as the parameter itself.
    """

    @staticmethod
    def _fit(fixture: dict, **kwargs) -> TPLS:
        d_matrix = fixture["D"]
        blocks = {key: value for key, value in fixture.items() if key != "D"}
        return TPLS(n_components=2, d_matrix=d_matrix, **kwargs).fit(DataFrameDict(blocks))

    def test_the_default_is_unchanged(self, fixture_tpls_example: dict) -> None:
        """``epsqrt`` is the value that was hard-coded, so the default fit does not move."""
        model = self._fit(copy.deepcopy(fixture_tpls_example))
        assert model.tol == epsqrt

    def test_tol_reaches_the_loop(self, fixture_tpls_example: dict) -> None:
        """A looser tolerance stops the per-component loop sooner, a tighter one later."""
        loose = self._fit(copy.deepcopy(fixture_tpls_example), tol=1e-2)
        default = self._fit(copy.deepcopy(fixture_tpls_example))
        tight = self._fit(copy.deepcopy(fixture_tpls_example), tol=1e-12)
        assert sum(loose.fitting_statistics["iterations"]) < sum(default.fitting_statistics["iterations"])
        assert sum(default.fitting_statistics["iterations"]) < sum(tight.fitting_statistics["iterations"])

    @pytest.mark.parametrize("bad", [0.0, -1e-3])
    def test_a_non_positive_tolerance_is_refused(self, fixture_tpls_example: dict, bad: float) -> None:
        """A tolerance of zero or less can never be met, so the loop would always run to the cap."""
        with pytest.raises(ValueError, match="tol must be positive"):
            self._fit(copy.deepcopy(fixture_tpls_example), tol=bad)

    def test_the_old_attribute_is_gone(self) -> None:
        """``tolerance_``, deprecated since 1.96.0, was removed in 2.0: ``tol`` is the only name."""
        model = TPLS(n_components=1, d_matrix={"g": pd.DataFrame({"p": [1.0]})}, tol=1e-5)
        # Checked by name, not read, so a static analyser sees no use of a removed attribute.
        assert not hasattr(model, "tolerance_")

    def test_a_loose_tolerance_does_not_start_discarding_columns(self) -> None:
        """The separation, stated as a test.

        The zero-variance check compares a column's standard deviation against a floor. It
        used to read the same attribute as the convergence test, so raising the tolerance to
        converge sooner also raised the bar a column had to clear to stay in the model. A
        column with a standard deviation of 1e-4 is far above machine precision and carries
        information; a caller asking for ``tol=1e-2`` must not lose it.
        """
        rng = np.random.default_rng(seed=7)
        y = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
        y["c"] = y["c"] * 1e-4  # small, but far above epsqrt and not constant

        model = TPLS(n_components=1, d_matrix={"g": pd.DataFrame({"p": [1.0]})}, tol=1e-2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            centering, scaling = model._learn_center_and_scaling_parameters(y, label="Y")

        assert not scaling.isna().any(), "a column with real variance was dropped as degenerate"
        assert centering.notna().all()


class TestIterationSettingsAreConsistentAcrossEstimators:
    """One contract for `tol` and `max_iter`, checked across every estimator that has them.

    #588 found five spellings of the same two settings. These tests are written over the
    set of estimators rather than one each, so an estimator added later that forgets the
    contract fails here rather than passing quietly.
    """

    #: Every estimator that iterates per component, with a factory that builds one with
    #: the given keyword arguments and fits it. Kept as callables because the fit
    #: signatures differ: PLS and MBPLS need a Y, the multiblock pair take dicts.
    @staticmethod
    def _blocks() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed=21)
        x = pd.DataFrame(rng.normal(size=(30, 5)), columns=[f"x{i}" for i in range(5)])
        y = pd.DataFrame(rng.normal(size=(30, 1)), columns=["y"])
        blocks = {"a": x.iloc[:, :3], "b": x.iloc[:, 3:]}
        return blocks, x, y

    def _fitters(self) -> dict:
        blocks, x, y = self._blocks()
        return {
            "PCA": lambda **kw: PCA(n_components=2, algorithm="nipals", **kw).fit(x),
            "PLS": lambda **kw: PLS(n_components=2, **kw).fit(x, y),
            "MBPCA": lambda **kw: MBPCA(n_components=2, **kw).fit(blocks),
            "MBPLS": lambda **kw: MBPLS(n_components=2, **kw).fit(blocks, y),
        }

    @pytest.mark.parametrize("name", ["PCA", "PLS", "MBPCA", "MBPLS"])
    def test_the_defaults_are_the_same_everywhere(self, name: str) -> None:
        """`tol` is epsqrt and `max_iter` is 500 on every one of them."""
        model = self._fitters()[name]()
        assert model.tol == epsqrt, f"{name}.tol is not epsqrt"
        assert model.max_iter == 500, f"{name}.max_iter is not 500"

    @pytest.mark.parametrize("name", ["PCA", "PLS", "MBPCA", "MBPLS", "OPLS", "TPLS"])
    def test_no_estimator_uses_a_none_sentinel(self, name: str) -> None:
        """A default of `None` would hide the value `get_params` should report.

        `MBPCA` and `MBPLS` used to declare `tol: float | None = None` and substitute
        `epsqrt` inside `fit`, so a caller inspecting a cloned estimator saw `None` and
        could not tell what the fit would use.
        """
        classes = {"PCA": PCA, "PLS": PLS, "MBPCA": MBPCA, "MBPLS": MBPLS, "OPLS": OPLS, "TPLS": TPLS}
        signature = inspect.signature(classes[name].__init__)
        for parameter_name in ("tol", "max_iter"):
            parameter = signature.parameters.get(parameter_name)
            if parameter is None:  # OPLS is closed form and has neither
                continue
            assert parameter.default is not None, f"{name}.{parameter_name} still defaults to None"
            assert isinstance(parameter.default, (int, float)), (
                f"{name}.{parameter_name} default is {parameter.default!r}, not a number"
            )

    @pytest.mark.parametrize("name", ["PCA", "PLS", "MBPCA", "MBPLS"])
    def test_max_iter_caps_the_loop_on_every_estimator(self, name: str) -> None:
        """A cap low enough to bite stops the loop there, and says so."""
        with pytest.warns(SpecificationWarning):
            model = self._fitters()[name](max_iter=2)
        iterations = np.asarray(model.fitting_info_["iterations"], dtype=float)
        assert np.all(iterations <= 2), f"{name} ran past max_iter: {iterations}"

    @pytest.mark.parametrize("name", ["PCA", "PLS", "MBPCA", "MBPLS"])
    def test_tol_reaches_the_loop_on_every_estimator(self, name: str) -> None:
        """A looser tolerance converges in no more iterations than a tighter one.

        Stated as "no more" rather than "strictly fewer" because an estimator whose
        default already converges in two iterations on this data has no room to improve;
        the strict version is checked per estimator elsewhere.
        """
        fit = self._fitters()[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpecificationWarning)
            loose = np.sum(np.asarray(fit(tol=1e-1).fitting_info_["iterations"], dtype=float))
            tight = np.sum(np.asarray(fit(tol=1e-10).fitting_info_["iterations"], dtype=float))
        assert loose <= tight, f"{name}: tol=1e-1 took {loose} iterations, tol=1e-10 took {tight}"

    @pytest.mark.parametrize(("key", "replacement"), [("md_tol", "tol"), ("md_max_iter", "max_iter")])
    @pytest.mark.parametrize("name", ["PCA", "PLS", "MBPCA", "MBPLS"])
    def test_the_removed_md_keys_are_refused(self, name: str, key: str, replacement: str) -> None:
        """Removed in 2.0, and refused rather than ignored: ignoring a convergence setting changes the fit."""
        value = 1e-1 if key == "md_tol" else 2
        message = rf"{name}\(missing_data_settings=\.\.\.\) no longer takes '{key}'.*Pass {replacement}="
        with pytest.raises(ValueError, match=message):
            self._fitters()[name](missing_data_settings={key: value})

    @pytest.mark.parametrize("name", ["PCA", "PLS", "MBPCA", "MBPLS"])
    def test_a_default_fit_does_not_warn(self, name: str) -> None:
        """None of the deprecations fire for a caller who passes nothing."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter("ignore", SpecificationWarning)
            self._fitters()[name]()


class TestEkfLoopKnobsWereRenamed:
    """`select_n_components` took `n_iter` and `tol` for the ekf EM loop (#588).

    `tol` in particular collided once `PCA` itself gained one, so the name meant the EM
    loop here and the per-component loop everywhere else. The old names were removed in
    2.0; `test_pca_select_n_components_refuses_the_removed_keywords` pins that.
    """

    @staticmethod
    def _x() -> pd.DataFrame:
        rng = np.random.default_rng(seed=4)
        return pd.DataFrame(rng.normal(size=(30, 5)))

    def test_the_new_names_are_silent(self) -> None:
        """No deprecation fires for a caller who has moved to `ekf_max_iter` / `ekf_tol`."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            PCA.select_n_components(self._x(), max_components=3, cv=3, ekf_max_iter=5, ekf_tol=1e-4, random_state=0)

    def test_the_ekf_cap_actually_bounds_the_em_loop(self) -> None:
        """Otherwise the rename would be cosmetic: the parameter has to reach the loop.

        A cap of one EM iteration leaves the imputation far from converged, so the
        cross-validated Q-squared differs from a well-converged run. Equality would mean
        the value is being ignored.
        """
        x = self._x()
        stingy = PCA.select_n_components(x, max_components=3, cv=3, ekf_max_iter=1, random_state=0)
        patient = PCA.select_n_components(x, max_components=3, cv=3, ekf_max_iter=50, random_state=0)
        assert not np.allclose(stingy.q2.to_numpy(), patient.q2.to_numpy()), "ekf_max_iter did not reach the EM loop"


# ---- PCA worked by hand: the estimator's API, its three fitting paths and its diagnostics (#1026) ----


def _named_block(seed: int = 0, n: int = 12) -> pd.DataFrame:
    """Return ``n`` centred rows of three named, correlated columns, labelled r0, r1, ..."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3)) @ np.array([[2.0, 1.0, 0.5], [0.0, 1.0, -1.0], [0.0, 0.0, 0.3]])
    X -= X.mean(axis=0)
    return pd.DataFrame(X, columns=["a", "b", "c"], index=[f"r{i}" for i in range(n)])


def _svd_reference(X: np.ndarray, A: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the first ``A`` loadings and scores of ``X`` by numpy's SVD, each loading's largest entry positive."""
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = Vt[:A].T.copy()
    for a in range(A):
        if P[np.argmax(np.abs(P[:, a])), a] < 0:
            P[:, a] = -P[:, a]
    return P, X @ P


def test_pca_constructor_keeps_its_defaults_and_declares_that_it_takes_nan() -> None:
    """The constructor stores what it is given; the estimator tells sklearn that NaN may reach it."""
    assert PCA(n_components=2).get_params() == {
        "n_components": 2,
        "algorithm": "auto",
        "tol": epsqrt,
        "max_iter": 500,
        "missing_data_settings": None,
    }
    assert PCA(n_components=2).__sklearn_tags__().input_tags.allow_nan is True


def test_pca_whose_fit_failed_is_not_fitted() -> None:
    """A fit that raised part way leaves no loadings, and every method that reads them says so.

    The failed fit has set ``n_samples_`` and other attributes ending in ``_`` before it
    refused the missing cell, so a check for any fitted attribute would wrongly pass.
    """
    X = pd.DataFrame([[1.0, 2.0], [np.nan, 1.0], [3.0, 0.5]], columns=["a", "b"])
    pca = PCA(n_components=1, algorithm="svd")
    with pytest.raises(ValueError, match=r"^SVD algorithm cannot handle missing data"):
        pca.fit(X)
    assert pca.n_samples_ == 3
    complete = X.fillna(0.0)
    calls: list[Callable] = [
        lambda: pca.transform(complete),
        lambda: pca.diagnose(complete),
        lambda: pca.project(complete),
        lambda: pca.score(complete),
        lambda: pca.projection_matrix([True, True]),
        pca.get_feature_names_out,
        pca.detect_outliers,
    ]
    for call in calls:
        with pytest.raises(NotFittedError, match="This PCA instance is not fitted yet"):
            call()


def test_pca_reads_new_rows_by_column_name_and_labels_every_result() -> None:
    """New rows with their columns in another order are matched by name, and keep their own row labels.

    Every value is worked from the fitted loadings: ``t = x P``, ``T2 = sum_a t_a^2 / s_a^2``
    cumulated over the components, ``SPE = ||x - t P'||``.
    """
    X = _named_block()
    pca = PCA(n_components=2).fit(X)
    P = pca.loadings_.to_numpy()
    s = pca.scaling_factor_for_scores_.to_numpy()
    new = pd.DataFrame([[0.5, -1.0, 2.0], [1.5, 0.2, -0.4]], columns=["a", "b", "c"], index=["x", "y"])
    reordered = new[["c", "a", "b"]]
    t = new.to_numpy() @ P
    t2 = np.cumsum(t**2 / s**2, axis=1)
    spe = np.linalg.norm(new.to_numpy() - t @ P.T, axis=1)

    expected_scores = pd.DataFrame(t, index=["x", "y"], columns=[1, 2])
    pd.testing.assert_frame_equal(pca.transform(reordered), expected_scores, rtol=1e-12)
    diagnosed = pca.diagnose(reordered)
    pd.testing.assert_frame_equal(diagnosed.scores, expected_scores, rtol=1e-12)
    pd.testing.assert_frame_equal(
        diagnosed.hotellings_t2, pd.DataFrame(t2, index=["x", "y"], columns=[1, 2]), rtol=1e-12
    )
    pd.testing.assert_series_equal(diagnosed.spe, pd.Series(spe, index=["x", "y"], name="SPE"), rtol=1e-12)
    projected = pca.project(reordered)
    pd.testing.assert_frame_equal(projected.scores, expected_scores, rtol=1e-12)
    pd.testing.assert_series_equal(
        projected.hotellings_t2, pd.Series(t2[:, -1], index=["x", "y"], name="Hotelling's T2"), rtol=1e-12
    )
    pd.testing.assert_series_equal(projected.spe, pd.Series(spe, index=["x", "y"], name="SPE"), rtol=1e-12)
    pd.testing.assert_series_equal(
        projected.condition_number, pd.Series([1.0, 1.0], index=["x", "y"], name="condition_number")
    )
    pd.testing.assert_series_equal(projected.n_observed, pd.Series([3, 3], index=["x", "y"], name="n_observed"))
    assert pca.score(reordered) == pytest.approx(-np.mean((new.to_numpy() - t @ P.T) ** 2), rel=1e-12)

    # An array has no labels of its own: its rows are numbered from 0, its columns read in the fitted order.
    for result in (
        pca.transform(new.to_numpy()),
        pca.diagnose(new.to_numpy()).scores,
        pca.project(new.to_numpy()).scores,
    ):
        pd.testing.assert_frame_equal(result, pd.DataFrame(t, columns=[1, 2]), rtol=1e-12)
    assert pca.diagnose(new.to_numpy()).hotellings_t2.index.equals(pd.RangeIndex(2))
    assert pca.diagnose(new.to_numpy()).spe.index.equals(pd.RangeIndex(2))
    for series in ("hotellings_t2", "spe", "condition_number", "n_observed"):
        assert pca.project(new.to_numpy())[series].index.equals(pd.RangeIndex(2))
    np.testing.assert_array_equal(pca.get_feature_names_out(), ["PC1", "PC2"])


def test_pca_score_reads_a_dataframe_by_column_name() -> None:
    """``score`` matches columns by name, as ``transform`` does (#1026).

    It took the scores from ``transform``, which realigns a DataFrame's columns, but the
    data from the DataFrame as given, so columns in another order were subtracted from the
    reconstruction of the wrong variables: -4.84 for a block whose reconstruction error is
    0.82.
    """
    X = _named_block()
    pca = PCA(n_components=1).fit(X)
    P = pca.loadings_.to_numpy()
    residual = X.to_numpy() - X.to_numpy() @ P @ P.T
    for frame in (X, X[["c", "b", "a"]], X[["b", "c", "a"]]):
        assert pca.score(frame) == pytest.approx(-np.mean(residual**2), rel=1e-12)


def test_pca_diagnose_and_score_read_a_pandas_missing_value_as_nan() -> None:
    """A nullable ``Float64`` column's ``pd.NA`` is a missing cell, as ``transform`` and ``project`` read it (#1026).

    ``diagnose`` and ``score`` converted the DataFrame as given with ``float()``, which
    raises TypeError on ``pd.NA``. The rows without one get the values worked by hand.
    """
    X = _named_block()
    pca = PCA(n_components=2).fit(X)
    P = pca.loadings_.to_numpy()
    nullable = X.astype("Float64")
    nullable.iloc[1, 0] = pd.NA
    diagnosed = pca.diagnose(nullable)
    spe = np.linalg.norm(X.to_numpy() - X.to_numpy() @ P @ P.T, axis=1)
    assert np.isnan(diagnosed.spe.iloc[1])
    assert np.isnan(diagnosed.hotellings_t2.iloc[1]).all()
    spe_complete = np.delete(spe, 1)
    np.testing.assert_allclose(diagnosed.spe.drop("r1"), spe_complete, rtol=1e-12)
    assert np.isnan(pca.score(nullable))
    complete = nullable.drop(index="r1")
    assert pca.score(complete) == pytest.approx(-np.mean(spe_complete**2) / 3, rel=1e-12)


def test_pca_refuses_new_rows_that_do_not_fit_the_model() -> None:
    """New rows are checked against the fit, never fitted to: the width, sparse input and infinities.

    A missing cell is allowed through ``transform`` (its score is NaN), but an infinite one
    is not. The expected width stays the fitted one after a refusal.
    """
    X = _named_block()
    pca = PCA(n_components=2).fit(X.to_numpy())
    calls: list[Callable] = [pca.transform, pca.diagnose, pca.project, pca.score]
    for call in calls:
        with pytest.raises(ValueError, match=r"^X has 2 features, but PCA is expecting 3 features as input\.$"):
            call(np.ones((2, 2)))
        message = (
            "Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy "
            "array."
        )
        with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
            call(csr_matrix(np.ones((2, 3))))
        with pytest.raises(ValueError, match=r"^Input X contains infinity"):
            call(np.array([[1.0, np.inf, 0.0]]))
    assert pca.n_features_in_ == 3
    scores = pca.transform(np.array([[1.0, np.nan, 0.0], [1.0, 2.0, 0.0]]))
    assert np.isnan(scores.iloc[0]).all()
    assert np.isfinite(scores.iloc[1]).all()


def test_pca_project_and_projection_matrix_work_without_training_residuals() -> None:
    """A model that keeps no training residuals (one pickled before they were kept) still projects.

    Its estimator for a row with a missing cell is the one ``project_rows`` and
    ``operator_for_pattern`` build when they are given no residuals.
    """
    X = _named_block()
    pca = PCA(n_components=2).fit(X)
    row = np.array([[0.5, np.nan, 2.0]])
    with_residuals = pca.project(row).scores.to_numpy()
    del pca._x_residuals
    loadings = pca.loadings_.to_numpy()
    expected = project_rows(loadings, loadings, pca.explained_variance_, row, method="tsr", x_residuals=None)
    np.testing.assert_allclose(pca.project(row).scores.to_numpy(), expected.scores, rtol=1e-12)
    assert not np.allclose(with_residuals, expected.scores), (
        "the residuals must change the estimate for the test to see them"
    )
    mask = np.array([True, False, True])
    op = operator_for_pattern(loadings, loadings, pca.explained_variance_, mask, method="tsr", x_residuals=None)
    built = pca.projection_matrix(["a", "c"])
    pd.testing.assert_frame_equal(built.matrix, pd.DataFrame(op.matrix, index=[1, 2], columns=["a", "c"]))
    assert built.condition_number == op.condition_number
    assert built.method == "tsr"
    assert pca.projection_matrix(["a", "c"], method="SCP").method == "scp"


def _block_with_a_flat_and_a_small_column() -> pd.DataFrame:
    """Return 8 centred rows: two correlated columns, a column of zeros and one with a sum of squares of 0.5."""
    rng = np.random.default_rng(3)
    big = rng.standard_normal((8, 2)) @ np.array([[3.0, 1.0], [0.0, 2.0]])
    small = rng.standard_normal(8)
    small = (small - small.mean()) * np.sqrt(0.5 / np.sum((small - small.mean()) ** 2))
    X = np.column_stack([big[:, 0], np.zeros(8), small, big[:, 1]])
    return pd.DataFrame(X - X.mean(axis=0), columns=["a", "flat", "small", "d"])


@pytest.mark.parametrize("algorithm", ["svd", "nipals", "tsr"])
def test_pca_fit_paths_by_hand_on_complete_data(algorithm: str) -> None:
    """Each path fits the truncated SVD on complete data, and reports its R2, SPE and variance by hand.

    The loadings are numpy's right singular vectors, signed so the largest entry of each is
    positive; scores ``X P``; explained variance ``diag(T'T) / (N - 1)``; cumulative R2
    ``1 - ||X - T P'||^2 / ||X||^2``, its increments the per-component R2; a column's
    cumulative R2 the same ratio over that column, NaN for the column of zeros, and an
    ordinary value for the column whose sum of squares is only 0.5; SPE the row norms of
    the residual. None of it warns.
    """
    X = _block_with_a_flat_and_a_small_column()
    A = 3
    P, T = _svd_reference(X.to_numpy(), A)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pca = PCA(n_components=A, algorithm=algorithm).fit(X)
    tol = 1e-12 if algorithm != "nipals" else 1e-6
    np.testing.assert_allclose(pca.loadings_.to_numpy(), P, atol=tol)
    np.testing.assert_allclose(pca.scores_.to_numpy(), T, atol=tol * 10)
    np.testing.assert_allclose(pca.explained_variance_, np.sum(T**2, axis=0) / 7, rtol=tol)
    total = np.sum(X.to_numpy() ** 2)
    r2_cumulative = []
    for a in range(1, A + 1):
        residual = X.to_numpy() - T[:, :a] @ P[:, :a].T
        r2_cumulative.append(1 - np.sum(residual**2) / total)
        np.testing.assert_allclose(pca.spe_.iloc[:, a - 1], np.linalg.norm(residual, axis=1), atol=tol * 10)
        kept = [0, 2, 3]
        per_column = 1 - np.sum(residual[:, kept] ** 2, axis=0) / np.sum(X.to_numpy()[:, kept] ** 2, axis=0)
        assert np.isnan(pca.r2_per_variable_.loc["flat", a])
        np.testing.assert_allclose(pca.r2_per_variable_.iloc[kept, a - 1], per_column, atol=tol)
    np.testing.assert_allclose(pca.r2_cumulative_, r2_cumulative, atol=tol)
    np.testing.assert_allclose(pca.r2_per_component_, np.diff(r2_cumulative, prepend=0.0), atol=tol)


def test_pca_fit_paths_on_two_rows_divide_the_variance_by_one() -> None:
    """Two rows have one degree of freedom: the explained variance is ``t't / 1``."""
    X = np.array([[1.0, -2.0, 0.5], [-1.0, 2.0, -0.5]])
    for algorithm in ("svd", "nipals", "tsr"):
        pca = PCA(n_components=1, algorithm=algorithm).fit(X)
        np.testing.assert_allclose(pca.explained_variance_, [np.sum(X**2)], rtol=1e-10)


def test_pca_svd_fit_of_a_block_of_zeros_has_no_r2_and_no_warning() -> None:
    """With nothing to explain, every R2 is NaN rather than a 0/0 warning; T2 and SPE are 0.

    A block whose sum of squares is only 0.02 has its R2 as any other.
    """
    small = _named_block().to_numpy() * 0.01
    assert np.sum(small**2) < 1
    P, T = _svd_reference(small, 1)
    expected = 1 - np.sum((small - T @ P.T) ** 2) / np.sum(small**2)
    assert PCA(n_components=1, algorithm="svd").fit(small).r2_cumulative_.iloc[0] == pytest.approx(expected, rel=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pca = PCA(n_components=2, algorithm="svd").fit(np.zeros((4, 3)))
    assert pca.r2_cumulative_.isna().all()
    assert pca.r2_per_component_.isna().all()
    assert pca.r2_per_variable_.isna().all().all()
    assert (pca.hotellings_t2_.to_numpy() == 0).all()
    assert (pca.spe_.to_numpy() == 0).all()


def test_pca_fitting_info_of_each_path() -> None:
    """SVD records no timing or iterations; NIPALS one of each per component; TSR its EM rounds, 0 on complete data.

    A time is a duration within the fit's own, not a clock reading.
    """
    X = _named_block()
    A = 2
    pca = PCA(n_components=A, algorithm="svd").fit(X)
    assert set(pca.fitting_info_) == {"timing", "iterations"}
    for value in pca.fitting_info_.values():
        assert value.shape == (A,)
        assert np.isnan(value).all()
    started = time.time()
    pca = PCA(n_components=A, algorithm="nipals").fit(X)
    elapsed = time.time() - started
    assert ((pca.fitting_info_["timing"] >= 0) & (pca.fitting_info_["timing"] <= elapsed)).all()
    iterations = pca.fitting_info_["iterations"]
    assert iterations.dtype == float
    assert pca.fitting_info_["timing"].dtype == float
    assert ((iterations >= 1) & (iterations == np.round(iterations))).all()
    started = time.time()
    pca = PCA(n_components=A, algorithm="tsr").fit(X)
    elapsed = time.time() - started
    assert pca.fitting_info_["iterations"] == 0
    assert 0 <= pca.fitting_info_["timing"] <= elapsed


def test_pca_nipals_after_one_iteration_by_hand() -> None:
    """With ``max_iter=1`` NIPALS stops after one pass, which can be worked by hand.

    The seed is the column with the largest sum of squares (the third here, not the first),
    its missing cell set to 0, plus 1. Then ``p_k = sum_i x_ik t_i / sum_i t_i^2`` over the
    rows observed in column k, normalised, and ``t_i = sum_k x_ik p_k / sum_k p_k^2`` over
    the columns observed in row i; the largest loading entry is made positive. The second
    component repeats it on the deflated block. Each component says it stopped at the cap.
    """
    X = np.array(
        [
            [0.5, -1.0, 4.0],
            [-1.0, 0.5, -3.0],
            [1.5, np.nan, 2.0],
            [0.0, 1.0, -2.5],
            [-1.0, -0.5, -0.5],
        ]
    )
    observed = ~np.isnan(X)
    Xd = np.nan_to_num(X)
    loadings, scores = [], []
    for _ in range(2):
        seed = Xd[:, int(np.argmax(np.sum(Xd**2, axis=0)))] + 1.0
        p = (Xd.T @ seed) / (observed.T @ seed**2)
        p /= np.linalg.norm(p)
        t = (Xd @ p) / (observed @ p**2)
        if p[np.argmax(np.abs(p))] < 0:
            p, t = -p, -t
        loadings.append(p)
        scores.append(t)
        Xd = np.where(observed, Xd - np.outer(t, p), 0.0)
    assert np.argmax(np.sum(np.nan_to_num(X) ** 2, axis=0)) == 2
    with pytest.warns(SpecificationWarning) as record:
        pca = PCA(n_components=2, algorithm="nipals", max_iter=1).fit(X)
    assert [str(w.message) for w in record] == [
        f"PCA NIPALS: component {a} reached the maximum number of iterations (1) without converging." for a in (1, 2)
    ]
    np.testing.assert_allclose(pca.loadings_.to_numpy(), np.column_stack(loadings), rtol=1e-12)
    np.testing.assert_allclose(pca.scores_.to_numpy(), np.column_stack(scores), rtol=1e-12)
    np.testing.assert_array_equal(pca.fitting_info_["iterations"], [1, 1])


def test_pca_nipals_seeds_from_the_largest_column_not_the_first() -> None:
    """A first column of zeros, orthogonal to every score, does not stall NIPALS (#195).

    Seeded from it, the score would be a column of ones, which a centred block maps to a
    zero loading. Seeded from the largest column, NIPALS finds the SVD's components.
    """
    X = _block_with_a_flat_and_a_small_column()[["flat", "a", "small", "d"]].to_numpy()
    P, _ = _svd_reference(X, 2)
    pca = PCA(n_components=2, algorithm="nipals").fit(X)
    np.testing.assert_allclose(pca.loadings_.to_numpy(), P, atol=1e-6)


def test_pca_nipals_signs_each_component_whatever_its_seed() -> None:
    """NIPALS converges to the sign its seed points to; the convention then fixes the sign.

    The seed is the third column, the largest, which leans against the first component
    (along the first two columns), so that component converges with its largest entries
    negative and is flipped, loading and scores together, to agree with the SVD's.
    """
    rng = np.random.default_rng(3)
    f = rng.standard_normal(30)
    X = np.column_stack(
        [
            2.2 * f + 0.3 * rng.standard_normal(30),
            2.2 * f + 0.3 * rng.standard_normal(30),
            2.8 * rng.standard_normal(30) - 0.4 * f,
        ]
    )
    X -= X.mean(axis=0)
    assert np.argmax(np.sum(X**2, axis=0)) == 2
    P, T = _svd_reference(X, 2)
    pca = PCA(n_components=2, algorithm="nipals").fit(X)
    np.testing.assert_allclose(pca.loadings_.to_numpy(), P, atol=1e-6)
    np.testing.assert_allclose(pca.scores_.to_numpy(), T, atol=1e-5)


def test_pca_nipals_refines_a_large_seed_at_least_once() -> None:
    """On a column of size 1e9 the ``+ 1`` offset is lost, so the seed passes the convergence test unrefined.

    The first pass runs regardless, so the fitted loading is the SVD's, not zero.
    """
    X = _named_block().to_numpy() * 1e9
    P, _ = _svd_reference(X, 1)
    pca = PCA(n_components=1, algorithm="nipals").fit(X)
    np.testing.assert_allclose(pca.loadings_.to_numpy(), P, atol=1e-6)


def test_pca_nipals_logs_each_component(caplog: pytest.LogCaptureFixture) -> None:
    """Each component's iteration count and the tolerance are logged at DEBUG level."""
    X = _named_block()
    with caplog.at_level(logging.DEBUG, logger="process_improve.multivariate._pca"):
        pca = PCA(n_components=2, algorithm="nipals").fit(X)
    iterations = pca.fitting_info_["iterations"].astype(int)
    assert [record.getMessage() for record in caplog.records] == [
        f"PCA NIPALS: component {a + 1} converged in {iterations[a]} iterations (tol=1.49012e-08)" for a in range(2)
    ]


def test_pca_nipals_needs_a_sum_of_squares_of_at_least_epsqrt() -> None:
    """A block whose sum of squares is exactly epsqrt (``2**-26``) still fits; one below it is refused.

    The single non-zero cell ``2**-13`` gives the loading (1, 0) and the scores (2**-13, 0).
    A second component finds nothing left, and the refusal says after which component.
    """
    X = np.array([[2.0**-13, 0.0], [0.0, 0.0]])
    assert np.sum(X**2) == epsqrt
    pca = PCA(n_components=1, algorithm="nipals").fit(X)
    np.testing.assert_allclose(pca.loadings_.to_numpy(), [[1.0], [0.0]], atol=1e-15)
    np.testing.assert_allclose(pca.scores_.to_numpy(), [[2.0**-13], [0.0]], atol=1e-20)
    message = "There is no variance left in the data array: cannot compute any more components beyond component {}."
    with pytest.raises(NotEnoughVarianceError, match=f"^{re.escape(message.format(0))}$"):
        PCA(n_components=1, algorithm="nipals").fit(X * 0.5)
    with pytest.raises(NotEnoughVarianceError, match=f"^{re.escape(message.format(1))}$"):
        PCA(n_components=2, algorithm="nipals").fit(X)


def _tsr_by_hand(X: np.ndarray, A: int, max_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Trimmed score regression (Arteaga and Ferrer, 2002; Folch-Fortuny et al., 2015), written out.

    Start the missing cells at 0. Each round: take the column means and covariance S of the
    current block, and the first A loadings V of its centred SVD; fill each row's missing
    cells, centred, with ``S_mo V_o (V_o' S_oo V_o)^+ V_o' x_o``, using at most as many
    loadings as the row has observed cells. Stop once the mean squared change of the filled
    cells is at most ``tol``, or after ``max_iter`` rounds. The loadings are then the first
    A eigenvectors of the filled block's covariance, signed so the largest entry is positive,
    and the scores the filled block times them.
    """
    missing = np.isnan(X)
    filled = np.where(missing, 0.0, X)
    rounds = 0
    for rounds in range(1, max_iter + 1):  # noqa: B007 - the count is returned
        before = filled[missing]
        mean = filled.mean(axis=0)
        S = np.cov(filled, rowvar=False)
        centred = filled - mean
        V = np.linalg.svd(centred, full_matrices=False)[2][:A].T
        for i in np.flatnonzero(missing.any(axis=1)):
            o, m = ~missing[i], missing[i]
            L = V[o, : min(A, o.sum())]
            centred[i, m] = S[np.ix_(m, o)] @ L @ np.linalg.pinv(L.T @ S[np.ix_(o, o)] @ L) @ L.T @ centred[i, o]
        filled = centred + mean
        if np.mean((filled[missing] - before) ** 2) <= tol:
            break
    values, vectors = np.linalg.eigh(np.cov(filled, rowvar=False))
    P = vectors[:, np.argsort(values)[::-1][:A]]
    P = P * np.where(P[np.argmax(np.abs(P), axis=0), np.arange(A)] < 0, -1.0, 1.0)
    return P, filled @ P, rounds


@pytest.mark.parametrize(("shape", "max_iter"), [((9, 4), 1), ((9, 4), 3), ((9, 4), 500), ((4, 6), 500)])
def test_pca_tsr_by_hand(shape: tuple[int, int], max_iter: int) -> None:
    """TSR on a block with missing cells, one row with fewer observed cells than components, tall and wide.

    The rounds, loadings, scores, R2 and SPE match the algorithm written out, and the
    reconstruction error of the observed cells only enters R2 and SPE.
    """
    rng = np.random.default_rng(11)
    N, K = shape
    X = rng.standard_normal((N, 2)) @ rng.standard_normal((2, K)) + 0.3 * rng.standard_normal((N, K)) + 2.0
    X[0, 1] = X[2, 3] = X[3, 0] = np.nan
    X[1, : K - 1] = np.nan  # one observed cell, fewer than the two components
    A = 2
    P, T, rounds = _tsr_by_hand(X, A, max_iter, epsqrt)
    pca = PCA(n_components=A, algorithm="tsr", max_iter=max_iter).fit(X)
    assert pca.fitting_info_["iterations"] == rounds
    assert rounds == max_iter or 1 < rounds < max_iter
    np.testing.assert_allclose(pca.loadings_.to_numpy(), P, atol=1e-8)
    np.testing.assert_allclose(pca.scores_.to_numpy(), T, atol=1e-7)
    np.testing.assert_allclose(pca.explained_variance_, np.sum(T**2, axis=0) / (N - 1), rtol=1e-8)
    observed = ~np.isnan(X)
    for a in range(1, A + 1):
        residual = np.where(observed, T[:, :a] @ P[:, :a].T - np.nan_to_num(X), 0.0)
        np.testing.assert_allclose(pca.spe_.iloc[:, a - 1], np.linalg.norm(residual, axis=1), atol=1e-7)
        np.testing.assert_allclose(pca.r2_cumulative_.iloc[a - 1], 1 - np.sum(residual**2) / np.nansum(X**2), atol=1e-8)
        np.testing.assert_allclose(
            pca.r2_per_variable_.iloc[:, a - 1], 1 - np.sum(residual**2, axis=0) / np.nansum(X**2, axis=0), atol=1e-8
        )
    np.testing.assert_allclose(pca.r2_per_component_, np.diff(pca.r2_cumulative_, prepend=0.0), atol=1e-12)


def test_pca_tsr_on_complete_data_runs_no_imputation_and_does_not_warn() -> None:
    """With nothing missing, TSR takes the covariance's eigenvectors at once: no rounds, no warning."""
    X = _block_with_a_flat_and_a_small_column()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pca = PCA(n_components=2, algorithm="tsr").fit(X)
    assert pca.fitting_info_["iterations"] == 0
    np.testing.assert_allclose(pca.loadings_.to_numpy(), _svd_reference(X.to_numpy(), 2)[0], atol=1e-12)


def _ekf_fill(X: np.ndarray, unknown: np.ndarray, a: int, constants: tuple, settings: dict) -> np.ndarray:
    """Fill the ``unknown`` cells from rank-``a`` reconstructions until they settle, as the ekf EM does.

    ``constants`` holds the fold's column centres and spreads.
    """
    centre, spread = constants
    sd = np.broadcast_to(spread, X.shape)[unknown]
    Xa = np.where(unknown, centre, X)
    previous = Xa[unknown] / sd
    for _ in range(settings["max_iter"]):
        if settings["scale"]:
            recon = _low_rank_reconstruction((Xa - centre) / spread, a) * spread + centre
        else:
            mean = Xa.mean(axis=0)
            recon = _low_rank_reconstruction(Xa - mean, a) + mean
        Xa = np.where(unknown, recon, Xa)
        current = Xa[unknown] / sd
        if np.linalg.norm(current - previous) < settings["tol"] * max(1.0, np.linalg.norm(previous)):
            break
        previous = current
    return Xa


def _ekf_by_hand(X: np.ndarray, A: int, *, n_folds: int, n_repeats: int, seed: int, **settings: object) -> tuple:
    """Element-wise k-fold cross-validation (Bro et al., 2008), written out cell by cell.

    Each repeat deals the observed cells, in an order drawn from ``seed``, into ``n_folds``
    folds of ``n_cells // n_folds`` cells, the last fold taking the rest. In a fold, each
    column's centre is the mean of its cells neither held out nor missing, and, when scaling,
    its spread their sd, kept at 1 for fewer than two cells or an sd of at most epsqrt. The
    unknown cells start at the centre and are refilled from the rank-``a`` reconstruction
    (``_low_rank_reconstruction``, which its own test checks against the SVD) until their
    scaled change is below ``tol`` times the norm of their scaled values (at least 1), or
    for ``max_iter`` rounds. PRESS sums the held-out cells' scaled squared errors, the null
    reference their scaled squared deviations from the centre; totals are averaged over
    the repeats. ``settings`` holds ``max_iter``, ``tol`` and ``scale``. Returns the six
    fields of ``_pca_ekf_press`` in its order.
    """
    n, p = X.shape
    rng = np.random.default_rng(seed)
    cells = np.flatnonzero(~np.isnan(X))
    size = cells.size // n_folds
    per_fold = np.full((A, n_folds * n_repeats), np.nan)
    per_column, per_column_null, press_units, null = np.zeros((A, p)), np.zeros(p), np.zeros(A), 0.0
    column = np.broadcast_to(np.arange(p), X.shape)
    for r in range(n_repeats):
        order = cells[rng.permutation(cells.size)]
        for k in range(n_folds):
            held = order[k * size : (k + 1) * size] if k < n_folds - 1 else order[k * size :]
            mask = np.isin(np.arange(n * p), held).reshape(n, p)
            unknown = mask | np.isnan(X)
            known = [X[~unknown[:, j], j] for j in range(p)]
            centre = np.array([cells_j.mean() if cells_j.size else 0.0 for cells_j in known])
            spread = np.array(
                [
                    cells_j.std(ddof=1)
                    if settings["scale"] and cells_j.size > 1 and cells_j.std(ddof=1) > epsqrt
                    else 1.0
                    for cells_j in known
                ]
            )
            null_residual = ((X - centre) / spread)[mask]
            null += np.sum(null_residual**2)
            np.add.at(per_column_null, column[mask], null_residual**2)
            for a in range(1, A + 1) if held.size else ():
                error = (X - _ekf_fill(X, unknown, a, (centre, spread), settings))[mask]
                scaled = error / np.broadcast_to(spread, X.shape)[mask]
                per_fold[a - 1, r * n_folds + k] = np.sum(scaled**2)
                np.add.at(per_column[a - 1], column[mask], scaled**2)
                press_units[a - 1] += np.sum(error**2)
    m = n_repeats
    return np.nansum(per_fold, axis=1) / m, per_fold, per_column / m, null / m, per_column_null / m, press_units / m


def _gappy_block() -> np.ndarray:
    """Return 7 rows of 4 columns with missing cells, a constant column, and a column observed twice."""
    rng = np.random.default_rng(21)
    X = rng.standard_normal((7, 2)) @ rng.standard_normal((2, 4)) * [1.0, 5.0, 1.0, 0.2] + [0.0, 3.0, 1.0, -1.0]
    X[:, 2] = 1.0  # constant
    X[[0, 4], 1] = np.nan
    X[2:, 3] = np.nan  # observed in two rows only
    X[5, 0] = np.nan
    return X


_EKF_CASES = {
    "complete, two repeats": (
        np.random.default_rng(5).standard_normal((8, 4)) @ np.diag([3.0, 1.0, 1.0, 0.5]),
        3,
        3,
        2,
        1e-6,
        True,
    ),
    "gappy, loose tolerance": (_gappy_block(), 3, 4, 1, 0.05, True),
    "gappy, unscaled": (_gappy_block(), 3, 4, 1, 0.05, False),
    "fewer cells than folds": (np.array([[1.0, 2.0], [3.0, 4.5]]), 1, 5, 1, 1e-6, True),
    "an in-fold sd of exactly epsqrt": (
        np.array([[-(2.0**-26), 1.0], [0.0, 2.0], [2.0**-26, 0.5], [1.0, -1.0]]),
        1,
        8,
        1,
        1e-6,
        True,
    ),
}


@pytest.mark.parametrize("case", list(_EKF_CASES))
def test_ekf_press_by_hand(case: str) -> None:
    """Every field of the element-wise k-fold pass agrees with the scheme written out.

    The cases reach each branch: two repeats; a constant column (spread 1), a column with
    one or two cells left in a fold, a fold missing a column altogether; the unscaled
    variant; folds with no cells; and, one cell per fold, a column whose in-fold sd is
    ``2**-26`` exactly, which is not above epsqrt and so is not divided by.
    """
    X, A, n_folds, n_repeats, tol, scale = _EKF_CASES[case]
    settings = {"n_folds": n_folds, "n_repeats": n_repeats, "ekf_max_iter": 50, "ekf_tol": tol}
    result = _pca_ekf_press(X, A, scale_inside_folds=scale, random_state=3, **settings)
    expected = _ekf_by_hand(X, A, n_folds=n_folds, n_repeats=n_repeats, seed=3, max_iter=50, tol=tol, scale=scale)
    for got, want in zip(result, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_ekf_null_reference_without_in_fold_scaling_is_the_centred_sum_of_squares() -> None:
    """Without in-fold scaling, Q2 is measured against each column's sum of squares about its mean.

    Missing cells are left out of both the mean and the sum.
    """
    X = _gappy_block()
    X[:, 2] = [0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 3.0]  # not constant, so every column has a reference
    centred_ss = np.nansum((X - np.nanmean(X, axis=0)) ** 2, axis=0)
    with pytest.warns(SpecificationWarning):
        result = PCA.select_n_components(X, max_components=2, cv=3, scale_inside_folds=False, random_state=4)
    ekf = _pca_ekf_press(
        X, 2, n_folds=3, n_repeats=1, ekf_max_iter=50, ekf_tol=1e-6, scale_inside_folds=False, random_state=4
    )
    np.testing.assert_allclose(result.q2, 1 - ekf.press / centred_ss.sum(), rtol=1e-12)
    np.testing.assert_allclose(result.q2_per_variable, 1 - ekf.per_column_press / centred_ss, rtol=1e-12)


def test_cell_scheme_preprocessing_by_hand() -> None:
    """Each column is centred and divided by its sd, unless the sd is below epsqrt.

    An sd of exactly ``2**-26`` (epsqrt) still divides; ``2**-27`` does not, and a constant
    column is all zeros either way.
    """
    from process_improve.multivariate._pca import _preprocess_for_cell_schemes

    d = 2.0**-26
    X = np.array([[1.0, -d, -d / 2, 4.0], [2.0, 0.0, 0.0, 4.0], [6.0, d, d / 2, 4.0]])
    assert X[:, 1].std(ddof=1) == epsqrt
    expected = np.column_stack([(X[:, 0] - 3.0) / X[:, 0].std(ddof=1), [-1.0, 0.0, 1.0], X[:, 2], np.zeros(3)])
    np.testing.assert_array_equal(_preprocess_for_cell_schemes(X, "ek"), expected)


@pytest.mark.parametrize(
    ("missing", "where"),
    [
        ([(0, 0), (1, 0), (2, 1)], "2 of them in column 0"),
        ([(0, 0), (1, 1)], "spread over 2 columns"),
        ([(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)], "spread over 3 columns"),
    ],
)
def test_cell_schemes_say_where_the_missing_cells_are(missing: list[tuple[int, int]], where: str) -> None:
    """One column holding more than half of the missing cells is named; otherwise the count of columns is.

    Exactly half is not more than half, and two of five is less.
    """
    X = np.arange(18.0).reshape(6, 3) ** 1.5
    for cell in missing:
        X[cell] = np.nan
    message = (
        f"cv_scheme='sacv' cannot take a block with missing cells, because it factorises the matrix directly. "
        f"This block has {len(missing)} of 18, {where}. Use cv_scheme='ekf', which imputes an unmeasured cell "
        "alongside the held-out ones and never scores it, or drop what is missing first."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        PCA.select_n_components(X, cv_scheme="sacv")


def _ckf_by_hand(Z: np.ndarray, A: int, n_folds: int, seed: int) -> tuple[np.ndarray, ...]:
    """Column-wise k-fold, written out: each group's columns predicted from scores of the others."""
    p = Z.shape[1]
    groups = np.array_split(np.random.default_rng(seed).permutation(p), max(1, min(n_folds, p)))
    V = np.linalg.svd(Z)[2].T
    per_fold = np.full((A, len(groups)), np.nan)
    per_column = np.full((A, p), np.nan)
    for a in range(1, A + 1):
        for g, held in enumerate(groups):
            kept = np.setdiff1d(np.arange(p), held)
            if kept.size >= a:
                residual = Z[:, held] - Z[:, kept] @ np.linalg.pinv(V[kept, :a]).T @ V[held, :a].T
                per_column[a - 1, held] = np.sum(residual**2, axis=0)
                per_fold[a - 1, g] = np.sum(residual**2)
    press = np.where(np.isnan(per_fold).any(axis=1), np.nan, per_fold.sum(axis=1))
    return press, per_fold, per_column


def test_ckf_by_hand_through_select_n_components() -> None:
    """The ckf curve, its folds, its per-variable Q2 and its standard error, worked by hand.

    Five columns in three groups of 2, 2 and 1: a count of three leaves at least three
    columns in, so every group evaluates it. The SE is the per-group spread over sqrt(3).
    """
    rng = np.random.default_rng(8)
    X = rng.standard_normal((9, 2)) @ rng.standard_normal((2, 5)) + 0.4 * rng.standard_normal((9, 5))
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    press, per_fold, per_column = _ckf_by_hand(Z, 3, 3, seed=6)
    result = PCA.select_n_components(X, max_components=3, cv=3, cv_scheme="ckf", random_state=6)
    np.testing.assert_allclose(result.press, press, rtol=1e-10)
    np.testing.assert_allclose(result.per_fold_press, per_fold, rtol=1e-10)
    np.testing.assert_allclose(result.q2, 1 - press / np.sum(Z**2), rtol=1e-10)
    np.testing.assert_allclose(result.q2_per_variable, 1 - per_column / np.sum(Z**2, axis=0), rtol=1e-10)
    np.testing.assert_allclose(result.se_press, per_fold.std(axis=1, ddof=1) / np.sqrt(3), rtol=1e-10)
    np.testing.assert_allclose(result.press_input_units, press, rtol=1e-10)


def test_ckf_cannot_evaluate_a_count_with_one_group() -> None:
    """``cv=1`` puts every column in one group, which leaves none to predict from: no count can be scored."""
    X = np.random.default_rng(1).standard_normal((8, 3))
    message = "Cross-validation produced NaN PRESS for every component count; no recommendation can be made."
    with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
        PCA.select_n_components(X, cv=1, cv_scheme="ckf", random_state=0)


def test_ckf_reports_the_spread_of_the_groups_that_evaluate_a_count_and_no_more() -> None:
    """A count only one group can evaluate has no spread: its SE is NaN, and computing it does not warn.

    Five columns in groups of 3 and 2: three components leave 2 columns beside the first
    group, too few, and 3 beside the second.
    """
    X = np.random.default_rng(2).standard_normal((9, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = PCA.select_n_components(X, max_components=3, cv=2, cv_scheme="ckf", random_state=0)
    per_fold = result.per_fold_press.to_numpy()
    counted = (~np.isnan(per_fold)).sum(axis=1)
    np.testing.assert_array_equal(counted, [2, 2, 1])
    np.testing.assert_allclose(result.se_press.iloc[:2], per_fold[:2].std(axis=1, ddof=1) / np.sqrt(2), rtol=1e-12)
    assert np.isnan(result.se_press.iloc[2])


def _ek_fold_models(X: np.ndarray, groups: list, axis: int, reference: np.ndarray, dots: list[float]) -> list:
    """Fit the SVD of X without each group of rows (axis 0) or columns (axis 1).

    Keeps the group, the singular values, and the right vectors (rows left out) or the left
    vectors (columns left out), each turned to agree in sign with the full SVD's vector of
    the same rank, and left as it is when the two are orthogonal. Every inner product the
    signs were read from is appended to ``dots``.
    """
    models = []
    for group in groups:
        u, d, vt = np.linalg.svd(np.delete(X, group, axis=axis), full_matrices=False)
        vectors = vt.T.copy() if axis == 0 else u.copy()
        for k in range(vectors.shape[1]):
            dots.append(float(vectors[:, k] @ reference[:, k]))
            if dots[-1] < 0:
                vectors[:, k] = -vectors[:, k]
        models.append((group, d, vectors))
    return models


def _ek_by_hand(X: np.ndarray, A: int, n_folds: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Eastment and Krzanowski's two models, written out cell by cell.

    ``x_ij`` is predicted by ``sum_k u_ik sqrt(d_k) sqrt(e_k) v_jk``, the score and singular
    value from the SVD of X without the column group of j, the loading and singular value from
    the SVD without the row group of i. Also returns every inner product the signs were read
    from.
    """
    n, p = X.shape
    rng = np.random.default_rng(seed)
    row_groups = np.array_split(rng.permutation(n), n_folds)
    column_groups = np.array_split(rng.permutation(p), n_folds)
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    dots: list[float] = []
    row_models = _ek_fold_models(X, row_groups, 0, Vt.T, dots)
    column_models = _ek_fold_models(X, column_groups, 1, U, dots)
    usable = min(model[1].size for model in row_models + column_models)
    press = np.full(A, np.nan)
    per_column = np.full((A, p), np.nan)
    for a in range(1, min(A, usable) + 1):
        predicted = np.zeros_like(X)
        for (rows, e, v), (columns, d, u) in itertools.product(row_models, column_models):
            for i, j in itertools.product(rows, columns):
                predicted[i, j] = sum(u[i, k] * np.sqrt(d[k]) * np.sqrt(e[k]) * v[j, k] for k in range(a))
        press[a - 1] = np.sum((X - predicted) ** 2)
        per_column[a - 1] = np.sum((X - predicted) ** 2, axis=0)
    return press, per_column, dots


def test_ek_press_by_hand() -> None:
    """The two-model prediction of every cell, on a general block and on one whose columns are orthogonal.

    On the second, whose columns are orthogonal, a fold model's vector can be exactly
    orthogonal to the full fit's vector of the same rank; its sign is then left as it is.
    """
    from process_improve.multivariate._pca import _eastment_krzanowski_press

    rng = np.random.default_rng(13)
    general = rng.standard_normal((8, 2)) @ rng.standard_normal((2, 5)) + 0.3 * rng.standard_normal((8, 5))
    general -= general.mean(axis=0)
    # Orthogonal columns of a Hadamard matrix: some fold models' vectors are exactly orthogonal to the full fit's.
    hadamard = scipy.linalg.hadamard(8)[:, 1:4] * [3.0, 2.0, 1.0]
    for X, A, folds, seed in ((general, 3, 3, 2), (hadamard, 1, 3, 1)):
        press, per_column, dots = _ek_by_hand(X, A, folds, seed=seed)
        got = _eastment_krzanowski_press(X, A, n_folds=folds, random_state=seed)
        np.testing.assert_allclose(got[0], press, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(got[1], per_column, rtol=1e-10, atol=1e-12)
        assert got[2] == pytest.approx(np.sum(X**2), rel=1e-15)
        np.testing.assert_allclose(got[3], np.sum(X**2, axis=0), rtol=1e-15)
    assert 0.0 in dots, "the Hadamard block must give a fold vector at right angles to the full fit's"


def test_ek_warns_only_when_its_folds_cannot_reach_the_counts_asked_for() -> None:
    """Two folds of 4 rows leave 2 rows per row model: 2 components can be evaluated, not 3.

    Asked for exactly 2 it is silent; asked for 3 it says so, and the third count is NaN.
    """
    from process_improve.multivariate._pca import _eastment_krzanowski_press

    X = np.random.default_rng(3).standard_normal((4, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _eastment_krzanowski_press(X, 2, n_folds=2, random_state=0)
    message = (
        "cv_scheme='ek' can evaluate at most 2 component(s) with 2 folds on a 4 by 6 block, not 3: each fold model "
        "is fitted on fewer rows or columns than the whole. Component counts above that are reported as NaN. "
        "Use fewer folds to evaluate more."
    )
    with pytest.warns(SpecificationWarning, match=f"^{re.escape(message)}$"):
        press, per_column, _, _ = _eastment_krzanowski_press(X, 3, n_folds=2, random_state=0)
    assert np.isfinite(press[:2]).all()
    assert np.isnan(press[2])
    assert np.isnan(per_column[2]).all()


def _leverage_block() -> np.ndarray:
    """Return a centred 4 x 4 block of rank 3 whose first row has no part in the third left singular vector.

    ``X = U diag(3, 2, 1) V'`` with orthonormal U orthogonal to the ones vector and
    ``U[0, 2] = 0``: with two components the first row's leverage is the largest a centred
    row can have, 1 - 1/n, so its cells are not scored while the other rows' are.
    """
    U = np.column_stack(
        [
            np.array([0.0, 1, 1, -2]) / np.sqrt(6),
            np.array([3.0, -1, -1, -1]) / np.sqrt(12),
            np.array([0.0, 1, -1, 0]) / np.sqrt(2),
        ]
    )
    V = np.linalg.qr(np.random.default_rng(9).standard_normal((4, 3)))[0]
    return U @ np.diag([3.0, 2.0, 1.0]) @ V.T


def test_sacv_by_hand_with_cells_it_cannot_score() -> None:
    """The smoothing approximation of Josse and Husson (2012), cell by cell.

    Each residual is divided by ``(1 - 1/n - h_i)(1 - b_j)``; a cell whose divisor is at most
    epsqrt is dropped, and each column's error is rescaled from the sum of squares of its
    scored cells to its whole. With two components the first row is dropped, so every column
    is rescaled; with three, every row is, and nothing is reported.
    """
    X = _leverage_block()
    n, p = X.shape
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    press, per_column, null_ss, per_column_null = _leverage_corrected_press(X, 3, method="sacv")
    for a in (1, 2):
        residual = X - (U[:, :a] * S[:a]) @ Vt[:a]
        divisor = np.outer(1 - 1 / n - np.sum(U[:, :a] ** 2, axis=1), 1 - np.sum(Vt[:a] ** 2, axis=0))
        scored = divisor > epsqrt
        assert scored.all() == (a == 1)
        corrected = np.where(scored, residual, 0.0) / np.where(scored, divisor, 1.0)
        scored_ss = np.sum(np.where(scored, X**2, 0.0), axis=0)
        expected = np.sum(corrected**2, axis=0) * np.sum(X**2, axis=0) / scored_ss
        np.testing.assert_allclose(per_column[a - 1], expected, rtol=1e-10)
        assert press[a - 1] == pytest.approx(expected.sum(), rel=1e-10)
    assert not (np.outer(1 - 1 / n - np.sum(U[:, :3] ** 2, axis=1), np.ones(p)) > epsqrt).any()
    assert np.isnan(press[2])
    assert np.isnan(per_column[2]).all()
    assert null_ss == pytest.approx(np.sum(X**2), rel=1e-15)
    np.testing.assert_allclose(per_column_null, np.sum(X**2, axis=0), rtol=1e-15)


def test_sacv_leaves_a_column_of_zeros_unscaled() -> None:
    """A column with no sum of squares to rescale onto keeps its own (zero) error, not 0/0."""
    X = np.column_stack([_leverage_block(), np.zeros(4)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        press, per_column, _, _ = _leverage_corrected_press(X, 2, method="sacv")
    assert per_column[1, 4] == 0.0
    assert np.isfinite(press).all()


def test_gcv_by_hand() -> None:
    """Generalised cross-validation: residuals inflated by ``np / [(n - 1)p - a(n + p - a - 1)]``.

    On 3 rows and 2 columns the first count leaves exactly one degree of freedom, an
    inflation of 6; the second leaves none, so it is not reported.
    """
    X = np.array([[1.0, 2.0], [-2.0, 0.5], [1.0, -2.5]])
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    residual = X - (U[:, :1] * S[:1]) @ Vt[:1]
    press, per_column, _, _ = _leverage_corrected_press(X, 2, method="gcv")
    np.testing.assert_allclose(per_column[0], np.sum((6.0 * residual) ** 2, axis=0), rtol=1e-12)
    assert press[0] == pytest.approx(36.0 * np.sum(residual**2), rel=1e-12)
    assert np.isnan(press[1])
    X = _leverage_block()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    press, _, _, _ = _leverage_corrected_press(X, 2, method="gcv")
    for a, remaining in ((1, 6), (2, 2)):
        residual = X - (U[:, :a] * S[:a]) @ Vt[:a]
        assert press[a - 1] == pytest.approx(np.sum((16.0 / remaining * residual) ** 2), rel=1e-12)


def _raw_block(n: int = 12, k: int = 4, seed: int = 0) -> pd.DataFrame:
    """Return an uncentred block of named columns with two latent factors, in units far from 1."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2)) @ rng.standard_normal((2, k)) + 0.3 * rng.standard_normal((n, k))
    return pd.DataFrame(X * np.arange(1, k + 1) * 3.0 + 10.0, columns=[f"v{j}" for j in range(k)])


@pytest.mark.parametrize("scheme", ["ekf", "ckf", "ek", "sacv", "gcv"])
def test_select_n_components_labels_every_result(scheme: str) -> None:
    """Every curve is a Series indexed by the component count, named for what it holds.

    The PRESS ratio is ``PRESS_a / PRESS_(a-1)`` from the second count; ``cv_scores`` is
    ``per_fold_press``; Q2's standard error is PRESS's over the same reference as Q2.
    """
    X = _raw_block()
    result = PCA.select_n_components(X, max_components=3, cv=3, cv_scheme=scheme, random_state=0)
    counts = pd.Index([1, 2, 3], name="n_components")
    for key, name in (
        ("press", "PRESS"),
        ("press_input_units", "PRESS (input units)"),
        ("q2", "Q2"),
        ("se_press", "SE(PRESS)"),
        ("q2_se", "SE(Q2)"),
    ):
        assert result[key].name == name
        assert result[key].index.equals(counts)
        assert result[key].index.name == "n_components"
    assert result.per_fold_press.index.equals(counts)
    folds = {"ekf": 3, "ckf": 3}.get(scheme, 1)
    assert list(result.per_fold_press.columns) == [f"fold_{i}" for i in range(1, folds + 1)]
    assert result.cv_scores is result.per_fold_press
    assert result.q2_per_variable.index.equals(counts)
    assert list(result.q2_per_variable.columns) == list(X.columns)
    press = result.press.to_numpy()
    expected_ratio = pd.Series([press[1] / press[0], press[2] / press[1]], index=[2, 3], name="PRESS ratio")
    expected_ratio.index.name = "n_components"
    pd.testing.assert_series_equal(result.press_ratio, expected_ratio, rtol=1e-15)
    reference = press[0] / (1.0 - result.q2.iloc[0])  # Q2 = 1 - PRESS / reference
    np.testing.assert_allclose(result.q2_se, result.se_press / reference, rtol=1e-10)
    if scheme == "ekf":
        assert not np.allclose(result.press_input_units, result.press)
        np.testing.assert_allclose(
            result.se_press, result.per_fold_press.std(axis=1, ddof=1) / np.sqrt(3) * 3, rtol=1e-12
        )
    else:
        np.testing.assert_array_equal(result.press_input_units, result.press)
    if scheme not in ("ekf", "ckf"):
        assert result.per_fold_press.isna().all().all()
        assert result.se_press.isna().all()
    assert result.cv_scheme == scheme
    assert result.selection_rule == "min"


def test_select_n_components_caps_the_counts_at_n_minus_1_and_k() -> None:
    """At most ``min(N - 1, K)`` counts are evaluated, whether the cap or the request is the binding one."""
    wide, tall = _raw_block(n=5, k=8), _raw_block(n=10, k=3)
    for X, cap in ((wide, 4), (tall, 3)):
        for requested in (None, 100):
            result = PCA.select_n_components(X, max_components=requested, cv_scheme="gcv")
            assert list(result.press.index) == list(range(1, cap + 1))
        result = PCA.select_n_components(X, max_components=2, cv_scheme="gcv")
        assert list(result.press.index) == [1, 2]


@pytest.mark.parametrize("scheme", ["ekf", "ckf", "ek"])
def test_select_n_components_reads_a_splitter_as_five_folds(scheme: str) -> None:
    """A cross-validation splitter is not read by these schemes: five folds are used, as for ``cv=5``."""
    X = _raw_block(n=12, k=6)
    with_splitter = PCA.select_n_components(X, max_components=2, cv=KFold(3), cv_scheme=scheme, random_state=1)
    with_five = PCA.select_n_components(X, max_components=2, cv=5, cv_scheme=scheme, random_state=1)
    pd.testing.assert_series_equal(with_splitter.press, with_five.press)
    pd.testing.assert_frame_equal(with_splitter.per_fold_press, with_five.per_fold_press)


def test_select_n_components_ekf_defaults_by_hand() -> None:
    """By default ekf deals the cells into 5 folds once, scales inside each, and caps EM at 50 rounds.

    With a tolerance of 0 the EM never converges early, so the cap shows in the curve.
    """
    X = _raw_block()
    result = PCA.select_n_components(X, max_components=2, ekf_tol=0.0, random_state=7)
    press = _ekf_by_hand(X.to_numpy(), 2, n_folds=5, n_repeats=1, seed=7, max_iter=50, tol=0.0, scale=True)[0]
    np.testing.assert_allclose(result.press, press, rtol=1e-10)


def test_select_n_components_reads_integer_data_as_numbers() -> None:
    """An integer block gives the curve of the same block as floats: a held-out cell is not truncated."""
    X = _raw_block().round().astype(int)
    for scheme in ("ekf", "ckf", "sacv"):
        as_int = PCA.select_n_components(X, max_components=2, cv=3, cv_scheme=scheme, random_state=0)
        as_float = PCA.select_n_components(X.astype(float), max_components=2, cv=3, cv_scheme=scheme, random_state=0)
        pd.testing.assert_series_equal(as_int.press, as_float.press)


def test_select_n_components_warns_about_a_prescaled_block_in_full() -> None:
    """The ekf folds re-standardise inside every element-fold, which erases a scaling the caller chose."""
    X = MCUVScaler().fit_transform(_raw_block())
    message = (
        "X is already centred and unit-variance scaled, so scale_inside_folds=True is not protecting you from "
        "leakage: it re-standardises inside every element-fold and overwrites the scaling you chose. Two "
        "deliberately different scalings (say autoscale versus Pareto) collapse onto the same standardised model "
        "this way, so a comparison of their PRESS reflects only the units the blocks arrived in, never which "
        "scaling suits the data. Pass the raw, unscaled X and let the folds scale it, or keep your own scaling and "
        "set scale_inside_folds=False (accepting the optimism that flag warns about)."
    )
    with pytest.warns(SpecificationWarning, match=f"^{re.escape(message)}$"):
        PCA.select_n_components(X, max_components=1, cv=3, random_state=0)


@pytest.mark.parametrize(
    ("scheme", "removed"),
    [
        ("bogus", ""),
        (
            "row_wise",
            (
                " 'row_wise' was removed in 2.0: it scored a held-out row through its own values, so it could not "
                "pick a component count."
            ),
        ),
    ],
)
def test_select_n_components_refuses_an_unknown_scheme_in_full(scheme: str, removed: str) -> None:
    """An unknown scheme is named, with the five there are; the removed ``"row_wise"`` says why it went."""
    message = f"Unknown cv_scheme {scheme!r}; expected one of 'ekf', 'ckf', 'ek', 'sacv' or 'gcv'.{removed}"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        PCA.select_n_components(_raw_block(), cv_scheme=scheme)


def _select_with_press(monkeypatch: pytest.MonkeyPatch, press: list[float], **kwargs: object) -> Bunch:
    """Run ``select_n_components`` with sacv's criterion replaced by ``press``, against a reference of 10.

    Q2 is then ``1 - press / 10``, and every column carries a quarter of each count's PRESS.
    """
    curve = np.asarray(press, dtype=float)

    def stand_in(Z: np.ndarray, max_components: int, *, method: str) -> tuple:
        assert method == "sacv"
        assert max_components == curve.size
        return curve, np.tile(curve[:, np.newaxis] / 4, (1, 4)), 10.0, np.full(4, 2.5)

    monkeypatch.setattr("process_improve.multivariate._pca._leverage_corrected_press", stand_in)
    return PCA.select_n_components(_raw_block(), max_components=curve.size, cv_scheme="sacv", **kwargs)


def test_select_n_components_press_ratio_with_zeros_is_inf_or_nan_without_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PRESS of 0 gives a ratio of 0 after it, NaN for 0/0 and inf for a positive PRESS over 0, silently."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _select_with_press(monkeypatch, [2.0, 0.0, 0.0, 3.0])
    np.testing.assert_array_equal(result.press_ratio, [0.0, np.nan, np.inf])
    np.testing.assert_allclose(result.q2, [0.8, 1.0, 1.0, 0.7])
    np.testing.assert_allclose(result.q2_per_variable, 1 - np.tile([[0.2], [0.0], [0.0], [0.3]], (1, 4)))


@pytest.mark.parametrize(
    ("press", "warns"),
    [
        ([4.0, 3.0, 2.0, 1.0], True),  # rises at every count and ends at the last: no optimum
        ([4.0, 3.0], True),  # two counts are enough to see it
        ([4.0], False),  # one count does not rise at all
        ([3.0, 3.0, 2.0], False),  # flat once, so it is not rising at every count
        ([2.0, 3.0, 1.0], False),  # the best is the last, but the curve turned over before it
        ([4.0, 1.0, 2.0], False),  # rising, then turning over: an optimum
    ],
)
def test_select_n_components_says_when_q2_never_turns_over(
    monkeypatch: pytest.MonkeyPatch, press: list[float], warns: bool
) -> None:
    """The warning fires only when Q2 rises at every count evaluated and the best count is the last."""
    message = (
        f"cv_scheme='sacv' did not turn over: its Q2 rises at every one of the {len(press)} component counts it "
        f"could evaluate, so {len(press)} is the largest count tried rather than an optimum. Evaluate more "
        "components, or use a scheme that holds data out (cv_scheme='ekf' or cv_scheme='ek'), before reading "
        "this as an answer."
    )
    if warns:
        with pytest.warns(SpecificationWarning, match=f"^{re.escape(message)}$"):
            _select_with_press(monkeypatch, press)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _select_with_press(monkeypatch, press)


def test_select_n_components_passes_the_q2_threshold_to_its_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Q2 of 0.5, 0.55 and 0.6 keeps all three counts at the default step of 0.01, and one at 0.1."""
    with pytest.warns(SpecificationWarning, match="did not turn over"):
        default = _select_with_press(monkeypatch, [5.0, 4.5, 4.0], selection_rule="q2_increment")
    assert default.n_components == 3
    strict = _select_with_press(monkeypatch, [5.0, 4.5, 4.0], selection_rule="q2_increment", min_q2_increase=0.1)
    assert strict.n_components == 1


@pytest.mark.parametrize(
    ("minka", "parallel", "consensus"), [(2, 2, "agree"), (3, 2, "agree"), (4, 2, "disagree"), (2, 0, "disagree")]
)
def test_select_n_components_consensus_by_hand(
    monkeypatch: pytest.MonkeyPatch, minka: int, parallel: int, consensus: str
) -> None:
    """The three counts agree when they span at most one; parallel analysis gets the caller's seed."""
    seeds: list[object] = []
    monkeypatch.setattr(PCA, "minka_mle", classmethod(lambda _cls, _X: minka))

    def parallel_analysis(cls: type, X: object, **kwargs: object) -> Bunch:
        seeds.append(kwargs)
        return Bunch(n_components=parallel)

    monkeypatch.setattr(PCA, "parallel_analysis", classmethod(parallel_analysis))
    result = _select_with_press(monkeypatch, [4.0, 1.0, 2.0], return_consensus=True, random_state=17)
    assert result.n_components == 2
    assert seeds == [{"random_state": 17}]
    assert result.minka_n_components == minka
    assert result.parallel_analysis_n_components == parallel
    assert result.consensus == consensus
    assert result.consensus_counts == (2, minka, parallel)


def _parallel_analysis_by_hand(
    X: np.ndarray, n_simulations: int, quantile: float, surrogate: str, seed: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Horn (1965), written out: covariance eigenvalues of X against a quantile of those of random blocks."""
    n, p = X.shape

    def eigenvalues(block: np.ndarray) -> np.ndarray:
        return np.sort(np.linalg.eigvalsh(np.atleast_2d(np.cov(block, rowvar=False))))[::-1][: min(n, p)]

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_simulations):
        if surrogate == "normal":
            null.append(eigenvalues(rng.standard_normal((n, p))))
        else:
            null.append(eigenvalues(np.column_stack([rng.permutation(X[:, j]) for j in range(p)])))
    observed = eigenvalues(X)
    threshold = np.quantile(np.array(null), quantile, axis=0)
    retained = 0
    while retained < observed.size and observed[retained] > threshold[retained]:
        retained += 1
    return observed, threshold, retained


@pytest.mark.parametrize("surrogate", ["normal", "permutation"])
def test_parallel_analysis_by_hand(surrogate: str) -> None:
    """Unscaled, the eigenvalues are those of X's covariance (about the column means), against the null quantile.

    Scaled, they are those of the MCUVScaler-scaled block. The default draws 200 null blocks.
    """
    X = _raw_block(n=15, k=5, seed=4).to_numpy()
    observed, threshold, retained = _parallel_analysis_by_hand(X, 40, 0.9, surrogate, seed=5)
    result = PCA.parallel_analysis(X, n_simulations=40, quantile=0.9, surrogate=surrogate, scale=False, random_state=5)
    np.testing.assert_allclose(result.observed_eigenvalues, observed, rtol=1e-10)
    np.testing.assert_allclose(result.null_threshold, threshold, rtol=1e-10)
    assert result.n_components == retained
    assert result.surrogate == surrogate
    scaled = MCUVScaler().fit_transform(pd.DataFrame(X)).to_numpy()
    observed, threshold, retained = _parallel_analysis_by_hand(scaled, 200, 0.95, surrogate, seed=6)
    result = PCA.parallel_analysis(X, surrogate=surrogate, random_state=6)
    np.testing.assert_allclose(result.observed_eigenvalues, observed, rtol=1e-10)
    np.testing.assert_allclose(result.null_threshold, threshold, rtol=1e-10)
    assert result.n_components == retained


def test_parallel_analysis_retains_only_the_leading_run() -> None:
    """A later eigenvalue above its null does not count once an earlier one fell below.

    One strong column and two of about unit spread: the second eigenvalue (1.269) falls just
    short of its null threshold (1.281) while the third (0.907) clears its own (0.847).
    """
    rng = np.random.default_rng(2)
    n = 40
    X = np.column_stack([5 * rng.standard_normal(n), 1.05 * rng.standard_normal(n), rng.standard_normal(n)])
    result = PCA.parallel_analysis(X, n_simulations=60, scale=False, random_state=0)
    above = result.observed_eigenvalues > result.null_threshold
    np.testing.assert_array_equal(above, [True, False, True])
    assert result.n_components == 1


def test_parallel_analysis_reads_its_input_as_numbers_in_columns() -> None:
    """A 1-D input is one column, an object array is read as numbers, and two rows have one degree of freedom."""
    x = np.array([1.0, 3.0, 2.0, 7.0, 4.0])
    one = PCA.parallel_analysis(x, n_simulations=10, scale=False, random_state=0)
    column = PCA.parallel_analysis(x[:, np.newaxis], n_simulations=10, scale=False, random_state=0)
    np.testing.assert_array_equal(one.observed_eigenvalues, column.observed_eigenvalues)
    assert one.observed_eigenvalues == pytest.approx([np.var(x, ddof=1)], rel=1e-12)
    X = _raw_block(n=8, k=3).to_numpy()
    as_objects = PCA.parallel_analysis(X.astype(object), n_simulations=10, scale=False, random_state=0)
    as_floats = PCA.parallel_analysis(X, n_simulations=10, scale=False, random_state=0)
    np.testing.assert_array_equal(as_objects.observed_eigenvalues, as_floats.observed_eigenvalues)
    two = np.array([[1.0, 2.0], [3.0, -1.0]])
    result = PCA.parallel_analysis(two, n_simulations=10, scale=False, random_state=0)
    assert result.observed_eigenvalues[0] == pytest.approx(np.sum((two - two.mean(axis=0)) ** 2), rel=1e-12)
    observed, threshold, _ = _parallel_analysis_by_hand(two, 10, 0.95, "normal", seed=0)
    np.testing.assert_allclose(result.observed_eigenvalues, observed, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.null_threshold, threshold, rtol=1e-10, atol=1e-12)


def test_minka_mle_is_sklearn_mle_on_the_block() -> None:
    """Minka's estimate is sklearn's, which centres the block itself; a 1-D input is one column."""
    from sklearn import decomposition

    X = _raw_block(n=60, k=6, seed=2).to_numpy()
    expected = decomposition.PCA(n_components="mle", svd_solver="full").fit(X).n_components_
    assert PCA.minka_mle(X) == expected
    assert PCA.minka_mle(pd.DataFrame(X)) == expected
    x = np.random.default_rng(0).standard_normal(20)
    column = x[:, np.newaxis]
    assert PCA.minka_mle(x) == decomposition.PCA(n_components="mle", svd_solver="full").fit(column).n_components_


def _outlier_block(units: float = 1.0) -> pd.DataFrame:
    """Return 12 rows near a plane in three columns, one far along it and one far off it."""
    rng = np.random.default_rng(31)
    X = rng.standard_normal((12, 2)) @ np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]) + 0.05 * rng.standard_normal(
        (12, 3)
    )
    X[3] = [6.0, 9.0, 3.0]  # far along the plane: a T2 outlier
    X[8] += [0.8, -0.8, 0.8]  # off the plane: an SPE outlier
    X -= X.mean(axis=0)
    return pd.DataFrame(X * units, index=[f"r{i}" for i in range(12)], columns=["a", "b", "c"])


@pytest.mark.parametrize("units", [10.0, 1e-3])
def test_detect_outliers_by_hand(units: float) -> None:
    """Each flagged row with its reasons, values, limits and severity ``max(SPE / limit, T2 / limit)``, worst first.

    The limits are the model's own at the confidence asked for; in thousandths of the units
    the SPE limit is below 1 and the severities are unchanged.
    """
    pca = PCA(n_components=2).fit(_outlier_block(units))
    spe, t2 = pca.spe_.iloc[:, -1], pca.hotellings_t2_.iloc[:, -1]
    spe_limit, t2_limit = pca.spe_limit(conf_level=0.95), pca.hotellings_t2_limit(conf_level=0.95)
    assert (spe_limit < 1) == (units < 1)
    found = pca.detect_outliers(conf_level=0.95)
    for outlier in found:
        label = outlier["observation"]
        assert outlier["spe"] == spe[label]
        assert outlier["hotellings_t2"] == t2[label]
        assert outlier["spe_limit"] == spe_limit
        assert outlier["hotellings_t2_limit"] == t2_limit
        assert outlier["severity"] == round(max(spe[label] / spe_limit, t2[label] / t2_limit), 4)
    assert "hotellings_t2" in next(o for o in found if o["observation"] == "r3")["outlier_types"]
    assert "spe" in next(o for o in found if o["observation"] == "r8")["outlier_types"]
    severities = [o["severity"] for o in found]
    assert severities == sorted(severities, reverse=True)
    assert pca.detect_outliers() == found


def test_detect_outliers_runs_the_esd_test_on_each_statistic(monkeypatch: pytest.MonkeyPatch) -> None:
    """ESD looks for up to ``max(1, N // 5)`` outliers in SPE and in T2, at ``alpha = 1 - conf_level``.

    With ESD finding nothing, only the rows above a limit are flagged.
    """
    calls: list[tuple[np.ndarray, dict]] = []

    def esd(values: np.ndarray, **kwargs: object) -> tuple[list, dict]:
        calls.append((values, kwargs))
        return [], {}

    monkeypatch.setattr("process_improve.multivariate._pca.detect_outliers_esd", esd)
    for rows, most in ((10, 2), (4, 1)):
        pca = PCA(n_components=1).fit(_outlier_block().iloc[:rows])
        calls.clear()
        found = pca.detect_outliers(conf_level=0.9)
        assert len(calls) == 2
        np.testing.assert_array_equal(calls[0][0], pca.spe_.iloc[:, -1].to_numpy())
        np.testing.assert_array_equal(calls[1][0], pca.hotellings_t2_.iloc[:, -1].to_numpy())
        for _, kwargs in calls:
            assert kwargs == {"max_outliers_detected": most, "alpha": pytest.approx(0.1)}
            assert type(kwargs["max_outliers_detected"]) is int
        above = (pca.spe_.iloc[:, -1] > pca.spe_limit(conf_level=0.9)) | (
            pca.hotellings_t2_.iloc[:, -1] > pca.hotellings_t2_limit(conf_level=0.9)
        )
        assert sorted(o["observation"] for o in found) == sorted(above[above].index)


def test_detect_outliers_accepts_confidence_levels_from_0_8_to_0_999() -> None:
    """Both ends are allowed; a level just outside either is refused, in full."""
    pca = PCA(n_components=1).fit(_outlier_block())
    pca.detect_outliers(conf_level=0.8)
    pca.detect_outliers(conf_level=0.999)
    for level in (0.7999, 0.9991):
        with pytest.raises(ValueError, match=rf"^conf_level must be between 0\.8 and 0\.999, got {level}\.$"):
            pca.detect_outliers(conf_level=level)


def test_detect_outliers_ignores_a_limit_that_carries_no_information(monkeypatch: pytest.MonkeyPatch) -> None:
    """A perfect fit has an SPE limit of 0, and as many components as rows an infinite T2 limit.

    Neither ratio counts towards the severity then: a perfect fit is ranked by T2 alone, and
    no row is above an SPE limit of 0. With an infinite T2 limit no row is above it, and a
    row the ESD test flags is ranked by its SPE alone.
    """
    # NIPALS normalises a one-entry loading to exactly 1, so every residual is exactly 0.
    one_column = PCA(n_components=1, algorithm="nipals").fit(_outlier_block()[["b"]])
    assert (one_column.spe_.to_numpy() == 0).all()
    assert one_column.spe_limit(conf_level=0.95) == 0.0
    t2 = one_column.hotellings_t2_.iloc[:, -1]
    limit = one_column.hotellings_t2_limit(conf_level=0.95)
    found = one_column.detect_outliers()
    assert found
    for outlier in found:
        assert outlier["outlier_types"] == ["hotellings_t2"]
        assert outlier["severity"] == round(t2[outlier["observation"]] / limit, 4)
    # A row the ESD test flags well inside the T2 limit is ranked below 1, not at 1.
    quietest = int(np.argmin(t2.to_numpy()))
    with monkeypatch.context() as patch:
        patch.setattr("process_improve.multivariate._pca.detect_outliers_esd", lambda _values, **_: ([quietest], {}))
        (outlier,) = (o for o in one_column.detect_outliers() if o["observation"] == t2.index[quietest])
    assert outlier["severity"] == round(t2.iloc[quietest] / limit, 4) < 1
    wide = pd.DataFrame(np.random.default_rng(4).standard_normal((4, 6)))
    full = PCA(n_components=4).fit(wide - wide.mean())
    assert full.hotellings_t2_limit(conf_level=0.95) == np.inf
    spe, spe_limit = full.spe_.iloc[:, -1], full.spe_limit(conf_level=0.95)
    assert spe_limit > 0
    monkeypatch.setattr("process_improve.multivariate._pca.detect_outliers_esd", lambda _values, **_: ([0], {}))
    found = full.detect_outliers()
    assert [outlier["observation"] for outlier in found] == [0]
    assert found[0]["severity"] == round(spe[0] / spe_limit, 4)


# ---- PLS worked by hand: the estimator's API, its fit, its diagnostics and its cross-validation (#1027) ----


def _scale_of(expected: object) -> float:
    """Return the largest size among ``expected``'s finite values, 0 when it has none."""
    values = np.abs(np.asarray(expected, dtype=float))
    return float(np.nanmax(values)) if np.isfinite(values).any() else 0.0


def _assert_near(actual: object, expected: object, rtol: float = 1e-7, atol: float | None = None) -> None:
    """Assert agreement to ``rtol`` of each value, or of the largest one for a value near 0.

    Two iterative fits, ours and sklearn's or two routes to ours, agree to their convergence
    tolerances, about 1e-8 of the values they reach; a prediction near 0 then misses a purely
    relative tolerance on some platforms' BLAS. The absolute floor is that tolerance scaled
    to the largest value compared, so it stays far below any setting or model that is wrong.
    """
    floor = rtol * _scale_of(expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=floor if atol is None else max(atol, floor))


def _assert_frame_near(left: pd.DataFrame, right: pd.DataFrame, rtol: float = 1e-5, **kwargs: object) -> None:
    """``pd.testing.assert_frame_equal`` with the absolute floor of ``_assert_near``."""
    floor = rtol * _scale_of(right.select_dtypes("number"))
    atol = max(float(kwargs.pop("atol", 1e-8)), floor)
    pd.testing.assert_frame_equal(left, right, rtol=rtol, atol=atol, **kwargs)


def _assert_series_near(left: pd.Series, right: pd.Series, rtol: float = 1e-5, **kwargs: object) -> None:
    """``pd.testing.assert_series_equal`` with the absolute floor of ``_assert_near``."""
    floor = rtol * _scale_of(right) if pd.api.types.is_numeric_dtype(right) else 0.0
    atol = max(float(kwargs.pop("atol", 1e-8)), floor)
    pd.testing.assert_series_equal(left, right, rtol=rtol, atol=atol, **kwargs)


def _pls_by_hand(X: np.ndarray, Y: np.ndarray, A: int) -> dict[str, np.ndarray]:
    """Return NIPALS PLS written out (Geladi and Kowalski, 1986), each component's largest loading positive.

    Each component starts from the column of Y with the largest sum of squares and iterates
    ``w = X'u / ||X'u||``, ``t = Xw``, ``c = Y't / t't``, ``u = Yc / c'c`` until ``u`` stops
    moving; then ``p = X't / t't`` and both blocks are deflated by ``t``.
    """
    X, Y = np.array(X, dtype=float), np.array(Y, dtype=float)
    out = {name: np.zeros((size, A)) for name, size in (("T", len(X)), ("U", len(X)), ("W", X.shape[1]))}
    out |= {"P": np.zeros((X.shape[1], A)), "C": np.zeros((Y.shape[1], A))}
    for a in range(A):
        u = Y[:, [np.argmax((Y**2).sum(axis=0))]]
        for _ in range(10_000):
            w = X.T @ u
            w /= np.linalg.norm(w)
            t = X @ w
            c = Y.T @ t / (t.T @ t)
            u_next = Y @ c / (c.T @ c)
            settled = np.linalg.norm(u_next - u) <= 1e-14 * np.linalg.norm(u_next)
            u = u_next
            if settled:
                break
        p = X.T @ t / (t.T @ t)
        X, Y = X - t @ p.T, Y - t @ c.T
        sign = np.sign(p[np.argmax(np.abs(p))])
        for name, vector in (("T", t), ("U", u), ("W", w), ("P", p), ("C", c)):
            out[name][:, a] = (sign * vector).ravel()
    return out


def _autoscaled(values: np.ndarray) -> np.ndarray:
    """Centre each column and divide it by its standard deviation with ddof=1, as PLS(scale=True) does."""
    return (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)


def _pls_block(seed: int = 0, n: int = 12, k: int = 4, m: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``n`` rows of ``k`` named, correlated X columns and ``m`` named responses that depend on them."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k)) @ (np.eye(k) + 0.5 * np.triu(np.ones((k, k)), 1))
    Y = X @ rng.standard_normal((k, m)) + 0.3 * rng.standard_normal((n, m))
    index = [f"r{i}" for i in range(n)]
    return (
        pd.DataFrame(X + 3.0, index=index, columns=[f"x{j}" for j in range(k)]),
        pd.DataFrame(10.0 * Y - 4.0, index=index, columns=[f"y{j}" for j in range(m)]),
    )


def test_pls_constructor_keeps_its_defaults_and_declares_nan_and_several_targets() -> None:
    """The constructor stores what it is given; the estimator tells sklearn it takes NaN and several targets."""
    assert PLS(n_components=2).get_params() == {
        "n_components": 2,
        "scale": True,
        "max_iter": 500,
        "tol": epsqrt,
        "copy": True,
        "warn_on_uncentred": True,
        "missing_data_settings": None,
    }
    tags = PLS(n_components=2).__sklearn_tags__()
    assert tags.input_tags.allow_nan is True
    assert tags.target_tags.multi_output is True


_UNCENTRED_HEAD = "PLS(scale=False) fits no intercept and centres nothing, but column(s) "
_UNCENTRED_MIDDLE = " block have a mean that is large relative to their own spread. The consequence is not an error: "
_UNCENTRED_TAIL = (
    ". Centre both blocks first (MCUVScaler().fit_transform(...), or X - X.mean()), or pass scale=True and let the "
    "model do it. Fitting un-centred on purpose? Pass warn_on_uncentred=False to silence this for this model alone."
)
_UNCENTRED_Y = (
    "every prediction is offset by roughly the response mean, so R2 and Q2 go large and negative and the fit reads "
    "as 'no relationship in this data' even when the relationship is strong"
)
_UNCENTRED_X = (
    "the un-centred columns act as an uncontrolled constant term, so the loadings and R2 describe the offset rather "
    "than the variation around it"
)


@pytest.mark.parametrize(
    ("x_offsets", "y_offset", "messages"),
    [
        (4, 0.0, [("'0', '1', '2', '3'", "X", _UNCENTRED_X)]),
        (5, 0.0, [("'0', '1', '2', '3', ... (5 columns in total)", "X", _UNCENTRED_X)]),
        (0, 7.0, [("'y'", "Y", _UNCENTRED_Y)]),
        (1, 7.0, [("'y'", "Y", _UNCENTRED_Y), ("'0'", "X", _UNCENTRED_X)]),
    ],
    ids=["four-x-columns", "five-x-columns", "y-alone", "y-then-x"],
)
def test_pls_names_the_uncentred_columns_of_each_block_in_full(
    x_offsets: int, y_offset: float, messages: list[tuple[str, str, str]]
) -> None:
    """``scale=False`` warns once per block with a mean far from 0, Y first, naming four columns at most.

    The labels are quoted as strings, so the integer label 0 reads as '0'.
    """
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.standard_normal((20, 6)))
    x -= x.mean()
    y = pd.DataFrame({"y": x[0] - x[1]})
    x.iloc[:, :x_offsets] += 10.0
    with pytest.warns(UncentredDataWarning) as record:
        PLS(n_components=2, scale=False).fit(x, y + y_offset)
    expected = [
        _UNCENTRED_HEAD + labels + " of the " + block + _UNCENTRED_MIDDLE + symptom + _UNCENTRED_TAIL
        for labels, block, symptom in messages
    ]
    assert [str(w.message) for w in record if w.category is UncentredDataWarning] == expected


def test_pls_refuses_an_md_method_it_cannot_fit_in_full_and_reads_any_case() -> None:
    """Only "nipals", "tsr" and "pmp" are fit-time methods; each is read whatever its case."""
    X, Y = _pls_block()
    X.iloc[0, 0] = np.nan
    message = (
        "md_method must be one of ['nipals', 'pmp', 'tsr']; got 'scp'. This is the fit-time setting, a smaller set "
        "than the method= accepted by project() and the contribution helpers."
    )
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        PLS(n_components=2, missing_data_settings={"md_method": "scp"}).fit(X, Y)
    nipals = PLS(n_components=2).fit(X, Y)
    shouted = PLS(n_components=2, missing_data_settings={"md_method": "NIPALS"}).fit(X, Y)
    np.testing.assert_array_equal(shouted.beta_coefficients_.to_numpy(), nipals.beta_coefficients_.to_numpy())
    assert "md_method" not in shouted.fitting_info_
    tsr = PLS(n_components=2, missing_data_settings={"md_method": "tsr"}).fit(X, Y)
    loud_tsr = PLS(n_components=2, missing_data_settings={"md_method": "TSR"}).fit(X, Y)
    assert loud_tsr.fitting_info_["md_method"] == "tsr"
    np.testing.assert_array_equal(loud_tsr.beta_coefficients_.to_numpy(), tsr.beta_coefficients_.to_numpy())


def _pls_outlier_blocks(units: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the 12 rows of ``_outlier_block`` and a response along its plane, in ``units``."""
    X = _outlier_block(units)
    y = pd.DataFrame({"y": X["a"] + X["b"] + 0.01 * units * np.random.default_rng(3).standard_normal(12)})
    return X, y


@pytest.mark.parametrize("units", [10.0, 1e-3])
def test_pls_detect_outliers_by_hand(units: float) -> None:
    """Each flagged row with its reasons, values, limits and severity ``max(SPE / limit, T2 / limit)``, worst first.

    The limits are the model's own at the confidence asked for; in thousandths of the units
    the SPE limit is below 1 and the severities are unchanged.
    """
    pls = PLS(n_components=2, scale=False, warn_on_uncentred=False).fit(*_pls_outlier_blocks(units))
    spe, t2 = pls.spe_.iloc[:, -1], pls.hotellings_t2_.iloc[:, -1]
    spe_limit, t2_limit = pls.spe_limit(conf_level=0.95), pls.hotellings_t2_limit(conf_level=0.95)
    assert (spe_limit < 1) == (units < 1)
    found = pls.detect_outliers(conf_level=0.95)
    for outlier in found:
        label = outlier["observation"]
        assert outlier["spe"] == spe[label]
        assert outlier["hotellings_t2"] == t2[label]
        assert outlier["spe_limit"] == spe_limit
        assert outlier["hotellings_t2_limit"] == t2_limit
        assert outlier["severity"] == round(max(spe[label] / spe_limit, t2[label] / t2_limit), 4)
    assert "hotellings_t2" in next(o for o in found if o["observation"] == "r3")["outlier_types"]
    assert "spe" in next(o for o in found if o["observation"] == "r8")["outlier_types"]
    severities = [o["severity"] for o in found]
    assert severities == sorted(severities, reverse=True)
    assert pls.detect_outliers() == found


def test_pls_detect_outliers_runs_the_esd_test_on_each_statistic(monkeypatch: pytest.MonkeyPatch) -> None:
    """ESD looks for up to ``max(1, N // 5)`` outliers in SPE and in T2, at ``alpha = 1 - conf_level``.

    With ESD finding nothing, only the rows above a limit are flagged.
    """
    calls: list[tuple[np.ndarray, dict]] = []

    def esd(values: np.ndarray, **kwargs: object) -> tuple[list, dict]:
        calls.append((values, kwargs))
        return [], {}

    monkeypatch.setattr("process_improve.multivariate._pls.detect_outliers_esd", esd)
    X, y = _pls_outlier_blocks()
    for rows, most in ((10, 2), (4, 1)):
        pls = PLS(n_components=1).fit(X.iloc[:rows], y.iloc[:rows])
        calls.clear()
        found = pls.detect_outliers(conf_level=0.9)
        assert len(calls) == 2
        np.testing.assert_array_equal(calls[0][0], pls.spe_.iloc[:, -1].to_numpy())
        np.testing.assert_array_equal(calls[1][0], pls.hotellings_t2_.iloc[:, -1].to_numpy())
        for _, kwargs in calls:
            assert kwargs == {"max_outliers_detected": most, "alpha": pytest.approx(0.1)}
            assert type(kwargs["max_outliers_detected"]) is int
        above = (pls.spe_.iloc[:, -1] > pls.spe_limit(conf_level=0.9)) | (
            pls.hotellings_t2_.iloc[:, -1] > pls.hotellings_t2_limit(conf_level=0.9)
        )
        assert sorted(o["observation"] for o in found) == sorted(above[above].index)


def test_pls_detect_outliers_accepts_confidence_levels_from_0_8_to_0_999() -> None:
    """Both ends are allowed; a level just outside either is refused, in full."""
    pls = PLS(n_components=1).fit(*_pls_outlier_blocks())
    pls.detect_outliers(conf_level=0.8)
    pls.detect_outliers(conf_level=0.999)
    for level in (0.7999, 0.9991):
        with pytest.raises(ValueError, match=rf"^conf_level must be between 0\.8 and 0\.999, got {level}\.$"):
            pls.detect_outliers(conf_level=level)


def test_pls_detect_outliers_ignores_a_limit_that_carries_no_information(monkeypatch: pytest.MonkeyPatch) -> None:
    """A perfect fit has an SPE limit of 0, and as many components as rows an infinite T2 limit.

    Neither ratio counts towards the severity then: a perfect fit is ranked by T2 alone, and
    no row is above an SPE limit of 0. With an infinite T2 limit no row is above it, and a
    row the ESD test flags is ranked by its SPE alone.
    """
    X, y = _pls_outlier_blocks()
    # One centred X column: its weight is exactly 1, so the score is the column and every residual exactly 0.
    one_column = PLS(n_components=1, scale=False).fit(X[["b"]], y - y.mean())
    assert (one_column.spe_.to_numpy() == 0).all()
    assert one_column.spe_limit(conf_level=0.95) == 0.0
    t2 = one_column.hotellings_t2_.iloc[:, -1]
    limit = one_column.hotellings_t2_limit(conf_level=0.95)
    found = one_column.detect_outliers()
    assert found
    for outlier in found:
        assert outlier["outlier_types"] == ["hotellings_t2"]
        assert outlier["severity"] == round(t2[outlier["observation"]] / limit, 4)
    # A row the ESD test flags well inside the T2 limit is ranked below 1, not at 1.
    quietest = int(np.argmin(t2.to_numpy()))
    with monkeypatch.context() as patch:
        patch.setattr("process_improve.multivariate._pls.detect_outliers_esd", lambda _values, **_: ([quietest], {}))
        (outlier,) = (o for o in one_column.detect_outliers() if o["observation"] == t2.index[quietest])
    assert outlier["severity"] == round(t2.iloc[quietest] / limit, 4) < 1
    rng = np.random.default_rng(4)
    wide, targets = pd.DataFrame(rng.standard_normal((4, 6)) + 2.0), pd.DataFrame(rng.standard_normal((4, 4)))
    full = PLS(n_components=4, scale=False, warn_on_uncentred=False).fit(wide, targets)
    assert full.hotellings_t2_limit(conf_level=0.95) == np.inf
    spe, spe_limit = full.spe_.iloc[:, -1], full.spe_limit(conf_level=0.95)
    assert spe_limit > 0
    monkeypatch.setattr("process_improve.multivariate._pls.detect_outliers_esd", lambda _values, **_: ([0], {}))
    found = full.detect_outliers()
    assert [outlier["observation"] for outlier in found] == [0]
    assert found[0]["severity"] == round(spe[0] / spe_limit, 4)


def _pls_new_rows_by_hand(pls: PLS, X: pd.DataFrame, Y: pd.DataFrame, new: np.ndarray) -> dict[str, np.ndarray]:
    """Return what a PLS fitted with ``scale=True`` on ``X``, ``Y`` gives complete new rows, from its W, P and C.

    ``z = (x - mean) / sd`` with the training statistics (ddof=1); ``t = z W (P'W)^-1``;
    ``T2 = sum_a t_a^2 / s_a^2`` with ``s_a^2 = t_a't_a / (N - 1)`` over the training scores;
    ``SPE = ||z - t P'||``; ``y_hat = t C'`` back in the units of Y.
    """
    W, P, C = pls.x_weights_.to_numpy(), pls.x_loadings_.to_numpy(), pls.y_loadings_.to_numpy()
    T = pls.scores_.to_numpy()
    z = (new - X.mean().to_numpy()) / X.std(ddof=1).to_numpy()
    t = z @ W @ np.linalg.inv(P.T @ W)
    return {
        "scores": t,
        "t2": np.sum(t**2 / (np.sum(T**2, axis=0) / (len(X) - 1)), axis=1),
        "spe": np.linalg.norm(z - t @ P.T, axis=1),
        "y_hat": t @ C.T * Y.std(ddof=1).to_numpy() + Y.mean().to_numpy(),
        "z": z,
    }


def test_pls_reads_new_rows_by_column_name_and_labels_every_result() -> None:
    """New rows with their columns in another order are matched by name, and keep their own row labels.

    Every value is worked from the fitted W, P and C and the training statistics; see
    ``_pls_new_rows_by_hand``.
    """
    X, Y = _pls_block()
    pls = PLS(n_components=2).fit(X, Y)
    new = pd.DataFrame([[3.5, 2.0, 5.0, 1.0], [2.5, 3.2, 2.6, 4.0]], columns=X.columns, index=["p", "q"])
    reordered = new[["x2", "x0", "x3", "x1"]]
    by_hand = _pls_new_rows_by_hand(pls, X, Y, new.to_numpy())
    expected_scores = pd.DataFrame(by_hand["scores"], index=["p", "q"], columns=[1, 2])
    expected_y = pd.DataFrame(by_hand["y_hat"], index=["p", "q"], columns=["y0", "y1"])

    pd.testing.assert_frame_equal(pls.transform(reordered), expected_scores, rtol=1e-10)
    diagnosed = pls.diagnose(reordered)
    pd.testing.assert_frame_equal(diagnosed.scores, expected_scores, rtol=1e-10)
    pd.testing.assert_series_equal(
        diagnosed.hotellings_t2, pd.Series(by_hand["t2"], index=["p", "q"], name="Hotelling's T²"), rtol=1e-10
    )
    pd.testing.assert_series_equal(diagnosed.spe, pd.Series(by_hand["spe"], index=["p", "q"], name="SPE"), rtol=1e-10)
    pd.testing.assert_frame_equal(diagnosed.y_hat, expected_y, rtol=1e-10)
    pd.testing.assert_frame_equal(pls.predict(reordered), expected_y, rtol=1e-10)
    projected = pls.project(reordered)
    pd.testing.assert_frame_equal(projected.scores, expected_scores, rtol=1e-10)
    pd.testing.assert_frame_equal(projected.y_hat, expected_y, rtol=1e-10)
    pd.testing.assert_series_equal(
        projected.hotellings_t2, pd.Series(by_hand["t2"], index=["p", "q"], name="Hotelling's T2"), rtol=1e-10
    )
    pd.testing.assert_series_equal(projected.spe, pd.Series(by_hand["spe"], index=["p", "q"], name="SPE"), rtol=1e-10)
    pd.testing.assert_series_equal(
        projected.condition_number, pd.Series([1.0, 1.0], index=["p", "q"], name="condition_number")
    )
    pd.testing.assert_series_equal(projected.n_observed, pd.Series([4, 4], index=["p", "q"], name="n_observed"))
    residual = Y.loc[["r0", "r1"]].to_numpy() - pls.predict(X.loc[["r0", "r1"]]).to_numpy()
    total = ((Y.loc[["r0", "r1"]] - Y.loc[["r0", "r1"]].mean()) ** 2).sum().to_numpy()
    assert pls.score(X.loc[["r0", "r1"]], Y.loc[["r0", "r1"]]) == pytest.approx(
        np.mean(1 - (residual**2).sum(axis=0) / total), rel=1e-12
    )

    # An array has no labels of its own: its rows are numbered from 0, its columns read in the fitted order.
    for result in (
        pls.transform(new.to_numpy()),
        pls.diagnose(new.to_numpy()).scores,
        pls.project(new.to_numpy()).scores,
    ):
        pd.testing.assert_frame_equal(result, pd.DataFrame(by_hand["scores"], columns=[1, 2]), rtol=1e-10)
    pd.testing.assert_frame_equal(
        pls.predict(new.to_numpy()), pd.DataFrame(by_hand["y_hat"], columns=["y0", "y1"]), rtol=1e-10
    )
    for series in ("hotellings_t2", "spe"):
        assert pls.diagnose(new.to_numpy())[series].index.equals(pd.RangeIndex(2))
    for series in ("hotellings_t2", "spe", "condition_number", "n_observed"):
        assert pls.project(new.to_numpy())[series].index.equals(pd.RangeIndex(2))
    np.testing.assert_array_equal(pls.get_feature_names_out(), ["T1", "T2"])


@pytest.mark.parametrize("scale", [True, False])
def test_pls_reads_columns_without_string_labels_by_position(scale: bool) -> None:
    """Columns labelled other than by strings are read by position, whether or not the model scales (#1027).

    sklearn records no feature names for such columns, so ``validate_data`` reads them by
    position, as the scaler and ``project`` did. Without scaling, ``transform``,
    ``diagnose`` and ``predict`` multiplied by the direct weights by label instead: integer
    labels in another order gave the scores of the columns by label but an SPE of them by
    position, which was neither, and labels the fit had not seen raised "matrices are not
    aligned". Every value here is ``t = x R``, ``SPE = ||x - t P'||``, ``y_hat = t C'``.
    """
    X, Y = _pls_block()
    X, Y = pd.DataFrame(X.to_numpy()), pd.DataFrame(Y.to_numpy())
    if not scale:
        X, Y = X - X.mean(), Y - Y.mean()
    pls = PLS(n_components=2, scale=scale).fit(X, Y)
    rows = np.array([[0.5, -1.0, 2.0, 1.0], [1.5, 0.2, -0.4, -2.0]])
    z = rows if not scale else (rows - X.mean().to_numpy()) / X.std(ddof=1).to_numpy()
    W, P, C = pls.x_weights_.to_numpy(), pls.x_loadings_.to_numpy(), pls.y_loadings_.to_numpy()
    t = z @ W @ np.linalg.inv(P.T @ W)
    y_hat = t @ C.T
    if scale:
        y_hat = y_hat * Y.std(ddof=1).to_numpy() + Y.mean().to_numpy()
    spe = np.linalg.norm(z - t @ P.T, axis=1)
    for labels in ([3, 2, 1, 0], [10, 11, 12, 13]):
        new = pd.DataFrame(rows, columns=labels)
        np.testing.assert_allclose(pls.transform(new).to_numpy(), t, rtol=1e-10)
        diagnosed = pls.diagnose(new)
        np.testing.assert_allclose(diagnosed.scores.to_numpy(), t, rtol=1e-10)
        np.testing.assert_allclose(diagnosed.spe.to_numpy(), spe, rtol=1e-10)
        np.testing.assert_allclose(pls.predict(new).to_numpy(), y_hat, rtol=1e-10)
        np.testing.assert_allclose(pls.project(new).scores.to_numpy(), t, rtol=1e-10)
        np.testing.assert_allclose(pls.project(new).spe.to_numpy(), spe, rtol=1e-10)


def _pls_training_residuals(pls: PLS, X: pd.DataFrame) -> np.ndarray:
    """Return ``E = Z - T P'`` of the autoscaled training block."""
    Z = _autoscaled(X.to_numpy())
    return Z - pls.scores_.to_numpy() @ pls.x_loadings_.to_numpy().T


@pytest.mark.parametrize(("method", "ridge"), [("tsr", 0.0), ("scp", 0.0), ("pmp", 0.5), ("tsr", 0.5)])
def test_pls_projects_a_row_with_a_gap_by_each_estimator(method: str, ridge: float) -> None:
    """A row with a missing cell is scored by the estimator asked for, with PLS's weights and training residuals.

    The reference is ``project_rows`` (and ``operator_for_pattern`` for the matrix) given
    the model's P, R = W (P'W)^-1, score variances, W and E = Z - T P', worked here from
    the training data. The estimators differ, so each argument is seen.
    """
    X, Y = _pls_block(n=15)
    pls = PLS(n_components=2).fit(X, Y)
    W, P = pls.x_weights_.to_numpy(), pls.x_loadings_.to_numpy()
    R = W @ np.linalg.inv(P.T @ W)
    variances = np.sum(pls.scores_.to_numpy() ** 2, axis=0) / 14
    E = _pls_training_residuals(pls, X)
    new = pd.DataFrame([[3.5, np.nan, 5.0, 1.0], [2.5, 3.2, 2.6, 4.0]], columns=X.columns, index=["p", "q"])
    z = (new.to_numpy() - X.mean().to_numpy()) / X.std(ddof=1).to_numpy()
    expected = project_rows(P, R, variances, z, method=method, ridge=ridge, x_weights=W, x_residuals=E)
    others = [
        project_rows(P, R, variances, z, method=other, ridge=other_ridge, x_weights=W, x_residuals=E).scores
        for other, other_ridge in (("tsr", 0.0), ("scp", 0.0), ("pmp", 0.5), ("tsr", 0.5), ("pmp", 0.0))
        if (other, other_ridge) != (method, ridge)
    ]
    assert all(not np.allclose(other, expected.scores) for other in others)
    kwargs = {} if (method, ridge) == ("tsr", 0.0) else {"method": method, "ridge": ridge}
    projected = pls.project(new, **kwargs)
    np.testing.assert_allclose(projected.scores.to_numpy(), expected.scores, rtol=1e-10)
    np.testing.assert_allclose(projected.spe.to_numpy(), expected.spe, rtol=1e-10)
    np.testing.assert_allclose(projected.condition_number.to_numpy(), expected.condition_number, rtol=1e-10)
    np.testing.assert_array_equal(projected.n_observed.to_numpy(), [3, 4])
    t2 = np.sum(expected.scores**2 / variances, axis=1)
    np.testing.assert_allclose(projected.hotellings_t2.to_numpy(), t2, rtol=1e-10)
    y_hat = expected.scores @ pls.y_loadings_.to_numpy().T * Y.std(ddof=1).to_numpy() + Y.mean().to_numpy()
    np.testing.assert_allclose(projected.y_hat.to_numpy(), y_hat, rtol=1e-10)

    mask = np.array([True, False, True, True])
    op = operator_for_pattern(P, R, variances, mask, method=method, ridge=ridge, x_weights=W, x_residuals=E)
    built = pls.projection_matrix(["x0", "x2", "x3"], **kwargs)
    pd.testing.assert_frame_equal(
        built.matrix, pd.DataFrame(op.matrix, index=[1, 2], columns=["x0", "x2", "x3"]), rtol=1e-10
    )
    assert built.condition_number == pytest.approx(op.condition_number, rel=1e-10)
    assert built.method == method


def test_pls_project_and_projection_matrix_work_without_training_residuals() -> None:
    """A model that keeps no training residuals (one pickled before they were kept) still projects.

    Its estimator for a row with a missing cell is the one ``project_rows`` and
    ``operator_for_pattern`` build when they are given no residuals.
    """
    X, Y = _pls_block(n=15)
    pls = PLS(n_components=2).fit(X, Y)
    row = pd.DataFrame([[3.5, np.nan, 5.0, 1.0]], columns=X.columns)
    with_residuals = pls.project(row).scores.to_numpy()
    del pls._x_residuals
    W, P = pls.x_weights_.to_numpy(), pls.x_loadings_.to_numpy()
    R = pls.direct_weights_.to_numpy()
    z = (row.to_numpy() - X.mean().to_numpy()) / X.std(ddof=1).to_numpy()
    with pytest.warns(SpecificationWarning):
        expected = project_rows(P, R, pls.explained_variance_, z, x_weights=W, x_residuals=None)
    with pytest.warns(SpecificationWarning):
        np.testing.assert_allclose(pls.project(row).scores.to_numpy(), expected.scores, rtol=1e-12)
    assert not np.allclose(with_residuals, expected.scores), "the residuals must change the estimate"
    mask = np.array([True, False, True, True])
    with pytest.warns(SpecificationWarning):
        op = operator_for_pattern(P, R, pls.explained_variance_, mask, x_weights=W, x_residuals=None)
    with pytest.warns(SpecificationWarning):
        built = pls.projection_matrix(mask)
    pd.testing.assert_frame_equal(built.matrix, pd.DataFrame(op.matrix, index=[1, 2], columns=["x0", "x2", "x3"]))


def test_pls_whose_fit_failed_is_not_fitted() -> None:
    """A fit that raised part way leaves no weights, and every method that reads them says so.

    The failed fit has set ``n_samples_`` and other attributes ending in ``_`` before it
    refused the short Y, so a check for any fitted attribute would wrongly pass.
    """
    X, Y = _pls_block()
    pls = PLS(n_components=1)
    with pytest.raises(ValueError, match=r"^The X and Y arrays must have the same number of rows"):
        pls.fit(X, Y.iloc[:5])
    assert pls.n_samples_ == 12
    calls: list[Callable] = [
        lambda: pls.transform(X),
        lambda: pls.diagnose(X),
        lambda: pls.predict(X),
        lambda: pls.project(X),
        lambda: pls.projection_matrix([True] * 4),
        lambda: pls.invert(1.0),
        lambda: pls.cross_validate(X, Y),
        lambda: pls.prediction_interval(X),
        pls.get_feature_names_out,
        pls.detect_outliers,
    ]
    for call in calls:
        with pytest.raises(NotFittedError, match="This PLS instance is not fitted yet"):
            call()
    # Being unfitted is said first, before a setting out of range.
    with pytest.raises(NotFittedError, match="This PLS instance is not fitted yet"):
        pls.prediction_interval(X, conf_level=2.0)


def test_pls_new_rows_are_refused_as_validate_data_refuses_them() -> None:
    """New rows must have the fitted number of columns and no infinity; a missing cell is allowed."""
    X, Y = _pls_block()
    pls = PLS(n_components=2).fit(X.to_numpy(), Y.to_numpy())
    for call in (pls.transform, pls.diagnose, pls.project):
        with pytest.raises(ValueError, match=r"^X has 3 features, but PLS is expecting 4 features as input\.$"):
            call(np.ones((2, 3)))
        with pytest.raises(ValueError, match=r"^Input X contains infinity"):
            call(np.array([[1.0, np.inf, 0.0, 1.0]]))
    assert pls.n_features_in_ == 4
    scores = pls.transform(np.array([[1.0, np.nan, 0.0, 1.0], [1.0, 2.0, 0.0, 1.0]]))
    assert np.isnan(scores.iloc[0]).all()
    assert np.isfinite(scores.iloc[1]).all()


def _gapped_pls_block() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``_pls_block(n=15)`` with two X cells and one Y cell missing."""
    X, Y = _pls_block(n=15)
    X.iloc[1, 2] = X.iloc[6, 0] = np.nan
    Y.iloc[3, 0] = np.nan
    return X, Y


def _compare_pls_with_by_hand(pls: PLS, reference: dict[str, np.ndarray], rtol: float) -> None:
    """Check a fitted PLS's T, W, P and C against ``_pls_by_hand``'s."""
    for name, attribute in (("T", "scores_"), ("W", "x_weights_"), ("P", "x_loadings_"), ("C", "y_loadings_")):
        np.testing.assert_allclose(np.asarray(getattr(pls, attribute)), reference[name], rtol=rtol, atol=rtol)


@pytest.mark.parametrize("method", ["tsr", "pmp"])
def test_pls_imputed_fit_by_hand(method: str) -> None:
    """``md_method="tsr"`` / ``"pmp"``: impute ``[X Y]`` jointly, then fit NIPALS once to the completed blocks.

    The reference fills the cells with ``impute_low_rank`` at the model's own tolerance and
    cap, autoscales the completed blocks, and fits PLS by hand. The model keeps the column
    labels, and says how the imputation went.
    """
    X, Y = _gapped_pls_block()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pls = PLS(n_components=2, tol=1e-9, max_iter=300, missing_data_settings={"md_method": method}).fit(X, Y)
    imputation = impute_low_rank(pd.concat([X, Y], axis=1).to_numpy(), 2, method=method, tol=1e-9, max_iter=300)
    completed = imputation.completed
    reference = _pls_by_hand(_autoscaled(completed[:, :4]), _autoscaled(completed[:, 4:]), 2)
    _compare_pls_with_by_hand(pls, reference, rtol=1e-6)
    assert pls.fitting_info_["md_method"] == method
    assert pls.fitting_info_["md_rounds"] == imputation.rounds
    assert pls.fitting_info_["md_converged"] is True
    assert pls.fitting_info_["md_shift"] == imputation.shift
    assert list(pls.x_loadings_.index) == list(X.columns)
    assert list(pls.y_loadings_.index) == list(Y.columns)
    predictions = reference["T"] @ reference["C"].T * completed[:, 4:].std(axis=0, ddof=1) + completed[:, 4:].mean(
        axis=0
    )
    _assert_near(pls.predictions_.to_numpy(), predictions, rtol=1e-6)
    # The diagnostics count the observed cells only: row 1 has its third X cell missing.
    z = (X.to_numpy() - completed[:, :4].mean(axis=0)) / completed[:, :4].std(axis=0, ddof=1)
    residual = z - reference["T"] @ reference["P"].T
    assert pls.spe_.iloc[1, -1] == pytest.approx(np.sqrt(np.nansum(residual[1] ** 2)), rel=1e-6)


def test_pls_imputed_fit_reads_a_pandas_missing_value_and_any_case() -> None:
    """A nullable ``Float64`` response's ``pd.NA`` is a missing cell, and ``"TSR"`` is ``"tsr"``."""
    X, Y = _gapped_pls_block()
    nullable = Y.astype("Float64")
    nullable.iloc[3, 0] = pd.NA
    plain = PLS(n_components=2, missing_data_settings={"md_method": "tsr"}).fit(X, Y)
    for y, method in ((nullable, "tsr"), (Y, "TSR")):
        other = PLS(n_components=2, missing_data_settings={"md_method": method}).fit(X, y)
        np.testing.assert_array_equal(other.beta_coefficients_.to_numpy(), plain.beta_coefficients_.to_numpy())
        assert other.fitting_info_["md_method"] == "tsr"


def test_pls_weighted_imputed_fit_scales_and_fits_on_the_weighted_rows_only() -> None:
    """With ``sample_weight``, the imputation models every row, and the scaling and the fit only the weighted ones.

    A weight of 0 drops a row from the fit: the reference autoscales the completed rows of
    weight 1 with their own statistics and fits PLS to them by hand.
    """
    X, Y = _gapped_pls_block()
    weights = np.ones(15)
    weights[[0, 5, 9]] = 0.0
    pls = PLS(n_components=2, missing_data_settings={"md_method": "tsr"}).fit(X, Y, sample_weight=weights)
    completed = impute_low_rank(pd.concat([X, Y], axis=1).to_numpy(), 2, method="tsr", tol=epsqrt, max_iter=500)
    kept = completed.completed[weights > 0]
    reference = _pls_by_hand(_autoscaled(kept[:, :4]), _autoscaled(kept[:, 4:]), 2)
    for name, attribute in (("W", "x_weights_"), ("P", "x_loadings_"), ("C", "y_loadings_")):
        np.testing.assert_allclose(np.asarray(getattr(pls, attribute)), reference[name], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pls.scores_.to_numpy()[weights > 0], reference["T"], rtol=1e-6, atol=1e-6)


def test_pls_imputed_fit_says_in_full_when_the_cells_still_move() -> None:
    """An imputation stopped by ``max_iter`` before it settles is reported with its last shift and rounds."""
    X, Y = _gapped_pls_block()
    with pytest.warns(SpecificationWarning) as record:
        PLS(n_components=2, max_iter=2, missing_data_settings={"md_method": "pmp"}).fit(X, Y)
    imputation = impute_low_rank(pd.concat([X, Y], axis=1).to_numpy(), 2, method="pmp", tol=epsqrt, max_iter=2)
    assert not imputation.converged
    expected = (
        f"PLS PMP: the imputed cells were still moving by {imputation.shift:.3g} standard deviations after 2 rounds "
        f"(tol={epsqrt:g}). Raise max_iter, or loosen tol if that change is already negligible for your data."
    )
    assert expected in [str(w.message) for w in record]


def test_pls_invert_by_hand() -> None:
    """``invert``: the minimum-norm scores for a target, the null space of C', and the input they give.

    With ``Q = C'`` (components x targets) and ``y`` autoscaled with the training statistics,
    ``tau = y pinv(Q)``; the null space is spanned by the left singular vectors of ``Q`` past
    its rank; ``x_new = (tau + N c) P'`` and ``y_hat = (tau + N c) Q``, both back in units;
    ``T2 = sum tau_a^2 / s_a^2``. A target given as a dict or a DataFrame is read by name.
    """
    X, Y = _pls_block(n=15)
    pls = PLS(n_components=3).fit(X, Y)
    Q, P = pls.y_loadings_.to_numpy().T, pls.x_loadings_.to_numpy()
    variances = np.sum(pls.scores_.to_numpy() ** 2, axis=0) / 14
    target = np.array([5.0, -3.0])
    tau = ((target - Y.mean().to_numpy()) / Y.std(ddof=1).to_numpy()) @ np.linalg.pinv(Q)
    U = np.linalg.svd(Q)[0]
    null_space = U[:, 2:]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for given in (
            target,
            {"y1": -3.0, "y0": 5.0},
            pd.Series({"y1": -3.0, "y0": 5.0}),
            pd.DataFrame({"y1": [-3.0], "y0": [5.0]}, index=["wanted"]),
        ):
            result = pls.invert(given)
            pd.testing.assert_series_equal(result.scores, pd.Series(tau, index=[1, 2, 3], name="scores"), rtol=1e-10)
            x_new = tau @ P.T * X.std(ddof=1).to_numpy() + X.mean().to_numpy()
            pd.testing.assert_series_equal(result.x_new, pd.Series(x_new, index=X.columns, name="x_new"), rtol=1e-10)
            pd.testing.assert_series_equal(result.y_hat, pd.Series(target, index=Y.columns, name="y_hat"), rtol=1e-10)
            assert result.null_space_dimension == 1
            basis = result.null_space_basis
            assert list(basis.index) == [1, 2, 3]
            assert list(basis.columns) == ["NS1"]
            np.testing.assert_allclose(np.abs(basis.to_numpy()), np.abs(null_space), rtol=1e-10)
            assert result.hotellings_t2 == pytest.approx(np.sum(tau**2 / variances), rel=1e-10)
        moved = pls.invert(target, null_space_coordinates=[2.0])
    shifted = tau + 2.0 * moved.null_space_basis.to_numpy().ravel()
    np.testing.assert_allclose(moved.scores.to_numpy(), shifted, rtol=1e-10)
    x_moved = shifted @ P.T * X.std(ddof=1).to_numpy() + X.mean().to_numpy()
    np.testing.assert_allclose(moved.x_new.to_numpy(), x_moved, rtol=1e-10)
    np.testing.assert_allclose(moved.y_hat.to_numpy(), target, rtol=1e-10)
    assert moved.hotellings_t2 == pytest.approx(np.sum(shifted**2 / variances), rel=1e-10)


def test_pls_invert_null_space_has_numpys_rank_tolerance() -> None:
    """The null space is the dimensions past the rank of C', at numpy's ``matrix_rank`` tolerance.

    In units of 1e-4, two targets nearly proportional to each other give C' a second singular
    value about 1e-9 times the first: far above that tolerance (``3 * eps * s_1``), so the rank
    is 2, though below ``3 * eps / s_1``.
    """
    X, _ = _pls_block(n=15)
    X = X - X.mean()
    y = X.to_numpy() @ np.array([1.0, -1.0, 0.5, 2.0])
    Y = pd.DataFrame({"a": y, "b": 2.0 * y + 1e-9 * X.to_numpy()[:, 1]}) * 1e-4
    pls = PLS(n_components=3, scale=False).fit(X, Y - Y.mean())
    Q = pls.y_loadings_.to_numpy().T
    singular = np.linalg.svd(Q, compute_uv=False)
    assert 3 * np.finfo(float).eps * singular[0] < singular[1] < 3 * np.finfo(float).eps / singular[0]
    assert np.linalg.matrix_rank(Q) == 2
    assert pls.invert([1e-4, 2e-4]).null_space_dimension == 1


def test_pls_invert_refuses_a_target_or_a_move_it_cannot_use_in_full() -> None:
    """The wrong number of target values, a missing one, and the wrong number of null-space coordinates."""
    X, Y = _pls_block(n=15)
    pls = PLS(n_components=3).fit(X, Y)
    with pytest.raises(
        ValueError, match=r"^y_desired has 3 value\(s\); expected 2 to match the fitted target\(s\): \['y0', 'y1'\]\.$"
    ):
        pls.invert([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"^y_desired is missing value\(s\) for target\(s\): \['y0', 'y1'\]\.$"):
        pls.invert({"y0": 1.0})
    message = r"^null_space_coordinates has 2 value\(s\); expected 1 to match the null-space dimension \(A - r\)\.$"
    with pytest.raises(ValueError, match=message):
        pls.invert([1.0, 2.0], null_space_coordinates=[1.0, 2.0])


def test_pls_invert_reads_decimal_values_as_the_floats_they_hold() -> None:
    """A target and a null-space move given as ``Decimal`` are read as floats, by a model that scales nothing too."""
    X, Y = _pls_block(n=15)
    X, Y = X - X.mean(), Y[["y0"]] - Y[["y0"]].mean()
    pls = PLS(n_components=2, scale=False).fit(X, Y)
    exact = pls.invert(1.5, null_space_coordinates=[0.25])
    given = pls.invert(Decimal("1.5"), null_space_coordinates=[Decimal("0.25")])
    pd.testing.assert_series_equal(given.x_new, exact.x_new)
    pd.testing.assert_series_equal(given.scores, exact.scores)


def test_pls_prediction_interval_by_hand() -> None:
    """``y_hat -/+ t * s_E * sqrt(1 + 1/N + T2 / (N - 1))``, with the t quantile on ``N - A - 1`` degrees of freedom.

    ``s_E`` is the calibration RMSE on that many degrees of freedom, ``RMSE * sqrt(N / (N - A - 1))``,
    or a cross-validated RMSE as given. The bounds keep the new rows' labels.
    """
    X, Y = _pls_block(n=15)
    pls = PLS(n_components=2).fit(X, Y)
    new = pd.DataFrame([[3.5, 2.0, 5.0, 1.0], [2.5, 3.2, 2.6, 4.0]], columns=X.columns, index=["p", "q"])
    by_hand = _pls_new_rows_by_hand(pls, X, Y, new.to_numpy())
    fitted = _pls_new_rows_by_hand(pls, X, Y, X.to_numpy())["y_hat"]
    rmse = np.sqrt(np.mean((Y.to_numpy() - fitted) ** 2, axis=0))
    quantile = t_dist.ppf(0.95, 12)
    spread = np.sqrt(1 + 1 / 15 + by_hand["t2"] / 14)[:, None]
    for cv_result, error_sd in ((None, rmse * np.sqrt(15 / 12)), (Bunch(rmse_cv=pd.Series([0.5, 2.0])), [0.5, 2.0])):
        interval = pls.prediction_interval(new, conf_level=0.9, cv_result=cv_result)
        expected = pd.DataFrame(by_hand["y_hat"], index=["p", "q"], columns=["y0", "y1"])
        pd.testing.assert_frame_equal(interval.y_hat, expected, rtol=1e-10)
        half = quantile * spread * np.asarray(error_sd)
        pd.testing.assert_frame_equal(interval.lower, expected - half, rtol=1e-10)
        pd.testing.assert_frame_equal(interval.upper, expected + half, rtol=1e-10)
        assert interval.conf_level == 0.9
    assert pls.prediction_interval(new).lower.iloc[0, 0] == pytest.approx(
        by_hand["y_hat"][0, 0] - t_dist.ppf(0.975, 12) * spread[0, 0] * rmse[0] * np.sqrt(15 / 12), rel=1e-10
    )


@pytest.mark.parametrize(("components", "scale"), [(2, True), (3, False)])
def test_pls_prediction_interval_keeps_at_least_one_degree_of_freedom(components: int, scale: bool) -> None:
    """With ``N - A - 1`` at 1, or at 0 (four rows left uncentred), the interval uses 1 degree of freedom."""
    X, Y = _pls_block(n=4)
    pls = PLS(n_components=components, scale=scale, warn_on_uncentred=False).fit(X, Y[["y0"]])
    interval = pls.prediction_interval(X, conf_level=0.9)
    rmse = pls.rmse_.iloc[:, -1].to_numpy()
    assert rmse > 1e-3
    t2 = pls.diagnose(X).hotellings_t2.to_numpy()
    half = t_dist.ppf(0.95, 1) * np.sqrt(1 + 1 / 4 + t2 / 3) * rmse * np.sqrt(4 / 1)
    np.testing.assert_allclose((interval.upper - interval.y_hat).to_numpy().ravel(), half, rtol=1e-10)


def test_pls_prediction_interval_refuses_a_level_outside_0_5_to_1_in_full() -> None:
    """Both ends are refused, as for ``cross_validate``."""
    X, Y = _pls_block()
    pls = PLS(n_components=1).fit(X, Y)
    for level in (0.5, 1.0):
        with pytest.raises(ValueError, match=rf"^conf_level must be between 0\.5 and 1\.0, got {level}\.$"):
            pls.prediction_interval(X, conf_level=level)


def _record_calls(monkeypatch: pytest.MonkeyPatch, name: str) -> list[tuple]:
    """Stand in for ``process_improve.multivariate._pls.<name>``, recording each call's arguments."""
    calls: list[tuple] = []
    monkeypatch.setattr(f"process_improve.multivariate._pls.{name}", lambda *args: calls.append(args) or Bunch())
    return calls


def test_pls_compare_cv_criteria_passes_its_settings_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The method and the function hand the class, the blocks, every setting and the PLS keywords to the criteria."""
    calls = _record_calls(monkeypatch, "_compare_cv_criteria")
    X, Y = _pls_block()
    PLS.compare_cv_criteria(X, Y)
    defaults = _CompareSettings(
        max_components=None,
        cv=7,
        random_state=None,
        n_permutations=999,
        n_cv_permutations=199,
        alpha=0.05,
        angle_threshold=30.0,
        conf_level=0.95,
    )
    settings = {
        "max_components": 3,
        "cv": 4,
        "random_state": 11,
        "n_permutations": 49,
        "n_cv_permutations": 9,
        "alpha": 0.1,
        "angle_threshold": 20.0,
        "conf_level": 0.9,
    }
    PLS.compare_cv_criteria(X, Y, **settings, max_iter=50)
    compare_cv_criteria(X, Y, **settings, tol=1e-6)
    (cls, x, y, given, kwargs), (_, _, _, chosen, more), (function_cls, x2, y2, again, other) = calls
    assert cls is PLS
    assert function_cls is PLS
    assert x is X
    assert y is Y
    assert x2 is X
    assert y2 is Y
    assert given == defaults
    assert kwargs == {}
    assert chosen == _CompareSettings(**settings)
    assert more == {"max_iter": 50}
    assert again == _CompareSettings(**settings)
    assert other == {"tol": 1e-6}


def test_pls_pseudo_validation_set_passes_its_settings_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The method and the function hand the class, the blocks, every setting and the PLS keywords to the builder."""
    calls = _record_calls(monkeypatch, "_pseudo_validation_set")
    X, Y = _pls_block()
    PLS.pseudo_validation_set(X, Y, n_components=2)
    PLS.pseudo_validation_set(X, Y, n_components=3, cv=4, scope="local", random_state=5, max_iter=50)
    pseudo_validation_set(X, Y, n_components=1, cv=3, tol=1e-6)
    (cls, x, y, given, kwargs), (_, _, _, chosen, more), (function_cls, x2, y2, again, other) = calls
    assert cls is PLS
    assert function_cls is PLS
    assert x is X
    assert y is Y
    assert x2 is X
    assert y2 is Y
    assert given == _PseudoValidationSettings(n_components=2, cv=7, scope="global", random_state=None)
    assert kwargs == {}
    assert chosen == _PseudoValidationSettings(n_components=3, cv=4, scope="local", random_state=5)
    assert more == {"max_iter": 50}
    assert again == _PseudoValidationSettings(n_components=1, cv=3, scope="global", random_state=None)
    assert other == {"tol": 1e-6}


def test_pls_score_weighs_the_rows_it_is_given() -> None:
    """``score`` is the mean over targets of ``1 - sum w e^2 / sum w (y - ybar_w)^2``, the weighted R²."""
    X, Y = _pls_block(n=15)
    pls = PLS(n_components=1).fit(X, Y)
    weights = np.linspace(0.1, 2.0, 15)
    y, residual = Y.to_numpy(), Y.to_numpy() - pls.predict(X).to_numpy()
    centre = weights @ y / weights.sum()
    r2 = 1 - (weights @ residual**2) / (weights @ (y - centre) ** 2)
    assert pls.score(X, Y, sample_weight=weights) == pytest.approx(r2.mean(), rel=1e-12)
    assert pls.score(X, Y, sample_weight=weights) != pytest.approx(pls.score(X, Y), rel=1e-6)


def _sklearn_pls(n_components: int) -> PLSRegression:
    """Return sklearn's PLSRegression, converged far past its default tolerance (a squared change of 1e-6)."""
    return PLSRegression(n_components=n_components, tol=1e-24, max_iter=10_000)


def _stand_in_selection(monkeypatch: pytest.MonkeyPatch, counts: list[int]) -> list[tuple]:
    """Stand in for ``PLS.select_n_components``: record each call and recommend the next of ``counts``."""
    calls: list[tuple] = []

    def select(cls: type, X: pd.DataFrame, Y: pd.DataFrame, **kwargs: object) -> Bunch:
        calls.append((cls, X, Y, kwargs))
        return Bunch(n_components=counts[len(calls) - 1])

    monkeypatch.setattr(PLS, "select_n_components", classmethod(select))
    return calls


def _nested_scores_by_hand(Y: np.ndarray, oof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return RMSEP and Q2 (per column, then total) of out-of-fold predictions, over the rows predicted.

    Only observed cells count; the total sum of squares is about each column's mean over all rows.
    """
    covered = ~np.isnan(oof).any(axis=1)
    residual, y = (Y - oof)[covered], Y[covered]
    observed = ~np.isnan(residual)
    press = np.array([np.sum(residual[observed[:, m], m] ** 2) for m in range(Y.shape[1])])
    counts = observed.sum(axis=0)
    tss = np.array([np.sum((y[observed[:, m], m] - np.nanmean(Y[:, m])) ** 2) for m in range(Y.shape[1])])
    rmsep = [np.sqrt(p / c) if c else np.nan for p, c in zip(press, counts, strict=True)]
    q2 = [1 - p / t if t > 0 else np.nan for p, t in zip(press, tss, strict=True)]
    rmsep.append(np.sqrt(press.sum() / counts.sum()) if counts.sum() else np.nan)
    q2.append(1 - press.sum() / tss.sum() if tss.sum() > 0 else np.nan)
    return np.array(rmsep), np.array(q2)


def test_pls_nested_cv_by_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each outer fold asks the inner selection for a count, fits PLS at it, and predicts its held-out rows.

    The selection is stood in for, so the defaults it is handed can be read. Each fold's
    predictions are sklearn's ``PLSRegression`` at the chosen count on the fold's training rows
    (autoscaled with ddof=1 there too); RMSEP and Q2 are worked from them.
    """
    calls = _stand_in_selection(monkeypatch, [1, 2, 1])
    X, Y = _pls_block(n=15)
    result = PLS.nested_cv(X, Y, outer_cv=3, random_state=4)
    oof = np.full((15, 2), np.nan)
    splits = list(KFold(n_splits=3, shuffle=True, random_state=4).split(X))
    assert len(calls) == 3
    for i, ((train, test), count) in enumerate(zip(splits, [1, 2, 1], strict=True)):
        cls, x, y, kwargs = calls[i]
        assert cls is PLS
        _assert_frame_near(x, X.iloc[train])
        _assert_frame_near(y, Y.iloc[train])
        assert kwargs == {
            "max_components": None,
            "cv": 5,
            "n_repeats": 10,
            "selection_rule": "1se",
            "scale_inside_folds": True,
            "min_q2_increase": 0.01,
            "n_permutations": 999,
            "alpha": 0.01,
            "random_state": 4 + i,
        }
        oof[test] = _sklearn_pls(count).fit(X.iloc[train], Y.iloc[train]).predict(X.iloc[test])
    _assert_frame_near(result.cv_predictions, pd.DataFrame(oof, index=Y.index, columns=Y.columns))
    rmsep, q2 = _nested_scores_by_hand(Y.to_numpy(), oof)
    _assert_series_near(result.rmsep, pd.Series(rmsep, index=["y0", "y1", "total"], name="RMSEP"))
    _assert_series_near(result.q2y, pd.Series(q2, index=["y0", "y1", "total"], name="Q2Y"))
    assert result.selected_components_per_fold == [1, 2, 1]
    expected = pd.Series([2 / 3, 1 / 3], index=pd.Index([1, 2], name="n_components"), name="vote_share")
    _assert_series_near(result.selected_components_distribution, expected)


def test_pls_nested_cv_passes_its_settings_on_and_predicts_as_the_fold_models_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every setting reaches the inner selection, the PLS keywords the final fits too (#1027).

    Without in-fold scaling the fold models autoscale themselves (``scale=True``), and each
    held-out row is predicted as its fold model predicts it, which is sklearn's
    ``PLSRegression`` on the rows as given.
    """
    calls = _stand_in_selection(monkeypatch, [2, 1])
    fitted: list[tuple[int, int]] = []
    fit = PLS.fit

    def recording_fit(self: PLS, *args: object, **kwargs: object) -> PLS:
        fitted.append((self.n_components, self.max_iter))
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(PLS, "fit", recording_fit)
    X, Y = _pls_block(n=12)
    settings = {
        "max_components": 2,
        "inner_cv": 3,
        "n_inner_repeats": 2,
        "selection_rule": "min",
        "scale_inside_folds": False,
        "min_q2_increase": 0.05,
        "n_permutations": 19,
        "alpha": 0.05,
    }
    result = PLS.nested_cv(X, Y, outer_cv=2, **settings, max_iter=300)
    for _, _, _, kwargs in calls:
        assert kwargs == {
            "max_components": 2,
            "cv": 3,
            "n_repeats": 2,
            "selection_rule": "min",
            "scale_inside_folds": False,
            "min_q2_increase": 0.05,
            "n_permutations": 19,
            "alpha": 0.05,
            "random_state": None,
            "max_iter": 300,
        }
    assert fitted == [(2, 300), (1, 300)]
    oof = np.full((12, 2), np.nan)
    # The outer folds are shuffled without a seed: each one's training rows are those it handed the selection.
    for (_, x_train, _, _), count in zip(calls, [2, 1], strict=True):
        rows = X.index.get_indexer(x_train.index)
        held = np.setdiff1d(np.arange(12), rows)
        oof[held] = _sklearn_pls(count).fit(X.iloc[rows], Y.iloc[rows]).predict(X.iloc[held])
    _assert_near(result.cv_predictions.to_numpy(), oof, rtol=1e-6)


class _RecordingSplitter(BaseCrossValidator):
    """Two folds of alternate rows, recording what ``split`` is given."""

    def __init__(self) -> None:
        self.seen: list[tuple[object, object]] = []

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:  # noqa: ARG002
        return 2

    def split(self, X: object, y: object = None, groups: object = None):
        self.seen.append((X, y))
        yield from super().split(X, y, groups)

    def _iter_test_indices(self, X: object = None, y: object = None, groups: object = None):  # noqa: ARG002
        n = len(X)  # type: ignore[arg-type]
        yield from (np.arange(0, n, 2), np.arange(1, n, 2))


def test_pls_nested_cv_hands_both_blocks_to_its_splitter_and_needs_two_folds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A splitter is given X and Y, as one that stratifies needs; an integer below 2 is refused in full."""
    _stand_in_selection(monkeypatch, [1, 1, 1, 1])
    X, Y = _pls_block()
    splitter = _RecordingSplitter()
    PLS.nested_cv(X, Y, outer_cv=splitter)
    ((x, y),) = splitter.seen
    pd.testing.assert_frame_equal(x, X)
    pd.testing.assert_frame_equal(y, Y)
    assert PLS.nested_cv(X, Y, outer_cv=2).selected_components_per_fold == [1, 1]
    with pytest.raises(ValueError, match=r"^outer_cv must be >= 2 when given as an int; got 1\.$"):
        PLS.nested_cv(X, Y, outer_cv=1)
    _stand_in_selection(monkeypatch, [1] * 5)
    assert PLS.nested_cv(X, Y).selected_components_per_fold == [1] * 5  # five outer folds by default


@pytest.mark.parametrize(
    ("blanks", "missed"),
    [
        ([(1, 1), (2, 1)], 0),
        ([(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)], 1),
        ([(0, 1), (1, 1), (2, 1)], 1),
        ([(r, c) for r in range(3) for c in range(2)], 3),
    ],
    ids=["one-cell-in-y1", "one-cell-in-all", "no-cell-in-y1", "none"],
)
def test_pls_nested_cv_scores_only_the_observed_cells_of_the_rows_it_predicts(
    monkeypatch: pytest.MonkeyPatch, blanks: list[tuple[int, int]], missed: int
) -> None:
    """One outer fold holds out three rows. RMSEP and Q2 count their observed responses only, NaN where none.

    The responses are in hundredths, so a total sum of squares below 1 still counts, and one
    observed cell is enough for an RMSEP. Nothing warns.
    """
    _stand_in_selection(monkeypatch, [1])
    X, Y = _pls_block(n=12)
    Y = Y / 100.0
    for row, column in blanks:
        Y.iloc[row, column] = np.nan
    one_fold = PredefinedSplit([0, 0, 0] + [-1] * 9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = PLS.nested_cv(X, Y, outer_cv=one_fold)
    oof = np.full((12, 2), np.nan)
    oof[:3] = _sklearn_pls(1).fit(X.iloc[3:], Y.iloc[3:]).predict(X.iloc[:3])
    _assert_near(result.cv_predictions.to_numpy(), oof, rtol=1e-6)
    rmsep, q2 = _nested_scores_by_hand(Y.to_numpy(), oof)
    _assert_near(result.rmsep.to_numpy(), rmsep, rtol=1e-6)
    _assert_near(result.q2y.to_numpy(), q2, rtol=1e-6)
    assert np.isnan(rmsep).sum() == missed
    assert np.nansum((Y.to_numpy()[:3] - np.nanmean(Y.to_numpy(), axis=0)) ** 2) < 1


def test_pls_nested_cv_scales_inside_folds_for_a_model_that_does_not_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """``scale=False`` fold models on autoscaled folds predict as sklearn's autoscaling PLSRegression does."""
    calls = _stand_in_selection(monkeypatch, [2, 2])
    X, Y = _pls_block(n=12)
    result = PLS.nested_cv(X, Y, outer_cv=2, random_state=5, scale=False)
    assert all(kwargs["scale"] is False for _, _, _, kwargs in calls)
    oof = np.full((12, 2), np.nan)
    for train, test in KFold(n_splits=2, shuffle=True, random_state=5).split(X):
        oof[test] = _sklearn_pls(2).fit(X.iloc[train], Y.iloc[train]).predict(X.iloc[test])
    _assert_near(result.cv_predictions.to_numpy(), oof, rtol=1e-6)


def test_pls_nested_cv_projects_a_held_out_row_with_a_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A held-out row with a missing cell is predicted as its fold model's ``project`` predicts it."""
    _stand_in_selection(monkeypatch, [2, 2])
    X, Y = _pls_block(n=12)
    X.iloc[[0, 1], [1, 2]] = np.nan
    splitter = PredefinedSplit([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    result = PLS.nested_cv(X, Y, outer_cv=splitter)
    for held in (np.arange(6), np.arange(6, 12)):
        trained = np.setdiff1d(np.arange(12), held)
        model = PLS(n_components=2).fit(X.iloc[trained], Y.iloc[trained])
        expected = model.project(X.iloc[held]).y_hat.to_numpy()
        _assert_near(result.cv_predictions.iloc[held].to_numpy(), expected, rtol=1e-6)


def _cross_validated_by_hand(
    X: pd.DataFrame, Y: pd.DataFrame, resamples: list, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return sklearn's coefficients on each resample's rows of weight above 0, and its held-out predictions."""
    betas, predictions = [], np.full(Y.shape, np.nan)
    for train, test in resamples:
        rows = train if weights is None else train[weights[train] > 0]
        model = _sklearn_pls(2).fit(X.iloc[rows], Y.iloc[rows])
        betas.append(model.coef_.T)
        if test is not None:
            predictions[test] = model.predict(X.iloc[test])
    return np.array(betas), predictions


@pytest.mark.parametrize(("cv", "weighted"), [("loo", False), (4, False), ("loo", True)])
def test_pls_cross_validate_by_hand(cv: str | int, weighted: bool, capsys: pytest.CaptureFixture) -> None:
    """Jackknife and K-fold: the coefficients of each resample's fit, their jackknife spread, and the CV predictions.

    With G resamples the standard error is ``sqrt((G - 1) / G * sum (b_g - b_mean)^2)`` and the
    interval ``b_mean -/+ t(1 - alpha / 2, G - 1) * se``; a weight of 0 drops a row from every fit
    it is in, but not from the predictions. A constant column has a coefficient of exactly 0,
    whose interval is [0, 0]: not significant. Arrays are numbered from 0, as frames keep their
    labels. The progress bar names the scheme.
    """
    X, Y = _pls_block(n=10)
    X["flat"] = 3.0
    weights = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1.0]) if weighted else None
    pls = PLS(n_components=2).fit(X, Y)
    given = (X, Y) if weighted else (X.to_numpy(), Y.to_numpy())
    result = pls.cross_validate(*given, cv=cv, conf_level=0.9, random_state=2, sample_weight=weights)
    rows, targets = (X.index, Y.columns) if weighted else (pd.RangeIndex(10), pd.RangeIndex(2))
    if cv == "loo":
        resamples = [(np.delete(np.arange(10), i), np.array([i])) for i in range(10)]
    else:
        resamples = list(KFold(n_splits=4, shuffle=True, random_state=2).split(X))
    betas, predictions = _cross_validated_by_hand(X, Y, resamples, weights)
    G = len(resamples)
    _assert_near(result.beta_samples, betas, rtol=1e-6)
    mean = betas.mean(axis=0)
    se = np.sqrt((G - 1) / G * np.sum((betas - mean) ** 2, axis=0))
    lower, upper = mean - t_dist.ppf(0.95, G - 1) * se, mean + t_dist.ppf(0.95, G - 1) * se
    for name, expected in (("beta_mean", mean), ("beta_std", se), ("beta_ci_lower", lower), ("beta_ci_upper", upper)):
        _assert_frame_near(result[name], pd.DataFrame(expected, index=X.columns, columns=Y.columns), rtol=1e-5)
    _assert_frame_near(result.significant, pd.DataFrame((lower > 0) | (upper < 0), index=X.columns, columns=Y.columns))
    assert (lower[4] == 0).all()
    assert (upper[4] == 0).all()
    assert not result.significant.loc["flat"].any()
    _assert_frame_near(result.y_hat_cv, pd.DataFrame(predictions, index=rows, columns=targets), rtol=1e-6)
    residual = Y.to_numpy() - predictions
    assert result.press == pytest.approx(np.sum(residual**2), rel=1e-6)
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    _assert_series_near(result.rmse_cv, pd.Series(rmse, index=targets, name="RMSE_CV"), rtol=1e-6)
    q2 = 1 - np.sum(residual**2, axis=0) / np.sum((Y.to_numpy() - Y.to_numpy().mean(axis=0)) ** 2, axis=0)
    _assert_series_near(result.q_squared, pd.Series(q2, index=targets, name="Q_squared"), rtol=1e-6)
    assert result.n_resamples == G
    assert result.method == {"loo": "jackknife", 4: "kfold"}[cv]
    assert result.conf_level == 0.9
    bar = capsys.readouterr().err
    assert (
        ("Jackknife (LOO): 100%" in bar and "10/10" in bar)
        if cv == "loo"
        else ("4-Fold CV: 100%" in bar and "4/4" in bar)
    )


def test_pls_cross_validate_bootstrap_by_hand(capsys: pytest.CaptureFixture) -> None:
    """Bootstrap: rows drawn with replacement by ``default_rng(random_state)``, a percentile interval, no CV."""
    X, Y = _pls_block(n=10)
    pls = PLS(n_components=2).fit(X, Y)
    result = pls.cross_validate(X, Y, n_bootstrap=5, conf_level=0.8, random_state=3, show_progress=False)
    assert capsys.readouterr().err == ""
    rng = np.random.default_rng(3)
    draws = [(rng.choice(10, size=10, replace=True), None) for _ in range(5)]
    betas, _ = _cross_validated_by_hand(X, Y, draws)
    _assert_near(result.beta_samples, betas, rtol=1e-6)
    lower, upper = np.percentile(betas, 10, axis=0), np.percentile(betas, 90, axis=0)
    for name, expected in (
        ("beta_mean", betas.mean(axis=0)),
        ("beta_std", betas.std(axis=0, ddof=1)),
        ("beta_ci_lower", lower),
        ("beta_ci_upper", upper),
    ):
        _assert_frame_near(result[name], pd.DataFrame(expected, index=X.columns, columns=Y.columns), rtol=1e-5)
    _assert_frame_near(result.significant, pd.DataFrame((lower > 0) | (upper < 0), index=X.columns, columns=Y.columns))
    assert (result.y_hat_cv, result.press, result.rmse_cv, result.q_squared) == (None, None, None, None)
    assert (result.n_resamples, result.method, result.conf_level) == (5, "bootstrap", 0.8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # one resample has no spread: its sd is NaN
        single = pls.cross_validate(X, Y, n_bootstrap=1, random_state=3)
    assert (single.method, single.n_resamples) == ("bootstrap", 1)
    assert "Bootstrap: 100%" in capsys.readouterr().err


def test_pls_cross_validate_refuses_a_level_outside_0_5_to_1_in_full() -> None:
    """Both ends are refused."""
    X, Y = _pls_block()
    pls = PLS(n_components=1).fit(X, Y)
    for level in (0.5, 1.0):
        with pytest.raises(ValueError, match=rf"^conf_level must be between 0\.5 and 1\.0, got {level}\.$"):
            pls.cross_validate(X, Y, conf_level=level, show_progress=False)


def test_pls_fit_transform_names_the_missing_response_in_full() -> None:
    """``fit_transform`` without Y says why the default exists and what to pass."""
    X, _ = _pls_block()
    message = (
        "Y is required by PLS.fit_transform; got None. The None default exists only to match the sklearn "
        "fit_transform(X, y=None) signature, and PLS cannot be fitted without responses. Pass the response block, "
        "e.g. model.fit_transform(X, Y)."
    )
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        PLS(n_components=1).fit_transform(X)


def _masked_regression(block: np.ndarray, vector: np.ndarray, per: str) -> np.ndarray:
    """Regress ``vector`` on each column (``per="column"``) or row of ``block``, over its observed cells only."""
    present, values = ~np.isnan(block), np.nan_to_num(block)
    if per == "column":
        return (values.T @ vector) / (present.T @ vector**2)
    return (values @ vector) / (present @ vector**2)


def _pls_nipals_as_fitted(X: np.ndarray, Y: np.ndarray, A: int, tol: float = epsqrt, max_iter: int = 500) -> dict:
    """NIPALS PLS as ``PLS`` runs it, written out, with the iterations each component took.

    Each component is seeded from the Y column of largest sum of squares (a missing cell as
    0), plus 1; an iteration is ``w = X'u / u'u`` normalised, ``t = Xw / w'w``,
    ``c = Y't / t't``, ``u = Yc / c'c``, each over the observed cells; the loop stops once
    ``||u_old - u|| / ||u|| < tol``, or after ``max_iter`` iterations. Then ``p = X't / t't``,
    both blocks are deflated, and the component is turned so that its largest loading is
    positive.
    """
    X, Y = np.array(X, dtype=float), np.array(Y, dtype=float)
    out: dict = {name: [] for name in ("T", "U", "W", "P", "C", "iterations")}
    for _ in range(A):
        seed = np.nan_to_num(Y[:, [np.argmax(np.nansum(Y**2, axis=0))]])
        u, iterations = seed + 1.0, 0
        while True:
            previous = u
            w = _masked_regression(X, u, "column")
            w = w / np.linalg.norm(w)
            t = _masked_regression(X, w, "row")
            c = _masked_regression(Y, t, "column")
            u = _masked_regression(Y, c, "row")
            iterations += 1
            if np.linalg.norm(previous - u) / np.linalg.norm(u) < tol or iterations >= max_iter:
                break
        p = _masked_regression(X, t, "column")
        X, Y = X - t @ p.T, Y - t @ c.T
        sign = np.sign(p[np.argmax(np.abs(p))])
        for name, vector in (("T", t), ("U", u), ("W", w), ("P", p), ("C", c)):
            out[name].append((sign * vector).ravel())
        out["iterations"].append(iterations)
    return {name: np.array(values) if name == "iterations" else np.column_stack(values) for name, values in out.items()}


def _by_hand_diagnostics(Z: np.ndarray, Zy: np.ndarray, ref: dict, y_scale: np.ndarray) -> dict:
    """Return what ``fit`` reports from a model's T, P and C on its scaled blocks, worked over the observed cells.

    After ``a`` components: SPE is the root of each row's residual sum of squares; R2 per
    variable is ``1 - SS(residual) / SS(column)`` (NaN for a column of zeros); cumulative R2Y
    is pooled over Y; RMSE is per target, over its observed cells, in Y's units.
    """
    T, P, C = ref["T"], ref["P"], ref["C"]
    out: dict = {name: [] for name in ("spe", "r2x", "r2y", "r2", "rmse")}
    for a in range(1, T.shape[1] + 1):
        ex, ey = Z - T[:, :a] @ P[:, :a].T, Zy - T[:, :a] @ C[:, :a].T
        out["spe"].append(np.sqrt(np.nansum(ex**2, axis=1)))
        x_total, y_total = np.nansum(Z**2, axis=0), np.nansum(Zy**2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["r2x"].append(np.where(x_total > 0, 1 - np.nansum(ex**2, axis=0) / x_total, np.nan))
            out["r2y"].append(np.where(y_total > 0, 1 - np.nansum(ey**2, axis=0) / y_total, np.nan))
        out["r2"].append(1 - np.nansum(ey**2) / np.nansum(Zy**2))
        out["rmse"].append(np.sqrt(np.nanmean(ey**2, axis=0)) * y_scale)
    return {name: np.array(values) if name == "r2" else np.column_stack(values) for name, values in out.items()}


@pytest.mark.parametrize("scale", [True, False])
def test_pls_fit_by_hand(scale: bool) -> None:
    """Every fitted attribute of a three-component model of two responses, against NIPALS written out.

    With ``scale=True`` both blocks are autoscaled (ddof=1) first and the coefficients,
    predictions and RMSE are reported back in their units; the coefficients are sklearn's.
    Each block carries a constant column, whose R2 per variable is NaN without a warning;
    with ``scale=False`` each also has one whose sum of squares is below 1, an ordinary value.
    """
    X, Y = _pls_block(n=15)
    X["flat"], Y["none"] = (7.0, 3.0) if scale else (0.0, 0.0)
    if not scale:
        X, Y = X - X.mean(), Y - Y.mean()
        X["small"] = np.r_[0.5, -0.5, np.zeros(13)]
        Y["tiny"] = 0.05 * X["x0"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pls = PLS(n_components=3, scale=scale).fit(X, Y)
    with np.errstate(invalid="ignore"):  # the constant columns are 0 / 0 here, set to 0 below
        Z, Zy = (_autoscaled(X.to_numpy()), _autoscaled(Y.to_numpy())) if scale else (X.to_numpy(), Y.to_numpy())
    x_sd = X.std(ddof=1).to_numpy(copy=True) if scale else np.ones(X.shape[1])
    y_sd, y_mean = (Y.std(ddof=1).to_numpy(copy=True), Y.mean().to_numpy()) if scale else (np.ones(4), np.zeros(4))
    if scale:  # a constant column is centred and left unscaled
        Z[:, 4], Zy[:, 2], x_sd[4], y_sd[2] = 0.0, 0.0, 1.0, 1.0
    ref = _pls_nipals_as_fitted(Z, Zy, 3)
    components = [1, 2, 3]
    frames = {
        "scores_": (ref["T"], X.index),
        "y_scores_": (ref["U"], Y.index),
        "x_weights_": (ref["W"], X.columns),
        "x_loadings_": (ref["P"], X.columns),
        "y_loadings_": (ref["C"], Y.columns),
        "y_weights_": (ref["C"], Y.columns),
    }
    for name, (values, index) in frames.items():
        _assert_frame_near(getattr(pls, name), pd.DataFrame(values, index=index, columns=components), rtol=1e-6)
    R = ref["W"] @ np.linalg.inv(ref["P"].T @ ref["W"])
    _assert_frame_near(pls.direct_weights_, pd.DataFrame(R, index=X.columns, columns=components), rtol=1e-6)
    beta = R @ ref["C"].T * y_sd / x_sd[:, None]
    _assert_frame_near(pls.beta_coefficients_, pd.DataFrame(beta, index=X.columns, columns=Y.columns), rtol=1e-6)
    if scale:
        sklearn = _sklearn_pls(3).fit(X.drop(columns="flat"), Y.drop(columns="none"))
        _assert_near(pls.beta_coefficients_.drop(index="flat", columns="none").to_numpy(), sklearn.coef_.T, rtol=1e-6)
    predictions = ref["T"] @ ref["C"].T * y_sd + y_mean
    _assert_frame_near(pls.predictions_, pd.DataFrame(predictions, index=Y.index, columns=Y.columns), rtol=1e-6)
    variance = np.sum(ref["T"] ** 2, axis=0) / 14
    _assert_near(pls.explained_variance_, variance, rtol=1e-6)
    _assert_series_near(
        pls.scaling_factor_for_scores_,
        pd.Series(np.sqrt(variance), index=components, name="Standard deviation per score"),
        rtol=1e-6,
    )
    t2 = np.cumsum(ref["T"] ** 2 / variance, axis=1)
    _assert_frame_near(pls.hotellings_t2_, pd.DataFrame(t2, index=X.index, columns=components), rtol=1e-6)
    _assert_near(pls._x_residuals, Z - ref["T"] @ ref["P"].T, rtol=1e-6, atol=1e-8)
    hand = _by_hand_diagnostics(Z, Zy, ref, y_sd)
    _assert_frame_near(pls.spe_, pd.DataFrame(hand["spe"], index=X.index, columns=components), rtol=1e-6)
    _assert_frame_near(pls.r2_per_variable_, pd.DataFrame(hand["r2x"], index=X.columns, columns=components), rtol=1e-6)
    assert pls.r2_per_variable_.loc["flat"].isna().all()
    assert pls.r2y_per_variable_.loc["none"].isna().all()
    _assert_frame_near(pls.r2y_per_variable_, pd.DataFrame(hand["r2y"], index=Y.columns, columns=components), rtol=1e-6)
    _assert_series_near(
        pls.r2_cumulative_, pd.Series(hand["r2"], index=components, name="Output cumulative R²"), rtol=1e-6
    )
    _assert_series_near(
        pls.r2_per_component_,
        pd.Series(np.diff(hand["r2"], prepend=0.0), index=components, name="Output R² per component"),
        rtol=1e-6,
    )
    _assert_frame_near(pls.rmse_, pd.DataFrame(hand["rmse"], index=Y.columns, columns=components), rtol=1e-6)
    np.testing.assert_array_equal(pls.fitting_info_["iterations"], ref["iterations"])
    assert pls.fitting_info_["timing"].shape == (3,)
    assert (pls.fitting_info_["timing"] >= 0).all()
    assert (pls.fitting_info_["timing"] < 60).all()  # seconds this fit took, not a clock reading
    assert (pls.n_samples_, pls.n_features_in_, pls.n_targets_, pls.n_components_) == (15, X.shape[1], Y.shape[1], 3)
    assert pls.has_missing_data_ is False


def test_pls_fit_with_missing_cells_by_hand() -> None:
    """NIPALS skips each missing cell in its sums, and every diagnostic counts the observed cells only (#1027).

    One X cell and one Y cell are missing; the seed column's missing cell counts as 0. The
    RMSE of the response with a gap is over its observed cells: it was NaN.
    """
    X, Y = _gapped_pls_block()
    X, Y = X - X.mean(), Y - Y.mean()
    pls = PLS(n_components=2, scale=False).fit(X, Y)
    ref = _pls_nipals_as_fitted(X.to_numpy(), Y.to_numpy(), 2)
    for name, attribute in (("T", "scores_"), ("W", "x_weights_"), ("P", "x_loadings_"), ("C", "y_loadings_")):
        _assert_near(np.asarray(getattr(pls, attribute)), ref[name], rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(pls.fitting_info_["iterations"], ref["iterations"])
    hand = _by_hand_diagnostics(X.to_numpy(), Y.to_numpy(), ref, np.ones(2))
    _assert_near(pls.spe_.to_numpy(), hand["spe"], rtol=1e-6)
    _assert_near(pls.r2y_per_variable_.to_numpy(), hand["r2y"], rtol=1e-6)
    _assert_near(pls.r2_cumulative_.to_numpy(), hand["r2"], rtol=1e-6)
    _assert_near(pls.rmse_.to_numpy(), hand["rmse"], rtol=1e-6)
    assert np.isfinite(pls.rmse_.to_numpy()).all()
    assert pls.has_missing_data_ is True
    _assert_near(pls._x_residuals, np.nan_to_num(X.to_numpy() - ref["T"] @ ref["P"].T), atol=1e-9)
    for gapped in ((X, Y.fillna(0.0)), (X.fillna(0.0), Y)):
        assert PLS(n_components=1, scale=False).fit(*gapped).has_missing_data_ is True


def test_pls_fit_labels_its_results_from_the_blocks_it_is_given() -> None:
    """A labelled X labels the rows of every result; a 1-D response array follows it, a Series keeps its own labels.

    A 1-D array of responses was labelled 0..N-1 while a 2-D one took X's row labels (#1027).
    Arrays are numbered from 0.
    """
    X, Y = _pls_block()
    for y in (Y["y0"].to_numpy(), Y[["y0"]].to_numpy()):
        pls = PLS(n_components=1).fit(X, y)
        assert pls.predictions_.index.equals(X.index)
        assert pls.y_scores_.index.equals(X.index)
        assert list(pls.predictions_.columns) == [0]
    series = pd.Series(Y["y0"].to_numpy(), index=[f"s{i}" for i in range(12)], name="q")
    pls = PLS(n_components=1).fit(X, series)
    assert list(pls.y_loadings_.index) == ["q"]
    assert list(pls.predictions_.columns) == ["q"]
    assert list(pls.predictions_.index) == list(series.index)
    bare = PLS(n_components=1).fit(X.to_numpy(), Y.to_numpy())
    assert bare.scores_.index.equals(pd.RangeIndex(12))
    assert bare.x_weights_.index.equals(pd.RangeIndex(4))
    assert bare.y_loadings_.index.equals(pd.RangeIndex(2))
    np.testing.assert_allclose(bare.scores_.to_numpy(), PLS(n_components=1).fit(X, Y).scores_.to_numpy())


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda X, Y: PLS(n_components=1).fit(X, Y, sample_weight=np.r_[np.nan, np.ones(11)]),
            ValueError,
            "sample_weight must be finite (no NaN / inf).",
        ),
        (
            lambda X, Y: PLS(n_components=1).fit(X, Y, sample_weight=np.r_[np.inf, np.ones(11)]),
            ValueError,
            "sample_weight must be finite (no NaN / inf).",
        ),
        (
            lambda X, Y: PLS(n_components=1).fit(X, Y, sample_weight=np.r_[-0.1, np.ones(11)]),
            ValueError,
            "sample_weight must be non-negative.",
        ),
        (
            lambda X, Y: PLS(n_components=1).fit(X, Y, sample_weight=np.ones(11)),
            ValueError,
            "sample_weight has 11 entries; expected 12 to match X / Y.",
        ),
        (lambda X, Y: PLS(n_components=0).fit(X, Y), ValueError, "n_components must be >= 1; got 0."),
        (
            lambda X, Y: PLS(n_components=1).fit(X.iloc[:1], Y.iloc[:1]),
            ValueError,
            "Found array with 1 sample(s) (shape=(1, 4)) while a minimum of 2 is required by PLS.",
        ),
        (
            lambda X, Y: PLS(n_components=1, missing_data_settings={"md_tol": 1e-3}).fit(X, Y),
            ValueError,
            (
                "PLS(missing_data_settings=...) no longer takes 'md_tol', removed in 2.0. Pass tol=0.001 to the "
                "constructor instead."
            ),
        ),
        (
            lambda X, Y: PLS(n_components=1).fit(csr_matrix(X.to_numpy()), Y),
            TypeError,
            "PLS does not accept sparse input: NIPALS centres and scales every column",
        ),
    ],
    ids=[
        "nan-weight",
        "infinite-weight",
        "negative-weight",
        "short-weights",
        "no-components",
        "one-row",
        "removed-key",
        "sparse",
    ],
)
def test_pls_fit_refuses_input_it_cannot_use_in_full(call: Callable, error: type[Exception], message: str) -> None:
    """Each refusal names what is wrong."""
    X, Y = _pls_block()
    with pytest.raises(error, match="^" + re.escape(message)):
        call(X, Y)


def test_pls_fit_takes_infinity_nowhere_but_missing_cells_anywhere() -> None:
    """An infinite X cell is refused; NaN is a missing cell."""
    X, Y = _pls_block()
    X.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"^Input X contains infinity"):
        PLS(n_components=1).fit(X, Y)
    X.iloc[0, 0] = np.nan
    assert PLS(n_components=1).fit(X, Y).has_missing_data_ is True


def test_pls_fit_caps_the_components_at_the_rows_or_columns_in_full() -> None:
    """More components than ``min(N, K)`` warn and fit that many; exactly that many, or None, fit it silently."""
    X, Y = _pls_block(n=6, k=4)
    message = (
        "The requested number of components is more than can be computed from data. The maximum number of "
        "components is the minimum of either the number of rows (6) or the number of columns (4)."
    )
    with pytest.warns(SpecificationWarning, match="^" + re.escape(message) + "$"):
        assert PLS(n_components=5).fit(X, Y).n_components_ == 4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert PLS(n_components=4).fit(X, Y).n_components_ == 4
        none = PLS(n_components=None).fit(X, Y)
    assert none.n_components_ == 4
    assert none.n_components is None


def test_pls_fit_reads_decimal_weights_as_the_floats_they_hold() -> None:
    """Row weights given as ``Decimal`` fit as the same floats."""
    X, Y = _pls_block()
    weights = [Decimal("0.5")] * 6 + [Decimal(2)] * 6
    given = PLS(n_components=2).fit(X, Y, sample_weight=weights)
    exact = PLS(n_components=2).fit(X, Y, sample_weight=[0.5] * 6 + [2.0] * 6)
    np.testing.assert_array_equal(given.beta_coefficients_.to_numpy(), exact.beta_coefficients_.to_numpy())


def test_pls_weighted_fit_by_hand() -> None:
    """``sample_weight``: NIPALS on ``sqrt(w)`` times the rows, scaled by the rows of weight above 0 alone.

    The weights, loadings and iterations are those of the rescaled rows; the scores are the
    rows' own, ``t = z R``, so a row of weight 0 still has one; the Y scores are
    ``u_a = Y_a c_a / c_a'c_a`` with Y deflated by those scores.
    """
    X, Y = _pls_block(n=15)
    weights = np.linspace(0.0, 2.0, 15)
    pls = PLS(n_components=2).fit(X, Y, sample_weight=weights)
    kept = weights > 0
    Z = (X.to_numpy() - X[kept].mean().to_numpy()) / X[kept].std(ddof=1).to_numpy()
    Zy = (Y.to_numpy() - Y[kept].mean().to_numpy()) / Y[kept].std(ddof=1).to_numpy()
    root = np.sqrt(weights)[:, None]
    ref = _pls_nipals_as_fitted(root * Z, root * Zy, 2)
    for name, attribute in (("W", "x_weights_"), ("P", "x_loadings_"), ("C", "y_loadings_")):
        _assert_near(np.asarray(getattr(pls, attribute)), ref[name], rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(pls.fitting_info_["iterations"], ref["iterations"])
    T = Z @ ref["W"] @ np.linalg.inv(ref["P"].T @ ref["W"])
    _assert_near(pls.scores_.to_numpy(), T, rtol=1e-6, atol=1e-9)
    C = ref["C"]
    U = np.column_stack([(Zy - T[:, :a] @ C[:, :a].T) @ C[:, a] / (C[:, a] @ C[:, a]) for a in range(2)])
    _assert_near(pls.y_scores_.to_numpy(), U, rtol=1e-6, atol=1e-9)
    assert T[0] @ T[0] > 0.01  # the row of weight 0 has a score of its own


@pytest.mark.parametrize("block", ["X", "Y"])
def test_pls_fit_needs_a_sum_of_squares_of_at_least_epsqrt_in_each_block(block: str) -> None:
    """A block whose sum of squares is exactly epsqrt (2**-26) is fitted; one smaller is refused in full."""
    big = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    for size, fits in ((2.0**-13, True), (2.0**-14, False)):
        tiny = pd.DataFrame({"b": [size, 0.0, 0.0, 0.0]})
        X, Y = (tiny, big) if block == "X" else (big, tiny)
        model = PLS(n_components=1, scale=False, warn_on_uncentred=False)
        if fits:
            model.fit(X, Y)
            continue
        message = (
            f"There is no variance left in the data array for {block}: cannot compute any more components beyond "
            "component 0."
        )
        with pytest.raises(NotEnoughVarianceError, match="^" + re.escape(message) + "$"):
            model.fit(X, Y)


def test_pls_fit_says_in_full_when_x_runs_out_part_way() -> None:
    """Two proportional X columns hold one component: asking for two is refused at the second."""
    rng = np.random.default_rng(8)
    x = rng.standard_normal(10)
    X = pd.DataFrame({"a": x, "b": 2 * x})
    Y = pd.DataFrame(rng.standard_normal((10, 2)), columns=["p", "q"])
    message = (
        "There is no variance left in the data array for X: cannot compute any more components beyond component 1."
    )
    with pytest.raises(NotEnoughVarianceError, match="^" + re.escape(message) + "$"):
        PLS(n_components=2, scale=False, warn_on_uncentred=False).fit(X - X.mean(), Y - Y.mean())


def test_pls_nipals_says_when_it_stops_at_max_iter_and_logs_each_component(caplog: pytest.LogCaptureFixture) -> None:
    """A component that reaches ``max_iter`` warns in full; each one logs its iterations and tolerance."""
    X, Y = _pls_block()
    with caplog.at_level(logging.DEBUG, logger="process_improve.multivariate._pls"):
        with pytest.warns(SpecificationWarning) as record:
            capped = PLS(n_components=2, max_iter=2).fit(X, Y)
        settled = PLS(n_components=1).fit(X, Y)
    messages = [str(w.message) for w in record if w.category is SpecificationWarning]
    assert messages == [
        f"PLS NIPALS: component {a} reached the maximum number of iterations (2) without converging." for a in (1, 2)
    ]
    np.testing.assert_array_equal(capped.fitting_info_["iterations"], [2, 2])
    count = int(settled.fitting_info_["iterations"][0])
    assert count > 2
    logged = [r.getMessage() for r in caplog.records if r.name == "process_improve.multivariate._pls"]
    assert f"PLS NIPALS: component 1 converged in {count} iterations (tol=1.49012e-08)" in logged
    assert "PLS NIPALS: component 2 converged in 2 iterations (tol=1.49012e-08)" in logged


@pytest.mark.parametrize("gapped_training", [False, True])
def test_held_out_scores_by_hand(gapped_training: bool) -> None:
    """A held-out row's scores under the first ``a`` components of a fold model (#676).

    A complete row's are ``x R_a``, where ``R_a`` is the first ``a`` columns of the model's
    ``R = W (P'W)^-1`` on complete training data, and ``W_a (P_a'W_a)^-1`` with gaps in it. A row
    with gaps is scored by ``operator_for_pattern`` over its observed columns, with the smaller
    model's P, R, score variances, W and training residuals; a row with fewer observed columns
    than components, or whose observed columns cannot tell the components apart (two equal
    columns), has no score.
    """
    X, Y = _pls_block(n=15)
    X["x3"] = X["x2"]
    X, Y = X - X.mean(), Y - Y.mean()
    if gapped_training:
        X.iloc[4, 0] = np.nan
    pls = PLS(n_components=3, scale=False).fit(X, Y)
    x_train = X.to_numpy()
    x_test = np.array(
        [[1.0, -0.5, 0.3, 0.3], [np.nan, np.nan, 0.3, 0.3], [np.nan, 1.0, 0.2, 0.5], [np.nan] * 3 + [0.4]]
    )
    W, P, T = pls.x_weights_.to_numpy(), pls.x_loadings_.to_numpy(), pls.scores_.to_numpy()
    full_R = W @ np.linalg.inv(P.T @ W)
    for a in (1, 2, 3):
        R = W[:, :a] @ np.linalg.inv(P[:, :a].T @ W[:, :a]) if gapped_training else full_R[:, :a]
        if not gapped_training:
            np.testing.assert_allclose(R, W[:, :a] @ np.linalg.inv(P[:, :a].T @ W[:, :a]), atol=1e-10)
        variances = np.sum(T[:, :a] ** 2, axis=0) / 14
        E = np.nan_to_num(x_train - T[:, :a] @ P[:, :a].T)
        expected = np.full((4, a), np.nan)
        expected[0] = x_test[0] @ R
        for row in (1, 2, 3):
            observed = ~np.isnan(x_test[row])
            if observed.sum() < a:
                continue
            try:
                op = operator_for_pattern(P[:, :a], R, variances, observed, x_weights=W[:, :a], x_residuals=E)
            except np.linalg.LinAlgError:
                continue
            expected[row] = op.matrix @ x_test[row, observed]
        got = _held_out_scores(pls, x_train, x_test, a)
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-12)
        assert (
            np.isnan(got).any(axis=1).tolist()
            == {1: [False] * 4, 2: [False, True, False, True], 3: [False] + [True] * 3}[a]
        )
    complete = x_test[[0]]
    np.testing.assert_allclose(
        _held_out_scores(pls, x_train, complete, 2),
        complete @ (W[:, :2] @ np.linalg.inv(P[:, :2].T @ W[:, :2])),
        rtol=1e-8,
    )


def _uncentred_fold_model(x: np.ndarray, y: np.ndarray, a: int) -> tuple[Callable, Callable]:
    """Return the predictions and reconstruction of PLS fitted by hand to rows as given, centred or not."""
    fitted = _pls_nipals_as_fitted(x, y, a)
    R = fitted["W"] @ np.linalg.inv(fitted["P"].T @ fitted["W"])
    return (lambda rows: rows @ R @ fitted["C"].T), (lambda rows: rows @ R @ fitted["P"].T)


def _select_by_hand(  # noqa: PLR0913 - the folds, their first pass and the kind of fold model
    X: pd.DataFrame, Y: pd.DataFrame, splits: list, A: int, first_repeat: int, *, centred: bool = True
) -> dict:
    """Cross-validate PLS by hand: sklearn's PLSRegression at each count on each fold's training rows.

    Each fold's model autoscales its training rows (ddof=1), as in-fold scaling does; with
    ``centred=False`` it is fitted to the rows as given. Returns the PRESS of Y and of X per
    count, each fold's PRESS and RMSE, the first repeat's out-of-fold predictions, and the
    curves ``select_n_components`` reports.
    """
    x, y = X.to_numpy(), Y.to_numpy()
    N, K = x.shape
    M, F = y.shape[1], len(splits)
    out = {
        "press_y": np.zeros((A, M)),
        "press_x": np.zeros((A, K)),
        "fold_press": np.zeros((A, F)),
        "fold_rmse": np.zeros((A, F)),
        "oof": np.full((A, N, M), np.nan),
    }
    for f, (train, test) in enumerate(splits):
        for a in range(1, A + 1):
            if centred:
                model = _sklearn_pls(a).fit(x[train], y[train])
                y_hat, x_hat = model.predict(x[test]), model.inverse_transform(model.transform(x[test]))
            else:
                predict, reconstruct = _uncentred_fold_model(x[train], y[train], a)
                y_hat, x_hat = predict(x[test]), reconstruct(x[test])
            out["press_y"][a - 1] += np.sum((y[test] - y_hat) ** 2, axis=0)
            out["press_x"][a - 1] += np.sum((x[test] - x_hat) ** 2, axis=0)
            out["fold_press"][a - 1, f] = np.sum((y[test] - y_hat) ** 2)
            out["fold_rmse"][a - 1, f] = np.sqrt(out["fold_press"][a - 1, f] / (len(test) * M))
            if f < first_repeat:
                out["oof"][a - 1, test] = y_hat
    coverage = np.zeros(N)
    for _, test in splits:
        coverage[test] += 1
    n_eff = coverage.sum()
    tss_y = coverage @ (y - y.mean(axis=0)) ** 2
    tss_x = coverage @ (x - x.mean(axis=0)) ** 2
    out["rmsecv"] = np.column_stack(
        [np.sqrt(out["press_y"] / n_eff), np.sqrt(out["press_y"].sum(axis=1) / (n_eff * M))]
    )
    r2y = 1 - out["press_y"] / tss_y
    out["r2y"] = np.column_stack([r2y, 1 - out["press_y"].sum(axis=1) / tss_y.sum(), r2y.mean(axis=1)])
    out["r2x"] = np.column_stack([1 - out["press_x"] / tss_x, 1 - out["press_x"].sum(axis=1) / tss_x.sum()])
    out["se"] = out["fold_rmse"].std(axis=1, ddof=1) / np.sqrt(F)
    repeats = F // first_repeat
    out["q2_se"] = out["fold_press"].std(axis=1, ddof=1) / np.sqrt(F) * first_repeat / (tss_y.sum() / repeats)
    return out


def test_pls_select_n_components_by_hand_with_repeated_folds() -> None:
    """Every curve of a repeated 3-fold cross-validation, by hand; the "min" rule overall and in each repeat.

    The out-of-fold predictions are the first repeat's at the count recommended, and each
    repeat votes for its own argmin; the vote shares, their mode, and whether the mode
    reaches ``stability_threshold`` are reported.
    """
    X, Y = _pls_block(n=12)
    result = PLS.select_n_components(
        X, Y, cv=3, n_repeats=2, random_state=1, max_components=3, selection_rule="min", stability_threshold=0.75
    )
    splits = list(RepeatedKFold(n_splits=3, n_repeats=2, random_state=1).split(X))
    hand = _select_by_hand(X, Y, splits, 3, 3)
    index = pd.Index([1, 2, 3], name="n_components")
    _assert_frame_near(
        result.rmsecv, pd.DataFrame(hand["rmsecv"], index=index, columns=["y0", "y1", "total"]), rtol=1e-6
    )
    folds = [f"fold_{i}" for i in range(1, 7)]
    _assert_frame_near(result.per_fold_rmsecv, pd.DataFrame(hand["fold_rmse"], index=index, columns=folds), rtol=1e-6)
    _assert_series_near(result.se_rmsecv, pd.Series(hand["se"], index=index, name="SE(RMSECV)"), rtol=1e-6)
    _assert_series_near(result.q2_se, pd.Series(hand["q2_se"], index=index, name="SE(Q2)"), rtol=1e-6)
    _assert_frame_near(
        result.r2y_validated,
        pd.DataFrame(hand["r2y"], index=index, columns=["y0", "y1", "total", "scaled_total"]),
        rtol=1e-6,
    )
    _assert_frame_near(
        result.r2x_validated, pd.DataFrame(hand["r2x"], index=index, columns=[*X.columns, "total"]), rtol=1e-6
    )
    _assert_series_near(result.press, pd.Series(hand["press_y"].sum(axis=1), index=index, name="PRESS"), rtol=1e-6)
    best = int(np.argmin(hand["rmsecv"][:, -1])) + 1
    assert result.n_components == best
    _assert_frame_near(
        result.cv_predictions, pd.DataFrame(hand["oof"][best - 1], index=Y.index, columns=Y.columns), rtol=1e-6
    )
    votes = [int(np.argmin(hand["fold_rmse"][:, 3 * r : 3 * r + 3].mean(axis=1))) + 1 for r in range(2)]
    shares = pd.Series([votes.count(a) / 2 for a in (1, 2, 3)], index=index, name="vote_share")
    _assert_series_near(result.selection_distribution, shares)
    assert result.selection_mode == int(np.argmax(shares.to_numpy())) + 1
    assert result.selection_is_stable is bool(shares.max() >= 0.75)
    assert result.selection_rule == "min"
    assert result.randomization_pvalues is None


def _record_rule(monkeypatch: pytest.MonkeyPatch, picks: list[int]) -> list[tuple[str, dict]]:
    """Stand in for the shared selection rule: record each call and return the next of ``picks``."""
    calls: list[tuple[str, dict]] = []

    def rule(name: str, **kwargs: object) -> int:
        calls.append((name, kwargs))
        return picks[len(calls) - 1]

    monkeypatch.setattr("process_improve.multivariate._pls._select_n_components", rule)
    return calls


def test_pls_select_n_components_hands_its_curves_to_the_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """The defaults (1-SE on 10 repeats of shuffled 5-fold CV) give the rule the pooled curves, then each repeat's.

    Each repeat's vote reads that repeat's per-fold RMSE only. Votes split evenly reach a
    ``stability_threshold`` of exactly one half; the mode is the smaller count.
    """
    calls = _record_rule(monkeypatch, [2] + [1, 3] * 5)
    X, Y = _pls_block(n=12)
    result = PLS.select_n_components(X, Y, max_components=3, random_state=0, stability_threshold=0.5)
    hand = _select_by_hand(X, Y, list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=0).split(X)), 3, 5)
    (rule, pooled), *per_repeat = calls
    assert rule == "1se"
    assert set(pooled) == {"mean_error", "se_error", "q2_cumulative", "min_q2_increase"}
    _assert_near(pooled["mean_error"], hand["rmsecv"][:, -1], rtol=1e-6)
    _assert_near(pooled["se_error"], hand["se"], rtol=1e-6)
    _assert_near(pooled["q2_cumulative"], hand["r2y"][:, 2], rtol=1e-6)
    assert pooled["min_q2_increase"] == 0.01
    _assert_near(result.q2_se.to_numpy(), hand["q2_se"], rtol=1e-6)
    assert len(per_repeat) == 10
    for r, (name, kwargs) in enumerate(per_repeat):
        folds = hand["fold_rmse"][:, 5 * r : 5 * r + 5]
        assert name == "1se"
        assert set(kwargs) == {"mean_error", "se_error"}
        _assert_near(kwargs["mean_error"], folds.mean(axis=1), rtol=1e-6)
        _assert_near(kwargs["se_error"], folds.std(axis=1, ddof=1) / np.sqrt(5), rtol=1e-6)
    assert result.n_components == 2
    _assert_frame_near(result.cv_predictions, pd.DataFrame(hand["oof"][1], index=Y.index, columns=Y.columns), rtol=1e-6)
    shares = pd.Series([0.5, 0.0, 0.5], index=pd.Index([1, 2, 3], name="n_components"), name="vote_share")
    _assert_series_near(result.selection_distribution, shares)
    assert result.selection_mode == 1
    assert result.selection_is_stable is True


@pytest.mark.parametrize(("rule", "cv"), [("q2_increment", 3), ("1se", KFold(n_splits=3))])
def test_pls_select_n_components_votes_only_for_the_rmse_rules_over_repeats(
    monkeypatch: pytest.MonkeyPatch, rule: str, cv: object
) -> None:
    """The Q2 rule gets its threshold and no vote is taken; nor is one over a single pass of folds."""
    calls = _record_rule(monkeypatch, [1])
    X, Y = _pls_block(n=12)
    result = PLS.select_n_components(
        X, Y, cv=cv, n_repeats=3, max_components=2, selection_rule=rule, min_q2_increase=0.05, random_state=0
    )
    assert len(calls) == 1
    assert calls[0][0] == rule
    assert calls[0][1]["min_q2_increase"] == 0.05
    assert (result.selection_distribution, result.selection_mode, result.selection_is_stable) == (None, None, None)
    assert result.selection_rule == rule


def test_pls_select_n_components_hands_the_scored_counts_to_van_der_voet(monkeypatch: pytest.MonkeyPatch) -> None:
    """The randomization rule compares each row's squared error, summed over Y, at every count; its settings pass on."""
    calls: list[tuple] = []

    def test(per_obs_sse: np.ndarray, **kwargs: object) -> tuple[int, np.ndarray]:
        calls.append((per_obs_sse, kwargs))
        return 2, np.array([0.004, 0.3])

    monkeypatch.setattr("process_improve.multivariate._pls._vandervoet_randomization", test)
    X, Y = _pls_block(n=12)
    settings = {"n_permutations": 19, "alpha": 0.05, "random_state": 3}
    result = PLS.select_n_components(
        X, Y, cv=4, n_repeats=1, max_components=2, selection_rule="randomization", **settings
    )
    hand = _select_by_hand(X, Y, list(KFold(n_splits=4, shuffle=True, random_state=3).split(X)), 2, 4)
    ((sse, kwargs),) = calls
    _assert_near(sse, np.sum((Y.to_numpy() - hand["oof"]) ** 2, axis=2), rtol=1e-6)
    _assert_near(kwargs.pop("total_rmsecv"), hand["rmsecv"][:, -1], rtol=1e-6)
    assert kwargs == settings
    assert result.n_components == 2
    _assert_series_near(
        result.randomization_pvalues,
        pd.Series([0.004, 0.3], index=pd.Index([1, 2], name="n_components"), name="p-value (Van der Voet)"),
    )
    assert result.selection_distribution is None
    PLS.select_n_components(X, Y, cv=4, n_repeats=1, max_components=2, selection_rule="randomization")
    assert {key: calls[1][1][key] for key in settings} == {"n_permutations": 999, "alpha": 0.01, "random_state": None}


def test_pls_select_n_components_leaves_out_a_count_that_cannot_score_a_held_out_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A held-out row observed in one X column has no score from two components: that count gets no PRESS (#676).

    Its curves are NaN, Van der Voet is handed the first count alone, and its p-value is NaN.
    """
    calls: list[tuple] = []

    def test(per_obs_sse: np.ndarray, **kwargs: object) -> tuple[int, np.ndarray]:
        calls.append((per_obs_sse, kwargs))
        return 1, np.array([0.5])

    monkeypatch.setattr("process_improve.multivariate._pls._vandervoet_randomization", test)
    X, Y = _pls_block(n=12)
    X.iloc[0, 1:] = np.nan
    result = PLS.select_n_components(X, Y, cv=4, n_repeats=1, max_components=2, selection_rule="randomization")
    ((sse, kwargs),) = calls
    assert sse.shape == (1, 12)
    np.testing.assert_allclose(kwargs["total_rmsecv"], result.rmsecv["total"].to_numpy()[:1])
    for frame in (result.rmsecv, result.per_fold_rmsecv, result.r2y_validated, result.r2x_validated):
        assert frame.loc[2].isna().all()
        assert frame.loc[1].notna().all()
    for series in (result.press, result.se_rmsecv, result.q2_se):
        assert np.isnan(series[2])
        assert np.isfinite(series[1])
    np.testing.assert_array_equal(result.randomization_pvalues.to_numpy(), [0.5, np.nan])
    assert result.n_components == 1


@pytest.mark.parametrize(("scale_inside_folds", "scale"), [(True, True), (True, False), (False, True), (False, False)])
def test_pls_select_n_components_with_a_splitter_by_hand(scale_inside_folds: bool, scale: bool) -> None:
    """A splitter that holds out some rows twice and others never: PRESS and the TSS count each row as often as tested.

    Every held-out row is predicted as its fold model predicts it: autoscaled by the fold or
    by the model itself (#1027), or, with neither, fitted to the rows as given. Without in-fold
    scaling the leak is announced in full. The first pass is every fold, and a row tested
    twice keeps the later prediction.
    """
    X, Y = _pls_block(n=12)
    splitter = ShuffleSplit(n_splits=3, test_size=5, random_state=0)
    message = (
        "scale_inside_folds=False leaks centring/scaling estimated on the full dataset into every CV fold, making the "
        "reported RMSECV optimistic. The default scale_inside_folds=True is preferred."
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = PLS.select_n_components(
            X,
            Y,
            cv=splitter,
            n_repeats=4,
            max_components=3,
            selection_rule="min",
            scale_inside_folds=scale_inside_folds,
            scale=scale,
            warn_on_uncentred=False,
        )
    warned = [str(w.message) for w in record if w.category is SpecificationWarning]
    assert warned == ([] if scale_inside_folds else [message])
    hand = _select_by_hand(X, Y, list(splitter.split(X)), 3, 3, centred=scale_inside_folds or scale)
    _assert_near(result.rmsecv.to_numpy(), hand["rmsecv"], rtol=1e-6)
    _assert_near(result.r2y_validated.to_numpy(), hand["r2y"], rtol=1e-6)
    _assert_near(result.r2x_validated.to_numpy(), hand["r2x"], rtol=1e-6)
    _assert_near(result.q2_se.to_numpy(), hand["q2_se"], rtol=1e-6)
    assert list(result.per_fold_rmsecv.columns) == ["fold_1", "fold_2", "fold_3"]
    _assert_near(result.cv_predictions.to_numpy(), hand["oof"][result.n_components - 1], rtol=1e-6)
    assert result.selection_distribution is None


@pytest.mark.parametrize(
    ("scale_inside_folds", "scale", "count"), [(True, True, 7), (False, True, 7), (False, False, 8)]
)
def test_pls_select_n_components_caps_the_count_where_a_fold_runs_out(
    scale_inside_folds: bool, scale: bool, count: int
) -> None:
    """A training fold of 8 rows centred, by the fold or by its model, holds 7 components; uncentred, 8 (#1027).

    Without in-fold scaling the cap was 8 whether or not the model centres, and a wide
    block with the default ``PLS(scale=True)`` ran out of X at the eighth component.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((10, 12)))
    Y = pd.DataFrame({"y": X[0] + 0.1 * rng.standard_normal(10)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SpecificationWarning)
        result = PLS.select_n_components(
            X, Y, cv=5, n_repeats=1, scale_inside_folds=scale_inside_folds, scale=scale, random_state=0
        )
    assert list(result.rmsecv.index) == list(range(1, count + 1))
    assert PLS.select_n_components(X, Y, cv=5, n_repeats=1, max_components=3, random_state=0).rmsecv.shape[0] == 3


def test_pls_select_n_components_refuses_and_warns_in_full() -> None:
    """Fewer than two folds or one repeat are refused; an X that arrives autoscaled is warned about."""
    X, Y = _pls_block(n=12)
    with pytest.raises(ValueError, match=r"^cv must be >= 2 when given as an int; got 1\.$"):
        PLS.select_n_components(X, Y, cv=1)
    with pytest.raises(ValueError, match=r"^n_repeats must be >= 1; got 0\.$"):
        PLS.select_n_components(X, Y, n_repeats=0)
    assert PLS.select_n_components(X, Y, cv=2, n_repeats=1, max_components=1).per_fold_rmsecv.shape == (1, 2)
    with pytest.warns(SpecificationWarning, match=r"re-standardises inside every CV fold.* comparison of their RMSECV"):
        PLS.select_n_components(_autoscaled(X.to_numpy()), Y, cv=3, n_repeats=1, max_components=1)


def test_pls_select_n_components_reads_arrays_and_a_series() -> None:
    """Arrays are numbered from 0; a Series keeps its name and its row labels."""
    X, Y = _pls_block(n=12)
    series = pd.Series(Y["y0"].to_numpy(), index=X.index, name="q")
    named = PLS.select_n_components(X, series, cv=3, n_repeats=1, max_components=2, random_state=0)
    bare = PLS.select_n_components(X.to_numpy(), series.to_numpy(), cv=3, n_repeats=1, max_components=2, random_state=0)
    assert list(named.rmsecv.columns) == ["q", "total"]
    assert list(named.cv_predictions.columns) == ["q"]
    assert named.cv_predictions.index.equals(X.index)
    assert list(bare.rmsecv.columns) == [0, "total"]
    assert list(bare.r2x_validated.columns) == [0, 1, 2, 3, "total"]
    np.testing.assert_allclose(bare.rmsecv.to_numpy(), named.rmsecv.to_numpy())


@pytest.mark.parametrize("scale_inside_folds", [True, False])
def test_pls_select_n_components_scores_rows_as_a_fold_model_that_recentres_predicts_them(
    scale_inside_folds: bool,
) -> None:
    """PRM centres every fold at its robust centre, after any in-fold scaling: its held-out rows are read in its space.

    The reference is each fold's PRM predicting its held-out rows, after the fold's own
    autoscaling when there is one, and back in the data's units (#1027).
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((15, 4)) + 3.0, columns=list("abcd"))
    Y = pd.DataFrame({"y": X.to_numpy() @ [1.0, 2.0, 0.0, -1.0] + 0.5 * rng.standard_normal(15)})
    X.iloc[2, 1] += 6.0  # pulls the fold mean away from the robust centre
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SpecificationWarning)
        result = PRM.select_n_components(X, Y, cv=cv, max_components=1, scale_inside_folds=scale_inside_folds)
    for train, test in cv.split(X):
        if scale_inside_folds:
            sx, sy = MCUVScaler().fit(X.iloc[train]), MCUVScaler().fit(Y.iloc[train])
            fold = PRM(n_components=1).fit(sx.transform(X.iloc[train]), sy.transform(Y.iloc[train]))
            expected = sy.inverse_transform(fold.predict(sx.transform(X.iloc[test]))).to_numpy()
        else:
            expected = PRM(n_components=1).fit(X.iloc[train], Y.iloc[train]).predict(X.iloc[test]).to_numpy()
        _assert_near(result.cv_predictions.iloc[test].to_numpy(), expected, rtol=1e-10)


def test_pls_nipals_iterates_at_least_once_on_a_huge_response() -> None:
    """In units of 1e20 the seed's ``+ 1`` is lost, so its first comparison sees no change: one iteration is forced.

    The model is the same as in ordinary units, scaled.
    """
    X, Y = _pls_block()
    X, Y = X - X.mean(), Y - Y.mean()
    huge = PLS(n_components=2, scale=False).fit(X, Y * 1e20)
    plain = PLS(n_components=2, scale=False).fit(X, Y)
    _assert_near(huge.x_weights_.to_numpy(), plain.x_weights_.to_numpy(), rtol=1e-6)
    _assert_near(huge.y_loadings_.to_numpy(), 1e20 * plain.y_loadings_.to_numpy(), rtol=1e-6)


def test_pls_unscaled_fit_reads_a_pandas_missing_value_as_nan() -> None:
    """``scale=False`` hands the response to NIPALS as given: a nullable column's ``pd.NA`` is a missing cell."""
    X, Y = _gapped_pls_block()
    X, Y = X - X.mean(), Y - Y.mean()
    nullable = Y.astype("Float64")
    nullable.iloc[3, 0] = pd.NA
    # The centring check reads the block itself, and does not take pd.NA: it is skipped here.
    plain = PLS(n_components=2, scale=False, warn_on_uncentred=False).fit(X, Y)
    other = PLS(n_components=2, scale=False, warn_on_uncentred=False).fit(X, nullable)
    np.testing.assert_array_equal(other.beta_coefficients_.to_numpy(), plain.beta_coefficients_.to_numpy())
    np.testing.assert_array_equal(other.rmse_.to_numpy(), plain.rmse_.to_numpy())


def test_pls_weighted_fit_leaves_the_y_scores_of_a_component_along_x_at_zero() -> None:
    """A component along X alone (#913) has a Y-loading of about 0: its rebuilt Y scores stay 0, not 0 / 0.

    Two replicates of a 2x2 design, the second weighted 0: after one component what is left
    of y is the interaction, which no column of X can reach.
    """
    x1, x2 = np.tile([-1.0, 1.0, -1.0, 1.0], 2), np.tile([-1.0, -1.0, 1.0, 1.0], 2)
    X = pd.DataFrame({"a": x1, "b": x2})
    y = pd.DataFrame({"y": x1 + 2.0 * x2 + 0.5 * x1 * x2})
    weights = np.r_[np.ones(4), np.zeros(4)]
    with pytest.warns(SpecificationWarning, match="nothing left in Y that X can explain"):
        pls = PLS(n_components=2).fit(X, y, sample_weight=weights)
    assert abs(pls.y_loadings_.iloc[0, 1]) < 1e-8
    np.testing.assert_array_equal(pls.y_scores_.iloc[:, 1].to_numpy(), np.zeros(8))
    c = pls.y_loadings_.iloc[0, 0]
    z = (y["y"].to_numpy() - y["y"].iloc[:4].mean()) / y["y"].iloc[:4].std(ddof=1)
    np.testing.assert_allclose(pls.y_scores_.iloc[:, 0].to_numpy(), z * c / c**2, rtol=1e-10)


def test_pls_select_n_components_standard_errors_need_two_folds() -> None:
    """Over two folds the SE of the RMSECV is half their difference; over one there is none, and 1-SE is argmin."""
    X, Y = _pls_block(n=12)
    two = PLS.select_n_components(X, Y, cv=KFold(n_splits=2), max_components=2)
    folds = two.per_fold_rmsecv.to_numpy()
    np.testing.assert_allclose(two.se_rmsecv.to_numpy(), np.abs(folds[:, 0] - folds[:, 1]) / 2, rtol=1e-10)
    assert np.isfinite(two.q2_se.to_numpy()).all()
    one = PLS.select_n_components(X, Y, cv=PredefinedSplit([0] * 4 + [-1] * 8), max_components=3)
    assert one.per_fold_rmsecv.shape == (3, 1)
    for series in (one.se_rmsecv, one.q2_se):
        assert series.dtype == float
        assert series.isna().all()
    assert one.n_components == int(np.argmin(one.rmsecv["total"].to_numpy())) + 1


def test_pls_select_n_components_hands_both_blocks_to_its_splitter() -> None:
    """A splitter is given X and Y, as one that stratifies needs."""
    X, Y = _pls_block()
    splitter = _RecordingSplitter()
    PLS.select_n_components(X, Y, cv=splitter, max_components=1)
    ((x, y),) = splitter.seen
    pd.testing.assert_frame_equal(x, X)
    pd.testing.assert_frame_equal(y, Y)


def test_pls_select_n_components_projects_a_held_out_row_with_a_gap() -> None:
    """A held-out row with missing cells is predicted as its fold model's ``project`` predicts it."""
    X, Y = _pls_block(n=12)
    X.iloc[[0, 7], [1, 2]] = np.nan
    cv = KFold(n_splits=3, shuffle=True, random_state=2)
    result = PLS.select_n_components(X, Y, cv=cv, max_components=1)
    for train, test in cv.split(X):
        expected = PLS(n_components=1).fit(X.iloc[train], Y.iloc[train]).project(X.iloc[test]).y_hat.to_numpy()
        _assert_near(result.cv_predictions.iloc[test].to_numpy(), expected, rtol=1e-6)


def test_pls_select_n_components_leave_one_out_with_one_response_by_hand() -> None:
    """Each fold holds one row and one response: its RMSE is the size of its one residual."""
    X, Y = _pls_block(n=8)
    result = PLS.select_n_components(X, Y[["y0"]], cv=LeaveOneOut(), max_components=2, selection_rule="min")
    hand = _select_by_hand(X, Y[["y0"]], list(LeaveOneOut().split(X)), 2, 8)
    _assert_near(result.per_fold_rmsecv.to_numpy(), hand["fold_rmse"], rtol=1e-6)
    _assert_near(result.rmsecv.to_numpy(), hand["rmsecv"], rtol=1e-6)


def test_pls_select_n_components_in_small_units_by_hand() -> None:
    """In thousandths every sum of squares is below 1, and every validated R2 and the Q2 band still count."""
    X, Y = _pls_block(n=12)
    X, Y = X / 1000.0, Y / 1000.0
    cv = KFold(n_splits=3, shuffle=True, random_state=4)
    result = PLS.select_n_components(X, Y, cv=cv, max_components=2, selection_rule="min")
    hand = _select_by_hand(X, Y, list(cv.split(X)), 2, 3)
    assert (Y.to_numpy() ** 2).sum() < 1
    _assert_near(result.r2y_validated.to_numpy(), hand["r2y"], rtol=1e-6)
    _assert_near(result.r2x_validated.to_numpy(), hand["r2x"], rtol=1e-6)
    _assert_near(result.q2_se.to_numpy(), hand["q2_se"], rtol=1e-6)


def test_pls_select_n_components_has_no_r2_where_the_held_out_rows_do_not_vary() -> None:
    """Held-out rows at the response's mean, and a constant X column, leave no spread to explain: NaN, silently."""
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.standard_normal((8, 3)), columns=["a", "b", "c"])
    X["flat"] = 2.0
    Y = pd.DataFrame({"y": [5.0, 5.0, 3.0, 7.0, 4.0, 6.0, 2.0, 8.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = PLS.select_n_components(X, Y, cv=PredefinedSplit([0, 0] + [-1] * 6), max_components=2)
    assert result.r2y_validated.isna().all(axis=None)
    assert result.r2x_validated["flat"].isna().all()
    assert result.r2x_validated[["a", "b", "c", "total"]].notna().all(axis=None)
    assert result.q2_se.isna().all()
    # Over two such folds the PRESS has a spread, and still no response spread to scale it by.
    Y = pd.DataFrame({"y": [5.0, 5.0, 5.0, 5.0, 3.0, 7.0, 4.0, 6.0, 2.0, 8.0]})
    X = pd.DataFrame(rng.standard_normal((10, 3)), columns=["a", "b", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = PLS.select_n_components(X, Y, cv=PredefinedSplit([0, 0, 1, 1] + [-1] * 6), max_components=2)
    assert np.isfinite(result.se_rmsecv.to_numpy()).all()
    assert result.q2_se.isna().all()


def test_pls_select_n_components_says_in_full_when_no_count_can_be_judged() -> None:
    """Folds that hold out nothing leave every RMSECV NaN: no recommendation, and the likely causes."""
    X, Y = _pls_block(n=10)
    message = (
        "Cross-validation produced NaN total-RMSECV for every component count; no recommendation can be made. "
        "Likely cause: a per-fold zero-variance Y column, or every fold trivially degenerate."
    )
    with (
        np.errstate(invalid="ignore"),
        pytest.warns(SpecificationWarning, match=r"scale_inside_folds=False leaks"),
        pytest.raises(RuntimeError, match="^" + re.escape(message) + "$"),
    ):
        PLS.select_n_components(
            X, Y, cv=_TrainsOnEverythingTestsNothing(), scale_inside_folds=False, max_components=2, n_permutations=19
        )


def test_held_out_scores_names_a_singular_p_w_of_the_smaller_model() -> None:
    """With gaps in the training rows the smaller model's ``P'W`` is inverted; a singular one is named in the error."""
    model = SimpleNamespace(
        x_loadings_=pd.DataFrame([[1.0, 1.0], [1.0, 1.0]]),
        x_weights_=pd.DataFrame([[1.0, 0.0], [0.0, 1.0]]),
    )
    x_train = np.array([[np.nan, 1.0], [1.0, 2.0]])
    with pytest.raises(np.linalg.LinAlgError, match=r"^\(P'W of the smaller model\) is singular or ill-conditioned"):
        _held_out_scores(model, x_train, np.ones((1, 2)), 2)


def _design_with_an_interaction(replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a replicated 2x2 design and ``y = x1 + 2 x2 + x1 x2 / 2``, whose interaction no column of X reaches."""
    x1, x2 = np.tile([-1.0, 1.0, -1.0, 1.0], replicates), np.tile([-1.0, -1.0, 1.0, 1.0], replicates)
    return pd.DataFrame({"a": x1, "b": x2}), pd.DataFrame({"y": x1 + 2.0 * x2 + 0.5 * x1 * x2})


def test_pls_fits_a_component_along_x_once_y_is_exhausted_in_full() -> None:
    """After one component the rest of y is the interaction: the second component is X's leading direction left (#913).

    The warning says so in full: R²Y stays at (1 + 4) / (1 + 4 + 1/4). The second weight is the
    leading right singular vector of X deflated by the first component, its loading's largest
    entry positive; its Y-loading is about 0 and its Y scores are 0.
    """
    X, y = _design_with_an_interaction(2)
    message = (
        "PLS: from component 2 on there is nothing left in Y that X can explain: what is left of Y is uncorrelated "
        f"with what is left of X (R²Y stays at {5 / 5.25:.4g}). The data support 1 component(s); each later one "
        "takes the largest direction left in X and leaves the predictions unchanged."
    )
    with pytest.warns(SpecificationWarning, match="^" + re.escape(message) + "$"):
        pls = PLS(n_components=2).fit(X, y)
    Z = _autoscaled(X.to_numpy())
    w1 = pls.x_weights_.to_numpy()[:, 0]
    t1 = Z @ w1
    left = Z - np.outer(t1, Z.T @ t1 / (t1 @ t1))
    w2 = np.linalg.svd(left)[2][0]
    p2 = left.T @ (left @ w2)
    w2 = w2 * np.sign(p2[np.argmax(np.abs(p2))])
    np.testing.assert_allclose(pls.x_weights_.to_numpy()[:, 1], w2, rtol=1e-6, atol=1e-8)
    assert abs(pls.y_loadings_.iloc[0, 1]) < 1e-8
    np.testing.assert_array_equal(pls.y_scores_.iloc[:, 1].to_numpy(), np.zeros(8))


def test_pls_fits_a_component_along_x_with_as_many_responses_as_rows() -> None:
    """Four rows and four responses: the Y-loading of the component along X is regressed over Y's columns."""
    X, y = _design_with_an_interaction(1)
    Y = pd.DataFrame(np.column_stack([k * y["y"].to_numpy() for k in (1.0, -2.0, 0.5, 3.0)]), columns=list("pqrs"))
    with pytest.warns(SpecificationWarning, match="nothing left in Y that X can explain"):
        pls = PLS(n_components=2).fit(X, Y)
    assert np.abs(pls.y_loadings_.iloc[:, 1].to_numpy()).max() < 1e-8


def test_pls_nipals_seeds_a_missing_cell_of_its_seed_column_as_zero() -> None:
    """After one iteration the component still shows its seed: the largest Y column, a missing cell as 0, plus 1."""
    X, Y = _pls_block(n=12)
    X, Y = X - X.mean(), Y - Y.mean()
    Y["y1"] *= 10.0  # the seed column
    Y.iloc[2, 1] = np.nan
    with pytest.warns(SpecificationWarning, match="reached the maximum number of iterations"):
        pls = PLS(n_components=1, max_iter=1, scale=False).fit(X, Y)
    ref = _pls_nipals_as_fitted(X.to_numpy(), Y.to_numpy(), 1, max_iter=1)
    for name, attribute in (("T", "scores_"), ("W", "x_weights_"), ("C", "y_loadings_")):
        np.testing.assert_allclose(np.asarray(getattr(pls, attribute)), ref[name], rtol=1e-10)


def test_pls_direct_weights_name_a_singular_p_w() -> None:
    """``R = W (P'W)^-1``; a ``P'W`` that cannot be inverted is named in the error."""
    W, P = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]), np.array([[1.0, 0.5], [0.0, 1.0], [0.2, 0.3]])
    np.testing.assert_allclose(_direct_weights(W, P), W @ np.linalg.inv(P.T @ W), rtol=1e-12)
    with pytest.raises(np.linalg.LinAlgError, match=r"^\(x_loadings' @ x_weights\) is singular or ill-conditioned"):
        _direct_weights(W, np.ones((3, 2)))


def test_pls_imputed_fit_reads_boolean_columns_as_numbers() -> None:
    """Indicator columns of booleans, as ``pd.get_dummies`` gives, are imputed with the responses as 0 and 1."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((15, 4)) > 0, columns=list("abcd"))
    Y = pd.DataFrame({"y": X.to_numpy() @ [1.0, 2.0, 3.0, 4.0] + rng.standard_normal(15)})
    Y.iloc[2, 0] = np.nan
    given = PLS(n_components=2, missing_data_settings={"md_method": "tsr"}).fit(X, Y)
    exact = PLS(n_components=2, missing_data_settings={"md_method": "tsr"}).fit(X.astype(float), Y)
    _assert_near(given.beta_coefficients_.to_numpy(), exact.beta_coefficients_.to_numpy(), rtol=1e-10)
