"""#978: a Hotelling's T² limit for the rows a model was fitted to, and one for new rows.

A fitted row's T² follows ((N - 1)^2 / N) * Beta(A/2, (N - A - 1)/2); a new row's follows
A (N^2 - 1) / (N (N - A)) * F(A, N - A). ``judged_rows`` picks one. Its default, ``None``, keeps
the new-row limit until 3.0 and warns, with a ``FutureWarning`` at the caller's line, on every call
whose number changes then.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from process_improve.batch._batch_monitor import BatchMonitor
from process_improve.batch._batch_pca import BatchPCA
from process_improve.batch.preprocessing import find_reference_batch
from process_improve.multivariate import tools
from process_improve.multivariate._limits import ellipse_coordinates, hotellings_t2_limit
from process_improve.multivariate.methods import PCA, PLS, MCUVScaler


def beta_limit_by_hand(conf_level: float, n_components: int, n_rows: int) -> float:
    """Return ((N - 1)^2 / N) times the beta quantile, from scipy's beta distribution."""
    a, n = n_components, n_rows
    return (n - 1) ** 2 / n * stats.beta.ppf(conf_level, a / 2, (n - a - 1) / 2)


def f_limit_by_hand(conf_level: float, n_components: int, n_rows: int) -> float:
    """Return A (N^2 - 1) / (N (N - A)) times the F quantile."""
    a, n = n_components, n_rows
    return a * (n**2 - 1) / (n * (n - a)) * stats.f.ppf(conf_level, a, n - a)


def _line_of(marker: str) -> int:
    """Return the number of the line of this file that ends with ``marker``: the call a warning must point at."""
    lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    return next(number for number, line in enumerate(lines, start=1) if line.endswith(marker))


# --------------------------------------------------------------------------- the two laws
@pytest.mark.parametrize(("conf_level", "n_components", "n_rows"), [(0.95, 1, 10), (0.95, 2, 20), (0.99, 5, 200)])
def test_the_fitted_row_limit_is_the_beta_quantile(conf_level: float, n_components: int, n_rows: int) -> None:
    """``judged_rows="fitted"`` gives the beta law's quantile; ``"new"`` the F law's."""
    fitted = hotellings_t2_limit(conf_level, n_components, n_rows, judged_rows="fitted")
    new = hotellings_t2_limit(conf_level, n_components, n_rows, judged_rows="new")
    assert fitted == pytest.approx(beta_limit_by_hand(conf_level, n_components, n_rows), rel=1e-10)
    assert new == pytest.approx(f_limit_by_hand(conf_level, n_components, n_rows), rel=1e-10)
    assert fitted < new < np.inf


@pytest.mark.parametrize(
    ("n_rows", "n_components", "beta_limit", "f_limit"),
    [(10, 2, 4.66, 11.04), (15, 2, 5.14, 8.74), (20, 2, 5.36, 7.88), (30, 3, 7.16, 9.86), (100, 5, 10.73, 12.16)],
)
def test_the_limits_match_the_table_of_the_issue(n_rows: int, n_components: int, beta_limit: float, f_limit: float):
    """The 95% limits #978 tabulates, to its two decimals."""
    assert hotellings_t2_limit(0.95, n_components, n_rows, judged_rows="fitted") == pytest.approx(beta_limit, abs=5e-3)
    assert hotellings_t2_limit(0.95, n_components, n_rows, judged_rows="new") == pytest.approx(f_limit, abs=5e-3)


def test_the_fitted_row_limit_lies_below_the_largest_fitted_t2() -> None:
    """At N = 10, A = 2 the F limit, 11.04, is above (N - 1)^2 / N = 8.1, so no fitted row can exceed it."""
    bound = 9**2 / 10
    assert (
        hotellings_t2_limit(0.95, 2, 10, judged_rows="fitted")
        < bound
        < hotellings_t2_limit(0.95, 2, 10, judged_rows="new")
    )


def test_the_fitted_row_limit_gives_the_stated_rate_on_the_rows_pca_was_fitted_to() -> None:
    """Over 600 PCA fits of 20 rows of 2 normal columns, 5% of the fitted rows lie above the 95% beta limit.

    With A = K the PCA scores are a rotation of the centred data, so T² is the rows' Mahalanobis
    distance from their own mean, which the beta law describes exactly. The F limit flags under 1%.
    """
    rng = np.random.default_rng(978)
    n_rows, n_fits = 20, 600
    t2 = np.concatenate(
        [
            PCA(n_components=2)
            .fit(pd.DataFrame(rng.normal(size=(n_rows, 2)) @ [[2.0, 0.5], [0.0, 1.0]]))
            .hotellings_t2_.iloc[:, -1]
            .to_numpy()
            for _ in range(n_fits)
        ]
    )
    fitted_limit = hotellings_t2_limit(0.95, 2, n_rows, judged_rows="fitted")
    new_limit = hotellings_t2_limit(0.95, 2, n_rows, judged_rows="new")
    assert np.mean(t2 > fitted_limit) == pytest.approx(0.05, abs=0.01)
    assert np.mean(t2 > new_limit) < 0.015


# --------------------------------------------------------------------------- edge cases
def test_with_one_row_more_than_components_every_fitted_row_sits_on_the_bound() -> None:
    """N = A + 1 leaves the beta law no second shape: every fitted T² is (N - 1)^2 / N, and that is the limit."""
    assert hotellings_t2_limit(0.95, 2, 3, judged_rows="fitted") == pytest.approx(4 / 3)
    assert hotellings_t2_limit(0.95, 1, 2, judged_rows="fitted") == pytest.approx(1 / 2)
    model = PCA(n_components=2).fit(
        MCUVScaler().fit_transform(pd.DataFrame([[1.0, 2.0, 0.0], [3.0, 1.0, 1.0], [0.0, 0.0, 5.0]]))
    )
    np.testing.assert_allclose(model.hotellings_t2_.iloc[:, -1], 4 / 3)
    assert model.hotellings_t2_limit(judged_rows="fitted") == pytest.approx(4 / 3)


def test_as_many_components_as_rows_is_infinite_for_both_and_silent() -> None:
    """A == N gives an infinite limit, as the F limit did, whichever rows; the default does not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hotellings_t2_limit(0.95, 3, 3) == float("inf")
    assert hotellings_t2_limit(0.95, 3, 3, judged_rows="fitted") == float("inf")


def test_no_components_is_zero_for_both_and_silent() -> None:
    """A == 0 gives 0 whichever rows, so the default has nothing to warn about."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hotellings_t2_limit(0.95, 0, 10) == 0.0
    assert hotellings_t2_limit(0.95, 0, 10, judged_rows="fitted") == 0.0


def test_one_component() -> None:
    """A = 1: Beta(1/2, (N - 2)/2), the square of a t statistic scaled to the fitted bound."""
    assert hotellings_t2_limit(0.95, 1, 12, judged_rows="fitted") == pytest.approx(beta_limit_by_hand(0.95, 1, 12))


def test_a_component_with_no_spread_and_missing_cells_count_as_the_f_limit_counts_them() -> None:
    """The limit reads ``n_components_`` and ``n_samples_``, as the F limit does, whatever the model holds."""
    rng = np.random.default_rng(3)
    raw = pd.DataFrame(rng.normal(size=(12, 2)) @ rng.normal(size=(2, 5)))
    raw.iloc[3, 1] = np.nan
    model = PCA(n_components=3).fit(MCUVScaler().fit_transform(raw))
    expected = beta_limit_by_hand(0.95, 3, 12)
    assert model.hotellings_t2_limit(judged_rows="fitted") == pytest.approx(expected)


def test_judged_rows_is_refused_by_name_when_it_is_not_one_of_its_values() -> None:
    """A misspelt value is an error, not a silent default."""
    with pytest.raises(ValueError, match=r'judged_rows must be "fitted", "new" or None; got \'training\''):
        hotellings_t2_limit(0.95, 2, 20, judged_rows="training")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- the coming default
def test_the_default_gives_the_new_row_limit_until_3_0_and_warns_at_the_callers_line() -> None:
    """``None`` gives today's number, with a FutureWarning, in full, that names 2.4.0, 3.0 and both values."""
    with pytest.warns(FutureWarning) as caught:
        limit = hotellings_t2_limit(0.95, 2, 20)  # the default call
    assert limit == hotellings_t2_limit(0.95, 2, 20, judged_rows="new")
    (warning,) = caught
    assert str(warning.message) == (
        "Hotelling's T² limit: judged_rows=None gives the F limit for new rows (7.879 here) until 3.0, and from 3.0 "
        "the beta limit for the rows the model was fitted to (5.361 here), as announced in 2.4.0. Pass "
        'judged_rows="fitted" to judge the rows the model was fitted to (hotellings_t2_, a score plot, outlier '
        'detection), or judged_rows="new" to judge rows it was not fitted to; either silences this warning.'
    )
    assert warning.filename == __file__
    assert warning.lineno == _line_of("# the default call")


@pytest.mark.parametrize("judged_rows", ["fitted", "new"])
def test_passing_either_value_is_silent(judged_rows: str) -> None:
    """Saying which rows are judged, either way, silences the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hotellings_t2_limit(0.95, 2, 20, judged_rows=judged_rows)  # type: ignore[arg-type]
        ellipse_coordinates(1, 2, n_components=2, scaling_factor_for_scores=pd.Series([1.0, 1.0]), n_rows=20,
                            judged_rows=judged_rows)  # type: ignore[arg-type]  # fmt: skip


@pytest.fixture(scope="module")
def pca() -> PCA:
    """Return a PCA of 25 rows."""
    rng = np.random.default_rng(7)
    return PCA(n_components=3).fit(MCUVScaler().fit_transform(pd.DataFrame(rng.normal(size=(25, 5)))))


def test_every_entry_point_that_judges_fitted_rows_warns_at_the_users_line(pca: PCA) -> None:
    """The model's limit, its ellipse, its score plot, its T² plot and its outlier detection."""
    pytest.importorskip("plotly")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pca.hotellings_t2_limit()  # entry: limit
        pca.ellipse_coordinates(1, 2)  # entry: ellipse
        pca.score_plot()  # entry: score plot
        pca.t2_plot()  # entry: t2 plot
        pca.detect_outliers()  # entry: outliers
    entries = ("limit", "ellipse", "score plot", "t2 plot", "outliers")
    assert [(w.filename, w.lineno) for w in caught] == [(__file__, _line_of(f"# entry: {name}")) for name in entries]
    assert all(w.category is FutureWarning and "judged_rows" in str(w.message) for w in caught)


def test_the_entry_points_use_the_limit_they_are_asked_for(pca: PCA) -> None:
    """``judged_rows`` reaches the limit each entry point draws or flags with."""
    pytest.importorskip("plotly")
    fitted = hotellings_t2_limit(0.95, 3, 25, judged_rows="fitted")
    two_scores = hotellings_t2_limit(0.95, 2, 25, judged_rows="fitted")
    assert pca.hotellings_t2_limit(judged_rows="fitted") == fitted
    x, _ = pca.ellipse_coordinates(1, 2, judged_rows="fitted")
    assert x[0] == pytest.approx(np.sqrt(two_scores) * pca.scaling_factor_for_scores_.iloc[0])
    figure = pca.score_plot(judged_rows="fitted")
    assert max(figure.data[-1].x) == pytest.approx(x[0])
    plot = pca.t2_plot(judged_rows="fitted")
    assert plot.layout.shapes[0].y0 == pytest.approx(fitted)
    for record in pca.detect_outliers(judged_rows="fitted"):
        assert record["hotellings_t2_limit"] == fitted
        assert "hotellings_t2_limit_note" not in record


def test_outlier_records_carry_the_warning_text_under_the_default(pca: PCA) -> None:
    """A caller that reads the records rather than the warnings finds the same text in each of them."""
    with pytest.warns(FutureWarning) as caught:
        records = pca.detect_outliers(conf_level=0.8)
    assert records
    assert {record["hotellings_t2_limit_note"] for record in records} == {str(caught[0].message)}


def test_the_rest_of_the_family_takes_judged_rows() -> None:
    """PLS, the batch models and TPLS forward ``judged_rows`` to the same limit."""
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.normal(size=(30, 4)))
    y = pd.DataFrame({"y": x.to_numpy() @ [1.0, 0.5, 0.0, -1.0] + 0.1 * rng.normal(size=30)})
    pls = PLS(n_components=2).fit(x, y)
    assert pls.hotellings_t2_limit(judged_rows="fitted") == pytest.approx(beta_limit_by_hand(0.95, 2, 30))
    assert pls.detect_outliers(judged_rows="fitted") is not None
    x_fit, _ = pls.ellipse_coordinates(1, 2, judged_rows="fitted")
    assert x_fit[0] == pytest.approx(np.sqrt(beta_limit_by_hand(0.95, 2, 30)) * pls.scaling_factor_for_scores_.iloc[0])


def test_the_callers_that_name_the_rows_they_judge_do_not_warn() -> None:
    """Batch monitoring judges new rows, and says so; the outlier tool and the reference-batch search fitted ones."""
    rng = np.random.default_rng(11)
    time = np.arange(8)
    batches = {
        i: pd.DataFrame({"u": a * np.sin(time / 3.0) + b + 0.1 * rng.normal(size=8), "r": a * np.cos(time / 2.0) - b})
        for i, (a, b) in enumerate(rng.normal(size=(12, 2)))
    }
    data = rng.normal(size=(20, 3)).tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        BatchMonitor(BatchPCA(n_components=2).fit(batches)).fit(batches)
        find_reference_batch(batches, ["u", "r"], settings={"n_components": 2, "judged_rows": "fitted"})
        result = tools.detect_multivariate_outliers(tools.DetectMultivariateOutliersInput(data=data, n_components=2))
    assert result["t2_limit"] == pytest.approx(beta_limit_by_hand(0.95, 2, 20))
