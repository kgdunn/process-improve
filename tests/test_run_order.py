"""Run orders robust to a time trend, checked against Núñez Ares and Goos (2019)."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import reference_catalogue, run_order_trends, time_trends, trend_robust_run_order


def _runs(name: str) -> pd.DataFrame:
    source = json.loads((reference_catalogue._CATALOGUE_DIR / "nunez-ares-goos-2019.json").read_text(encoding="utf-8"))
    entry = next(e for e in source["experiments"] + source.get("designs", []) if e["name"] == name)
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


def test_the_trends_are_orthonormal_and_orthogonal_to_the_intercept():
    trends = time_trends(9, degree=3)
    assert trends.T @ trends == pytest.approx(np.eye(3), abs=1e-12)
    assert trends.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-12)
    assert trends[:, 0] == pytest.approx((np.arange(9) - 4) / np.sqrt(60))  # the linear trend is the time count


def test_the_paper_s_run_orders_reproduce_its_table_of_cosines():
    """The 3^3 factorial in run order 2 (Table 9): every second-order term orthogonal to the linear trend."""
    found = run_order_trends(_runs("nunez-ares-goos-3-3-run-order-2"), factors=["X1", "X2", "X3"])["summary"]
    assert found["linear"]["interactions"]["max_cosine"] == pytest.approx(0, abs=1e-12)
    assert found["quadratic"]["interactions"]["max_cosine"] == pytest.approx(0.283, abs=5e-4)
    assert found["quadratic"]["quadratics"]["mean_cosine"] == pytest.approx(0.057, abs=5e-4)
    assert found["cubic"]["main_effects"]["max_cosine"] == pytest.approx(0.727, abs=5e-4)


def test_a_quadratic_column_is_measured_by_its_cosine_not_its_correlation():
    """A column that does not average zero has a cosine smaller than its correlation, by sqrt(1/3) here."""
    runs = _runs("nunez-ares-goos-3-3-run-order-1")
    found = run_order_trends(runs, factors=["X1", "X2", "X3"], degree=1)
    square = next(t for t in found["terms"] if t["term"] == "I(X1**2)")
    correlation = abs(np.corrcoef(runs["X1"] ** 2, np.arange(27))[0, 1])
    assert abs(square["cosine"]["linear"]) == pytest.approx(correlation * np.sqrt(1 / 3))


def test_the_standard_order_of_a_small_ccd_tilts_x1_by_75_degrees():
    """Table 1: X1 against the time count gives a dot product of 5 and an angle of about 75 degrees."""
    found = run_order_trends(_runs("nunez-ares-goos-fccd-2-factors-standard-order"), factors=["X1", "X2"])
    assert found["time_count_dot"]["X1"] == pytest.approx(5)
    x1 = next(t for t in found["terms"] if t["term"] == "X1")
    assert x1["angle"]["linear"] == pytest.approx(74.72, abs=0.005)


def test_an_order_given_as_positions_is_the_same_as_reordering_the_rows():
    runs = _runs("nunez-ares-goos-bbd-3-factors")[["X1", "X2", "X3"]]
    order = list(np.random.default_rng(3).permutation(len(runs)))
    by_order = run_order_trends(runs, order=order)["summary"]
    by_rows = run_order_trends(runs.iloc[order].reset_index(drop=True))["summary"]
    assert by_order == by_rows


def test_level_changes_count_the_factor_settings_altered_between_runs():
    design = pd.DataFrame({"A": [-1, -1, 1, 1], "B": [-1, 1, -1, 1]})
    found = run_order_trends(design, model="main_effects", degree=1)
    assert found["level_changes"] == {"A": 1, "B": 3}
    assert found["total_level_changes"] == 4


def test_the_search_finds_a_main_effect_order_orthogonal_to_the_linear_trend():
    full = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
    found = trend_robust_run_order(full, model="main_effects", degree=1, random_state=0)
    assert sorted(found["order"]) == list(range(8))
    assert found["trends"]["summary"]["linear"]["main_effects"]["max_cosine"] == pytest.approx(0, abs=1e-12)
    assert found["design"].to_numpy() == pytest.approx(full[found["order"]])


def test_the_search_improves_on_the_standard_order_and_is_reproducible():
    runs = _runs("nunez-ares-goos-3-3-standard-order")[["X1", "X2", "X3"]]
    before = run_order_trends(runs, degree=1)["summary"]["linear"]
    first = trend_robust_run_order(runs, degree=1, n_starts=2, n_kicks=5, random_state=7)
    again = trend_robust_run_order(runs, degree=1, n_starts=2, n_kicks=5, random_state=7)
    after = first["trends"]["summary"]["linear"]
    assert first["order"] == again["order"]
    assert before["main_effects"]["max_cosine"] > 0.9
    assert after["main_effects"]["max_cosine"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda d: run_order_trends(d, model="cubic"), "model must be one of"),
        (lambda d: run_order_trends(d, degree=5), "degree must be"),
        (lambda d: run_order_trends(d, order=[0, 0, 1, 2]), "each of the 4 row positions"),
        (lambda d: trend_robust_run_order(d, n_starts=0), "n_starts must be at least 1"),
    ],
)
def test_a_request_that_cannot_be_met_is_refused(call, message):
    design = pd.DataFrame({"A": [-1.0, 1.0, -1.0, 1.0], "B": [-1.0, -1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match=message):
        call(design)
