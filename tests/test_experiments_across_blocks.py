"""Models fitted across blocks: least squares with a block effect, the form chosen by leave-one-run-out error."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from process_improve.experiments import BlockedFit, fit_across_blocks

GRID = {f"r{i}": (a, b) for i, (a, b) in enumerate((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1))}
SHIFTS = {"d1": 0.6, "d2": -0.2, "d3": -0.4}
FORMS = {"linear": "A + B", "quadratic": "A + B + I(A**2)", "full": "A + B + A:B + I(A**2) + I(B**2)"}
ANCHORS = ["r0", "r4", "r8"]


def _truth(a: float, b: float) -> float:
    return 5 + 2 * a - b + 0.8 * a * a


def _blocked(seed: int, shifts: dict[str, float] = SHIFTS, noise: float = 0.1) -> pd.DataFrame:
    """Every grid run in the first block; the anchors and three other runs in each later block."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, (block, shift) in enumerate(shifts.items()):
        others = [r for r in GRID if r not in ANCHORS]
        chosen = list(GRID) if k == 0 else ANCHORS + list(rng.choice(others, 3, replace=False))
        for run in chosen:
            a, b = GRID[run]
            rows.append({"run": run, "A": a, "B": b, "day": block, "y": _truth(a, b) + shift + rng.normal(0, noise)})
    return pd.DataFrame(rows).set_index("run")


def test_the_fit_is_least_squares_with_a_block_effect_that_sums_to_zero() -> None:
    """Coefficients, block shifts and residual SD equal statsmodels' OLS with C(block, Sum)."""
    # Arrange
    data = _blocked(3)
    ols = smf.ols("y ~ A + B + I(A**2) + C(day, Sum)", data.reset_index()).fit()

    # Act
    fit = fit_across_blocks(data, ["y"], {"quadratic": FORMS["quadratic"]}, block="day")

    # Assert
    assert isinstance(fit, BlockedFit)
    for term in ["Intercept", "A", "B", "I(A ** 2)"]:
        assert fit.coefficients.loc[term, "y"] == pytest.approx(ols.params[term])
        assert fit.standard_errors.loc[term, "y"] == pytest.approx(ols.bse[term])
    assert fit.block_shift.loc["d1", "y"] == pytest.approx(ols.params["C(day, Sum)[S.d1]"])
    assert fit.block_shift.loc["d2", "y"] == pytest.approx(ols.params["C(day, Sum)[S.d2]"])
    assert fit.block_shift["y"].sum() == pytest.approx(0.0, abs=1e-12)
    assert fit.residual_sd["y"] == pytest.approx(np.sqrt(ols.scale))
    assert fit.df["y"] == ols.df_resid


def test_the_form_with_the_least_leave_one_run_out_error_is_chosen() -> None:
    """The true form beats the one missing its square and the one with terms it does not need."""
    # Arrange
    data = _blocked(3)

    # Act
    fit = fit_across_blocks(data, ["y"], FORMS, block="day")

    # Assert
    assert fit.chosen_form == "quadratic"
    assert fit.formula == FORMS["quadratic"]
    assert list(fit.selection.index) == ["quadratic", "full", "linear"]
    assert fit.selection["estimable"].all()
    assert (fit.selection["loo_rmse"] == fit.selection["y"]).all()


def test_leaving_out_a_run_removes_it_from_every_block_it_was_tested_in() -> None:
    """The leave-one-run-out RMSE equals refitting without all of a run's rows and predicting them in their blocks."""
    # Arrange
    data = _blocked(5).reset_index()
    errors = []
    for run in data["run"].unique():
        train, held = data[data["run"] != run], data[data["run"] == run]
        errors += list(held["y"] - smf.ols("y ~ A + B + I(A**2) + C(day, Sum)", train).fit().predict(held))

    # Act
    fit = fit_across_blocks(data.set_index("run"), ["y"], {"quadratic": FORMS["quadratic"]}, block="day")

    # Assert
    assert fit.selection.loc["quadratic", "loo_rmse"] == pytest.approx(np.sqrt(np.mean(np.square(errors))))


def test_a_run_that_alone_carries_a_term_is_named_and_an_unfittable_form_is_not_chosen() -> None:
    """Only one run varies C, so it cannot be left out; a form with more terms than rows cannot be fitted."""
    # Arrange
    data = _blocked(3).assign(C=0.0)
    data.loc[data.index == "r5", "C"] = 1.0
    forms = {"with C": "A + B + I(A**2) + C", "too many": " + ".join(f"I(A**{k})" for k in range(1, 30))}

    # Act
    fit = fit_across_blocks(data, ["y"], forms, block="day")

    # Assert
    assert fit.selection.loc["with C", "runs_alone"] == ["r5"]
    assert not fit.selection.loc["too many", "estimable"]
    assert fit.chosen_form == "with C"


def test_a_known_block_adds_its_shift_and_an_untested_block_widens_the_interval() -> None:
    """In a block of the fit the prediction moves by its shift; for a block not yet tested the interval is wider."""
    # Arrange
    fit = fit_across_blocks(_blocked(3), ["y"], FORMS, block="day")
    new = pd.DataFrame({"A": [0.5, -0.5], "B": [0.5, 0.0]})

    # Act
    average = fit.predict(new)
    in_d2 = fit.predict(new, block="d2")

    # Assert
    shift = in_d2[("y", "prediction")] - average[("y", "prediction")]
    np.testing.assert_allclose(shift, fit.block_shift.loc["d2", "y"])
    assert ((average[("y", "upper")] - average[("y", "lower")]) > (in_d2[("y", "upper")] - in_d2[("y", "lower")])).all()


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["anchors", "block"])
def test_the_95_percent_intervals_cover_about_95_percent_of_new_runs(mode: str) -> None:
    """Over 400 simulated experiments, a new run falls inside its interval about 95% of the time."""
    # Arrange
    rng = np.random.default_rng(11)
    inside = []

    # Act
    for seed in range(400):
        shifts = dict(zip(["d1", "d2", "d3", "d4"], rng.normal(0, 0.5, 4), strict=True))
        data = _blocked(seed, shifts, noise=0.2)
        a, b = rng.uniform(-1, 1, 2)
        if mode == "anchors":  # a coming block d4: its anchors measured, the new run predicted there
            fit = fit_across_blocks(data[data["day"] != "d4"], ["y"], {"q": FORMS["quadratic"]}, block="day")
            anchors = data[(data["day"] == "d4") & data.index.isin(ANCHORS)]
            interval = fit.predict(pd.DataFrame({"A": [a], "B": [b]}), anchors=anchors)
        else:
            fit = fit_across_blocks(data, ["y"], {"q": FORMS["quadratic"]}, block="day")
            interval = fit.predict(pd.DataFrame({"A": [a], "B": [b]}), block="d4")
        observed = _truth(a, b) + shifts["d4"] + rng.normal(0, 0.2)
        inside.append(interval[("y", "lower")].iloc[0] <= observed <= interval[("y", "upper")].iloc[0])

    # Assert
    assert 0.92 <= np.mean(inside) <= 0.98


def test_each_response_uses_the_rows_where_it_was_measured() -> None:
    """A response missing in some rows is fitted on the others; one never measured in a block has no shift there."""
    # Arrange
    data = _blocked(3).assign(z=lambda d: 2 * d["A"] + np.random.default_rng(1).normal(0, 0.1, len(d)))
    data.loc[data["day"] == "d3", "z"] = np.nan
    data.iloc[2, data.columns.get_loc("y")] = np.nan
    ols = smf.ols("y ~ A + B + I(A**2) + C(day, Sum)", data.reset_index().dropna(subset=["y"])).fit()

    # Act
    fit = fit_across_blocks(data, ["y", "z"], {"quadratic": FORMS["quadratic"]}, block="day")

    # Assert
    assert fit.coefficients.loc["A", "y"] == pytest.approx(ols.params["A"])
    assert np.isnan(fit.block_shift.loc["d3", "z"])
    assert fit.predict(pd.DataFrame({"A": [0.0], "B": [0.0]}), block="d3")[("z", "prediction")].isna().all()


def test_a_categorical_factor_with_effects_coding_matches_ols() -> None:
    """C(F, Sum) codes a text factor's levels to sum to zero, as statsmodels does."""
    # Arrange
    data = _blocked(3)
    data["F"] = np.where(data["B"] > 0, "high", np.where(data["B"] < 0, "low", "mid"))
    ols = smf.ols("y ~ A + C(F, Sum) + C(day, Sum)", data.reset_index()).fit()

    # Act
    fit = fit_across_blocks(data, ["y"], {"categorical": "A + C(F, Sum)"}, block="day")

    # Assert
    for term in ["A", "C(F, Sum)[S.high]", "C(F, Sum)[S.low]"]:
        assert fit.coefficients.loc[term, "y"] == pytest.approx(ols.params[term])


def test_one_block_is_fitted_without_a_block_effect() -> None:
    """With a single block there is nothing to shift: the fit is ordinary least squares."""
    # Arrange
    data = _blocked(3, {"d1": 0.0})
    ols = smf.ols("y ~ A + B + I(A**2)", data.reset_index()).fit()

    # Act
    fit = fit_across_blocks(data, ["y"], {"quadratic": FORMS["quadratic"]}, block="day")

    # Assert
    assert fit.coefficients.loc["A", "y"] == pytest.approx(ols.params["A"])
    assert fit.block_shift.loc["d1", "y"] == pytest.approx(0.0)


def test_what_cannot_be_fitted_is_refused_with_every_reason() -> None:
    """Missing columns, a two-sided formula, an unsafe or unknown name: each named; bad predict requests refused."""
    # Arrange
    data = _blocked(3)
    forms = {"two-sided": "y ~ A", "unsafe": "A + __import__", "unknown": "A + Z"}

    # Act, Assert
    with pytest.raises(ValueError, match="cannot be fitted across blocks") as refused:
        fit_across_blocks(data, ["y", "missing"], forms, block="week")
    for reason in ["'missing' is not a column", "'week' is not a column", "right-hand side only", "unsafe", "unknown"]:
        assert reason in str(refused.value)
    with pytest.raises(ValueError, match="at least one candidate form; give at least one response"):
        fit_across_blocks(data, [], {}, block="day")
    fit = fit_across_blocks(data, ["y"], FORMS, block="day")
    new = pd.DataFrame({"A": [0.0], "B": [0.0]})
    with pytest.raises(ValueError, match="not both"):
        fit.predict(new, block="d1", anchors=data.iloc[:2])
    with pytest.raises(ValueError, match="is not a block of the fit"):
        fit.predict(new, block="d9")


def test_with_nothing_fittable_no_form_is_chosen_and_nothing_is_predicted() -> None:
    """Every form has more terms than rows: no choice, and predict says why."""
    # Arrange
    data = _blocked(3).iloc[:4]

    # Act
    fit = fit_across_blocks(data, ["y"], {"full": FORMS["full"]}, block="day")

    # Assert
    assert fit.chosen_form is None
    assert fit.coefficients.empty
    with pytest.raises(ValueError, match="nothing to predict"):
        fit.predict(pd.DataFrame({"A": [0.0], "B": [0.0]}))
