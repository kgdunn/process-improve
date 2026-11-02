"""Tukey's test for nonadditivity and the estimates of missing runs (#800), against the printed sources."""

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import estimate_missing, tukey_nonadditivity

#: Box, Hunter and Hunter (1978), Table 7.11: survival times of four animals for each poison and treatment.
TOXIC = {
    "I": [[0.31, 0.45, 0.46, 0.43], [0.82, 1.10, 0.88, 0.72], [0.43, 0.45, 0.63, 0.76], [0.45, 0.71, 0.66, 0.62]],
    "II": [[0.36, 0.29, 0.40, 0.23], [0.92, 0.61, 0.49, 1.24], [0.44, 0.35, 0.31, 0.40], [0.56, 1.02, 0.71, 0.38]],
    "III": [[0.22, 0.21, 0.18, 0.23], [0.30, 0.37, 0.38, 0.29], [0.23, 0.25, 0.24, 0.22], [0.30, 0.36, 0.31, 0.33]],
}


def _two_way(table: list[list[float]], rows: str = "row", columns: str = "column") -> pd.DataFrame:
    """Return a two-way table, one list per row, as one run per cell, the rows and columns labelled 1, 2, ..."""
    return pd.DataFrame(
        [(str(i + 1), str(j + 1), value) for i, row in enumerate(table) for j, value in enumerate(row)],
        columns=[rows, columns, "y"],
    )


def _toxic() -> pd.DataFrame:
    cells = [(p, t, y) for p, rows in TOXIC.items() for t, ys in zip("ABCD", rows, strict=True) for y in ys]
    return pd.DataFrame(cells, columns=["poison", "treatment", "y"])


class TestTukeyNonadditivity:
    def test_montgomery_impurity_one_run_per_cell(self):
        """Montgomery (2013), Example 5.10: SSN 0.0985 tested against 1.9015 on 7 df."""
        impurity = pd.DataFrame(
            {
                "temperature": [100] * 5 + [125] * 5 + [150] * 5,
                "pressure": [25, 30, 35, 40, 45] * 3,
                "y": [5, 4, 6, 3, 5, 3, 1, 4, 2, 3, 1, 1, 3, 1, 2],
            }
        )
        found = tukey_nonadditivity(impurity, "y ~ C(temperature) + C(pressure)")
        assert (found["df"], found["df_error"], found["remainder_df"]) == (1, 7.0, 7.0)
        assert found["ss"] == pytest.approx(0.0985, abs=5e-5)
        assert found["ss_error"] == pytest.approx(1.9015, abs=5e-5)
        assert found["ms_error"] == pytest.approx(0.2716, abs=5e-5)
        assert found["f"] == pytest.approx(0.36, abs=0.005)
        assert found["p_value"] == pytest.approx(0.5660, abs=5e-5), "the book's 0.5674 is that of the rounded F"

    def test_bhh_penicillin_blocks(self):
        """Box, Hunter and Hunter, Section 7.2: SNA = 2.00, theta = 0.023, remainder 224 on 11 df, F = 0.10."""
        yields = [[89, 88, 97, 94], [84, 77, 92, 79], [81, 87, 87, 85], [87, 92, 89, 84], [79, 81, 80, 88]]
        found = tukey_nonadditivity(_two_way(yields, "blend", "treatment"), "y ~ C(blend) + C(treatment)")
        assert found["ss"] == pytest.approx(2.00, abs=0.005)
        assert found["coefficient"] == pytest.approx(0.023, abs=5e-4)
        assert (found["remainder_ss"], found["remainder_df"]) == (pytest.approx(224, abs=0.5), 11.0)
        assert found["f"] == pytest.approx(0.10, abs=0.005)

    @pytest.mark.parametrize(
        ("scale", "ss", "remainder_ms"),
        [
            pytest.param(lambda y: y, 0.1537, 0.0964 / 5, id="survival time: SNA 153.7, remainder 96.4 (thousandths)"),
            pytest.param(lambda y: 1 / y, 0.454, 0.223, id="rate: 45.4, remainder mean square 22.3 (hundredths)"),
        ],
    )
    def test_bhh_toxic_agents_against_the_pure_error(self, scale, ss, remainder_ms):
        """Box, Hunter and Hunter, Section 7.8: four animals a cell, so the test is against the error within cells."""
        runs = _toxic()
        found = tukey_nonadditivity(runs.assign(y=scale(runs["y"])), "y ~ C(poison) + C(treatment)", error="pure")
        assert (found["df_error"], found["remainder_df"]) == (36.0, 5.0)
        assert found["ss"] == pytest.approx(ss, abs=5e-4)
        assert found["remainder_ss"] / found["remainder_df"] == pytest.approx(remainder_ms, abs=5e-4)

    def test_the_toxic_agents_on_their_own_scale_fail_the_test(self):
        """Against the error within cells, F = 6.9 on the survival times: the effects do not add there."""
        found = tukey_nonadditivity(_toxic(), "y ~ C(poison) + C(treatment)", error="pure")
        assert found["f"] == pytest.approx(6.9, abs=0.05)
        assert found["p_value"] < 0.05

    @pytest.mark.parametrize("power", [-1.0, -0.5, 0.0, 0.5, 2.0])
    def test_the_power_that_makes_the_effects_add(self, power):
        """Effects that add exactly on the scale ``y**power`` (the log for 0) give that power back."""
        rows, columns = np.linspace(-0.15, 0.15, 5), np.linspace(-0.15, 0.15, 6)
        additive = 1.0 + rows[:, None] + columns[None, :]
        table = np.exp(additive) if power == 0 else additive ** (1 / power)
        found = tukey_nonadditivity(_two_way(table.tolist()), "y ~ C(row) + C(column)")
        assert found["power"] == pytest.approx(power, abs=0.02)

    def test_refusals(self):
        runs = _two_way([[1.0, 2.0, 3.0], [2.0, 4.0, 7.0], [3.0, 5.0, 8.0]])
        with pytest.raises(ValueError, match="error must be one of"):
            tukey_nonadditivity(runs, "y ~ C(row) + C(column)", error="within")
        with pytest.raises(ValueError, match="no pure error"):
            tukey_nonadditivity(runs, "y ~ C(row) + C(column)", error="pure")
        with pytest.raises(ValueError, match="must name the response"):
            tukey_nonadditivity(runs, "C(row) + C(column)")
        with pytest.raises(ValueError, match="no residual degrees of freedom"):
            tukey_nonadditivity(_two_way([[1.0, 2.0], [3.0, 5.0]]), "y ~ C(row) + C(column)")


class TestEstimateMissing:
    def test_davies_two_way_table_with_five_missing(self):
        """Davies (1956), Appendix 6B: five values estimated together, and the approximate analysis of the table.

        The book completes the table with the estimates as it prints them, to one decimal; from
        those, the rows' sum of squares is 103.07 and the columns' 51.17.
        """
        nan = np.nan
        table = [
            [13.4, 15.0, 13.3, nan, 17.4, 14.8],
            [11.0, nan, 9.7, nan, 14.6, 10.2],
            [9.8, 12.5, 12.2, 15.2, 14.1, 12.9],
            [nan, 18.1, 13.1, nan, 18.8, 14.2],
            [11.9, 12.0, 11.3, 10.2, 12.5, 8.8],
        ]
        found = estimate_missing(_two_way(table), "y ~ C(row) + C(column)", decimals=1)
        cells = {(row["row"], row["column"]): row["y"] for row in found["estimates"]}
        assert cells == {("1", "4"): 15.7, ("2", "2"): 12.4, ("2", "4"): 12.5, ("4", "1"): 14.8, ("4", "4"): 16.7}
        approximate = {row["term"]: row for row in found["approximate_anova"]}
        assert approximate["C(row)"]["sum_sq"] == pytest.approx(103.07, abs=0.005)
        assert approximate["C(column)"]["sum_sq"] == pytest.approx(51.17, abs=0.005)
        assert approximate["Residual"]["df"] == 15.0, "20 for the complete table, less one for each estimate"
        assert found["completed"]["y"].notna().all()

    def test_davies_blends_two_missing_in_different_blocks(self):
        """Davies (1956), Example 5-1, with batches 1 and 15 taken as missing: 17.34 and 17.28."""
        losses = {
            "I": {"B": 18.2, "A": 16.9, "C": 17.0, "E": 18.3, "D": 15.1},
            "II": {"A": 16.5, "E": 18.3, "B": 19.2, "C": 18.1, "D": 16.0},
            "III": {"B": 17.1, "D": 17.8, "C": 17.3, "E": 19.8, "A": 17.5},
        }
        runs = pd.DataFrame(
            [(b, blend, y) for b, row in losses.items() for blend, y in row.items()], columns=["block", "blend", "loss"]
        )
        runs.loc[[0, 14], "loss"] = np.nan
        found = estimate_missing(runs, "loss ~ C(block) + C(blend)")
        assert [row["index"] for row in found["estimates"]] == [0, 14]
        assert [row["loss"] for row in found["estimates"]] == pytest.approx([17.34, 17.28], abs=0.005)

    def test_the_estimates_leave_their_runs_no_residual(self):
        """Filled in, each estimate is where the model fitted to the completed table puts it."""
        runs = _two_way([[10.0, 12.0, 15.0, 9.0], [11.0, np.nan, 16.0, 8.5], [9.5, 12.5, np.nan, 7.0]])
        found = estimate_missing(runs, "y ~ C(row) + C(column)")
        completed = found["completed"]
        refit = estimate_missing(completed.assign(y=completed["y"].where(runs["y"].notna())), "y ~ C(row) + C(column)")
        assert [row["y"] for row in refit["estimates"]] == pytest.approx([row["y"] for row in found["estimates"]])
        residuals = (
            completed["y"]
            - completed.groupby("row")["y"].transform("mean")
            - completed.groupby("column")["y"].transform("mean")
            + completed["y"].mean()
        )
        assert residuals[runs["y"].isna()].to_numpy() == pytest.approx(0.0, abs=1e-9)

    def test_refusals(self):
        runs = _two_way([[1.0, 2.0], [3.0, 5.0]])
        with pytest.raises(ValueError, match="No run is missing"):
            estimate_missing(runs, "y ~ C(row) + C(column)")
        with pytest.raises(ValueError, match="is not a column"):
            estimate_missing(runs, "z ~ C(row) + C(column)")
