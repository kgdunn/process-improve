"""Tests for the provisional interaction-term helpers.

Unit tests only, as the module docstring says: none of this has been run against
a real product-by-compound block with a real sensory response.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from process_improve.interactions import interaction_terms, pair_coverage, stability_selection
from process_improve.multivariate._common import SpecificationWarning


def _standardised(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.mean()) / frame.std(ddof=1)


class TestPairCoverage:
    def test_a_full_factorial_pattern_is_covered(self) -> None:
        rng = np.random.default_rng(0)
        corners = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        points = np.repeat(corners, 5, axis=0) + rng.normal(scale=0.05, size=(20, 2))
        covered, detail = pair_coverage(points[:, 0], points[:, 1], min_per_corner=4)
        assert covered
        assert detail["low_low"] == 5
        assert detail["high_high"] == 5
        assert detail["n"] == 20

    def test_co_varying_variables_fail_and_the_correlation_says_why(self) -> None:
        """The correct answer, not a defect to work around."""
        rng = np.random.default_rng(0)
        a = rng.normal(size=40)
        b = a + rng.normal(scale=0.05, size=40)
        covered, detail = pair_coverage(a, b)
        assert not covered
        assert detail["low_high"] < 4
        assert detail["high_low"] < 4
        assert detail["correlation"] > 0.9

    def test_min_per_corner_is_respected(self) -> None:
        corners = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        points = np.repeat(corners, 3, axis=0) + np.linspace(0, 0.01, 24).reshape(12, 2)
        assert pair_coverage(points[:, 0], points[:, 1], min_per_corner=3)[0]
        assert not pair_coverage(points[:, 0], points[:, 1], min_per_corner=4)[0]

    def test_the_split_is_at_each_variable_s_own_median(self) -> None:
        a = np.array([10.0, 20.0, 30.0, 40.0])
        b = np.array([-5.0, -1.0, 1.0, 5.0])
        _covered, detail = pair_coverage(a, b, min_per_corner=1)
        assert detail["threshold_a"] == pytest.approx(25.0)
        assert detail["threshold_b"] == pytest.approx(0.0)

    def test_missing_values_are_dropped_pairwise(self) -> None:
        a = np.array([-1.0, -1.0, 1.0, 1.0, np.nan])
        b = np.array([-1.0, 1.0, -1.0, 1.0, 1.0])
        _covered, detail = pair_coverage(a, b, min_per_corner=1)
        assert detail["n"] == 4

    def test_an_empty_overlap_is_not_covered(self) -> None:
        covered, detail = pair_coverage(np.array([np.nan]), np.array([1.0]))
        assert not covered
        assert detail["n"] == 0
        assert np.isnan(detail["correlation"])

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            pair_coverage(np.zeros(4), np.zeros(5))

    def test_min_per_corner_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="min_per_corner"):
            pair_coverage(np.zeros(4), np.zeros(4), min_per_corner=0)


class TestInteractionTerms:
    @staticmethod
    def _parents(seed: int = 0, n: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        raw = pd.DataFrame(
            {
                "a": rng.normal(size=n),
                "b": rng.normal(size=n),
            }
        )
        raw["c"] = 0.8 * raw["a"] + 0.6 * rng.normal(size=n)
        return _standardised(raw)

    def test_products_are_re_centred_and_re_scaled(self) -> None:
        parents = self._parents()
        terms, _constants = interaction_terms(parents, [("a", "b")])
        assert terms["a_x_b"].mean() == pytest.approx(0.0, abs=1e-12)
        assert terms["a_x_b"].std(ddof=1) == pytest.approx(1.0)

    def test_the_raw_product_is_neither_centred_nor_unit_variance(self) -> None:
        """Why the second pass exists: mean r and variance 1 + r squared.

        Asserted at a sample size where the asymptotic result is clean; at the
        thirty products a real study has, the same quantities are visible but
        noisy, which is the reason the correction is applied unconditionally
        rather than only when it looks needed.
        """
        parents = self._parents(n=8000)
        raw_product = parents["a"] * parents["c"]
        r = float(parents["a"].corr(parents["c"]))
        assert abs(r) > 0.5
        assert raw_product.mean() == pytest.approx(r, abs=0.02)
        assert raw_product.var(ddof=1) == pytest.approx(1 + r**2, abs=0.1)
        # The correlated pair's raw product carries the larger variance, which is
        # the inflation the re-scaling removes.
        assert raw_product.var(ddof=1) > (parents["a"] * parents["b"]).var(ddof=1)

    def test_the_correction_removes_that_inflation(self) -> None:
        parents = self._parents(n=8000)
        terms, _constants = interaction_terms(parents, [("a", "b"), ("a", "c")])
        assert terms["a_x_b"].var(ddof=1) == pytest.approx(1.0)
        assert terms["a_x_c"].var(ddof=1) == pytest.approx(1.0)
        assert terms["a_x_c"].mean() == pytest.approx(0.0, abs=1e-12)

    def test_constants_record_the_centre_the_divisor_and_the_parent_correlation(self) -> None:
        parents = self._parents()
        _terms, constants = interaction_terms(parents, [("a", "c")])
        row = constants.set_index("term").loc["a_x_c"]
        raw_product = parents["a"] * parents["c"]
        assert row["center"] == pytest.approx(raw_product.mean())
        assert row["divisor"] == pytest.approx(raw_product.std(ddof=1))
        assert row["parent_correlation"] == pytest.approx(float(parents["a"].corr(parents["c"])))
        assert row["left"] == "a"
        assert row["right"] == "c"

    def test_constants_replay_onto_held_out_rows(self) -> None:
        parents = self._parents()
        # Standardise on the training rows, as a real nested pipeline would, so
        # the held-out rows never touch the parents' own constants either.
        train_raw, test_raw = parents.iloc[:40], parents.iloc[40:]
        centre, spread = train_raw.mean(), train_raw.std(ddof=1)
        train, test = (train_raw - centre) / spread, (test_raw - centre) / spread
        _terms, constants = interaction_terms(train, [("a", "b")])
        row = constants.set_index("term").loc["a_x_b"]
        replayed = (test["a"] * test["b"] - row["center"]) / row["divisor"]
        assert np.isfinite(replayed.to_numpy()).all()
        # A test row cannot have moved a training constant.
        _t2, constants2 = interaction_terms(train, [("a", "b")])
        pd.testing.assert_frame_equal(constants, constants2)

    def test_unscaled_parents_warn(self) -> None:
        raw = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [100.0, 250.0, 90.0, 300.0]})
        with pytest.warns(SpecificationWarning, match="does not look centred"):
            interaction_terms(raw, [("a", "b")])

    def test_standardised_parents_do_not_warn(self) -> None:
        parents = self._parents()
        with warnings.catch_warnings():
            warnings.simplefilter("error", SpecificationWarning)
            interaction_terms(parents, [("a", "b")])

    def test_a_self_pair_gives_a_quadratic_term(self) -> None:
        parents = self._parents()
        terms, _constants = interaction_terms(parents, [("a", "a")])
        assert list(terms.columns) == ["a_x_a"]
        assert terms["a_x_a"].mean() == pytest.approx(0.0, abs=1e-12)

    def test_an_unknown_column_raises(self) -> None:
        parents = self._parents()
        with pytest.raises(ValueError, match="not in x_log"):
            interaction_terms(parents, [("a", "zzz")])

    def test_a_repeated_pair_raises(self) -> None:
        parents = self._parents()
        with pytest.raises(ValueError, match="duplicate term columns"):
            interaction_terms(parents, [("a", "b"), ("a", "b")])

    def test_a_name_collision_raises(self) -> None:
        parents = self._parents()
        parents["a_x_b"] = parents["a"]
        with pytest.raises(ValueError, match="collide with existing columns"):
            interaction_terms(parents, [("a", "b")])

    def test_empty_pairs_raise(self) -> None:
        with pytest.raises(ValueError, match="pairs is empty"):
            interaction_terms(self._parents(), [])


class TestStabilitySelection:
    @staticmethod
    def _selector(threshold: float = 0.5) -> Callable[[pd.DataFrame, pd.DataFrame], list[str]]:
        def select(x: pd.DataFrame, y: pd.DataFrame) -> list[str]:
            correlations = x.apply(lambda column: abs(np.corrcoef(column, y.iloc[:, 0])[0, 1]))
            return sorted(correlations[correlations > threshold].index)

        return select

    @staticmethod
    def _blocks(seed: int = 0, n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        x = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"c{i}" for i in range(6)])
        y = pd.DataFrame({"y": 3 * x["c0"] + rng.normal(scale=0.3, size=n)})
        return x, y

    def test_the_real_driver_is_selected_far_more_often(self) -> None:
        x, y = self._blocks()
        table = stability_selection(self._selector(), x, y, n_iter=25, seed=0).set_index("name")
        assert table.loc["c0", "selection_frequency"] > 0.95
        assert table.drop(index="c0")["selection_frequency"].max() < 0.5

    def test_every_column_gets_a_row_even_when_never_selected(self) -> None:
        x, y = self._blocks()
        table = stability_selection(self._selector(0.95), x, y, n_iter=5, seed=0)
        assert sorted(table["name"]) == sorted(x.columns)
        assert (table["n_subsamples"] == 10).all()

    def test_the_halves_are_complementary(self) -> None:
        """Each split is used in both directions, sharing no rows."""
        seen: list[list] = []

        def recorder(x_part: pd.DataFrame, _y: pd.DataFrame) -> list[str]:
            seen.append(list(x_part.index))
            return []

        x, y = self._blocks(n=20)
        stability_selection(recorder, x, y, n_iter=3, seed=0)
        assert len(seen) == 6
        for first, second in zip(seen[0::2], seen[1::2], strict=True):
            assert len(first) == len(second) == 10
            assert not set(first) & set(second)
            assert set(first) | set(second) == set(x.index)

    def test_output_is_sorted_by_frequency(self) -> None:
        x, y = self._blocks()
        table = stability_selection(self._selector(), x, y, n_iter=10, seed=0)
        frequencies = table["selection_frequency"].tolist()
        assert frequencies == sorted(frequencies, reverse=True)

    def test_seed_makes_the_answer_reproducible(self) -> None:
        x, y = self._blocks()
        first = stability_selection(self._selector(), x, y, n_iter=10, seed=7)
        second = stability_selection(self._selector(), x, y, n_iter=10, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_a_name_from_outside_x_raises(self) -> None:
        x, y = self._blocks()
        with pytest.raises(ValueError, match="not columns of x"):
            stability_selection(lambda _x, _y: ["not_a_column"], x, y, n_iter=1)

    def test_too_few_products_to_split_raises(self) -> None:
        x, y = self._blocks(n=3)
        with pytest.raises(ValueError, match="at least 4 products"):
            stability_selection(self._selector(), x, y, n_iter=2)

    def test_mismatched_rows_raise(self) -> None:
        x, y = self._blocks()
        with pytest.raises(ValueError, match="same number of rows"):
            stability_selection(self._selector(), x, y.iloc[:10], n_iter=2)

    def test_n_iter_below_one_raises(self) -> None:
        x, y = self._blocks()
        with pytest.raises(ValueError, match="n_iter"):
            stability_selection(self._selector(), x, y, n_iter=0)
