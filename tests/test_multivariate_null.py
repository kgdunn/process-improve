"""Tests for the permutation nulls and the enrichment test.

The point of these three is that they discriminate: they separate data with a
relationship from data without one, which a VIP-exceedance count does not (see
``tests/test_multivariate_centring_traps.py::TestVipNormalisation``).
"""

from __future__ import annotations

import pathlib
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from process_improve.multivariate import (
    PLS,
    MCUVScaler,
    check_predictive_signal,
    class_enrichment,
    count_discoveries_under_null,
)
from process_improve.multivariate._common import SpecificationWarning
from process_improve.multivariate._null import _loo_fit_predict, _q2_per_column


def _blocks(
    n_products: int = 24, n_features: int = 6, noise: float = 0.4, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (x, y) with a real relationship: attribute 'real' loads on x0 and x1."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        rng.normal(size=(n_products, n_features)),
        columns=[f"c{i}" for i in range(n_features)],
        index=[f"p{i}" for i in range(n_products)],
    )
    y = pd.DataFrame(
        {
            "real": 2 * x["c0"] - x["c1"] + rng.normal(scale=noise, size=n_products),
            "noise": rng.normal(size=n_products),
        },
        index=x.index,
    )
    return x, y


def _cv_predict(n_components: int = 2, n_splits: int = 4) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Build a fit_predict that nests BOTH blocks' scaling inside each fold."""

    def fit_predict(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        for train_idx, test_idx in splitter.split(x):
            x_scaler = MCUVScaler().fit(x.iloc[train_idx])
            y_scaler = MCUVScaler().fit(y.iloc[train_idx])
            model = PLS(n_components=n_components, scale=False).fit(
                x_scaler.transform(x.iloc[train_idx]),
                y_scaler.transform(y.iloc[train_idx]),
            )
            scaled_prediction = model.predict(x_scaler.transform(x.iloc[test_idx]))
            # Back-transform with the training constants, so the score is on the
            # scale the fold actually saw.
            out.iloc[test_idx] = y_scaler.inverse_transform(np.asarray(scaled_prediction)).to_numpy()
        return out

    return fit_predict


class TestPermutationQ2:
    @pytest.mark.slow
    def test_signal_beats_the_null_and_noise_does_not(self) -> None:
        x, y = _blocks()
        table = check_predictive_signal(x, y, _cv_predict(), n_perm=99, seed=0).set_index("attribute")
        assert table.loc["real", "q2_observed"] > 0.5
        assert table.loc["real", "p_value"] <= 0.02
        assert table.loc["noise", "p_value"] > 0.10

    @pytest.mark.slow
    def test_the_null_mean_sits_below_zero(self) -> None:
        """A shuffled response predicts worse than the mean, so Q2 is negative."""
        x, y = _blocks()
        table = check_predictive_signal(x, y, _cv_predict(), n_perm=99, seed=0).set_index("attribute")
        assert table.loc["real", "q2_null_mean"] < 0
        assert table.loc["real", "q2_observed"] > table.loc["real", "q2_null_p95"]

    @pytest.mark.slow
    def test_p_value_can_never_be_zero(self) -> None:
        x, y = _blocks(noise=0.01)
        table = check_predictive_signal(x, y, _cv_predict(), n_perm=49, seed=0).set_index("attribute")
        assert table.loc["real", "p_value"] > 0
        assert table.loc["real", "p_value"] == pytest.approx(1 / 50)

    def test_columns_are_as_documented(self) -> None:
        x, y = _blocks()
        table = check_predictive_signal(x, y, _cv_predict(), n_perm=9, seed=0)
        assert list(table.columns) == [
            "attribute",
            "q2_observed",
            "q2_null_mean",
            "q2_null_p95",
            "p_value",
            "n_permutations",
        ]
        assert list(table["attribute"]) == ["real", "noise"]

    def test_whole_rows_are_permuted_not_columns(self) -> None:
        """Column-wise permutation would break the response correlation structure."""
        seen: list[pd.DataFrame] = []

        def recorder(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
            seen.append(y.copy())
            return pd.DataFrame(np.zeros(y.shape), index=y.index, columns=y.columns)

        x = pd.DataFrame({"a": np.arange(8.0)})
        y = pd.DataFrame({"u": np.arange(8.0), "v": np.arange(8.0) * 3})
        check_predictive_signal(x, y, recorder, n_perm=5, seed=0)
        for frame in seen[1:]:
            # v == 3 * u in every row of every permutation: the pairing survived.
            assert np.allclose(frame["v"].to_numpy(), 3 * frame["u"].to_numpy())

    @pytest.mark.slow
    def test_seed_makes_the_answer_reproducible(self) -> None:
        x, y = _blocks()
        first = check_predictive_signal(x, y, _cv_predict(), n_perm=19, seed=3)
        second = check_predictive_signal(x, y, _cv_predict(), n_perm=19, seed=3)
        pd.testing.assert_frame_equal(first, second)

    @pytest.mark.slow
    def test_a_different_seed_moves_the_null(self) -> None:
        x, y = _blocks()
        first = check_predictive_signal(x, y, _cv_predict(), n_perm=19, seed=1)
        second = check_predictive_signal(x, y, _cv_predict(), n_perm=19, seed=2)
        assert first["q2_null_mean"].iloc[0] != second["q2_null_mean"].iloc[0]

    def test_a_one_dimensional_return_is_accepted(self) -> None:
        x = pd.DataFrame({"a": np.arange(10.0)})
        y = pd.DataFrame({"u": np.arange(10.0)})
        table = check_predictive_signal(x, y, lambda _x, yy: np.asarray(yy).ravel(), n_perm=5)
        assert table["q2_observed"].iloc[0] == pytest.approx(1.0)

    def test_a_wrong_shape_return_raises(self) -> None:
        x, y = _blocks(n_products=8, n_features=2)
        with pytest.raises(ValueError, match="one out-of-sample prediction per row"):
            check_predictive_signal(x, y, lambda _x, _y: np.zeros((3, 3)), n_perm=2)

    def test_mismatched_rows_raise(self) -> None:
        x, y = _blocks(n_products=8, n_features=2)
        with pytest.raises(ValueError, match="same number of rows"):
            check_predictive_signal(x, y.iloc[:4], _cv_predict(), n_perm=2)

    def test_n_perm_below_one_raises(self) -> None:
        x, y = _blocks(n_products=8, n_features=2)
        with pytest.raises(ValueError, match="n_perm"):
            check_predictive_signal(x, y, _cv_predict(), n_perm=0)


class TestPipelineNull:
    @staticmethod
    def _selector(threshold: float = 0.6) -> Callable[[pd.DataFrame, pd.DataFrame], list[str]]:
        def select(x: pd.DataFrame, y: pd.DataFrame) -> list[str]:
            correlations = x.apply(lambda column: abs(np.corrcoef(column, y.iloc[:, 0])[0, 1]))
            return sorted(correlations[correlations > threshold].index)

        return select

    def test_a_real_relationship_gives_a_low_null_to_observed_ratio(self) -> None:
        x, y = _blocks(n_products=40, noise=0.2)
        result = count_discoveries_under_null(self._selector(), x, y[["real"]], n_perm=100, seed=0)
        assert result["observed"] >= 1
        assert result["null_to_observed_ratio"] < 0.2

    @pytest.mark.slow
    def test_pure_noise_gives_a_ratio_near_one(self) -> None:
        """Averaged over responses, everything a noise pipeline finds is what noise finds.

        Read over one response draw the ratio is noisy in both directions, which
        is a property of the estimator rather than of the data, so the claim is
        made across several draws.
        """
        rng = np.random.default_rng(1)
        x = pd.DataFrame(rng.normal(size=(15, 30)), columns=[f"c{i}" for i in range(30)])
        results = [
            count_discoveries_under_null(
                self._selector(threshold=0.45),
                x,
                pd.DataFrame({"noise": np.random.default_rng(100 + draw).normal(size=15)}),
                n_perm=100,
                seed=0,
            )
            for draw in range(7)
        ]
        ratios = [result["null_to_observed_ratio"] for result in results]
        assert float(np.median(ratios)) > 0.5
        # And the count itself never gets out beyond the null it was drawn from.
        assert all(result["observed"] <= result["null_p95"] for result in results)
        # The ratio is deliberately not clipped to [0, 1]: on pure noise some
        # draws have shuffling out-finding the real response, and that reading
        # is the point. Clipping it would round exactly this evidence away.
        assert max(ratios) > 1.0
        for result in results:
            assert result["null_to_observed_ratio"] == pytest.approx(result["null_mean"] / result["observed"])

    def test_a_real_relationship_puts_the_count_beyond_the_null(self) -> None:
        x, y = _blocks(n_products=40, noise=0.2)
        result = count_discoveries_under_null(self._selector(), x, y[["real"]], n_perm=100, seed=0)
        assert result["observed"] > result["null_p95"]

    def test_keys_are_as_documented(self) -> None:
        x, y = _blocks(n_products=20)
        result = count_discoveries_under_null(self._selector(), x, y[["real"]], n_perm=10, seed=0)
        assert set(result) == {
            "observed",
            "null_mean",
            "null_p95",
            "null_to_observed_ratio",
            "null_counts",
            "selected",
        }
        assert len(result["null_counts"]) == 10

    def test_selecting_nothing_gives_nan_rather_than_a_divide_by_zero(self) -> None:
        x, y = _blocks(n_products=20)
        result = count_discoveries_under_null(lambda _x, _y: [], x, y[["real"]], n_perm=5, seed=0)
        assert result["observed"] == 0
        assert np.isnan(result["null_to_observed_ratio"])

    def test_a_nondeterministic_selector_warns(self) -> None:
        x, y = _blocks(n_products=12, n_features=3)
        counter = {"n": 0}

        def flaky(_x: pd.DataFrame, _y: pd.DataFrame) -> list[str]:
            counter["n"] += 1
            return ["c0"] if counter["n"] % 2 else ["c1"]

        with pytest.warns(SpecificationWarning, match="different names on two identical calls"):
            count_discoveries_under_null(flaky, x, y, n_perm=3, seed=0)

    def test_a_deterministic_selector_does_not_warn(self) -> None:
        x, y = _blocks(n_products=12, n_features=3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SpecificationWarning)
            count_discoveries_under_null(self._selector(), x, y, n_perm=3, seed=0)

    def test_the_selector_is_re_run_end_to_end_per_permutation(self) -> None:
        """Response-independent steps are not hoisted: that would be an assumption."""
        calls: list[int] = []

        def counting(_x: pd.DataFrame, _y: pd.DataFrame) -> list[str]:
            calls.append(1)
            return ["c0"]

        x, y = _blocks(n_products=10, n_features=2)
        count_discoveries_under_null(counting, x, y, n_perm=7, seed=0)
        # One observed call, one determinism check, seven permutations.
        assert sum(calls) == 9


class TestClassEnrichment:
    ALL = [f"compound_{i}" for i in range(40)] + [f"ethyl_{i}_acetate" for i in range(8)]

    def test_a_clean_recovery_is_significant(self) -> None:
        ranked = [name for name in self.ALL if "acetate" in name] + [name for name in self.ALL if "acetate" not in name]
        result = class_enrichment(ranked, self.ALL, "acetate", top_n=10)
        assert result["in_top"] == 8
        assert result["class_size"] == 8
        assert result["n_compounds"] == 48
        assert result["p_value"] < 1e-6

    def test_a_ranking_that_ignores_the_class_is_not(self) -> None:
        ranked = [name for name in self.ALL if "acetate" not in name]
        result = class_enrichment(ranked, self.ALL, "acetate", top_n=10)
        assert result["in_top"] == 0
        assert result["p_value"] == pytest.approx(1.0)

    def test_the_pattern_is_a_regular_expression(self) -> None:
        result = class_enrichment(self.ALL, self.ALL, r"^ethyl_\d+_acetate$", top_n=48)
        assert result["class_size"] == 8

    def test_a_plain_substring_works_too(self) -> None:
        result = class_enrichment(self.ALL, self.ALL, "ethyl", top_n=48)
        assert result["class_size"] == 8

    def test_matched_names_come_back_in_ranked_order(self) -> None:
        ranked = ["ethyl_3_acetate", "compound_0", "ethyl_1_acetate"]
        result = class_enrichment(ranked, self.ALL, "acetate", top_n=3)
        assert result["matched"] == ["ethyl_3_acetate", "ethyl_1_acetate"]

    def test_top_n_is_truncated_to_the_ranking_length(self) -> None:
        result = class_enrichment(["compound_0", "compound_1"], self.ALL, "acetate", top_n=25)
        assert result["n_drawn"] == 2

    def test_an_absent_class_gives_nan_rather_than_a_meaningless_one(self) -> None:
        result = class_enrichment(self.ALL, self.ALL, "pyrazine", top_n=5)
        assert result["class_size"] == 0
        assert np.isnan(result["p_value"])

    def test_duplicates_in_the_population_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            class_enrichment(["a"], ["a", "a", "b"], "a")

    def test_a_ranked_name_outside_the_population_raises(self) -> None:
        with pytest.raises(ValueError, match="not in all_names"):
            class_enrichment(["z"], ["a", "b"], "a")

    def test_a_bad_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid regular expression"):
            class_enrichment(["a"], ["a", "b"], "[")

    def test_top_n_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            class_enrichment(["a"], ["a", "b"], "a", top_n=0)


class TestDegenerateInputs:
    """The paths where a Q2 cannot be formed, and the argument guards."""

    def test_a_constant_response_column_gives_nan_rather_than_a_divide_by_zero(self) -> None:
        x = pd.DataFrame({"a": np.arange(8.0)})
        y = pd.DataFrame({"flat": np.full(8, 3.0)})
        table = check_predictive_signal(x, y, lambda _x, yy: yy.to_numpy(), n_perm=3)
        assert np.isnan(table["q2_observed"].iloc[0])
        assert np.isnan(table["p_value"].iloc[0])
        assert table["n_permutations"].iloc[0] == 0

    def test_all_missing_predictions_give_nan(self) -> None:
        x = pd.DataFrame({"a": np.arange(8.0)})
        y = pd.DataFrame({"u": np.arange(8.0)})
        table = check_predictive_signal(x, y, lambda _x, yy: np.full(yy.shape, np.nan), n_perm=3)
        assert np.isnan(table["q2_observed"].iloc[0])

    def test_check_predictive_signal_rejects_a_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="must both be pandas DataFrames"):
            check_predictive_signal(np.zeros((8, 2)), pd.DataFrame({"y": np.zeros(8)}), lambda _x, _y: None)

    def test_check_predictive_signal_needs_at_least_two_products(self) -> None:
        with pytest.raises(ValueError, match="at least 2 products"):
            check_predictive_signal(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"y": [1.0]}), lambda _x, yy: yy)

    def test_count_discoveries_under_null_rejects_a_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="must both be pandas DataFrames"):
            count_discoveries_under_null(lambda _x, _y: [], np.zeros((8, 2)), pd.DataFrame({"y": np.zeros(8)}))

    def test_count_discoveries_under_null_rejects_mismatched_rows(self) -> None:
        x, y = _blocks(n_products=10, n_features=2)
        with pytest.raises(ValueError, match="same number of rows"):
            count_discoveries_under_null(lambda _x, _y: [], x, y.iloc[:4])

    def test_count_discoveries_under_null_rejects_n_perm_below_one(self) -> None:
        x, y = _blocks(n_products=10, n_features=2)
        with pytest.raises(ValueError, match="n_perm"):
            count_discoveries_under_null(lambda _x, _y: [], x, y, n_perm=0)


class TestDefaultFitPredict:
    """The built-in leave-one-out default, which is what most callers will use."""

    @pytest.mark.slow
    def test_the_default_separates_signal_from_noise(self) -> None:
        x, y = _blocks(n_products=20, n_features=5, noise=0.3)
        table = check_predictive_signal(x, y, n_perm=49, seed=0).set_index("attribute")
        assert table.loc["real", "q2_observed"] > 0.5
        assert table.loc["real", "p_value"] <= 0.05
        # The noise column must not look predictable: a negative Q-squared means
        # the model does worse than predicting the mean.
        assert table.loc["noise", "q2_observed"] < 0.0
        assert table.loc["noise", "p_value"] > 0.2

    @pytest.mark.slow
    def test_the_default_matches_an_explicit_equivalent_callable(self) -> None:
        x, y = _blocks(n_products=16, n_features=4)
        default = check_predictive_signal(x, y, n_perm=19, seed=0)
        explicit = check_predictive_signal(x, y, _loo_fit_predict(2), n_perm=19, seed=0)
        pd.testing.assert_frame_equal(default, explicit)

    @pytest.mark.slow
    def test_pre_scaling_the_blocks_does_not_change_the_answer(self) -> None:
        """The default must not be sensitive to preprocessing done by the caller.

        Q-squared is invariant to an affine transform of ``y``, and the default
        hands raw blocks to a PLS that re-derives its own constants inside every
        fold. A caller who scales first therefore gets the same number, rather
        than a quietly inflated one.
        """
        x, y = _blocks(n_products=16, n_features=4)
        raw = check_predictive_signal(x, y, n_perm=9, seed=0)
        scaled = check_predictive_signal(MCUVScaler().fit_transform(x), MCUVScaler().fit_transform(y), n_perm=9, seed=0)
        np.testing.assert_allclose(raw["q2_observed"].to_numpy(), scaled["q2_observed"].to_numpy(), atol=1e-8)

    def test_n_components_is_capped_at_what_the_block_supports(self) -> None:
        """A silly component count must not raise; it is clamped to the block."""
        x, y = _blocks(n_products=8, n_features=2)
        table = check_predictive_signal(x, y, n_components=99, n_perm=3, seed=0)
        assert len(table) == y.shape[1]
        assert table["q2_observed"].notna().all()


class TestRemovedAliases:
    """1.77.0 renamed these; the old spellings were removed in 2.0.0."""

    @pytest.mark.parametrize("name", ["permutation_q2", "pipeline_null"])
    def test_the_old_names_no_longer_import(self, name: str) -> None:
        from process_improve import multivariate as mv
        from process_improve.multivariate import methods

        for module in (mv, methods):
            assert name not in module.__all__
            with pytest.raises(AttributeError):
                getattr(module, name)

    def test_empirical_fdr_is_no_longer_returned(self) -> None:
        """``null_to_observed_ratio`` is the one key for the ratio."""
        x, y = _blocks(n_products=12, n_features=3)
        result = count_discoveries_under_null(lambda _x, _y: ["c0"], x, y, n_perm=5, seed=0)
        assert "empirical_fdr" not in result

    def test_an_unrelated_missing_attribute_still_raises_plainly(self) -> None:
        from process_improve.multivariate import _null

        with pytest.raises(AttributeError):
            _ = _null.not_a_real_name


def _shifted(shifts: list[float | None]) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Return a fit_predict whose k-th call predicts every row ``shifts[k]`` above its response.

    On a response of 0 to 7, whose total sum of squares is 42, a shift of d gives
    PRESS = 8 d^2 and Q2 = 1 - 8 d^2 / 42 in any row order, exactly for these shifts;
    None predicts nothing.
    """
    calls = iter(shifts)

    def fit_predict(_x: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
        shift = next(calls)
        return np.full(y.shape, np.nan) if shift is None else y.to_numpy() + shift

    return fit_predict


_EIGHT = pd.DataFrame({"a": np.arange(8.0)}, index=[f"p{i}" for i in range(8)])
_ZERO_TO_SEVEN = pd.DataFrame({"u": np.arange(8.0)}, index=_EIGHT.index)


class TestPermutationQ2ByHand:
    """Each number of the table, from Q2 values set by hand."""

    def test_the_null_summary(self) -> None:
        """Observed 34/42; the null 42, 40, 34, 34, 24, 10, -8 and -30, over 42.

        Four null values reach 34/42, ties included, so p = (1 + 4) / (8 + 1). The mean is
        146 / 8 / 42, and numpy's 95th percentile of the eight sits 0.95 * 7 = 6.65 places up
        the sorted values, 65% of the way from 40/42 to 42/42.
        """
        fit_predict = _shifted([1.0, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0])
        table = check_predictive_signal(_EIGHT, _ZERO_TO_SEVEN, fit_predict, n_perm=8)
        row = table.iloc[0]
        assert row["attribute"] == "u"
        assert row["q2_observed"] == pytest.approx(34 / 42, rel=1e-14)
        assert row["q2_null_mean"] == pytest.approx(146 / 8 / 42, rel=1e-14)
        assert row["q2_null_p95"] == pytest.approx(41.3 / 42, rel=1e-14)
        assert row["p_value"] == pytest.approx(5 / 9, rel=1e-14)
        assert row["n_permutations"] == 8

    def test_a_permutation_with_no_prediction_is_left_out(self) -> None:
        """Two usable draws, 42/42 and 10/42: one reaches the observed 34/42, so p = (1 + 1) / (2 + 1)."""
        table = check_predictive_signal(_EIGHT, _ZERO_TO_SEVEN, _shifted([1.0, None, 0.0, 2.0]), n_perm=3)
        row = table.iloc[0]
        assert row["n_permutations"] == 2
        assert row["q2_null_mean"] == pytest.approx(52 / 2 / 42, rel=1e-14)
        assert row["p_value"] == pytest.approx(2 / 3, rel=1e-14)

    def test_one_permutation_is_enough(self) -> None:
        table = check_predictive_signal(_EIGHT, _ZERO_TO_SEVEN, _shifted([1.0, 0.0]), n_perm=1)
        row = table.iloc[0]
        assert (row["q2_null_mean"], row["q2_null_p95"], row["p_value"], row["n_permutations"]) == (1.0, 1.0, 1.0, 1)

    def test_an_observed_q2_that_cannot_be_formed_leaves_the_null_unread(self) -> None:
        table = check_predictive_signal(_EIGHT, _ZERO_TO_SEVEN, _shifted([None, 0.0, 1.0]), n_perm=2)
        row = table.iloc[0]
        assert np.isnan([row["q2_observed"], row["q2_null_mean"], row["q2_null_p95"], row["p_value"]]).all()
        assert row["n_permutations"] == 2

    def test_two_products_are_enough(self) -> None:
        """Rows 0 and 1, each predicted 0.5 off: PRESS 0.5 over a total of 0.5, so Q2 = 0."""
        x = pd.DataFrame({"a": [0.0, 1.0]})
        y = pd.DataFrame({"u": [0.0, 1.0]})
        table = check_predictive_signal(x, y, lambda _x, yy: yy.to_numpy() + 0.5, n_perm=1)
        assert table["q2_observed"].iloc[0] == 0.0
        assert table["p_value"].iloc[0] == 1.0

    def test_an_integer_attribute_is_named_as_text(self) -> None:
        y = pd.DataFrame({7: np.arange(8.0)}, index=_EIGHT.index)
        table = check_predictive_signal(_EIGHT, y, _shifted([1.0, 0.0]), n_perm=1)
        assert table["attribute"].tolist() == ["7"]

    def test_a_response_with_no_attributes_gives_the_documented_columns(self) -> None:
        y = pd.DataFrame(index=_EIGHT.index)
        table = check_predictive_signal(_EIGHT, y, lambda _x, yy: np.empty(yy.shape), n_perm=1)
        assert table.empty
        assert list(table.columns) == [
            "attribute",
            "q2_observed",
            "q2_null_mean",
            "q2_null_p95",
            "p_value",
            "n_permutations",
        ]

    def test_a_prediction_of_none_is_read_as_missing(self) -> None:
        """Row 3 left out: the rest are predicted exactly, so Q2 = 1."""

        def fit_predict(_x: pd.DataFrame, y: pd.DataFrame) -> list[list[float | None]]:
            return [[value] if i != 3 else [None] for i, value in enumerate(y["u"])]

        table = check_predictive_signal(_EIGHT, _ZERO_TO_SEVEN, fit_predict, n_perm=1)
        assert table["q2_observed"].iloc[0] == 1.0

    def test_a_data_frame_of_the_wrong_shape_is_refused_by_name(self) -> None:
        """One column for two attributes: refused with the shape, as an array is, not an IndexError."""
        x, y = _blocks(n_products=8, n_features=2)
        with pytest.raises(
            ValueError,
            match=r"^fit_predict returned predictions of shape \(8, 1\), but the response block is \(8, 2\)\. "
            r"It must return one out-of-sample prediction per row and per attribute\.$",
        ):
            check_predictive_signal(x, y, lambda _x, yy: yy[["real"]], n_perm=1)

    def test_every_permuted_response_keeps_the_products_labels(self) -> None:
        """The responses move between products; the products stay where x has them."""
        seen: list[pd.Index] = []

        def recorder(_x: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
            seen.append(y.index)
            return y.to_numpy()

        check_predictive_signal(_EIGHT, _ZERO_TO_SEVEN, recorder, n_perm=3)
        assert len(seen) == 4
        for index in seen:
            pd.testing.assert_index_equal(index, _EIGHT.index)

    def test_the_defaults_are_500_permutations_from_seed_0(self) -> None:
        def shuffles(**kwargs: int) -> tuple[list[list[float]], pd.DataFrame]:
            seen: list[list[float]] = []

            def recorder(_x: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
                seen.append(y["u"].tolist())
                return y.to_numpy() + 1.0

            return seen, check_predictive_signal(_EIGHT, _ZERO_TO_SEVEN, recorder, **kwargs)

        default_order, table = shuffles()
        assert table["n_permutations"].iloc[0] == 500
        assert default_order == shuffles(n_perm=500, seed=0)[0]
        assert default_order != shuffles(n_perm=500, seed=1)[0]

    def test_a_non_frame_is_refused_in_full(self) -> None:
        with pytest.raises(TypeError, match=r"^x and y must both be pandas DataFrames, one row per product\.$"):
            check_predictive_signal(np.zeros((8, 2)), _ZERO_TO_SEVEN, _shifted([0.0]))


class TestQ2PerColumn:
    """_q2_per_column worked by hand: 1 - PRESS / TSS over the rows with both values."""

    def test_a_row_missing_either_value_is_left_out(self) -> None:
        """Rows 0, 2 and 3: values 0, 2, 3 (mean 5/3, TSS 14/3) and PRESS 0.25, so Q2 = 1 - 3/56."""
        truth = pd.DataFrame({"u": [0.0, 1.0, 2.0, 3.0, np.nan]})
        predicted = np.array([[0.5], [np.nan], [2.0], [3.0], [4.0]])
        assert _q2_per_column(truth, predicted)[0] == pytest.approx(53 / 56, rel=1e-14)

    def test_two_rows_are_enough_and_one_is_not(self) -> None:
        """0 and 2 predicted as 0.5 and 1.5: TSS 2, PRESS 0.5, so Q2 = 0.75."""
        truth = pd.DataFrame({"u": [0.0, 2.0, np.nan], "v": [1.0, np.nan, np.nan]})
        predicted = np.array([[0.5, 1.0], [1.5, 1.0], [1.0, 1.0]])
        out = _q2_per_column(truth, predicted)
        assert out[0] == 0.75
        assert np.isnan(out[1])

    def test_a_column_that_cannot_be_scored_does_not_stop_the_next(self) -> None:
        truth = pd.DataFrame({"one_row": [1.0, np.nan, np.nan], "flat": [2.0, 2.0, 2.0], "u": [0.0, 1.0, 2.0]})
        out = _q2_per_column(truth, truth.to_numpy() + 0.1)
        assert np.isnan(out[:2]).all()
        assert out[2] == pytest.approx(1 - 0.03 / 2, rel=1e-12)

    def test_a_small_spread_is_still_a_spread(self) -> None:
        """0, 0.1 and 0.2 have a TSS of 0.02; each predicted 0.01 high gives PRESS 3e-4, so Q2 = 0.985."""
        truth = pd.DataFrame({"u": [0.0, 0.1, 0.2]})
        assert _q2_per_column(truth, truth.to_numpy() + 0.01)[0] == pytest.approx(0.985, rel=1e-12)

    def test_values_held_as_objects_are_read_as_numbers(self) -> None:
        """None in an object column is missing, in the response and in the predictions alike."""
        truth = pd.DataFrame({"u": [0.0, 2.0, None, 5.0]}, dtype=object)
        predicted = np.array([[0.5], [1.5], [1.0], [None]], dtype=object)
        assert _q2_per_column(truth, predicted)[0] == 0.75


class TestDefaultFitPredictAgainstSklearn:
    """The default is leave-one-out PLS on the raw blocks, scaled per fold: sklearn's, with the capped count."""

    @pytest.mark.parametrize(
        ("n_rows", "n_features", "asked", "used"),
        [(8, 3, 1, 1), (8, 3, 0, 1), (8, 3, 5, 3), (6, 4, 9, 4), (5, 5, 9, 3)],
        ids=["as-asked", "at-least-one", "capped-by-features", "capped-by-rows", "capped-by-rows-not-features"],
    )
    def test_the_predictions_match_sklearn(
        self, capsys: pytest.CaptureFixture[str], n_rows: int, n_features: int, asked: int, used: int
    ) -> None:
        """Leave-one-out leaves n - 1 rows, whose centred rank is n - 2: the cap on the components."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import LeaveOneOut, cross_val_predict

        rng = np.random.default_rng(n_rows * 10 + n_features)
        x = pd.DataFrame(rng.normal(size=(n_rows, n_features)), columns=[f"c{i}" for i in range(n_features)])
        y = pd.DataFrame({"y": x.to_numpy() @ np.arange(1.0, n_features + 1) + rng.normal(scale=0.3, size=n_rows)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn notes a y residual that is constant to rounding
            expected = cross_val_predict(PLSRegression(n_components=used), x, y, cv=LeaveOneOut())
        with warnings.catch_warnings():
            # PLS would warn, and fit fewer, if asked for more components than the block holds.
            warnings.simplefilter("error")
            predicted = _loo_fit_predict(asked)(x, y)
        np.testing.assert_allclose(predicted.to_numpy(), expected.reshape(-1, 1), atol=1e-10)
        assert capsys.readouterr().err == ""


class TestPipelineNullByHand:
    """count_discoveries_under_null from selection counts set by hand."""

    @staticmethod
    def _scripted(counts: list[int]) -> Callable[[pd.DataFrame, pd.DataFrame], list[str]]:
        """Return a selector whose k-th call returns counts[k] names."""
        calls = iter(counts)
        return lambda _x, _y: [f"c{i}" for i in range(next(calls))]

    def test_the_null_summary(self) -> None:
        """Four names, twice, then 0, 1, 1, 2, 3, 3, 5 and 9 under the shuffles.

        Their mean is 24 / 8 = 3, a ratio of 3 / 4 to the observed count, and numpy's 95th
        percentile sits 0.95 * 7 = 6.65 places up, 65% of the way from 5 to 9: 7.6.
        """
        selector = self._scripted([4, 4, 0, 1, 1, 2, 3, 3, 5, 9])
        result = count_discoveries_under_null(selector, _EIGHT, _ZERO_TO_SEVEN, n_perm=8)
        assert result["observed"] == 4
        assert result["selected"] == ["c0", "c1", "c2", "c3"]
        np.testing.assert_array_equal(result["null_counts"], [0, 1, 1, 2, 3, 3, 5, 9])
        assert result["null_mean"] == 3.0
        assert result["null_p95"] == pytest.approx(7.6, rel=1e-14)
        assert result["null_to_observed_ratio"] == 0.75

    def test_one_permutation_is_enough(self) -> None:
        result = count_discoveries_under_null(self._scripted([2, 2, 1]), _EIGHT, _ZERO_TO_SEVEN, n_perm=1)
        assert result["null_to_observed_ratio"] == 0.5

    def test_the_defaults_are_500_permutations_from_seed_0(self) -> None:
        """A selector that returns as many names as the first product's shuffled response shows each shuffle."""

        def first_response(_x: pd.DataFrame, y: pd.DataFrame) -> list[str]:
            return [f"c{i}" for i in range(int(y.iloc[0, 0]))]

        default = count_discoveries_under_null(first_response, _EIGHT, _ZERO_TO_SEVEN)["null_counts"]
        assert len(default) == 500
        np.testing.assert_array_equal(
            default,
            count_discoveries_under_null(first_response, _EIGHT, _ZERO_TO_SEVEN, n_perm=500, seed=0)["null_counts"],
        )
        assert not np.array_equal(
            default,
            count_discoveries_under_null(first_response, _EIGHT, _ZERO_TO_SEVEN, n_perm=500, seed=1)["null_counts"],
        )

    def test_a_nondeterministic_selector_warns_in_full_and_from_the_caller(self) -> None:
        with pytest.warns(SpecificationWarning) as record:
            count_discoveries_under_null(self._scripted([1, 2, 0]), _EIGHT, _ZERO_TO_SEVEN, n_perm=1)
        assert str(record[0].message) == (
            "select() returned different names on two identical calls, so it carries its own randomness. The "
            "permutation counts below mix that randomness in with the null, and the empirical FDR cannot be read "
            "as a property of the data. Seed the selector (or make it deterministic) and re-run."
        )
        assert pathlib.Path(record[0].filename).name != "_null.py"

    def test_a_non_frame_is_refused_in_full(self) -> None:
        with pytest.raises(TypeError, match=r"^x and y must both be pandas DataFrames, one row per product\.$"):
            count_discoveries_under_null(self._scripted([0]), _EIGHT, np.arange(8.0))


class TestClassEnrichmentByHand:
    ALL = TestClassEnrichment.ALL

    def test_the_top_is_twelve_by_default(self) -> None:
        assert class_enrichment(self.ALL, self.ALL, "acetate")["n_drawn"] == 12

    def test_a_top_of_one(self) -> None:
        """One draw from 48 compounds, 8 of them acetates, finds one with probability 8 / 48."""
        ranked = ["ethyl_0_acetate", *[name for name in self.ALL if name != "ethyl_0_acetate"]]
        result = class_enrichment(ranked, self.ALL, "acetate", top_n=1)
        assert (result["in_top"], result["n_drawn"]) == (1, 1)
        assert result["p_value"] == pytest.approx(8 / 48, rel=1e-12)

    def test_an_empty_ranking_draws_nothing(self) -> None:
        result = class_enrichment([], self.ALL, "acetate")
        assert result["n_drawn"] == 0
        assert np.isnan(result["p_value"])

    def test_the_duplicates_are_named(self) -> None:
        with pytest.raises(ValueError, match=r"^all_names must not contain duplicates; repeated: \['a', 'c'\]\.$"):
            class_enrichment(["a"], ["a", "c", "a", "b", "c", "c"], "a")

    def test_the_names_outside_the_population_are_named(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"^ranked contains name\(s\) \['y', 'z'\] that are not in all_names\. The population has to "
            r"contain the sample, or the hypergeometric probability is not defined\.$",
        ):
            class_enrichment(["z", "a", "y"], ["a", "b"], "a")
