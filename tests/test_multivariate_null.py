"""Tests for the permutation nulls and the enrichment test.

The point of these three is that they discriminate: they separate data with a
relationship from data without one, which a VIP-exceedance count does not (see
``tests/test_multivariate_centring_traps.py::TestVipNormalisation``).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from process_improve.multivariate import (
    PLS,
    MCUVScaler,
    class_enrichment,
    permutation_q2,
    pipeline_null,
)
from process_improve.multivariate._common import SpecificationWarning


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
        table = permutation_q2(_cv_predict(), x, y, n_perm=99, seed=0).set_index("attribute")
        assert table.loc["real", "q2_observed"] > 0.5
        assert table.loc["real", "p_value"] <= 0.02
        assert table.loc["noise", "p_value"] > 0.10

    @pytest.mark.slow
    def test_the_null_mean_sits_below_zero(self) -> None:
        """A shuffled response predicts worse than the mean, so Q2 is negative."""
        x, y = _blocks()
        table = permutation_q2(_cv_predict(), x, y, n_perm=99, seed=0).set_index("attribute")
        assert table.loc["real", "q2_null_mean"] < 0
        assert table.loc["real", "q2_observed"] > table.loc["real", "q2_null_p95"]

    @pytest.mark.slow
    def test_p_value_can_never_be_zero(self) -> None:
        x, y = _blocks(noise=0.01)
        table = permutation_q2(_cv_predict(), x, y, n_perm=49, seed=0).set_index("attribute")
        assert table.loc["real", "p_value"] > 0
        assert table.loc["real", "p_value"] == pytest.approx(1 / 50)

    def test_columns_are_as_documented(self) -> None:
        x, y = _blocks()
        table = permutation_q2(_cv_predict(), x, y, n_perm=9, seed=0)
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
        permutation_q2(recorder, x, y, n_perm=5, seed=0)
        for frame in seen[1:]:
            # v == 3 * u in every row of every permutation: the pairing survived.
            assert np.allclose(frame["v"].to_numpy(), 3 * frame["u"].to_numpy())

    @pytest.mark.slow
    def test_seed_makes_the_answer_reproducible(self) -> None:
        x, y = _blocks()
        first = permutation_q2(_cv_predict(), x, y, n_perm=19, seed=3)
        second = permutation_q2(_cv_predict(), x, y, n_perm=19, seed=3)
        pd.testing.assert_frame_equal(first, second)

    @pytest.mark.slow
    def test_a_different_seed_moves_the_null(self) -> None:
        x, y = _blocks()
        first = permutation_q2(_cv_predict(), x, y, n_perm=19, seed=1)
        second = permutation_q2(_cv_predict(), x, y, n_perm=19, seed=2)
        assert first["q2_null_mean"].iloc[0] != second["q2_null_mean"].iloc[0]

    def test_a_one_dimensional_return_is_accepted(self) -> None:
        x = pd.DataFrame({"a": np.arange(10.0)})
        y = pd.DataFrame({"u": np.arange(10.0)})
        table = permutation_q2(lambda _x, yy: np.asarray(yy).ravel(), x, y, n_perm=5)
        assert table["q2_observed"].iloc[0] == pytest.approx(1.0)

    def test_a_wrong_shape_return_raises(self) -> None:
        x, y = _blocks(n_products=8, n_features=2)
        with pytest.raises(ValueError, match="one out-of-sample prediction per row"):
            permutation_q2(lambda _x, _y: np.zeros((3, 3)), x, y, n_perm=2)

    def test_mismatched_rows_raise(self) -> None:
        x, y = _blocks(n_products=8, n_features=2)
        with pytest.raises(ValueError, match="same number of rows"):
            permutation_q2(_cv_predict(), x, y.iloc[:4], n_perm=2)

    def test_n_perm_below_one_raises(self) -> None:
        x, y = _blocks(n_products=8, n_features=2)
        with pytest.raises(ValueError, match="n_perm"):
            permutation_q2(_cv_predict(), x, y, n_perm=0)


class TestPipelineNull:
    @staticmethod
    def _selector(threshold: float = 0.6) -> Callable[[pd.DataFrame, pd.DataFrame], list[str]]:
        def select(x: pd.DataFrame, y: pd.DataFrame) -> list[str]:
            correlations = x.apply(lambda column: abs(np.corrcoef(column, y.iloc[:, 0])[0, 1]))
            return sorted(correlations[correlations > threshold].index)

        return select

    def test_a_real_relationship_gives_a_low_empirical_fdr(self) -> None:
        x, y = _blocks(n_products=40, noise=0.2)
        result = pipeline_null(self._selector(), x, y[["real"]], n_perm=100, seed=0)
        assert result["observed"] >= 1
        assert result["empirical_fdr"] < 0.2

    @pytest.mark.slow
    def test_pure_noise_gives_an_fdr_near_one(self) -> None:
        """Averaged over responses, everything a noise pipeline finds is what noise finds.

        Read over one response draw the ratio is noisy in both directions, which
        is a property of the estimator rather than of the data, so the claim is
        made across several draws.
        """
        rng = np.random.default_rng(1)
        x = pd.DataFrame(rng.normal(size=(15, 30)), columns=[f"c{i}" for i in range(30)])
        results = [
            pipeline_null(
                self._selector(threshold=0.45),
                x,
                pd.DataFrame({"noise": np.random.default_rng(100 + draw).normal(size=15)}),
                n_perm=100,
                seed=0,
            )
            for draw in range(7)
        ]
        assert float(np.median([result["empirical_fdr"] for result in results])) > 0.5
        # And the count itself never gets out beyond the null it was drawn from.
        assert all(result["observed"] <= result["null_p95"] for result in results)

    def test_a_real_relationship_puts_the_count_beyond_the_null(self) -> None:
        x, y = _blocks(n_products=40, noise=0.2)
        result = pipeline_null(self._selector(), x, y[["real"]], n_perm=100, seed=0)
        assert result["observed"] > result["null_p95"]

    def test_keys_are_as_documented(self) -> None:
        x, y = _blocks(n_products=20)
        result = pipeline_null(self._selector(), x, y[["real"]], n_perm=10, seed=0)
        assert set(result) == {
            "observed",
            "null_mean",
            "null_p95",
            "empirical_fdr",
            "null_counts",
            "selected",
        }
        assert len(result["null_counts"]) == 10

    def test_selecting_nothing_gives_nan_rather_than_a_divide_by_zero(self) -> None:
        x, y = _blocks(n_products=20)
        result = pipeline_null(lambda _x, _y: [], x, y[["real"]], n_perm=5, seed=0)
        assert result["observed"] == 0
        assert np.isnan(result["empirical_fdr"])

    def test_a_nondeterministic_selector_warns(self) -> None:
        x, y = _blocks(n_products=12, n_features=3)
        counter = {"n": 0}

        def flaky(_x: pd.DataFrame, _y: pd.DataFrame) -> list[str]:
            counter["n"] += 1
            return ["c0"] if counter["n"] % 2 else ["c1"]

        with pytest.warns(SpecificationWarning, match="different names on two identical calls"):
            pipeline_null(flaky, x, y, n_perm=3, seed=0)

    def test_a_deterministic_selector_does_not_warn(self) -> None:
        x, y = _blocks(n_products=12, n_features=3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SpecificationWarning)
            pipeline_null(self._selector(), x, y, n_perm=3, seed=0)

    def test_the_selector_is_re_run_end_to_end_per_permutation(self) -> None:
        """Response-independent steps are not hoisted: that would be an assumption."""
        calls: list[int] = []

        def counting(_x: pd.DataFrame, _y: pd.DataFrame) -> list[str]:
            calls.append(1)
            return ["c0"]

        x, y = _blocks(n_products=10, n_features=2)
        pipeline_null(counting, x, y, n_perm=7, seed=0)
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
        table = permutation_q2(lambda _x, yy: yy.to_numpy(), x, y, n_perm=3)
        assert np.isnan(table["q2_observed"].iloc[0])
        assert np.isnan(table["p_value"].iloc[0])
        assert table["n_permutations"].iloc[0] == 0

    def test_all_missing_predictions_give_nan(self) -> None:
        x = pd.DataFrame({"a": np.arange(8.0)})
        y = pd.DataFrame({"u": np.arange(8.0)})
        table = permutation_q2(lambda _x, yy: np.full(yy.shape, np.nan), x, y, n_perm=3)
        assert np.isnan(table["q2_observed"].iloc[0])

    def test_permutation_q2_rejects_a_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="must both be pandas DataFrames"):
            permutation_q2(lambda _x, _y: None, np.zeros((8, 2)), pd.DataFrame({"y": np.zeros(8)}))

    def test_permutation_q2_needs_at_least_two_products(self) -> None:
        with pytest.raises(ValueError, match="at least 2 products"):
            permutation_q2(lambda _x, yy: yy, pd.DataFrame({"a": [1.0]}), pd.DataFrame({"y": [1.0]}))

    def test_pipeline_null_rejects_a_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="must both be pandas DataFrames"):
            pipeline_null(lambda _x, _y: [], np.zeros((8, 2)), pd.DataFrame({"y": np.zeros(8)}))

    def test_pipeline_null_rejects_mismatched_rows(self) -> None:
        x, y = _blocks(n_products=10, n_features=2)
        with pytest.raises(ValueError, match="same number of rows"):
            pipeline_null(lambda _x, _y: [], x, y.iloc[:4])

    def test_pipeline_null_rejects_n_perm_below_one(self) -> None:
        x, y = _blocks(n_products=10, n_features=2)
        with pytest.raises(ValueError, match="n_perm"):
            pipeline_null(lambda _x, _y: [], x, y, n_perm=0)
