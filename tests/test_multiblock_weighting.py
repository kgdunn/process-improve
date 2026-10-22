"""Block weighting by the columns that vary, and components with nothing left in Y to explain (#913)."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import PredefinedSplit

from process_improve.multivariate._common import informative_widths
from process_improve.multivariate.methods import MBPCA, MBPLS, PLS

N = 40


@pytest.fixture(scope="module")
def blocks() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Two blocks of 4 columns, the same with 12 constant columns added to the second, and a response."""
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.normal(size=(N, 4)), columns=[f"a{i}" for i in range(4)])
    b = pd.DataFrame(rng.normal(size=(N, 4)), columns=[f"b{i}" for i in range(4)])
    constants = pd.DataFrame(np.ones((N, 12)) * np.arange(12), columns=[f"c{i}" for i in range(12)])
    y = pd.DataFrame({"y": a["a0"] + b["b0"] + 0.1 * rng.normal(size=N)})
    return {"a": a, "b": b}, {"a": a, "b": pd.concat([b, constants], axis=1)}, y


@pytest.fixture(scope="module")
def factorial() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return a 2^4 factorial in two blocks: one PLS component already gives the least-squares fit."""
    runs = np.array(list(itertools.product([-1, 1], repeat=4)), dtype=float)
    rng = np.random.default_rng(0)
    y = pd.DataFrame({"y": runs @ [3.0, -2.0, 1.0, 0.5] + rng.normal(scale=0.5, size=16)})
    return {"F": pd.DataFrame(runs[:, :2], columns=["A", "B"]), "Z": pd.DataFrame(runs[:, 2:], columns=["C", "D"])}, y


class TestInformativeWidths:
    def test_constant_and_unobserved_columns_do_not_count(self) -> None:
        values = np.column_stack([np.arange(5.0), np.zeros(5), np.full(5, np.nan), [np.nan, 1, np.nan, np.nan, np.nan]])
        assert informative_widths({"x": values}) == {"x": 1}

    def test_a_block_of_constants_divides_by_one(self) -> None:
        assert informative_widths({"x": np.zeros((5, 3))}) == {"x": 1}


class TestBlockWeighting:
    def test_constant_columns_leave_an_mbpls_fit_unchanged(self, blocks: tuple) -> None:
        plain, padded, y = blocks
        before, after = MBPLS(n_components=2).fit(plain, y), MBPLS(n_components=2).fit(padded, y)
        assert after.block_widths_ == {"a": 4, "b": 16}
        assert after.block_informative_widths_ == {"a": 4, "b": 4}
        np.testing.assert_allclose(after.super_weights_, before.super_weights_)
        np.testing.assert_allclose(after.r2_y_cumulative_, before.r2_y_cumulative_)
        np.testing.assert_allclose(after.predict(padded), before.predict(plain))
        np.testing.assert_allclose(after.transform(padded), before.transform(plain))

    def test_the_vip_of_the_varying_columns_is_unchanged(self, blocks: tuple) -> None:
        plain, padded, y = blocks
        before, after = MBPLS(n_components=2).fit(plain, y), MBPLS(n_components=2).fit(padded, y)
        np.testing.assert_allclose(after.block_vip_["b"].iloc[:4], before.block_vip_["b"])
        assert (after.block_vip_["b"].iloc[4:] == 0).all()

    def test_score_contributions_are_unchanged(self, blocks: tuple) -> None:
        plain, padded, y = blocks
        before, after = MBPLS(n_components=2).fit(plain, y), MBPLS(n_components=2).fit(padded, y)
        np.testing.assert_allclose(
            after.score_contributions(padded, component=1)["b"].iloc[:, :4],
            before.score_contributions(plain, component=1)["b"],
        )

    def test_a_model_fitted_before_the_count_was_kept_divides_by_its_full_width(self, blocks: tuple) -> None:
        """A model pickled by an earlier release has no block_informative_widths_; it projects as it was fitted."""
        plain, _padded, y = blocks
        model = MBPLS(n_components=2).fit(plain, y)
        expected = model.transform(plain)
        del model.block_informative_widths_  # the widths are equal here, so the projection is too
        np.testing.assert_allclose(model.transform(plain), expected)

    def test_constant_columns_leave_an_mbpca_fit_unchanged(self, blocks: tuple) -> None:
        plain, padded, _y = blocks
        before, after = MBPCA(n_components=2).fit(plain), MBPCA(n_components=2).fit(padded)
        assert after.block_informative_widths_ == {"a": 4, "b": 4}
        np.testing.assert_allclose(after.super_scores_, before.super_scores_)
        np.testing.assert_allclose(after.transform(padded), before.transform(plain))


class TestNothingLeftInY:
    @staticmethod
    def _fit(model: PLS | MBPLS, x: object, y: pd.DataFrame) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x, y)
        return [str(w.message) for w in caught]

    @pytest.mark.parametrize("weighted", [False, True])
    def test_pls_says_what_happened_and_keeps_the_predictions(self, factorial: tuple, weighted: bool) -> None:
        blocks, y = factorial
        x = pd.concat(blocks.values(), axis=1)
        model = PLS(n_components=2)
        weights = np.ones(len(x)) if weighted else None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x, y, sample_weight=weights)
        messages = [str(w.message) for w in caught]
        assert any("from component 2 on there is nothing left in Y" in m for m in messages)
        assert not any("maximum number of iterations" in m for m in messages)
        assert model.n_components_ == 2
        assert model.fitting_info_["iterations"][1] < model.max_iter
        np.testing.assert_allclose(model.predict(x), PLS(n_components=1).fit(x, y).predict(x), atol=1e-10)
        assert np.isfinite(model.direct_weights_.to_numpy()).all()
        assert np.isfinite(model.hotellings_t2_.to_numpy()).all()

    @pytest.mark.parametrize("algorithm", ["dense", "nipals"])
    def test_mbpls_says_what_happened_and_keeps_the_predictions(self, factorial: tuple, algorithm: str) -> None:
        blocks, y = factorial
        model = MBPLS(n_components=2, algorithm=algorithm)
        messages = self._fit(model, blocks, y)
        assert any("from component 2 on there is nothing left in Y" in m for m in messages)
        assert not any("did not converge" in m for m in messages)
        assert model.fitting_info_["iterations"][1] < model.max_iter
        assert model.r2_y_cumulative_.iloc[1] == pytest.approx(model.r2_y_cumulative_.iloc[0])
        one = MBPLS(n_components=1, algorithm=algorithm).fit(blocks, y)
        np.testing.assert_allclose(model.predict(blocks), one.predict(blocks), atol=1e-10)

    def test_cross_validation_is_flat_beyond_the_supported_count(self) -> None:
        """Each training fold is a full 2^4 factorial, so every fold model has nothing left after one component."""
        runs = np.array(list(itertools.product([-1, 1], repeat=4)), dtype=float)
        x = pd.DataFrame(np.vstack([runs, runs]), columns=list("ABCD"))
        rng = np.random.default_rng(1)
        y = pd.DataFrame({"y": x.to_numpy() @ [3.0, -2.0, 1.0, 0.5] + rng.normal(scale=0.5, size=32)})
        replicate = PredefinedSplit(np.repeat([0, 1], 16))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = PLS.select_n_components(x, y, max_components=3, cv=replicate, n_permutations=0)
        rmsecv = result["rmsecv"]["total"].to_numpy()
        assert rmsecv[1:] == pytest.approx([rmsecv[0], rmsecv[0]], rel=1e-9)
        assert not any("maximum number of iterations" in str(w.message) for w in caught)

    def test_data_with_more_to_explain_are_not_flagged(self, blocks: tuple) -> None:
        plain, _padded, y = blocks
        y2 = y.assign(y2=plain["a"]["a1"] - plain["b"]["b2"])
        messages = self._fit(MBPLS(n_components=2), plain, y2) + self._fit(
            PLS(n_components=2), pd.concat(plain.values(), axis=1), y2
        )
        assert not any("nothing left in Y" in m for m in messages)
