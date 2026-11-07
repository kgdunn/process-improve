"""Block weighting by the columns that vary, and components with nothing left in Y to explain (#913)."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import PredefinedSplit

from process_improve.multivariate._common import informative_widths
from process_improve.multivariate.methods import MBPCA, MBPLS, PLS, NotEnoughVarianceError, SpecificationWarning

N = 40
#: Padded and plain fits are compared to near machine precision, beyond the default tolerance's ~1e-8 steps,
#: which a different numpy release can stop at a slightly different point.
TIGHT = 1e-12


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
        before, after = MBPLS(n_components=2, tol=TIGHT).fit(plain, y), MBPLS(n_components=2, tol=TIGHT).fit(padded, y)
        assert after.block_widths_ == {"a": 4, "b": 16}
        assert after.block_informative_widths_ == {"a": 4, "b": 4}
        np.testing.assert_allclose(after.super_weights_, before.super_weights_)
        np.testing.assert_allclose(after.r2_y_cumulative_, before.r2_y_cumulative_)
        np.testing.assert_allclose(after.predict(padded), before.predict(plain))
        np.testing.assert_allclose(after.transform(padded), before.transform(plain))

    def test_the_vip_of_the_varying_columns_is_unchanged(self, blocks: tuple) -> None:
        plain, padded, y = blocks
        before, after = MBPLS(n_components=2, tol=TIGHT).fit(plain, y), MBPLS(n_components=2, tol=TIGHT).fit(padded, y)
        np.testing.assert_allclose(after.block_vip_["b"].iloc[:4], before.block_vip_["b"])
        assert (after.block_vip_["b"].iloc[4:] == 0).all()

    def test_score_contributions_are_unchanged(self, blocks: tuple) -> None:
        plain, padded, y = blocks
        before, after = MBPLS(n_components=2, tol=TIGHT).fit(plain, y), MBPLS(n_components=2, tol=TIGHT).fit(padded, y)
        np.testing.assert_allclose(
            after.score_contributions(padded, component=1)["b"].iloc[:, :4],
            before.score_contributions(plain, component=1)["b"],
        )

    def test_a_model_fitted_before_the_count_was_kept_divides_by_its_full_width(self, blocks: tuple) -> None:
        """A model pickled by an earlier release has no block_informative_widths_; it projects as it was fitted."""
        plain, _padded, y = blocks
        model = MBPLS(n_components=2, tol=TIGHT).fit(plain, y)
        expected = model.transform(plain)
        del model.block_informative_widths_  # the widths are equal here, so the projection is too
        np.testing.assert_allclose(model.transform(plain), expected)

    def test_constant_columns_leave_an_mbpca_fit_unchanged(self, blocks: tuple) -> None:
        plain, padded, _y = blocks
        before, after = MBPCA(n_components=2, tol=TIGHT).fit(plain), MBPCA(n_components=2, tol=TIGHT).fit(padded)
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

    def test_mbpls_components_past_the_rank_of_x_leave_the_predictions_as_they_were(self) -> None:
        """Their Y-loadings are 0, so six components predict as four do on six columns of X of rank four (#1074).

        Once the four directions are spent, what is left of X is round-off, and so is the super score
        of each later component. Y's regression on that score, its Y-loading until then, was the
        ratio of two round-off errors, and it moved the predictions. Six columns, not four: more
        components than columns are cut to the number of columns, with a warning (#1149).
        """
        rng = np.random.default_rng(0)
        x = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
        y = pd.DataFrame({"y": x.to_numpy() @ [1.0, -0.5, 0.3, 0.0] + rng.normal(0, 0.1, 30)})
        x = x.assign(e=x["a"] + x["b"], f=x["c"] - x["d"])
        blocks = {"b1": x[["a", "b", "e"]], "b2": x[["c", "d", "f"]]}
        four = MBPLS(n_components=4).fit(blocks, y)
        with pytest.warns(SpecificationWarning, match="from component 5 on there is nothing left in Y"):
            six = MBPLS(n_components=6).fit(blocks, y)
        np.testing.assert_array_equal(six.super_y_loadings_.iloc[:, 4:].to_numpy(), np.zeros((1, 2)))
        np.testing.assert_allclose(six.predictions_, four.predictions_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(six.predict(blocks), four.predict(blocks), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(six.r2_y_per_component_.iloc[4:].to_numpy(), [0.0, 0.0])

    def test_pls_components_along_x_leave_the_predictions_as_they_were(self) -> None:
        """PLS's components along X have Y-loadings of 0 too, so three components predict as one does (#1074).

        A 2x2x2 design with columns scaled 1, 3 and 2, the third nudged by 1e-9 times the x1 x2
        interaction, and ``y = x1 + x1 x2 / 2``: after one component what is left of y is the
        interaction, which X reaches only through the nudge. Y's regression on the third score,
        its Y-loading until then, was -3.75e-10, and it moved the predictions by 7.5e-10.
        """
        levels = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
        x = levels * [1.0, 3.0, 2.0]
        x[:, 2] += 1e-9 * levels[:, 0] * levels[:, 1]
        x = pd.DataFrame(x, columns=["x1", "x2", "x3"])
        y = pd.DataFrame({"y": levels[:, 0] + 0.5 * levels[:, 0] * levels[:, 1]})
        one = PLS(n_components=1, scale=False).fit(x, y)
        with pytest.warns(SpecificationWarning, match="from component 2 on there is nothing left in Y"):
            three = PLS(n_components=3, scale=False).fit(x, y)
        np.testing.assert_array_equal(three.y_loadings_.iloc[:, 1:].to_numpy(), np.zeros((1, 2)))
        np.testing.assert_allclose(three.predictions_, one.predictions_, rtol=0, atol=1e-14)
        np.testing.assert_allclose(three.predict(x), one.predict(x), rtol=0, atol=1e-14)

    @pytest.mark.parametrize("algorithm", ["dense", "nipals"])
    def test_mbpls_refuses_a_constant_response_as_pls_does(self, blocks: tuple, algorithm: str) -> None:
        """A response with no variation leaves every weight 0, and a new row with no score (#1075).

        MBPLS fitted it, and predicted NaN for every new row while its own predictions were Y's
        mean. PLS refuses the same response, and so does MBPLS now.
        """
        plain, _padded, _y = blocks
        constant = pd.DataFrame({"y": np.full(N, 3.0)})
        message = (
            "^There is no variance in Y: every response is constant or has no observed value, "
            "so there is nothing for X to explain.$"
        )
        with pytest.raises(NotEnoughVarianceError, match=message):
            MBPLS(n_components=1, algorithm=algorithm).fit(plain, constant)
        with pytest.raises(NotEnoughVarianceError, match=message):
            PLS(n_components=1).fit(pd.concat(plain.values(), axis=1), constant)

    def test_mbpls_refuses_a_response_with_no_observed_value_as_pls_does(self, blocks: tuple) -> None:
        """Every response unobserved leaves nothing to explain either: named in a warning, then refused (#1075)."""
        plain, _padded, _y = blocks
        unobserved = pd.DataFrame({"y": np.full(N, np.nan)})
        with (
            pytest.warns(UserWarning, match="Block Y has columns with no observed values"),
            pytest.raises(NotEnoughVarianceError, match=r"^There is no variance in Y"),
        ):
            MBPLS(n_components=1).fit(plain, unobserved)

    @pytest.mark.parametrize("algorithm", ["dense", "nipals"])
    def test_mbpls_refuses_blocks_of_constants_as_pls_does(self, blocks: tuple, algorithm: str) -> None:
        """Blocks with nothing to explain Y by leave every weight 0, and a new row with no score (#1128).

        MBPLS fitted them, and predicted NaN for every new row. PLS refuses the same columns, and
        so does MBPLS now, in the same words (#1144). The constant columns of the padded block count
        as constant here too.
        """
        _plain, padded, y = blocks
        constant = {name: block * 0.0 + 2.0 for name, block in padded.items()}
        message = r"is constant or has no observed value, so there is nothing to explain Y by\.$"
        with pytest.raises(
            NotEnoughVarianceError, match=rf"^There is no variance in X: every column of every block {message}"
        ):
            MBPLS(n_components=1, algorithm=algorithm).fit(constant, y)
        with pytest.raises(NotEnoughVarianceError, match=rf"^There is no variance in X: every column {message}"):
            PLS(n_components=1).fit(pd.concat(constant.values(), axis=1), y)

    def test_mbpls_refuses_blocks_with_no_observed_value_as_pls_does(self, blocks: tuple) -> None:
        """A block of constants beside one with no observed value has nothing to explain Y by either (#1128)."""
        plain, _padded, y = blocks
        unobserved = {"a": plain["a"] * 0.0 + 2.0, "b": plain["b"] * np.nan}
        with (
            pytest.warns(UserWarning, match=r'^Block X\["b"\] has columns with no observed values'),
            pytest.raises(NotEnoughVarianceError, match=r"^There is no variance in X"),
        ):
            MBPLS(n_components=1).fit(unobserved, y)

    def test_mbpls_refuses_a_component_once_y_is_fitted_exactly_as_pls_does(self) -> None:
        """One component fits ``y = A + B`` exactly, so a second has nothing to explain, in PLS or MBPLS (#1128).

        The columns of a 2^3 design are exactly orthogonal, and ``y`` lies along the first
        component of each model. MBPLS went on with a component along X, where PLS refuses one.
        """
        design = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=3)), columns=["A", "B", "C"])
        y = pd.DataFrame({"y": design["A"] + design["B"]})
        split = {"AB": design[["A", "B"]], "C": design[["C"]]}
        np.testing.assert_allclose(MBPLS(n_components=1).fit(split, y).predictions_, y, rtol=0, atol=1e-14)
        message = (
            r"^There is no variance left in the data array for Y: cannot compute any more components beyond "
            r"component 1\.$"
        )
        with pytest.raises(NotEnoughVarianceError, match=message):
            PLS(n_components=2).fit(design, y)
        for algorithm in ("dense", "nipals"):
            with pytest.raises(NotEnoughVarianceError, match=message):
                MBPLS(n_components=2, algorithm=algorithm).fit(split, y)

    def test_data_with_more_to_explain_are_not_flagged(self, blocks: tuple) -> None:
        plain, _padded, y = blocks
        y2 = y.assign(y2=plain["a"]["a1"] - plain["b"]["b2"])
        messages = self._fit(MBPLS(n_components=2), plain, y2) + self._fit(
            PLS(n_components=2), pd.concat(plain.values(), axis=1), y2
        )
        assert not any("nothing left in Y" in m for m in messages)
