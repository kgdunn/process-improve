"""Tests for the centring and scaling traps in the PLS API.

Three of these pin behaviour that used to be silent:

* ``PLS(scale=False)`` fits no intercept, so an un-centred response makes every
  prediction wrong by roughly the response mean (A1);
* ``select_n_components(..., scale_inside_folds=True)`` re-standardises inside
  each fold, which erases a scaling the caller applied deliberately (A3);
* ``sum(VIP ** 2)`` is exactly the number of X variables, so counting how many
  variables exceed VIP 1 describes the shape of the VIP distribution rather than
  the strength of any relationship (A6).
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from process_improve.multivariate import PCA, PLS, vip
from process_improve.multivariate._common import SpecificationWarning
from process_improve.multivariate._preprocessing import _looks_prescaled


@contextlib.contextmanager
def _numbers_only() -> Iterator[None]:
    """Silence the new SpecificationWarnings where the test is about the numbers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SpecificationWarning)
        yield


@contextlib.contextmanager
def _no_specification_warning() -> Iterator[None]:
    """Turn a SpecificationWarning into a failure, for the quiet-path assertions."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", SpecificationWarning)
        yield


def _standardised_x(n_samples: int = 20, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    """Build a mean-0, unit-variance X block, so only the response offset varies below."""
    rng = np.random.default_rng(seed)
    columns = [chr(ord("a") + i) for i in range(n_features)]
    x = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=columns)
    return (x - x.mean()) / x.std(ddof=1)


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(1 - ((observed - predicted) ** 2).sum() / ((observed - observed.mean()) ** 2).sum())


class TestUncentredBlockWarning:
    """A1: ``scale=False`` fits no intercept, so an offset response is silently wrong."""

    def test_offset_response_warns(self) -> None:
        x = _standardised_x()
        y = pd.DataFrame({"y": 2 * x["a"] - x["b"] + 5.0})
        with pytest.warns(SpecificationWarning, match="fits no intercept"):
            PLS(n_components=2, scale=False).fit(x, y)

    def test_warning_names_the_symptom_and_the_block(self) -> None:
        x = _standardised_x()
        y = pd.DataFrame({"y": 2 * x["a"] - x["b"] + 5.0})
        with pytest.warns(SpecificationWarning) as record:
            PLS(n_components=2, scale=False).fit(x, y)
        message = str(record[0].message)
        # The symptom, not just the condition: a caller who only reads the first
        # sentence still learns that their R2 / Q2 cannot be believed.
        assert "Y block" in message
        assert "R2" in message
        assert "Q2" in message
        assert "'y'" in message

    def test_the_offset_really_does_destroy_r2(self) -> None:
        """The behaviour the warning is about, pinned so the two cannot drift apart."""
        x = _standardised_x()
        truth = 2 * x["a"] - x["b"]
        scores = {}
        for offset in (0.0, 5.0):
            y = pd.DataFrame({"y": truth + offset})
            with _numbers_only():
                model = PLS(n_components=2, scale=False).fit(x, y)
            scores[offset] = _r2(y["y"].to_numpy(), np.asarray(model.predict(x)).ravel())
        assert scores[0.0] > 0.98
        assert scores[5.0] < -4.0

    def test_centred_blocks_do_not_warn(self) -> None:
        x = _standardised_x()
        y = pd.DataFrame({"y": 2 * x["a"] - x["b"]})
        with _no_specification_warning():
            PLS(n_components=2, scale=False).fit(x, y)

    def test_uncentred_x_block_warns(self) -> None:
        x = _standardised_x() + 4.0
        truth = 2 * x["a"] - x["b"]
        y = pd.DataFrame({"y": truth - truth.mean()})
        with pytest.warns(SpecificationWarning, match="X block"):
            PLS(n_components=2, scale=False).fit(x, y)

    def test_scale_true_never_warns_about_centring(self) -> None:
        """``scale=True`` centres both blocks itself, so the trap cannot be sprung."""
        x = _standardised_x() + 4.0
        y = pd.DataFrame({"y": 2 * x["a"] - x["b"] + 5.0})
        with _no_specification_warning():
            PLS(n_components=2, scale=True).fit(x, y)

    def test_small_offset_relative_to_spread_does_not_warn(self) -> None:
        """A mean well inside the noise costs almost no R2 and should stay quiet."""
        x = _standardised_x()
        y = pd.DataFrame({"y": 2 * x["a"] - x["b"] + 0.05})
        with _no_specification_warning():
            PLS(n_components=2, scale=False).fit(x, y)

    def test_constant_nonzero_column_warns(self) -> None:
        """A column with no spread at all cannot have been centred."""
        x = _standardised_x()
        x["constant"] = 3.0
        y = pd.DataFrame({"y": 2 * x["a"] - x["b"]})
        with pytest.warns(SpecificationWarning, match="X block"):
            PLS(n_components=2, scale=False).fit(x, y)

    def test_all_zero_column_does_not_warn(self) -> None:
        x = _standardised_x()
        x["zeros"] = 0.0
        y = pd.DataFrame({"y": 2 * x["a"] - x["b"]})
        with _no_specification_warning():
            PLS(n_components=2, scale=False).fit(x, y)


class TestPreScaledInsideFoldsWarning:
    """A3: in-fold re-standardisation erases a scaling the caller chose deliberately."""

    @staticmethod
    def _blocks() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(0)
        raw = pd.DataFrame(
            rng.lognormal(size=(40, 8)) * (10.0 ** np.arange(8)),
            columns=[f"v{i}" for i in range(8)],
        )
        y = pd.DataFrame({"y": raw["v0"] * 2 + rng.normal(scale=0.5, size=40)})
        y = y - y.mean()
        autoscale = (raw - raw.mean()) / raw.std(ddof=1)
        pareto = (raw - raw.mean()) / np.sqrt(raw.std(ddof=1))
        return raw, autoscale, pareto, y

    def test_prescaled_x_warns(self) -> None:
        _raw, autoscale, _pareto, y = self._blocks()
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        with pytest.warns(SpecificationWarning, match="already centred and unit-variance"):
            PLS.select_n_components(autoscale, y, cv=cv, max_components=4, scale_inside_folds=True)

    def test_raw_x_does_not_warn(self) -> None:
        raw, _autoscale, _pareto, y = self._blocks()
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        with _no_specification_warning():
            PLS.select_n_components(raw, y, cv=cv, max_components=4, scale_inside_folds=True)

    def test_two_deliberate_scalings_are_indistinguishable_inside_folds(self) -> None:
        """The behaviour the warning is about: identical RMSECV from different scalings."""
        _raw, autoscale, pareto, y = self._blocks()
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        with _numbers_only():
            rmsecv = {
                name: float(
                    np.asarray(
                        PLS.select_n_components(block, y, cv=cv, max_components=4, scale_inside_folds=True)["rmsecv"]
                    ).ravel()[0]
                )
                for name, block in (("autoscale", autoscale), ("pareto", pareto))
            }
            unscaled = {
                name: float(
                    np.asarray(
                        PLS.select_n_components(block, y, cv=cv, max_components=4, scale_inside_folds=False)["rmsecv"]
                    ).ravel()[0]
                )
                for name, block in (("autoscale", autoscale), ("pareto", pareto))
            }
        assert np.isclose(rmsecv["autoscale"], rmsecv["pareto"])
        assert not np.isclose(unscaled["autoscale"], unscaled["pareto"])


class TestPreScaledInsideFoldsWarningPCA:
    """A3 for PCA: ``PCA.select_n_components`` carries the same two warnings as PLS."""

    _EQUAL_SPREADS = np.full(8, 25.0)
    _UNEQUAL_SPREADS = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0])

    @staticmethod
    def _blocks(column_spreads: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Raw, autoscaled and Pareto-scaled views of one low-rank block.

        Each raw column is given exactly the spread requested. With equal spreads
        the Pareto block is the autoscaled block times one constant, so the two
        in-fold standardised matrices are identical cell for cell and the collapse
        can be pinned exactly. With unequal spreads the two scalings weight the
        columns differently, which is what a caller comparing them wants to see.
        """
        rng = np.random.default_rng(0)
        scores = rng.normal(size=(40, 2)) * np.array([6.0, 3.0])
        loadings = rng.normal(size=(2, 8))
        raw = pd.DataFrame(scores @ loadings + 0.3 * rng.normal(size=(40, 8)), columns=[f"v{i}" for i in range(8)])
        raw = (raw - raw.mean()) / raw.std(ddof=1) * column_spreads + 100.0
        autoscale = (raw - raw.mean()) / raw.std(ddof=1)
        pareto = (raw - raw.mean()) / np.sqrt(raw.std(ddof=1))
        return raw, autoscale, pareto

    @staticmethod
    def _q2_curve(block: pd.DataFrame, *, scale_inside_folds: bool) -> np.ndarray:
        result = PCA.select_n_components(
            block, max_components=4, cv=5, random_state=0, scale_inside_folds=scale_inside_folds
        )
        return result["q2"].to_numpy()

    def test_prescaled_x_warns(self) -> None:
        _raw, autoscale, _pareto = self._blocks(self._UNEQUAL_SPREADS)
        with pytest.warns(SpecificationWarning, match="already centred and unit-variance"):
            PCA.select_n_components(autoscale, max_components=4, cv=5, random_state=0)

    def test_scale_inside_folds_false_warns(self) -> None:
        _raw, autoscale, _pareto = self._blocks(self._UNEQUAL_SPREADS)
        with pytest.warns(SpecificationWarning, match="leaks"):
            PCA.select_n_components(autoscale, max_components=4, cv=5, random_state=0, scale_inside_folds=False)

    def test_raw_x_does_not_warn(self) -> None:
        raw, _autoscale, _pareto = self._blocks(self._UNEQUAL_SPREADS)
        with _no_specification_warning():
            PCA.select_n_components(raw, max_components=4, cv=5, random_state=0)

    def test_row_wise_ignores_the_flag_and_so_does_the_warning(self) -> None:
        """``row_wise`` warns about itself only; the flag it ignores earns no second warning."""
        _raw, autoscale, _pareto = self._blocks(self._UNEQUAL_SPREADS)
        with pytest.warns(SpecificationWarning) as record:
            PCA.select_n_components(autoscale, max_components=4, cv=5, cv_scheme="row_wise", random_state=0)
        messages = [str(w.message) for w in record if issubclass(w.category, SpecificationWarning)]
        assert len(messages) == 1
        assert "row_wise" in messages[0]

    def test_two_deliberate_scalings_collapse_inside_folds(self) -> None:
        """The behaviour the warning is about: in-fold standardisation undoes the caller's scaling.

        With equal column spreads the Pareto block is a constant multiple of the
        autoscaled one, so the in-fold standardised matrices coincide exactly and
        so do the Q2 curves. The same two blocks with unequal spreads are told
        apart only when the folds leave the caller's scaling alone.
        """
        _raw, autoscale, pareto = self._blocks(self._EQUAL_SPREADS)
        with _numbers_only():
            inside = [self._q2_curve(block, scale_inside_folds=True) for block in (autoscale, pareto)]
        np.testing.assert_allclose(inside[0], inside[1], rtol=1e-10)

        _raw, autoscale, pareto = self._blocks(self._UNEQUAL_SPREADS)
        with _numbers_only():
            kept = [self._q2_curve(block, scale_inside_folds=False) for block in (autoscale, pareto)]
        assert not np.allclose(kept[0], kept[1])


class TestVipNormalisation:
    """A6: VIP is normalised so its mean square is exactly 1."""

    @pytest.mark.parametrize("n_components", [1, 2, 3])
    def test_sum_of_squared_vip_equals_n_features(self, n_components: int) -> None:
        x = _standardised_x()
        y = pd.DataFrame({"y": 2 * x["a"] - x["b"]})
        model = PLS(n_components=n_components, scale=False).fit(x, y)
        scores = vip(model)
        assert float((scores**2).sum()) == pytest.approx(x.shape[1])

    def test_exceedance_count_barely_responds_to_permuting_the_response(self) -> None:
        """Why an exceedance count is not a test statistic: the null looks like the data."""
        x = _standardised_x(n_samples=30, n_features=8)
        y = pd.DataFrame({"y": 2 * x["a"] - x["b"]})
        observed = int((vip(PLS(n_components=2, scale=False).fit(x, y)) > 1.0).sum())

        rng = np.random.default_rng(0)
        null_counts = []
        for _ in range(50):
            permuted = pd.DataFrame(y.to_numpy()[rng.permutation(len(y))], index=y.index, columns=y.columns)
            null_counts.append(int((vip(PLS(n_components=2, scale=False).fit(x, permuted)) > 1.0).sum()))
        # The observed count sits inside the null's range, so a test built on it
        # cannot separate signal from noise even though the signal here is strong.
        assert min(null_counts) <= observed <= max(null_counts)


class TestPreScaledDetection:
    """Direct tests of the helper the A3 warning is built on."""

    def test_a_standardised_block_is_recognised(self) -> None:
        assert _looks_prescaled(_standardised_x())

    def test_a_ddof_zero_scaling_still_counts_as_standardised(self) -> None:
        """Off by sqrt(n / (n - 1)), 2.6% at n=20, which is inside the tolerance."""
        raw = _standardised_x()
        assert _looks_prescaled((raw - raw.mean()) / raw.std(ddof=0))

    def test_a_raw_block_is_not(self) -> None:
        assert not _looks_prescaled(_standardised_x() * 100 + 50)

    def test_an_all_constant_block_carries_no_evidence_either_way(self) -> None:
        """No column has spread, so there is nothing to judge; the answer is False."""
        assert not _looks_prescaled(pd.DataFrame({"a": [2.0, 2.0, 2.0], "b": [0.0, 0.0, 0.0]}))

    def test_a_constant_column_alongside_standardised_ones_is_ignored(self) -> None:
        block = _standardised_x()
        block["constant"] = 7.0
        assert _looks_prescaled(block)
