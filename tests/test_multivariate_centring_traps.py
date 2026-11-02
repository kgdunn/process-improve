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
from sklearn.base import clone
from sklearn.model_selection import KFold

from process_improve.multivariate import PCA, PLS, vip
from process_improve.multivariate._common import SpecificationWarning, UncentredDataWarning
from process_improve.multivariate._preprocessing import _looks_prescaled, _uncentred_columns, _warn_scaling_traps


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


class TestUncentredWarningOptOut:
    """A deliberate un-centred fit needs a narrower escape than silencing the category."""

    @staticmethod
    def _offset_blocks() -> tuple[pd.DataFrame, pd.DataFrame]:
        x = _standardised_x()
        return x, pd.DataFrame({"y": 2 * x["a"] - x["b"] + 5.0})

    def test_warning_is_its_own_class_under_the_parent(self) -> None:
        """Filters and ``pytest.warns`` written against the parent keep matching."""
        assert issubclass(UncentredDataWarning, SpecificationWarning)
        x, y = self._offset_blocks()
        with pytest.warns(UncentredDataWarning, match="fits no intercept"):
            PLS(n_components=2, scale=False).fit(x, y)

    def test_flag_silences_the_warning_under_simplefilter_error(self) -> None:
        """The reported symptom: ``filterwarnings = error`` turned the warning into a fit failure."""
        x, y = self._offset_blocks()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PLS(n_components=2, scale=False, warn_on_uncentred=False).fit(x, y)

    def test_flag_does_not_silence_other_specification_warnings(self) -> None:
        """What blanket suppression costs: a second, unrelated warning from the same fit."""
        x, y = self._offset_blocks()
        with pytest.warns(SpecificationWarning, match="more than can be computed") as record:
            PLS(n_components=40, scale=False, warn_on_uncentred=False).fit(x, y)
        assert not [w for w in record if issubclass(w.category, UncentredDataWarning)]

    def test_category_filter_is_the_wider_opt_out(self) -> None:
        """For callers who cannot reach the constructor, e.g. a model built by other code."""
        x, y = self._offset_blocks()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", UncentredDataWarning)
            with pytest.warns(SpecificationWarning, match="more than can be computed"):
                PLS(n_components=40, scale=False).fit(x, y)
        assert not [w for w in record if issubclass(w.category, UncentredDataWarning)]

    def test_flag_changes_only_the_warning_never_the_fit(self) -> None:
        """Opting out must not quietly centre the data or otherwise move the numbers."""
        x, y = self._offset_blocks()
        with _numbers_only():
            loud = PLS(n_components=2, scale=False).fit(x, y)
        quiet = PLS(n_components=2, scale=False, warn_on_uncentred=False).fit(x, y)
        np.testing.assert_allclose(
            np.asarray(loud.predict(x), dtype=float),
            np.asarray(quiet.predict(x), dtype=float),
        )

    def test_flag_is_a_no_op_when_scale_is_true(self) -> None:
        """``scale=True`` centres both blocks itself, so there is nothing to opt out of."""
        x, y = self._offset_blocks()
        with _no_specification_warning():
            PLS(n_components=2, scale=True, warn_on_uncentred=False).fit(x, y)

    def test_flag_survives_get_params_and_clone(self) -> None:
        """It has to reach a fit that a Pipeline or a grid search constructs."""
        model = PLS(n_components=2, scale=False, warn_on_uncentred=False)
        assert model.get_params()["warn_on_uncentred"] is False
        x, y = self._offset_blocks()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clone(model).fit(x, y)

    def test_default_stays_on(self) -> None:
        """The warning is correct; the opt-out is opt-in."""
        assert PLS(n_components=2).get_params()["warn_on_uncentred"] is True

    def test_message_points_at_the_flag(self) -> None:
        """A caller who hits this by surprise should not have to search for the remedy."""
        x, y = self._offset_blocks()
        with pytest.warns(UncentredDataWarning) as record:
            PLS(n_components=2, scale=False).fit(x, y)
        assert "warn_on_uncentred=False" in str(record[0].message)


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
        """The behaviour the warning is about: identical RMSECV from different scalings.

        Without in-fold scaling the two stay apart only for a model that does not scale
        itself: a ``PLS(scale=True)`` autoscales each training fold on its own, and its
        held-out predictions are now read in its own space (#1027).
        """
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
                        PLS.select_n_components(
                            block, y, cv=cv, max_components=4, scale_inside_folds=False, scale=False
                        )["rmsecv"]
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

    def test_the_mean_tolerance_includes_its_edge(self) -> None:
        """A column whose mean is exactly 0.05 still counts as centred.

        The four values sum to exactly 0.2 in floating point, so their mean is exactly 0.05, and
        their deviations 1.5, -0.5, -0.5, -0.5 give a sample standard deviation of exactly 1.
        """
        column = [1.55, -0.45, -0.45, -0.4500000000000001]
        assert np.mean(column) == 0.05
        assert np.std(column, ddof=1) == 1.0
        assert _looks_prescaled(pd.DataFrame({"edge": column}))

    def test_each_criterion_is_needed_on_its_own(self) -> None:
        """Centred but with sd 1.2, or unit variance but with mean 0.1: neither is standardised."""
        block = _standardised_x()
        assert not _looks_prescaled(block * 1.2)
        assert not _looks_prescaled(block + 0.1)

    def test_numbers_held_as_objects_and_an_empty_column(self) -> None:
        """Object-dtype numbers are read as numbers; an all-missing column is skipped without a RuntimeWarning."""
        block = _standardised_x().astype(object)
        block["empty"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _looks_prescaled(block)


class TestUncentredColumns:
    """Direct tests of the helper the A1 warning is built on: ``|mean| / sd`` against 0.5."""

    def test_the_ratio_is_strict_and_set_at_one_half(self) -> None:
        """4, 0, 0, 0 has mean 1 and sd exactly 2: on the line, not over it. 4.4, 0.4, 0.4, 0.4 is at 0.7."""
        block = pd.DataFrame({"on_the_line": [4.0, 0.0, 0.0, 0.0], "over": [4.4, 0.4, 0.4, 0.4]})
        assert _uncentred_columns(block) == ["over"]

    def test_no_spread_is_judged_on_the_mean_alone(self) -> None:
        """At or below sqrt(tiny) = 2**-511 a spread counts as none, and so does a mean.

        The column 2**-510, 0, 2**-511 has mean and sample sd both exactly 2**-511: no spread, and a
        mean too small to report. Read as a spread, its mean would be twice the 0.5 ratio.
        """
        tiny = 2.0**-511
        block = pd.DataFrame(
            {
                "spread_at_the_edge": [2.0 * tiny, 0.0, tiny],
                "constant_at_the_edge": [tiny, tiny, tiny],
                "constant_above_it": [2.0 * tiny] * 3,
                "holds_inf": [np.inf, 1.0, 2.0],
                "empty": [np.nan] * 3,
            }
        )
        assert _uncentred_columns(block) == ["constant_above_it"]


class TestScalingTrapMessages:
    """The full text of both warnings, and that they are not attributed to the helper that raises them."""

    def test_the_leak_warning(self) -> None:
        with pytest.warns(SpecificationWarning) as record:
            _warn_scaling_traps(_standardised_x(), scale_inside_folds=False, fold="CV fold", metric="RMSECV")
        assert len(record) == 1
        assert str(record[0].message) == (
            "scale_inside_folds=False leaks centring/scaling estimated on the full dataset into every CV fold, "
            "making the reported RMSECV optimistic. The default scale_inside_folds=True is preferred."
        )
        assert not record[0].filename.endswith("_preprocessing.py")

    def test_the_prescaled_warning(self) -> None:
        with pytest.warns(SpecificationWarning) as record:
            _warn_scaling_traps(_standardised_x(), scale_inside_folds=True, fold="element-fold", metric="PRESS")
        assert len(record) == 1
        assert str(record[0].message) == (
            "X is already centred and unit-variance scaled, so scale_inside_folds=True is not protecting you "
            "from leakage: it re-standardises inside every element-fold and overwrites the scaling you chose. "
            "Two deliberately different scalings (say autoscale versus Pareto) collapse onto the same "
            "standardised model this way, so a comparison of their PRESS reflects only the units the blocks "
            "arrived in, never which scaling suits the data. Pass the raw, unscaled X and let the folds scale "
            "it, or keep your own scaling and set scale_inside_folds=False (accepting the optimism that flag "
            "warns about)."
        )
        assert not record[0].filename.endswith("_preprocessing.py")

    def test_a_raw_block_scaled_inside_the_folds_is_quiet(self) -> None:
        with _no_specification_warning():
            _warn_scaling_traps(_standardised_x() * 3 + 1, scale_inside_folds=True, fold="CV fold", metric="RMSECV")
