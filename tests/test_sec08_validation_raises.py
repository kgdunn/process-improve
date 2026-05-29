"""SEC-08: input/state validation uses explicit raises, not bare asserts.

Asserts are stripped under ``python -O``; these checks must remain active. The
tests assert the concrete exception type (ValueError / NotFittedError) at a few
representative converted sites. Other converted sites are covered by the
existing regression and monitoring suites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from process_improve.batch.data_input import check_valid_batch_dict, dict_to_melted, melted_to_dict, melted_to_wide
from process_improve.experiments.optimal import point_exchange
from process_improve.regression.methods import OLS, repeated_median_slope
from process_improve.univariate.metrics import detect_outliers_esd, median_absolute_deviation


class TestPointExchangeBounds:
    def _x(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [-1, 1, -1, 1], "b": [-1, -1, 1, 1], "c": [1, -1, 1, -1]})

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            point_exchange(self._x(), number_points=2)  # < n columns (3)

    def test_too_many_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at most"):
            point_exchange(self._x(), number_points=99)  # > n rows (4)


class TestOlsPredictBeforeFit:
    def test_predict_before_fit_raises_not_fitted(self) -> None:
        model = OLS()
        with pytest.raises(NotFittedError, match="must be fitted"):
            model.predict(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))


class TestMeltedToWideMissingColumn:
    def test_missing_batch_id_column_raises(self) -> None:
        df = pd.DataFrame({"not_the_id": [1, 2, 3], "value": [4.0, 5.0, 6.0]})
        with pytest.raises(ValueError, match="does not exist"):
            melted_to_wide(df, batch_id_col="batch_id")


class TestEsdOutlierValidation:
    def test_too_many_outliers_requested_raises(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="cannot exceed the sample size"):
            detect_outliers_esd(x, max_outliers_detected=99)

    def test_alpha_above_one_raises(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="alpha"):
            detect_outliers_esd(x, max_outliers_detected=1, alpha=1.5)


class TestMedianAbsoluteDeviationAxis:
    def test_axis_none_raises(self) -> None:
        with pytest.raises(ValueError, match="axis=None"):
            median_absolute_deviation(np.array([1.0, 2.0, 3.0]), axis=None)


class TestRepeatedMedianSlopeValidation:
    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="More than two samples"):
            repeated_median_slope(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            repeated_median_slope(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


class TestBatchDictValidation:
    def _batch(self, cols=("Tag01", "Tag02"), n=4) -> pd.DataFrame:
        return pd.DataFrame({c: np.arange(n, dtype=float) for c in cols})

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="At least 1 batch"):
            check_valid_batch_dict({})

    def test_column_mismatch_raises(self) -> None:
        batches = {"b1": self._batch(("Tag01", "Tag02")), "b2": self._batch(("Tag01", "TagX"))}
        with pytest.raises(ValueError, match="column names must be the same"):
            check_valid_batch_dict(batches)

    def test_non_numeric_column_raises(self) -> None:
        bad = pd.DataFrame({"Tag01": [1.0, 2.0], "Tag02": ["a", "b"]})
        with pytest.raises(ValueError, match="numeric type"):
            check_valid_batch_dict({"b1": bad})

    def test_missing_values_raises_when_no_nan(self) -> None:
        nan_batch = pd.DataFrame({"Tag01": [1.0, np.nan], "Tag02": [3.0, 4.0]})
        with pytest.raises(ValueError, match="No missing values"):
            check_valid_batch_dict({"b1": nan_batch}, no_nan=True)

    def test_unequal_row_counts_raises(self) -> None:
        batches = {"b1": self._batch(n=4), "b2": self._batch(n=5)}
        with pytest.raises(ValueError, match="same number of samples"):
            dict_to_melted(batches)

    def test_melted_to_dict_missing_column_raises(self) -> None:
        df = pd.DataFrame({"not_id": [1, 2], "value": [3.0, 4.0]})
        with pytest.raises(ValueError, match="does not exist"):
            melted_to_dict(df, batch_id_col="batch_id")
