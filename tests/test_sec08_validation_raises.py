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

from process_improve.batch.data_input import melted_to_wide
from process_improve.experiments.optimal import point_exchange
from process_improve.regression.methods import OLS


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
        from process_improve.univariate.metrics import detect_outliers_esd

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="cannot exceed the sample size"):
            detect_outliers_esd(x, max_outliers_detected=99)
