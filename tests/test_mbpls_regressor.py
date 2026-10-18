"""MBPLS meets scikit-learn's regressor contract: ``predict`` returns the predicted Y and ``score`` its R² (#781).

``MBPLS`` inherits ``RegressorMixin``, whose ``score`` calls ``predict``. Before 2.0 that ``predict``
was a deprecated forwarder to ``diagnose`` that returned a Bunch, so ``score`` raised a ``ValueError``;
2.0 removed it, so ``score`` raised an ``AttributeError``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_regressor
from sklearn.metrics import r2_score

from process_improve.multivariate.methods import MBPLS


def _blocks(n_rows: int, seed: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return two X-blocks and two responses, the first driven by block a, the second by block b."""
    rng = np.random.default_rng(seed)
    index = [f"r{i}" for i in range(n_rows)]
    X = {
        "a": pd.DataFrame(rng.standard_normal((n_rows, 3)), columns=list("abc"), index=index),
        "b": pd.DataFrame(rng.standard_normal((n_rows, 2)), columns=list("de"), index=index),
    }
    y = pd.DataFrame(
        {
            "y1": X["a"]["a"] + 0.1 * rng.standard_normal(n_rows),
            "y2": X["b"]["d"] - X["a"]["b"] + 0.3 * rng.standard_normal(n_rows),
        },
        index=index,
    )
    return X, y


@pytest.fixture(scope="module")
def fitted() -> tuple[MBPLS, dict[str, pd.DataFrame], pd.DataFrame]:
    X, y = _blocks(30, seed=0)
    return MBPLS(n_components=2).fit(X, y), X, y


def test_score_is_the_r_squared_of_the_in_sample_predictions(fitted) -> None:
    model, X, y = fitted
    found = model.score(X, y)
    assert isinstance(found, float)
    assert found == pytest.approx(r2_score(y, model.predictions_))


def test_score_averages_the_y_columns_with_equal_weights(fitted) -> None:
    model, X, y = fitted
    predicted = model.predict(X)
    per_column = [r2_score(y[name], predicted[name]) for name in y.columns]
    assert model.score(X, y) == pytest.approx(np.mean(per_column))


def test_score_passes_the_sample_weights_on(fitted) -> None:
    model, X, y = fitted
    weights = np.linspace(0.5, 2.0, len(y))
    assert model.score(X, y, sample_weight=weights) == pytest.approx(
        r2_score(y, model.predict(X), sample_weight=weights)
    )


def test_predict_is_the_diagnose_predictions_of_new_rows(fitted) -> None:
    model, _, _ = fitted
    new, _ = _blocks(7, seed=1)
    predicted = model.predict(new)
    pd.testing.assert_frame_equal(predicted, model.diagnose(new).predictions)
    assert list(predicted.index) == list(new["a"].index)
    assert list(predicted.columns) == ["y1", "y2"]


def test_it_is_still_a_regressor_and_a_clone_fits(fitted) -> None:
    model, X, y = fitted
    assert is_regressor(MBPLS(n_components=2))
    copy = clone(model).fit(X, y)
    assert copy.score(X, y) == pytest.approx(model.score(X, y))
