"""Golden-output equivalence test for the Holt-Winters ``ControlChart`` fit.

``tests/fixtures/control_chart_hw_golden/golden.json`` was recorded from the per-row pandas
implementation of ``_holt_winters_warmup_fit`` before it was rewritten over numpy arrays.
Every case must still produce the same ``chart.df`` (values, dtypes, columns, index), the
same fitted scalars and warm-up statistics, the same lambda grid, and the same exception
(type and message) on the inputs that raise. Values must agree to 1e-12; on the platform
that recorded the fixture the rewrite is bit-identical.

Regenerate the fixture only from an implementation known to be correct, see
``tests/fixtures/control_chart_hw_golden/README.md``.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from process_improve.monitoring.control_charts import ControlChart

GOLDEN = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "control_chart_hw_golden" / "golden.json").read_text()
)
CASES = {case["name"]: case for case in GOLDEN["cases"]}
TOLERANCE = {"rtol": 1e-12, "atol": 1e-12, "equal_nan": True}


def _decode_input(spec: dict[str, Any]) -> Any:  # noqa: ANN401
    """Rebuild the exact input object (ndarray, list or Series) that was recorded."""
    if spec["kind"] == "ndarray":
        return np.asarray(spec["values"], dtype=spec["dtype"])
    if spec["kind"] == "series":
        return pd.Series(spec["values"], index=spec["index"], dtype=spec["dtype"])
    return list(spec["values"])


def _as_float(values: Any) -> np.ndarray:  # noqa: ANN401
    """JSON null (``pd.NA`` / ``None``) maps to NaN for the numeric comparison."""
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def _fit(case: dict[str, Any]) -> tuple[ControlChart, BaseException | None, set[str]]:
    chart = ControlChart(variant="hw")
    error: BaseException | None = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            chart.calculate_limits(_decode_input(case["input"]), **case["kwargs"])
        except Exception as exc:  # noqa: BLE001  - compared against the recorded exception below
            error = exc
    return chart, error, {f"{w.category.__name__}: {w.message}" for w in caught}


@pytest.mark.parametrize("name", sorted(CASES))
def test_holt_winters_matches_golden_output(name: str) -> None:
    """The fit reproduces the recorded output of the original implementation."""
    case = CASES[name]
    chart, error, emitted = _fit(case)

    # No warning the original did not also emit (it may emit fewer).
    assert emitted <= set(case["warnings"]), f"new warnings: {emitted - set(case['warnings'])}"

    if "exception" in case:
        assert error is not None, f"expected {case['exception']['type']}, but the fit succeeded"
        assert type(error).__name__ == case["exception"]["type"]
        assert str(error) == case["exception"]["message"]
        return
    assert error is None, f"unexpected {type(error).__name__}: {error}"

    expected = case["result"]
    df = chart.df
    assert list(df.columns) == expected["df"]["columns"]
    assert {col: str(dtype) for col, dtype in df.dtypes.items()} == expected["df"]["dtypes"]
    index = expected["df"]["index"]
    assert isinstance(df.index, pd.RangeIndex)
    assert (df.index.start, df.index.stop, df.index.step) == (index["start"], index["stop"], index["step"])
    for col, values in expected["df"]["data"].items():
        # "y" always equals the input, so the fixture stores it once, as the sentinel "input".
        expected_values = case["input"]["values"] if values == "input" else values
        np.testing.assert_allclose(
            _as_float(df[col].tolist()), _as_float(expected_values), err_msg=f"df[{col!r}]", **TOLERANCE
        )

    for attr, recorded in expected["scalars"].items():
        actual = getattr(chart, attr)
        assert type(actual).__name__ == recorded["type"], attr
        np.testing.assert_allclose(float(actual), _as_float([recorded["value"]])[0], err_msg=attr, **TOLERANCE)

    assert set(chart.warm_up) == set(expected["warm_up"])
    for key, recorded in expected["warm_up"].items():
        actual = chart.warm_up[key]
        if "series" in recorded:
            assert isinstance(actual, pd.Series), key
            assert actual.index.tolist() == recorded["index"]
            assert str(actual.dtype) == recorded["dtype"]
            np.testing.assert_allclose(
                _as_float(actual.tolist()), _as_float(recorded["series"]), err_msg=key, **TOLERANCE
            )
        else:
            assert type(actual).__name__ == recorded["type"], key
            np.testing.assert_allclose(float(actual), _as_float([recorded["value"]])[0], err_msg=key, **TOLERANCE)

    assert hasattr(chart, "_residuals_HW") == ("residuals_HW" in expected)
    if "residuals_HW" in expected:
        np.testing.assert_allclose(
            chart._residuals_HW, np.asarray(expected["residuals_HW"], dtype=float), err_msg="_residuals_HW", **TOLERANCE
        )
    assert chart.idx_outside_3S == expected["idx_outside_3S"]
    assert getattr(chart, "train_samples", []) == expected["train_samples"]
