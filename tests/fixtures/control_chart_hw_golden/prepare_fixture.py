"""Record the golden output of the Holt-Winters ``ControlChart`` for an equivalence test.

The golden file pins every number ``ControlChart(variant="hw").calculate_limits`` produces
for a fixed set of input series, so a rewrite of the fit (for speed, say) can be checked
to reproduce the old output exactly, exceptions included. It must therefore be generated
from the implementation being replaced, BEFORE that implementation is touched.

Each case stores its input (exact float repr, so the test does not depend on any RNG
stream) and either the raised exception or, for a successful fit:

- every column of ``chart.df`` with its dtype and the index (``y``, which always equals the
  input, is stored once as the sentinel ``"input"``);
- ``target``, ``s``, ``ld_1``, ``ld_2``, ``ld_s``, ``_tau``, ``_target_calculated_best``,
  ``_delta_UCL_3sigma`` / ``_delta_LCL_3sigma``, ``N``, ``warm_up_M`` (value and type name);
- the full ``warm_up`` dict, ``_residuals_HW``, ``idx_outside_3S`` and ``train_samples``;
- the warnings emitted.

For every grid-search case the script also checks that the best lambda cell beats the
runner-up by a clear relative margin, so a last-bit platform difference in ``libm`` cannot
flip the ``argmin`` and make the test fail on another operating system.

Run from the repository root::

    uv run python tests/fixtures/control_chart_hw_golden/prepare_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import warnings
from typing import Any

import numpy as np
import pandas as pd

from process_improve.monitoring import control_charts
from process_improve.monitoring.control_charts import ControlChart

HERE = pathlib.Path(__file__).parent
GOLDEN_JSON = HERE / "golden.json"

#: Minimum relative gap between the best and second-best grid cell (see module docstring).
MIN_GRID_GAP = 1e-9

SCALAR_ATTRIBUTES = (
    "target",
    "s",
    "ld_1",
    "ld_2",
    "ld_s",
    "_tau",
    "_target_calculated_best",
    "_delta_UCL_3sigma",
    "_delta_LCL_3sigma",
    "N",
    "warm_up_M",
)


def _noise(seed: int, n: int, loc: float = 50.0, scale: float = 2.0) -> np.ndarray:
    """Return Gaussian noise from the legacy, stream-stable ``RandomState``."""
    return loc + scale * np.random.RandomState(seed).standard_normal(n)


def _set(values: np.ndarray, where: Any, value: float) -> np.ndarray:  # noqa: ANN401
    """Return a copy of ``values`` with ``values[where] = value``."""
    out = values.copy()
    out[where] = value
    return out


def _add(values: np.ndarray, where: Any, delta: float) -> np.ndarray:  # noqa: ANN401
    """Return a copy of ``values`` with ``delta`` added at ``where``."""
    out = values.copy()
    out[where] += delta
    return out


def _cases() -> list[dict[str, Any]]:
    """Return the golden cases: name, input series and ``calculate_limits`` kwargs."""
    ramp = 10.0 + 0.25 * np.arange(50) + _noise(3, 50, loc=0.0, scale=0.5)
    step = _add(_noise(4, 60), slice(40, None), 6.0)
    spikes = _add(_add(_noise(5, 45, loc=20.0, scale=1.0), 28, 15.0), 38, -12.0)
    inf_then_nan = _set(_set(_noise(9, 40), 25, np.inf), 28, np.nan)
    leading_nans = _set(_noise(6, 40, loc=100.0), slice(0, 4), np.nan)
    mad_zero = np.array([5.0] * 8 + [1.0, 9.0] + (5.0 + np.random.RandomState(11).standard_normal(20)).tolist())
    stable = _noise(1, 120)
    stable_40 = _noise(24, 40)  # shorter series for the pinned-parameter variants keeps the fixture small

    cases: list[dict[str, Any]] = [
        # Grid search (no lambdas supplied): the slow path this fixture exists for.
        {"name": "stable_noise_n120", "y": stable},
        {"name": "stable_noise_n200_warmup_cap", "y": _noise(2, 200)},
        {"name": "ramp_n50", "y": ramp},
        {"name": "step_change_n60", "y": step},
        {"name": "spikes_n45", "y": spikes},
        {"name": "warmup_outlier_n40", "y": _add(_noise(13, 40), 3, 20.0)},
        {"name": "leading_nan_n40", "y": _set(_noise(6, 40, loc=100.0), 0, np.nan)},
        {"name": "leading_nans_raise_n40", "y": leading_nans},
        {"name": "embedded_nans_n50", "y": _set(_noise(7, 50), [20, 21, 37], np.nan)},
        {"name": "nan_run_longer_than_window_n50", "y": _set(_noise(8, 50), slice(20, 32), np.nan)},
        {"name": "inf_then_nan_n40", "y": inf_then_nan},
        {"name": "near_constant_n40", "y": _noise(10, 40, loc=100.0, scale=1e-6)},
        {"name": "mad_zero_warmup_n30", "y": mad_zero},
        {"name": "constant_raise_n30", "y": np.full(30, 42.0)},
        {"name": "short_n2", "y": _noise(12, 2)},
        {"name": "short_n9", "y": _noise(12, 9)},
        {"name": "short_n10", "y": _noise(14, 10)},
        {"name": "short_n11", "y": _noise(15, 11)},
        {"name": "short_n19", "y": _noise(16, 19)},
        {"name": "short_n20", "y": _noise(17, 20)},
        {"name": "short_n21", "y": _noise(18, 21)},
        {"name": "int_list_n40", "y": [int(v) for v in np.round(_noise(19, 40, scale=4.0))]},
        {"name": "series_custom_index_n30", "y": pd.Series(_noise(20, 30), index=np.arange(100, 130))},
        {"name": "float32_n30", "y": _noise(21, 30).astype(np.float32)},
        {"name": "nullable_float64_n30", "y": pd.Series(_noise(22, 30), dtype="Float64")},
        # Pinned lambdas, and pinned target / s (the ``isinstance(..., float)`` branches).
        {"name": "fixed_lambdas_n40", "y": stable_40, "kwargs": {"ld_1": 0.4, "ld_2": 0.7}},
        {"name": "fixed_lambda_zero_n40", "y": stable_40, "kwargs": {"ld_1": 0.0, "ld_2": 0.5}},
        {"name": "fixed_lambdas_one_n40", "y": stable_40, "kwargs": {"ld_1": 1.0, "ld_2": 1.0}},
        {"name": "fixed_int_lambdas_n40", "y": stable_40, "kwargs": {"ld_1": 1, "ld_2": 0}},
        {"name": "given_target_and_s_n40", "y": stable_40, "kwargs": {"target": 50.0, "s": 2.0}},
        {"name": "given_target_n40", "y": stable_40, "kwargs": {"target": 50}},
        {"name": "given_s_n40", "y": stable_40, "kwargs": {"s": 2}},
        {
            "name": "mad_zero_warmup_given_target_n30",
            "y": mad_zero,
            "kwargs": {"target": 5.0, "ld_1": 0.5, "ld_2": 0.8},
        },
        {
            "name": "leading_nans_fixed_lambdas_raise_n40",
            "y": leading_nans,
            "kwargs": {"ld_1": 0.4, "ld_2": 0.7},
        },
        {
            "name": "nullable_float64_with_na_raise_n50",
            "y": _with_na(pd.Series(_noise(23, 50), dtype="Float64"), 25),
            "kwargs": {"ld_1": 0.4, "ld_2": 0.7},
        },
    ]
    for case in cases:
        case.setdefault("kwargs", {})
    return cases


def _with_na(series: pd.Series, position: int) -> pd.Series:
    out = series.copy()
    out.iloc[position] = pd.NA
    return out


def _json_value(value: Any) -> Any:  # noqa: ANN401
    """Convert a scalar to a JSON-native value, keeping NaN / inf and mapping ``pd.NA`` to null."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def _encode_input(y: Any) -> dict[str, Any]:  # noqa: ANN401
    if isinstance(y, pd.Series):
        return {
            "kind": "series",
            "dtype": str(y.dtype),
            "values": [_json_value(v) for v in y.tolist()],
            "index": [int(i) for i in y.index],
        }
    if isinstance(y, np.ndarray):
        return {"kind": "ndarray", "dtype": str(y.dtype), "values": [_json_value(v) for v in y.tolist()]}
    return {"kind": "list", "values": [_json_value(v) for v in y]}


def _encode_index(index: pd.Index) -> dict[str, Any]:
    if isinstance(index, pd.RangeIndex):
        return {"type": "RangeIndex", "start": index.start, "stop": index.stop, "step": index.step}
    return {"type": type(index).__name__, "values": [_json_value(v) for v in index]}


def _encode_warm_up(warm_up: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in warm_up.items():
        if isinstance(value, pd.Series):
            out[key] = {
                "series": [_json_value(v) for v in value.tolist()],
                "index": [int(i) for i in value.index],
                "dtype": str(value.dtype),
            }
        else:
            out[key] = {"value": _json_value(value), "type": type(value).__name__}
    return out


def _record(chart: ControlChart) -> dict[str, Any]:
    df = chart.df
    record: dict[str, Any] = {
        "df": {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "index": _encode_index(df.index),
            "data": {col: [_json_value(v) for v in df[col].tolist()] for col in df.columns},
        },
        "scalars": {},
        "warm_up": _encode_warm_up(chart.warm_up),
        "idx_outside_3S": [int(i) for i in chart.idx_outside_3S],
        "train_samples": [int(i) for i in getattr(chart, "train_samples", [])],
    }
    for name in SCALAR_ATTRIBUTES:
        if hasattr(chart, name):
            value = getattr(chart, name)
            record["scalars"][name] = {"value": _json_value(value), "type": type(value).__name__}
    if hasattr(chart, "_residuals_HW"):
        record["residuals_HW"] = [[_json_value(v) for v in row] for row in chart._residuals_HW.tolist()]
    return record


def _grid_gap(residuals: np.ndarray) -> float:
    """Relative margin by which the winning grid cell beats the runner-up."""
    finite = np.sort(residuals[np.isfinite(residuals)].ravel())
    if finite.size < 2:
        return float("inf")
    return float((finite[1] - finite[0]) / abs(finite[0]))


def _run(case: dict[str, Any]) -> dict[str, Any]:
    chart = ControlChart(variant="hw")
    entry: dict[str, Any] = {"name": case["name"], "input": _encode_input(case["y"]), "kwargs": case["kwargs"]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            chart.calculate_limits(case["y"], **case["kwargs"])
        except Exception as exc:  # noqa: BLE001  - recording the exception is the point
            entry["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        else:
            entry["result"] = _record(chart)
            # chart.df["y"] is the input itself; store it once, under "input", to keep the file small.
            data = entry["result"]["df"]["data"]
            if json.dumps(data["y"]) != json.dumps(entry["input"]["values"]):
                raise RuntimeError(f"{case['name']}: chart.df['y'] differs from the input")
            data["y"] = "input"
    entry["warnings"] = sorted({f"{w.category.__name__}: {w.message}" for w in caught})

    if "result" in entry and "residuals_HW" in entry["result"]:
        gap = _grid_gap(chart._residuals_HW)
        print(f"  {case['name']:<40s} grid gap {gap:.3e}")
        if gap < MIN_GRID_GAP:
            raise RuntimeError(
                f"{case['name']}: best and runner-up grid cells are within {gap:.1e}; pick another series"
            )
    else:
        print(f"  {case['name']:<40s} {entry.get('exception', {}).get('type', 'ok (no grid)')}")
    return entry


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=40"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=HERE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    """Run every case on the current implementation and write ``golden.json``."""
    print(f"recording from {control_charts.__file__}")
    entries = [_run(case) for case in _cases()]
    provenance = {
        "commit": _git_commit(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    # One compact case per line: keeps the fixture small yet greppable by case name.
    cases = ",\n".join(json.dumps(entry, separators=(",", ":")) for entry in entries)
    GOLDEN_JSON.write_text(f'{{"generated_from":{json.dumps(provenance)},"cases":[\n{cases}\n]}}\n')
    print(f"wrote {len(entries)} cases to {GOLDEN_JSON}")


if __name__ == "__main__":
    main()
