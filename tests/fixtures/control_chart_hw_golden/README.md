# Holt-Winters control-chart golden output

Equivalence fixture for `ControlChart(variant="hw").calculate_limits`. It was recorded
from the original implementation, whose Holt-Winters recursion indexed `chart.df` per
row and wrote each row back with `df.loc`, immediately before that recursion was
rewritten over numpy arrays for speed. `tests/test_control_chart_hw_golden.py` checks
that the current code reproduces it.

## Files

| File | Role |
| --- | --- |
| `prepare_fixture.py` | Defines the cases, runs them on the installed `process_improve`, and writes `golden.json`. Not run in CI. |
| `golden.json` | One case per line: the exact input, the `calculate_limits` kwargs, and either the raised exception (type and message) or the full fitted state, plus the warnings emitted. `chart.df["y"]` always equals the input, so it is stored once, as the sentinel `"input"` (the generator checks the equality). `generated_from` records the tree it ran on (`git describe --dirty`) and the library versions. |

## What is pinned

Every column of `chart.df` (values, dtype, column order, index), `target`, `s`,
`ld_1`, `ld_2`, `ld_s`, `_tau`, `_target_calculated_best`, the 3-sigma deltas, `N`,
`warm_up_M` (values and Python types), the whole `warm_up` dict, the 5x5 lambda grid
`_residuals_HW`, `idx_outside_3S` and `train_samples`.

The cases cover the lambda grid search and pinned lambdas; stable noise, a ramp, a step
change, spikes, an outlier inside the warm-up window; leading, embedded and long runs
of NaN (longer than the 10-row error-fallback window) and an infinity; near-constant,
MAD-zero and constant warm-ups; N from 2 to 200 around the warm-up boundaries
(`warm_up_M` = 10 to 20, `2 * warm_up_M`); a given `target` and/or `s`; and int,
float32, nullable `Float64` (with and without `pd.NA`) and indexed `pd.Series` inputs.

The test compares values to `rtol = atol = 1e-12`. On the platform that recorded the
fixture (see `generated_from`) the array implementation is bit-identical.

For every grid-search case the generator checks that the winning lambda cell beats the
runner-up by a relative margin of at least 1e-9 (the smallest is about 7e-3), so a
last-bit `libm` difference on another operating system cannot change which lambdas are
chosen.

## History

The file was first recorded from the original per-row pandas implementation, and the
array rewrite reproduced it bit for bit. Since then it has been regenerated only for
deliberate changes, each listed here with the cases it touched. The file holds one case
per line, so `git log -p` on it shows the exact scope of every regeneration.

- `beta_hat` is float64 rather than int64 when N < `warm_up_M` (`short_n2`, `short_n9`).
- A gap that starts at index 1 carries the forecast forward instead of making the fit
  raise (`leading_nans_n40` and `leading_nans_fixed_lambdas_n40`, which were named
  `..._raise_...` while they recorded the `ValueError`).

## Regenerating

Only regenerate from an implementation that is known to be correct, and only when a
change to the numbers is intended (a bug fix, say); the point of the fixture is that a
refactor must not move them. The script prints the path of the `control_charts` module
it records from. From the repository root:

```bash
uv run python tests/fixtures/control_chart_hw_golden/prepare_fixture.py
```
