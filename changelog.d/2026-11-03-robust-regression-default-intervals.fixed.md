- **`robust_regression` returns least-squares intervals by default again, and announces the bootstrap for
  3.0.** 2.3.0 (#953) made percentile-bootstrap intervals the default without the announcement that the
  deprecation policy requires for a changed default: `conf_intervals`, `conf_interval_intercept` and
  `pi_range` changed for every call that did not choose, and the call took about nine times as long.
  `intervals` now defaults to `None`, which means `"least_squares"`, the numbers of 2.2.2, until 3.0 and
  `"bootstrap"` from 3.0. Until then a call whose intervals that change would move issues a `FutureWarning`,
  and its result carries the same text as `intervals_note`. Pass `intervals="least_squares"` to keep these
  numbers, or `intervals="bootstrap"` for the new ones; either stops the warning. A call whose intervals are
  NaN either way (two complete pairs or fewer, no spread in x, a missing value kept with `na_rm=False`) does
  not warn. The `robust_regression` tool passes `intervals="bootstrap"`, so its output does not change.
  (#1131)
- **`robust_regression` with `conflevel=np.nan` returns the fit with NaN intervals again, as in 2.2.2.**
  2.3.0 drew all 999 resamples and then raised `ValueError: Percentiles must be in the range [0, 100]`. The
  slope and the rest of the fit are returned; `conf_intervals`, `conf_interval_intercept` and the prediction
  limits of `pi_range` are NaN, and `pi_range` keeps its x column. This holds whichever `intervals` is, and no
  resample is drawn. An infinite `conflevel`, or one larger than 1 in absolute value, gives the same. (#1131)
