- **The Holt-Winters `ControlChart` uses a given `sigma` when the data cannot estimate the scale.** On a
  series with no usable spread, such as one where every value is the same, `calculate_limits(y, sigma=...)`
  raised with pinned lambdas or in the lambda search, although the error told the caller to pass `sigma`,
  and ignored `sigma` and `target` below the warm-up length (fewer than 10 values), giving limits of zero
  width (`s = 0`) or `s = NaN` for one value. Now `sigma` is the scale of the limits, the lambdas that the
  data cannot choose are 0.5 and 0.8, and a short series uses a given `target`. Data with usable spread give
  the same results as before. Without `sigma`, a short series with no spread now raises `ValueError`, so
  the `control_chart` tool reports it as an error instead of charting it with zero-width limits.
