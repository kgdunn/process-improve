- **The `robust_regression` tool refuses an `n_boot` it cannot draw in time.** Its limit of 1,500
  points bounded the default resamples only: an explicit `n_boot=999` at 1,500 points passed the
  input contract and ran for about 26 s, past the 10 s timeout. The fit and each resample are a
  repeated median whose cost grows as the square of the number of points, so the contract now
  allows `(n_boot + 1)` times the number of points squared up to the work of the default at 1,500
  points (the fit and 39 resamples): at most 999 resamples up to 300 points, 89 at 1,000 and 39 at
  1,500. A larger call is refused before any worker starts, with a `ToolInputInvalidError` that
  says, for example, `robust_regression draws at most 39 bootstrap resamples of 1,500 points as a
  tool; got n_boot=999. ... Pass n_boot=39 or fewer, fit a random sample of fewer points, or call
  process_improve.regression.robust_regression in Python.` The `n_boot` description in the tool's
  schema gives the limit. The default resamples, and the Python function, are unchanged. (#1169)
