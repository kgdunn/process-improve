- **`OLS.prediction_interval` on an unfitted model raises `NotFittedError`, as `OLS.predict` does.**
  It checked with an `assert`, so it raised `AssertionError`, and under `python -O`, which removes
  asserts, it went on to an `AttributeError` for a model never fitted, and to NaN intervals without
  a word for a fit that had too few rows. It now raises sklearn's `NotFittedError: OLS must be
  fitted before calling prediction_interval().` in both cases, however Python runs. (#1169)
