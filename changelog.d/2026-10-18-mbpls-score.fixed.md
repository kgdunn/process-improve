- **`MBPLS.score` returns the R² of the predictions.** `MBPLS` is a scikit-learn regressor, whose
  `score` calls `predict`, but it had no `predict` that returned predictions, so `score` raised in
  every release: a `ValueError` in 1.99.0, where `predict` returned the `diagnose` Bunch, and an
  `AttributeError` in 2.0.0, which removed that method. `score` now averages the R² over the Y
  columns with equal weights, as `PLS.score` does, and takes `sample_weight`.
