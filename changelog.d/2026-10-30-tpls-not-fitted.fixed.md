- **`TPLS` raises `NotFittedError` before a fit.** `__init__` sets attributes that end in an
  underscore, so scikit-learn's `check_is_fitted` passed on a model never fitted, and
  `diagnose`, `score` and a `make_tpls_scorer` scorer failed later with an `AttributeError`.
  `TPLS` now tells `check_is_fitted` whether `fit` has completed. (#1028)
