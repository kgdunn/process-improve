- **`fit` checks `n_components`, `max_iter` and `tol`, however they were set.** `MBPCA`, `MBPLS`
  and `TPLS` checked some of them in `__init__`, which `set_params` and `clone` do not pass
  through, and no estimator checked `max_iter` in `fit`. A value set that way fitted with a
  convergence warning, or failed later with a numpy error. Now `PCA`, `PLS`, `PLSDA`, `PRM`,
  `MBPCA`, `MBPLS`, `TPLS` and `AdaptivePCA` check them in `fit`, with one message each across the
  family: `n_components must be >= 1; got 0.`, `max_iter must be >= 1; got 0.`, and the
  `Tolerance ...` messages for a `tol` that is not between machine precision and 10. `__init__`
  stores its arguments without checking them, as sklearn requires, so building `MBPCA`, `MBPLS` or
  `TPLS` with an out-of-range value no longer raises until `fit`; `TPLS` also checks `d_matrix` in
  `fit` now. Two values that fitted before are now refused: `PCA` checks `tol` and `max_iter` on
  the SVD path too, where they are not used, and `TPLS` refuses a `tol` of 10 or more, or one
  below machine precision, as the other iterative estimators do. (#1071)
