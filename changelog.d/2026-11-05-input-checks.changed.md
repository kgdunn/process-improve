- **Every multivariate estimator refuses a count that is not a whole number, as `PCA` does.** `PLS`,
  `PLSDA`, `PRM`, `OPLS`, `AdaptivePCA`, `AdaptivePLS`, `CA`, `MCA`, `FAMD`, `MFA`, `FPCA`, `ASCA` and
  `GPA` fitted `n_components=2.5` as 2 without a word, and so with a `max_iter`, `max_weight_iter`,
  `n_orthogonal_components`, `spe_limit_window` or `n_basis` of 2.5; `FPCA`'s `degree` and
  `penalty_order` failed inside numpy. `fit` now raises scikit-learn's `InvalidParameterError`, a
  `ValueError` and a `TypeError`: "The 'n_components' parameter of PLS must be an instance of 'int' or
  None. Got 2.5 instead." A float that holds a whole number, such as `max_iter=1e4`, is refused too; a
  numpy integer is taken. (#1144)
- **`BatchPCA`, `BatchPLS` and `BatchMonitor` check the settings they declare when they fit**, as the
  multivariate estimators do. A `spe_window` of 1.5 or a `scale` of `"yes"` is refused with the same
  message; `scale` and `group_by_batch` take numpy's `bool` as well as Python's. (#1144)
