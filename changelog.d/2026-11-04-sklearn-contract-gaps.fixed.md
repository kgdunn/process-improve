- **`PCA`, `MBPCA`, `MBPLS` and `TPLS` take a numpy integer or float wherever they take a Python one.**
  `n_components=np.int64(2)`, a count as `np.argmax` or a grid of `np.arange` values gives it, was refused
  with `InvalidParameterError`, and so were a numpy `max_iter` and `tol=np.float32(1e-6)`, though `PLS` took
  them. The constraints these estimators declare, and those of `BatchPCA`, `BatchPLS` and `BatchMonitor`, now
  name any integer and any real number, so a whole-number `tol` such as `tol=1` is accepted too. The range
  checks and their messages do not change. (#1127)
- **`TPLS.__init__` stores its arguments and nothing else, and `AdaptivePCA.fit` and `AdaptivePLS.fit` leave
  `n_components` as given.** `TPLS` set `is_fitted_` and `required_blocks_` in `__init__`; they now exist
  from `fit` on, and `plot` is a property, so `model.plot.scores()` and `model.plot.loadings()` work as
  before. The adaptive models wrote `int(n_components)` back into `n_components`, which `clone` and
  `get_params` read; the count they fitted is in `n_components_`, as for the rest of the family. (#1127)
- **`CA`, `MCA`, `FAMD`, `MFA`, `FPCA` and `impute_low_rank` refuse an `n_components` below 1 with the
  family's message,** `n_components must be >= 1; got 0.`, where they said `n_components must be at least
  1`. (#1127)
