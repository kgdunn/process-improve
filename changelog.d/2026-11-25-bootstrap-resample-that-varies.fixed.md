- **`PLS.cross_validate` and `PLSDA.cross_validate` with `n_bootstrap` no longer fail on a resample of one class.**
  A bootstrap resample drawn from the rows of a single class, or any resample on whose rows no column of Y varies,
  has nothing for X to explain, and its fit raised `NotEnoughVarianceError`: for a PLS-DA of 12 rows, 6 of each
  class, about one call in 500 without a seed. Such a resample is now drawn again. A seed whose resamples all varied
  gives the same result as before.
