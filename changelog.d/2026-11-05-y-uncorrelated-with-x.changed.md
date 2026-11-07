- **A Y uncorrelated with every column of X is refused across the PLS family.** With `X'Y` about
  zero there is no first component. `PLS` (and `PLSDA` and `PRM`) and `OPLS` raised a
  `numpy.linalg.LinAlgError` that blamed collinear columns, after numpy's division warnings, and
  `MBPLS` fitted every weight 0, with an R²Y of 0, and predicted NaN for every new row. All now
  raise `NotEnoughVarianceError`, as for a constant Y: "There is nothing in X to explain Y by: Y is
  uncorrelated with every column of X." `NotEnoughVarianceError` is a `ValueError` as well
  (#1166), so code that caught the old `LinAlgError`, a `ValueError`, still catches it. `X'Y` is
  judged against `||X|| ||Y||`, so the units of the data do not matter (#1144).
