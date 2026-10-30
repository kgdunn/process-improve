- **`TPLS` fits when the first column of `Y` is constant.** Its iteration, and the one-component
  PLS inside it, started from the first `Y` column. A constant column is all zero after
  preprocessing, so the scores were not finite and the fit raised `ValueError`, although the other
  `Y` columns varied. Both now start from the `Y` column with the largest sum of squares, as `PLS`
  and `MBPLS` do, and the model no longer depends on where the constant column sits. A fit with one
  `Y` column, or whose first `Y` column already had the largest sum of squares, is unchanged; any
  other fit can differ from before only within the convergence tolerance. (#954)
