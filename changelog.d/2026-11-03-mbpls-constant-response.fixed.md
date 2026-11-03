- **`MBPLS.fit` refuses a response with no variation, as `PLS.fit` does.** A `Y` whose every column is
  constant, or has no observed value, raises `NotEnoughVarianceError`: `There is no variance in Y: every
  response is constant or has no observed value, so there is nothing for X to explain.` `MBPLS` used to fit
  it with every weight 0, and `predict` then gave NaN for every new row, while `predictions_` gave Y's mean.
  A constant response beside one that varies is still fitted, and predicted at its mean. (#1075)
