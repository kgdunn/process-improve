- **`TPLS.diagnose` gives a new row the SPE the fit gives a training row.** It deflated every
  cell of a new row, so a missing F or Z cell, which has no residual in the fit, added the
  product of the row's score and the cell's loading to the row's SPE: the training rows with a
  missing cell, diagnosed again, had a larger SPE than the fitted `spe`. A row with nothing
  observed in F, whose scores are NaN, had an SPE of 0, the SPE of a row the model fits exactly;
  it is NaN now. The SPE frames hold floats, not objects. (#1028)
