- **An infinite response is named as Y.** `PLS`, `PRM`, `MBPLS` and `AdaptivePLS` said "Input X contains
  infinity or a value too large for dtype('float64')." for an infinite cell of Y; they now say "Input Y
  contains ...", as `OPLS` does. (#1114)
- **`PRM` refuses a Y of another length with the family's error**: "The X and Y arrays must have the same
  number of rows: X has 30 and Y has 20." It failed with an `IndexError` from its robust centring.
  `MBPLS` says the same, where it said "y has 20 rows; expected 30 to match X-blocks." (#1114)
- **`OPLS` refuses an X with no variance before it divides by zero**: an X of zeros raised
  `NotEnoughVarianceError` after a `RuntimeWarning` ("invalid value encountered in divide"); now the
  error comes alone. An X that varies but has no covariance with y, which gave a model of NaN after the
  same warning, is refused as the rest of the PLS family refuses it (see the entry on a Y uncorrelated
  with every column of X). (#1114)
- **`PLS.select_n_components`, `compare_cv_criteria` and `pseudo_validation_set` take a row and a column
  with nothing observed, as `fit` does.** `select_n_components` raised `RuntimeError` for such a row, and
  warned of a "Mean of empty slice" for such a column; the other two refused both. A held-out row with
  nothing observed in X has no prediction, as `predict` gives it none, and is left out of PRESS, of Q2's
  reference and of the pseudo-validation alarm rates; a column with nothing observed is left out of every
  model, with the warning `fit` gives. `PLS.cross_validate` and `MBPLS.select_n_components` already took
  such a row, but kept it in Q2's reference while leaving it out of PRESS; they now leave it out of both.
  The default component count of `PLS.select_n_components`, `MBPLS.select_n_components`,
  `compare_cv_criteria` and `pseudo_validation_set` no longer counts a column with nothing observed,
  which adds no component: with one, the folds were asked for one component too many, and refused. (#1114)
- **`PRM.vip()` is finite on contaminated data.** Its R² was measured on every row, outliers too, so a
  component could lower it, and VIP took the square root of a negative number: a `RuntimeWarning` and
  NaN for some variables. (#1114)
- **`PCA`, `PLS`, `PLSDA`, `PRM`, `OPLS`, `AdaptivePCA`, `AdaptivePLS`, `ASCA` and `FAMD` refuse an X with
  no columns by name**, as `MBPCA`, `MBPLS` and `TPLS` do: "Block X has no columns; a block needs at
  least one." So do `PLS`, `PRM` and `AdaptivePLS` for a Y with no columns. numpy's "at least one array
  or dtype is required" named neither. (#1114)
