- **The selectivity ratio's F critical value counts each feature's observed rows.** A feature's
  ratio is formed over its observed cells, but `attrs["f_critical"]` counted every row of `X`, so a
  feature with gaps was held to a lower bar than its rows support. It is now a Series indexed by
  feature, also when every feature has the same count, where it was one float: read one feature's
  value as `ratio.attrs["f_critical"]["name"]`, and `ratio > ratio.attrs["f_critical"]` compares
  each feature with its own (#1144).
- **`OPLS` contributions work.** `t2_contributions`, `spe_contributions`, `score_contributions`,
  `group_contributions`, `meda` and `omeda` raised `AttributeError` (no `loadings_`). They take the
  `X` that `transform` takes, and on the training data add up to the stored T², SPE and scores, as
  the rest of the family's do (#1144).
- **`PCA.select_n_components`, `PLS.select_n_components` and `PLS.nested_cv` read a pandas
  missing value (`pd.NA`) as a missing cell**, as `fit` does. A nullable column holding one raised
  `TypeError`; `PLSDA` followed `PLS` (#1144).
- **TPLS refuses an `F` group with no variation used as given on every numpy build.** A group
  whose every blend has the same row, with `skip_f_matrix_preprocessing=True`, gives the iteration
  only rounding error to follow. Depending on the numpy build, TPLS refused the fit after an
  overflow warning, or fitted it with F weights near 1e15 and no weight for F: one such group then
  took a varying group out of the model without a word, and an all-constant `F` had VIPs of NaN. It
  is now refused before the iteration follows the rounding error, with the message that asks for
  each D and F group to vary (#1144).
- **Adaptive history labels no longer repeat.** A row with no label is numbered one past the
  largest integer label recorded so far, or by the number of rows recorded when that is larger; it
  went on from the last label, so the labels 2 and "x" gave the next row 2 again. A DataFrame whose
  rows pandas numbered 0, 1, ... is numbered on as an array is: two such frames passed to
  `partial_fit` gave 0 to n - 1 twice (#1144).
