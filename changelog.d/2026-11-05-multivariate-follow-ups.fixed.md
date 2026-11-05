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
- **TPLS gives a block that no component explains a VIP of 0.** An `F` whose every blend has the
  same row, used as given (`skip_f_matrix_preprocessing=True`), has loadings and R² of 0; both of
  its VIPs divided 0 by 0, with numpy's `RuntimeWarning`, and were NaN (#1144).
- **Adaptive history labels no longer repeat.** A row with no label is numbered one past the
  largest integer label recorded so far, or by the number of rows recorded when that is larger; it
  went on from the last label, so the labels 2 and "x" gave the next row 2 again. A DataFrame whose
  rows pandas numbered 0, 1, ... is numbered on as an array is: two such frames passed to
  `partial_fit` gave 0 to n - 1 twice (#1144).
