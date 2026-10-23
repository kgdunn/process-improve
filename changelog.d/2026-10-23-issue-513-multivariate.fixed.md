- **`PCA.select_n_components(return_consensus=True)` scales the block for parallel analysis
  whatever `scale_inside_folds` is (#513).** The flag was passed on as
  `parallel_analysis(scale=...)`, so with `scale_inside_folds=False` a raw block was compared in its
  own units against a unit-variance null, and parallel analysis retained every component of loud
  noise. `parallel_analysis_n_components` and `consensus` can change for calls that passed
  `scale_inside_folds=False` on a block that was not already autoscaled.
- **`MBPLS` and `MBPCA` scale each block's Hotelling's T² as they scale the super T² (#513).** Each
  component is divided by the block scores' second moment about zero over the rows that have a
  score, the convention `explained_variance_` uses for the super scores. The block T² used the
  variance about the scores' own mean, which differs once missing cells leave them off-centre, and
  on an `MBPLS` fit with a row that has nothing observed in a block, that variance was NaN, so the
  block's T² went unscaled for every row. `block_hotellings_t2_` changes for fits with missing data
  (by under 1% for scattered missing cells, and by a large factor in the second case); on complete
  data it is unchanged.
- **`TPLS` fits a blend with nothing observed in F or Z, and a material with nothing observed in D
  (#513).** The row's regression had an empty denominator, which gave NaN, and the NaN spread to
  every score of every later component, so since #709 the fit was refused. The row now gets a
  zero score or loading in that block and no weight in its regressions, as PCA and PLS fit an
  all-missing row, so through that block it has no effect on the other rows. A fit still refused,
  because a block has no variation for the iteration to follow, now says that rather than naming
  empty rows.
  `regress_a_space_on_b_row` returns 0 rather than NaN for a row with nothing present.
- **`TPLS` super weights and block weights point the way of the super score (#513).** The inner
  PLS step's sign convention negated `t`, `w` and `q` but not `u`, and TPLS regresses its blocks on
  `u`, so on every component the convention flipped, `w_loadings_super`, `r_loadings_f`,
  `s_loadings_d`, `v_loadings_d` and `w_loadings_z` came out with the opposite sign to the super
  score: with no Z block the F super weight was -1. Those components change sign now; scores,
  predictions, SPE, Hotelling's T² and VIP do not change.
- **`TPLS` leaves a column with no variance out of every score (#513).** Such a column is scaled to
  NaN and zeroed, as it was, but its cells were still counted as observed, so in F they entered
  every row's denominator and shrank every score; the warning added earlier said the column was
  excluded when it was only partly so. The presence maps (`not_na_d`, `not_na_f`, `not_na_z`,
  `not_na_y`) are now taken after preprocessing, in `fit` and in `diagnose`. A fit whose F block has
  a constant column now matches the same fit with that column unobserved; its scores, predictions
  and loadings change.
- **`TPLS.diagnose` corrects a Z score for the row's missing cells, as `fit` does (#513).** It
  computed the correction and never applied it, so a row with a missing Z cell scored lower in
  `diagnose` than in `fit`, and so did its T², SPE and predictions. Rows with every Z cell
  observed are unchanged.
