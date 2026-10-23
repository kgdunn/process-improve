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
