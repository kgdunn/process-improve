- **NIPALS PCA, PLS, MBPCA and MBPLS are right when a block has as many rows as columns.**
  `quick_regress` chose whether to regress on the columns or the rows of a block from the block's shape,
  so a square block (N = K, or N = M for Y) was regressed on the wrong axis. NIPALS PLS then gave wrong
  weights, and NIPALS PCA wrong loadings, even on complete data; a cross-validation fold that happened
  to be square gave a wrong Q2. `quick_regress` takes a new keyword `per="column"` or `per="row"`, which
  every NIPALS loop now passes. Called without it on a square block, it raises `ValueError` instead of
  guessing; other shapes behave as before.
