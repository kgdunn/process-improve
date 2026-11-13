- **An explicit formula that opens with `-1` keeps no intercept when its blocks are added.**
  `analyze_experiment` puts the block contrasts in front of a formula that names `Block`, and
  patsy reads `Block1 + -1 + x1` as a model with an intercept, so a mixture formula written
  `y ~ -1 + x1 + x2 + x3 + Block` was fitted with one, and was rank-deficient for it. (#1229)
