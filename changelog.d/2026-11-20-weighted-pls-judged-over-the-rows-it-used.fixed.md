- **A weighted PLS fit, and PRM, judge their rows over the rows the fit used** (#1149). `PLS(sample_weight=...)`
  left a row of weight 0 out of the fit, as documented (`[1, 1, 0, 0, 1]` is the fit to rows 0, 1 and 4), but
  took `explained_variance_`, the T² limit and its ellipse, the SPE limit and the score limits over every row.
  For PRM, whose robust weights set an outlier aside, one row at 1e10 made the score variances about 1e18 and every
  other row's T² 0, and a row at 1e160 raised numpy overflow warnings. Each is now taken over the rows of positive
  weight, in their weights, with Kish's effective number of rows: 0/1 weights give the fit to the rows of weight
  1, and weights scaled together give the same limits. A far row's SPE is exact, and its T² past the largest float
  is `inf`, without a warning. An unweighted fit is unchanged.
