- **The `robust_regression` tool no longer times out above about 800 points.** Its default drew 999
  bootstrap resamples whatever the number of points, and each costs time in proportion to the square
  of that number, so a hosted call passed the 10 s tool timeout. `n_boot` now defaults to 999 up to
  197 points and to fewer above, so up to 1,000 points the bootstrap does at most the work of 39
  resamples of 1,000 points (156 resamples at 500 points). From 1,000 points up it draws 39, the
  fewest on which a 95% percentile interval rests, and the time again grows as the square of the
  number of points: a hosted call with more than about 1,500 points can still time out on a slow
  machine. Fewer resamples give rougher intervals; pass `n_boot` to set the number, as before. The
  output reports the number drawn as `n_boot`. Calls with 197 points or fewer, and calls that set
  `n_boot`, return the same numbers as before. The library function `robust_regression` keeps its
  default of 999. (#1095)
