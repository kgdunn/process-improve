- **The `robust_regression` tool no longer times out above about 800 points.** Its default drew 999
  bootstrap resamples whatever the number of points, and each costs time in proportion to the square
  of that number, so a hosted call passed the 10 s tool timeout. `n_boot` now defaults to 999 up to
  500 points and to fewer above, so the bootstrap does the work of 999 resamples of 500 points: 249
  resamples at 1,000 points, 9 at 5,000. Fewer resamples give rougher intervals; pass `n_boot` to set
  the number, as before. The output reports the number drawn as `n_boot`. Calls with 500 points or
  fewer, and calls that set `n_boot`, return the same numbers as before. The library function
  `robust_regression` keeps its default of 999. (#1095)
