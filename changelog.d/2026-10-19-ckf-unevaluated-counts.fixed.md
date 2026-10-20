- **`PCA.select_n_components(cv_scheme="ckf")` no longer recommends a component count it could not
  evaluate (#675).** A count that leaves a column group with fewer retained columns than components
  cannot be scored on that group. Its PRESS was the sum over the groups that could be scored: 0 when
  none could, so pure noise in five columns came out with Q2 = 1 at five components and that count
  was recommended, and too small when some could. Such a count now has a NaN PRESS and Q2, and is
  not recommended; the columns of the groups it skips have a NaN per-variable Q2, not 1.
