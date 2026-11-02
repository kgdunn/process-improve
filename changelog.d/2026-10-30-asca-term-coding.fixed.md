- **`ASCA` reads only the categories a categorical factor takes.** A category no row takes, as a
  subset of a categorical frame keeps, was coded as a level of its own, so the factor's effect was
  not centred and its sum of squares was wrong. (#1029)
- **`ASCA` with fewer response columns than a term has degrees of freedom** asked that term's PCA
  for more components than the effect has, and the PCA warned. Each term's PCA is now capped at the
  columns of `X` too. (#1029)
