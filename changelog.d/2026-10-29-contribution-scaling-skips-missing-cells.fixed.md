- **`score_contributions(scaling="maximum")` and `scaling="within"` now scale data with missing
  cells.** A missing cell's contribution is NaN, and both scalings took a maximum or a sum that
  the NaN turned into NaN: with any cell missing, `"maximum"` returned every row unscaled, and
  `"within"` returned each row that has a missing cell unscaled. Both now skip NaN, as the row
  sums do. Data with no missing cells give the same numbers as before. (#1024)
