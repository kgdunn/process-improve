- **Categorical process factors in mixture-process designs.** A mixture-process design can now hold a
  categorical process factor, such as a catalyst type or a supplier, beside continuous ones. It enters the
  model in `L - 1` sum-to-zero columns; the candidates take every level, and the I- and G-criteria average
  over the levels. Without an intercept, patsy would code such a factor with a column for every level
  wherever the term below it is missing from the formula (`T:catalyst` in the `"interactions"` model), which
  made the model inestimable. `evaluate_design` and `analyze_experiment` now code every categorical factor of
  a mixture-process model in `L - 1` columns (`ReducedRank`, `ReducedRankFormula`), so the recorded formula
  fits as it is. With the coolant settings as the two-level factors they are, the library computes Goos and
  Jones's (2011) p. 120 statements on Table 6.2: an average relative prediction variance of 0.483 ("a little
  less than a half") and a largest of two-thirds.
