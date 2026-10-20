- **`evaluate_design(..., metric="estimate_correlation")` reports the correlations between the estimated
  second-order effects**, `c_ij / sqrt(c_ii c_jj)` with `c = (X'X)^-1` (JMP's "correlation of estimates"). Where
  `correlation` compares the design's columns, this says how well the fit can tell a square from an interaction.
- **Both `correlation` and `estimate_correlation` now summarise their sizes by kind of pair** under `by_pair`: the
  largest and the mean |r|, and the number of pairs, for `two_squares`, `square_and_own_interaction`,
  `square_and_other_interaction` and `two_interactions`, as Jones and Nachtsheim (2011) tabulate them for definitive
  screening designs. Under the full quadratic model of a DSD, `correlation` gives their Table 3; in its three-factor
  projections, `estimate_correlation` gives their Table 4.
