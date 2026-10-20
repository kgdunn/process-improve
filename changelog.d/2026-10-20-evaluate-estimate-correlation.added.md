- **`evaluate_design(..., metric="estimate_correlation")` reports the correlations between the estimated
  second-order effects**, `c_ij / sqrt(c_ii c_jj)` with `c = (X'X)^-1` (JMP's "correlation of estimates"). Where
  `correlation` compares the design's columns, this says how well the fit can tell a square from an interaction.
  The sizes are summarised by kind of pair, as Jones and Nachtsheim (2011) tabulate them for definitive screening
  designs: `max_abs_r_<kind>` and `mean_abs_r_<kind>` for `two_squares`, `square_and_own_interaction`,
  `square_and_other_interaction`, `two_interactions` and `second_order` (every pair). With `evaluate_projections`
  and `model="quadratic"` it gives their Table 4 for each three-factor projection.
