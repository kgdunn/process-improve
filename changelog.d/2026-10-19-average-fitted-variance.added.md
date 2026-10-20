- **`model_summary` reports `average_fitted_variance` (#797 R1).** It is the average variance of
  the fitted values over the runs, p s²/n, because the runs' leverages sum to p whatever the design
  (Box and Wetz). With `sigma` it is p sigma²/n, which reproduces BHH's 0.965 (Exercise 15.8) and
  1.125 (the composite design of section 15.3). `adequate_precision` is the range of the fitted
  values over its square root, and is read against `sigma` when one is given. A model with no
  constant term now counts its own terms in both; it counted one too many.
