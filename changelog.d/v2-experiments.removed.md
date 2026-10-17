- **`random_seed` is removed; use `random_state`.** It was deprecated since 1.97.0. Passing
  `random_seed=` to `generate_design`, `evaluate_design`, `evaluate_all` or `generate_omars` now
  raises `TypeError`, and the `generate_design` agent tool refuses it as an unknown key, as it
  refuses any other. `random_state` takes an int, a `numpy.random.Generator` or `None`, and its
  default (42) gives the same designs as before. The parameters that followed `random_seed` in
  those signatures each move up one position.
- **The `i_efficiency` metric of `evaluate_design` is removed; use
  `"average_prediction_variance"`.** It was deprecated since 1.97.0. Asking for it now raises the
  `ValueError` any unknown metric raises, which lists the metrics there are. The replacement
  reports the I-criterion itself: the prediction variance averaged over the region, in units of
  the error variance, where lower is better. `"i_optimality"` and `"i_criterion"` remain aliases
  for it.
