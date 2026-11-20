- **`check_predictive_signal` and `count_discoveries_under_null` take `random_state`** (#1149), as
  `docs/development/reproducibility.rst` asks of every public function that draws: a seed or a
  `numpy.random.Generator`, in the place `seed` had, with the same default of 0, so a call gives the same
  numbers as before. Their `seed=` keyword is deprecated since 2.7.0 and will be removed in 3.0; it still works,
  with a `DeprecationWarning`. `PCA.parallel_analysis` and `PCA.select_n_components` now say in their annotations
  that they take a `Generator`, as they did.
