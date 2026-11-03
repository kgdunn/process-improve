- **`randomization_test_mbpls(seed=...)` is deprecated since 2.4.0 and will be removed in 3.0; use
  `random_state`.** `random_state` takes an int, a `numpy.random.Generator` or `None`, as every function of
  the package that draws random numbers does, and an int draws the same permutations as the same `seed` did.
  Passing `seed` warns with a `DeprecationWarning`; passing both raises a `TypeError`. (#1072)
