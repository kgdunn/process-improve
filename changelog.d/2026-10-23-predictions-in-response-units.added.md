- **A prediction from a transformed response is also given in the response's own units.** With
  `analyze_experiment(transform=...)`, each record of `predictions` holds `response_units`: the prediction put
  back (the response's median, which a monotone transform keeps, not its mean) and the ends of both intervals
  put back, low before high. An end the transform leaves unbounded, such as a 1/y interval through 0, is `None`.
