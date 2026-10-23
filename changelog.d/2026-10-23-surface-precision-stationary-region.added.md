- **`analyze_experiment` notes a fitted surface too imprecise to read (#797 R1).** When
  `model_summary["adequate_precision"]`, the range of the fitted values over their average standard
  error, is below 4, `model_summary["model_note"]` says the surface is not estimated precisely enough
  to read its contours or its optimum: Box and Wetz's check, at Design-Expert's threshold for the same
  ratio. It is read against `sigma` when one is given; a fit with no residual and no `sigma` gets no
  note. The reference catalogue now checks Box, Hunter and Hunter's average variance of the fitted
  values against their prior sigma, which a catalogue check may now carry as `"sigma"`.
- **`stationary_point_region`: the confidence region of a fitted surface's stationary point**
  (`process_improve.experiments`, #797 R5). Box and Hunter's (1954) region is the set of settings at
  which the gradient of the fitted second-order surface does not differ significantly from zero, an F
  test on k and n - p degrees of freedom. It is returned as functions of settings (the F statistic, and
  whether each setting is inside) that take a `numpy.meshgrid` as readily as one point, beside the
  stationary point, its classification and the delta-method standard error of each coordinate. Along a
  ridge the region runs out of the experiment, which the standard errors alone do not show.
