- **`experiments.analyze_functional_experiment`: designed experiments whose response is a curve.**
  The curves are reduced by `multivariate.FPCA`, and the components are kept up to 99% of their
  variance, plus any later one up to 99.9% whose score the design factors explain by an F-test: a
  factor can drive a shape that carries little of the variance. Each kept score is analysed with
  `analyze_experiment` under the same model, so coding, blocks, mixture models and a `WholePlot`
  column work as they do for one number. The `FunctionalFit` it returns holds the analyses and the
  residual covariance of the scores, and `FunctionalFit.predict` gives the predicted curve at new
  settings with pointwise and simultaneous bands from a wild bootstrap. The bootstrap refits the
  FPCA and the choice of components in each replicate, and draws one sign per whole plot. In a
  simulation of an 18-run design, 95% bands covered the true curve 87% of the time pointwise and
  88% over the whole curve.
  `FunctionalFit.match_target` finds the factor settings whose predicted curve comes closest to a
  target curve, by a bounded multistart search in the space of the kept scores, where the L2 distance
  between curves is the distance between their scores; a factor can be searched within a range or
  held at a value.
