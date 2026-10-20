- **`analyze_experiment(sigma=...)`: standard errors from an error standard deviation known before
  the runs (#789 B6).** Sources often take sigma from past records or earlier experiments, not
  from a small experiment's residual. With `sigma` given, the standard errors, tests and intervals
  of the coefficients, effects, significance, confidence intervals, predictions and confirmation
  test use it, and read the normal distribution, since sigma is not estimated from the runs. A
  saturated design gets standard errors this way: BHH's eight bicycle runs for seven factors give
  each effect 2.1 with a sigma of 3. The ANOVA, the lack-of-fit test, Lenth's method and a mixture
  model's significance and effects keep the residual. `model_summary` reports the value as
  `prior_sigma`.
