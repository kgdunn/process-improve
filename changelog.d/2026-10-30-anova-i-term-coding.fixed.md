- **`analyze_experiment` tests an `I()` term that is not a product of whole powers of the factors
  as given.** The ANOVA and the significance were worked on a refit with every numeric factor
  coded to -1/+1, also with `coding="actual"`. That re-expresses a polynomial model, but changes
  what a term such as `I(z2 - 0.2 * z1)` means: in Box and Draper's Table 3.6 the sum of squares
  of `z1` came out 14.97, not 12.0. Such a model is now tested on the factors as given.
- **`analyze_experiment` fits `I(x ** 0.5)`, `I(1 / x)` and `I(x ** -1)`.** The design's rank was
  worked on the coded factors, where these terms have no value, and the call failed with a
  `ValueError` or a `LinAlgError`. The rank is now worked on the factors as given. Such a model
  has no coded scale, so `coding="auto"` keeps its coefficients in actual units, and
  `coding="coded"` or coding ranges raise a `ValueError` that says so. (#963)
