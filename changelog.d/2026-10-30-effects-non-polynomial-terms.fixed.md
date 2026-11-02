- **`analyze_experiment` refuses effects for an `I()` term that has no coded scale.** The
  `"effects"` and `"lenth_method"` analyses refit the model on the factors coded to -1/+1. A term
  such as `I(x ** 0.5)` was NaN there, and the refit dropped those runs without a warning;
  `I(1 / x)` failed with a `MissingDataError`; and `I(z2 - 0.2 * z1)` gave the effect of another
  term. These analyses now raise a `ValueError` that says the model has no coded scale, as
  `coding="coded"` does for it since #1040. A polynomial model keeps its effects, and so does any
  model whose numeric factors are already coded.
