- **`PLSDA`'s cross-validation takes its class labels.** `select_n_components`,
  `compare_cv_criteria`, `pseudo_validation_set`, `nested_cv` and `cross_validate`, inherited from
  `PLS`, read the labels as a numeric response: text labels could not be read as floats, and labels
  0 and 1 failed on the shape of the two indicator columns the model is fitted to. Each now runs
  `PLS`'s method on that one-hot class-indicator matrix, and a splitter such as `StratifiedKFold` is
  given the labels, so it keeps each class's share in every fold. (#1029)
