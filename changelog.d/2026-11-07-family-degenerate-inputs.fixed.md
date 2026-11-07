- **`PLSDA` keeps every class in every training fold of an integer `cv`.** Its `select_n_components`,
  `compare_cv_criteria`, `pseudo_validation_set`, `nested_cv` (outer and inner folds) and `cross_validate`
  shuffled the rows into PLS's K-fold whatever their class, so a training fold could hold no row of a
  class. Its indicator matrix was then constant, and the fold's fit raised `NotEnoughVarianceError`:
  `nested_cv` on twelve rows of two classes failed for 7 seeds in 40. An integer `cv` now shuffles each
  class into the folds on its own, as `StratifiedKFold` does (`RepeatedStratifiedKFold` with
  `n_repeats`), with the seed the folds took before. The folds, and so the numbers, differ from those of
  earlier releases for the same seed. (#1149)
- **More components than the data support are cut to what they support, with a warning, across the
  family.** `PCA`, `PLS` and `OPLS` cut a count above the smaller of the number of rows and of columns,
  with a `SpecificationWarning`. `AdaptivePCA` and `AdaptivePLS` raised a raw `ValueError` or
  `IndexError`, `MBPCA` and `MBPLS` emitted a numpy `RuntimeWarning` or fitted components to rounding
  noise, and `TPLS` fitted them too. They now cut the count with the same warning, and `n_components_`
  is the count fitted. `TPLS` cuts to the rank its D can have, the fewer of a group's materials and
  properties in the group with most: every component's weights on the materials come through D, and a
  component past its rank had super scores near 1e15. (#1149)
- **`AdaptivePCA` names a column with no observed value, and both adaptive models leave it out.**
  `AdaptivePCA` fitted such a column without the warning the rest of the family gives (#1098). Both
  `AdaptivePCA` and `AdaptivePLS` added a new row's value in that column to the row's SPE, and an update
  moved the column's centre toward it. The value now adds nothing to the row's scores or SPE, as in
  `PCA` and `PLS`, and an update leaves the column out. (#1149)
