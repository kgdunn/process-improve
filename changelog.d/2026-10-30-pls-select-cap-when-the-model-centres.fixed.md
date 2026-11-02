- **`PLS.select_n_components(scale_inside_folds=False)` caps the components at one fewer than
  the smallest training fold's rows when the fold model centres its rows itself.** Without in-fold
  scaling the cap was the rows of the smallest fold, but the default `PLS(scale=True)`, and `PRM`,
  centre each fold and so leave one component fewer: with as many columns as rows in a fold, the
  default `max_components` raised `NotEnoughVarianceError` ("no variance left in the data array
  for X") instead of selecting. A model built with `scale=False` keeps the cap of the fold's rows.
  (#1027)
