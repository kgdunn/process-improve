- **`OPLS` no longer fits more components than the data can carry.** One predictive plus
  `n_orthogonal_components` orthogonal components above the rank of the centred `X` took the last
  of its variance from the predictive component, and the fit returned predictions off by whole
  units (27 on the cheddar-cheese data with four components from three columns), with no warning.
  As for `PCA` and `PLS`, the total is now capped at the number of rows or of columns, whichever
  is smaller, with the same `SpecificationWarning`, and a component with no variance left to take
  raises `NotEnoughVarianceError`, as a `PLS` component does. `n_components_` and the
  `orthogonal_space_dimension` of `invert` report the components fitted. (#1025)
