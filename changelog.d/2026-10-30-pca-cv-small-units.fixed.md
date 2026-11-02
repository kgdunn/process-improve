- **`PCA.select_n_components` standardises a column in any units.** In each fold of the
  element-wise scheme, and once for the `ckf`, `ek`, `sacv` and `gcv` schemes, a column whose
  standard deviation was below 1.5e-8, as in small units, was taken for a constant one and left
  unscaled, so it dropped out of PRESS and Q2: a column 1e9 times smaller moved the element-wise
  PRESS at one component from 93.6 to 87.9. A column is now constant only when its values are all
  equal, as `MCUVScaler` decides, which `ASCA` now uses too. (#1029)
