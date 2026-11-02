- **A `TPLS` fit with more than one Z group no longer fails with a shape error.** The fit gives each
  Z group its own weights and its own block score, but labelled the super weights for one Z group
  only, so a second group raised `ValueError: Shape of passed values is (3, 1), indices imply (2, 1)`.
  `w_loadings_super` now has one row per Z group, labelled `Z["<group>"]`, before the `"F"` row,
  and `diagnose` scores each Z group on its own, as the fit does, where it used to pool them into
  one score. A model with one Z group is unchanged, and keeps its `"Z"` row label. (#981)
