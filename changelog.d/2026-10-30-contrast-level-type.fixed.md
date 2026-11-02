- **A contrast that names a level its column does not have is refused with the levels there are.**
  `C(Block, Treatment('1'))` on a `Block` column of integers names the level as a string, and patsy
  failed with "specified level '1' not found". `analyze_experiment`, `lm`, `evaluate_design`,
  `augment_design`, `tukey_nonadditivity`, `estimate_missing` and `ASCA` now raise a `ValueError`
  that names the column, the level asked for and the column's levels, and shows a spelling that
  works, `C(Block, Treatment(1))`. The reference level of `Treatment` and the omitted level of `Sum`
  are checked. An integer that patsy reads as the position of a level still works. (#1043)
