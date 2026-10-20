- **`augment_design("upgrade_to_rsm")` reads `n_additional_runs`.** The documented argument had no
  effect: the centre runs were always topped up to five, so Davies's composite design (a 2^3
  completed with one centre run and six axial runs, 15 runs) came out with 19. It is now the number
  of centre runs in the second stage, and without it the centre runs are still brought to five. A
  `Block` column now marks the two stages, which are run at different times (an existing `Block`
  column keeps its labels), and the result gives `orthogonal_blocking_alpha`, the axial distance at
  which the stages block orthogonally with these centre runs, with a note when the chosen one is not
  it.
