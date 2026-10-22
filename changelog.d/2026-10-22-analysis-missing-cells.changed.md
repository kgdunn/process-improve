- **`analyze_experiment` fits the interactions a design can estimate when a combination of levels was never
  run.** The default interactions model used to fall back to main effects whenever the design could not
  estimate it in full, so one missing cell (a pair of levels the constraints disallow, or a continuous factor
  held at one value on a level of another) cost every interaction. Each pair of factors is now judged on the
  combinations of its levels that were run: an interaction those cells cannot estimate is left out of the
  formula, with a warning that names the missing combinations, and the rest are fitted. A design whose pairs
  each show every combination but which has too few runs, such as a Plackett-Burman design, still falls back
  to main effects.
