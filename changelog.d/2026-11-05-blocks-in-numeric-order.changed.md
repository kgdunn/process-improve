- **Numbered blocks are coded in numeric order, so block 10 follows block 9.** `evaluate_design`
  read a `Block` column named in its formula as text, and `analyze_experiment` sorted the blocks as
  text, so with 10 or more blocks block 10 came after block 1: `analyze_experiment`'s `Block2` was
  block 10, and the block coded -1 in every contrast column was block 9, not the last one. Both now
  sort the blocks by number. With 10 or more blocks the block contrasts' coefficients, names and
  VIFs are reordered, and the A-criterion and the condition numbers, which depend on how the blocks
  are coded, can change; the fit, its predictions and the D- and I-criteria do not. In
  `evaluate_design`, `C(Block, Treatment(1))` now names block 1, where it was refused. (#1129)
