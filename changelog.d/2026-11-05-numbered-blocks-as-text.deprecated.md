- **Naming a numbered block as text in an `evaluate_design` contrast is deprecated.**
  `C(Block, Treatment('1'))` is deprecated since 2.5.0 and will be removed in 3.0; write
  `C(Block, Treatment(1))`, which now names block 1. Until 3.0 the text spelling still reads the
  block, with a `DeprecationWarning` that points at the caller's line, and gives the numbers it
  gave in 2.4.0 for designs of up to 9 blocks. (#1129)
