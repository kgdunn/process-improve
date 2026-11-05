- **`TPLS` refuses data with no variance with `NotEnoughVarianceError`, as `MBPLS` and `PLS` do, not
  `ValueError`.** A first Y block with no variation, or F, D and Z with nothing that varies, is
  refused before the iteration, with the words `MBPLS` uses; a component whose scores are not
  finite, because a block has no variation for the iteration to follow, keeps its message. Code
  that caught `ValueError` for these should catch `NotEnoughVarianceError`, or the `RuntimeError`
  it derives from. (#1144)
