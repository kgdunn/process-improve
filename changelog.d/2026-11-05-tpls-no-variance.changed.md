- **`TPLS` refuses data with no variance with `NotEnoughVarianceError`, as `MBPLS` and `PLS` do.** A
  first Y block with no variation, or F, D and Z with nothing that varies, is refused before the
  iteration, with the words `MBPLS` uses; a component whose scores are not finite, because a block
  has no variation for the iteration to follow, keeps its message. `NotEnoughVarianceError` is now a
  `ValueError` as well as a `RuntimeError`, so code that caught the `ValueError` TPLS raised before
  still catches it, and so does code that catches `RuntimeError`. (#1144)
