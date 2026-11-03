- **PCA and PLS NIPALS fit the same components in any units.** The test for "no variance left"
  compared what was left of X (or Y) with an absolute 1.5e-8, so the same data in small units, such
  as `X * 1e-5` under `PLS(scale=False)` or `PCA(algorithm="nipals")`, stopped at the first
  component with `NotEnoughVarianceError`. What is left is now judged against the block's size
  before the first component, as `OPLS` judges it: a component is refused when the norm left is at
  most `sqrt(eps)` (about 1.5e-8) times the block's norm. A block of zeros, or one whose rank is used
  up, is still refused. (#1070)
