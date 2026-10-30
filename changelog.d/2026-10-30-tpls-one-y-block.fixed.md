- **`TPLS.fit` refuses a second Y or Z block.** Its inner PLS model regresses one Y block on
  one score per block of Z and F, and labels the super weights `["Z", "F"]`. A second Y block
  was deflated with the first one's loadings without a word (or failed, when its width
  differed), and a second Z block failed with a pandas shape error. Both are refused now, with
  a message naming the blocks given. (#1028)
