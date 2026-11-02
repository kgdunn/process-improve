- **`TPLS.fit` refuses a second Y block.** Its inner PLS model regresses the first Y block
  alone on the super scores. A second Y block was deflated with the first one's loadings
  without a word (or failed, when its width differed). It is refused now, with a message
  naming the blocks given. (#1028)
