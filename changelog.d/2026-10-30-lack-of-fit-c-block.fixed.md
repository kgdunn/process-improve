- **`analyze_experiment` takes pure error within blocks when the formula writes the block as
  `C(Block)`.** Only a bare `Block` term grouped the replicates within blocks; with `C(Block)`,
  `C(Block, Sum)` or `C(Block, Treatment(1))` the model fitted the blocks but pure error pooled
  the replicates across them, so their block difference counted twice and the lack-of-fit sum of
  squares went negative. The lack of fit and the curvature test now group the replicates within
  blocks whenever the formula refers to the block column. A pure error that still exceeds the
  residual is reported as an error, not as a negative lack of fit. (#968)
