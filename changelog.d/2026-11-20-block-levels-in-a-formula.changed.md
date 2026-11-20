- **`analyze_experiment` takes a formula's `Block` as levels wherever it stands** (#1236). `Block` was levels
  only as a term of its own: in `Block*A` it was the block's number, a straight line across the blocks, and
  `Block:A` was that number times A, so `Block*A` and `Block + A + Block:A` gave two different fits. Now both are
  the model of `C(Block)*A`: `Block:A` is a difference in A's effect for each block but one. The analysis of
  variance tests each block term as one row (`Block`, `Block:A`) on the blocks' degrees of freedom, the lists of
  significant terms leave block terms out, and model selection scores a formula's block terms as its refit does.
  A formula with `Block` in a product fits differently from before; one with `Block` alone, or `C(Block)`, does
  not change.
