- **`experiments.fit_across_blocks`: models fitted across blocks.** For experiments run in blocks
  (days, batches, panel sessions), it fits each candidate model form to every response by least
  squares with a block effect that sums to zero, chooses the form with the lowest leave-one-run-out
  error (a run left out of every block it was tested in), and returns a `BlockedFit` with the
  coefficients, each block's shift and `predict`: prediction intervals in a known block, the average
  block, or a coming block whose shift is read from the runs tested again in it.
