- **Split-plot designs in `evaluate_design`.** `whole_plot_variance_ratio` (the whole-plot variance
  over the run variance) evaluates a design with a `WholePlot` column through the generalized least
  squares information `X' V^-1 X`, `V = I + eta Z Z'`, in place of `X'X`, so every metric (the
  coefficient variances, the D-, A- and I-criteria, the FDS curve) is that of the split-plot model.
  It reproduces the variances Goos and Jones (2011) print for a 50-run wind tunnel design in
  10 whole plots. The `evaluate_design` tool takes the same field. Without it, the whole plots are
  ignored as before.
