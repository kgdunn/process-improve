- **Split-plot and two-way split-plot designs in `evaluate_design`.** `random_effects` maps each
  column that labels a random effect's groups (`{"WholePlot": 1.0}` for a split-plot design,
  `{"Row": 1.0, "Column": 1.0}` for a two-way split-plot, or strip-plot, design) to its variance
  over the run variance. Every metric (the coefficient variances, the D-, A- and I-criteria, the FDS
  curve) then uses the generalized least squares information `X' V^-1 X`,
  `V = I + sum_k eta_k Z_k Z_k'`, in place of `X'X`. It reproduces the variances Goos and Jones
  (2011) print for a 50-run split-plot design in ten whole plots (Table 10.3) and for a 48-run
  two-way split-plot design in 16 rows and six columns (Tables 11.3 and 11.4). The `evaluate_design`
  tool takes the same field. Without it, the runs are treated as independent, as before.
