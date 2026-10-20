- **A split-plot main effect's test no longer depends on how its factor is coded.**
  `analyze_experiment(..., analysis_type="split_plot")` tested a main effect beside its own
  interaction on its treatment-coded coefficients when the formula named the factor plainly
  (`method`) or as `C(method)`. That tested the effect at one reference level of the other
  factor, on degrees of freedom that mixed the strata, with no warning: Montgomery's paper
  experiment gave F = 2.05 on 2 and 15.5 df (p = 0.16) for the method, where Table 14.19 prints
  7.08 on 2 and 4 df (p = 0.049). The tests now code every categorical factor to sum-to-zero
  contrasts, whatever the formula writes, and the coefficients stay in the formula's coding. The
  two-way split-plot analysis is fixed in the same way.
