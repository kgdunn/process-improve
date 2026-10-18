- **Two-way split-plot (strip-plot) analysis.** `analyze_experiment(..., analysis_type="split_plot",
  whole_plot=["Row", "Column"])` fits crossed random effects, one per column named, by REML, with
  Satterthwaite's or Kenward and Roger's degrees of freedom and standard errors as for one whole-plot
  column. The result has each random effect's variance, error degrees of freedom and best linear
  unbiased predictions (BLUPs), named by its column, and each term's stratum. It reproduces JMP's
  analysis in Goos and Jones (2011), Tables 11.6 and 11.7, to every printed digit. The split-plot
  analysis with one whole-plot column now reports its whole plots' BLUPs as well.
