- **Kenward and Roger's method in the split-plot analysis.** `analyze_experiment(...,
  analysis_type="split_plot", df_method="kenward_roger")` inflates the covariance of the estimates
  for the uncertainty in the variance components and scales each term's F test, as Kenward and
  Roger (1997) do. `information="observed"` computes the degrees of freedom from the observed REML
  information, as JMP and R's lmerTest do, in place of the expected (SAS, pbkrtest). The two
  together reproduce JMP's analysis of a design in random blocks, which is the same model with the
  blocks named as the whole plots. The defaults, and the results they give, are unchanged.
