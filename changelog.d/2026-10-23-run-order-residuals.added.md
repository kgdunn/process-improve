- **`analyze_experiment` checks the residuals for serial dependence in run order** (#802 G3). With a
  `RunOrder` column, `residual_diagnostics["run_order"]` holds the residuals' lag-1 autocorrelation in run
  order, its standard error 1/sqrt(N), and the Durbin-Watson statistic in run order, and a `note` says when
  the autocorrelation is more than twice its standard error: successive runs then look serially dependent,
  which is why Box, Hunter and Hunter randomize the run order. The existing `durbin_watson` keeps its
  meaning, the rows' order; without a usable `RunOrder` column the `note` says that this is the only order
  known.
