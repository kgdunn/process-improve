- **Split-plot optimal designs without pyoptex.** `split_plot_design(factors, hard_to_change,
  n_runs, SplitPlotOptions(...))` finds a D-, I- or A-optimal split-plot design by coordinate
  exchange on the generalised least squares information X'V^-1 X: the hard-to-change factors
  are set once for each whole plot, the others for each run, and categorical factors take their
  own levels. `SplitPlotOptions` sets the criterion, the model, the number of whole plots, the
  ratio of the whole-plot to the run variance and the number of random starts. For Goos and
  Jones's wind tunnel (50 runs in ten whole plots, the full quadratic model, equal variances) it
  finds an I-optimal design with an average prediction variance of 0.51219, against the printed
  design's 0.51227, in about a second.
