- **Two-way split-plot (strip-plot) designs.** `designs_split_plot.strip_plot_design(factors,
  StripPlotLayout(row_factors, column_factors, cells), StripPlotOptions(...))` searches for a D-,
  A- or I-optimal design on a grid of runs: a row's factors are set once for its runs and a
  column's once for its runs, with the rows and the columns as crossed random effects of the
  ratios given. On Goos and Jones's 48-run battery cell grid it reaches the printed design's
  D-efficiency. On the cells of their twelve-run design it finds a design a little better than
  the printed one (which is 99.8% D-efficient), confirmed by enumerating every setting.
