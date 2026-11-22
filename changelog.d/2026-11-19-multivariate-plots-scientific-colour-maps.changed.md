- **Two multivariate plots are drawn in the package's Scientific colour maps** (Crameri, 2018), which are even to
  the eye and read in gray and without telling red from green. `confusion_matrix_plot` (and
  `PLSDA.confusion_matrix_plot`) defaults to devon, where it was Plotly's "Blues"; its `colorscale` setting takes
  the name of one of the package's maps ("batlow", "devon", "lajolla", "grayC") or, as before, any Plotly name, so
  `settings={"colorscale": "Blues"}` gives the earlier figure. `Resampler.plot_results` draws its ridges in batlow,
  where it was viridis.
