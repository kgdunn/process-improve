- **The contour plot covers the whole design, can be drawn in the factors' own units, and can show the
  canonical analysis of the model (#799).** The grid of the contour, 3D surface and prediction-variance
  plots reaches every run in `design_data`, so a composite design's axial runs lie inside the plot, and
  `extent=` sets it by hand. `units="actual"` draws the contour plot in the factors' own units, from
  `analyze_experiment`'s `factor_coding` or the ranges of the region's factors.
  `overlays=["stationary_point", "canonical_axes"]` marks the stationary point with its classification
  and draws each canonical axis through it with its eigenvalue, which shows where a ridge runs.
