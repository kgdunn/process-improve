- **Contour and surface plots are drawn in batlow.** `SURFACE_COLORSCALE`, the default colour scale of the
  library's contour and surface plots and of Factorial's charts, is now the Scientific colour map batlow (Crameri,
  2018), light for a low value and dark for a high one, where it was a ramp of blues. Factorial's correlation map
  follows it. `process_improve.visualization.colors` gains `COLOUR_MAPS` (batlow, devon, lajolla, grayC) and
  `colorscale(name, dark_page=False)`, which returns one as a Plotly colour scale; a plot's own `colorscale`
  style still takes any other.
