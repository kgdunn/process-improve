- **The desirability contour and the correlation map are drawn in the default colour map.** The desirability
  contour (`desirability_contour`) was red, amber and green, which a reader who does not tell red from green cannot
  read; it is now batlow, light for a desirability of 0 and dark for 1 (`DESIRABILITY_COLORSCALE`). The library's
  correlation map (`correlation_map`) follows `SURFACE_COLORSCALE`, where it named Plotly's own blues, and so does
  the Python that Factorial shows for a mixture's response over its blends.
