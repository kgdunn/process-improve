- **A correlation map gives each correlation its sign.** `compare_designs(...).correlations` holds the Pearson
  correlation between a design's terms from -1 to +1, where it held the absolute value; `max_r` is still the
  largest size among the squares and interactions. The library's `correlation_map` plot, Factorial's map of a
  study's runs, the maps of the designs it compares and the design catalog's map are drawn from -1 to +1 in a
  Scientific colour map of two colors (Crameri, 2018): vik on a light page, blue through near white to red, and
  berlin on a dark page, whose zero is near black. Each map has a key. The words and the table of pairs still go by
  the size of a correlation. `process_improve.visualization.colors` gains `DIVERGING_COLOR_MAPS` and
  `diverging_colorscale(dark_page=False)`.
