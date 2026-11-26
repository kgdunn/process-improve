- **The plot themes draw slim color bars, contour numbers with a halo, and no box around the journal theme's
  legend** (#1242). Every theme's contours, heatmaps and surfaces take a color bar 12 pixels thick with its title
  along its side, where a wide one with its title on top took the plot's room; a contour's numbers are in the
  theme's ink with a halo in its paper's color, so they read on a ramp's darkest bands. The default `pi_journal`
  theme no longer draws a black box around the legend. The runs on a response surface (`visualize_doe(...,
  plot_type="contour")`) are drawn with a white face and a dark edge, which read on every color of the surface,
  where the dark dots were lost on its dark bands.
