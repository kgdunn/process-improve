- **Curve responses in the browser app.** A study can read a curve on each run, such as a colour over
  time or a spectrum. The reader pastes the runs' curves under the run sheet, and the workbook carries
  them on a "Curves" sheet. In Learn, the what-if panel draws the curve predicted at its settings, with
  95% bands at each point and over the whole curve, and redraws it as a setting moves without asking
  Python again. A target curve pasted there moves the panel to the settings whose curve comes closest.
  `FunctionalFit.curve_maps` gives the fitted curve and each bootstrap replicate as `mean + row @ map`,
  the numbers the page predicts from; an optional `grid` keeps a spectrum's maps small. Each bootstrap
  replicate now reuses the fit's basis, which makes the bands about ten times faster to compute.
