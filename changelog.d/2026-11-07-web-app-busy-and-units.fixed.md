- **factori.al's status line is no longer left busy.** In a study whose only results are its curves,
  "Installing plotly for the charts..." stayed on the status line, busy, after the curve was drawn.
  The note now ends when plotly is installed ("Installed plotly for the charts."), or says why it
  could not be. "Analysing..." also stayed when a new study was started while an analysis was on its
  way; it now goes. (#1169)
- **A factor's new units reach every part of factori.al that shows them.** A change of a factor's
  units alone left the old units on the what-if panel's controls, on the factors held while looking
  inside the limits' picture, and on the questions that adjust a proposed limit, and in their labels
  for screen readers. A design in one factor kept its old name and units in its note on how it
  compares the levels ("The runs compare T at 150 and at 200 degC"), after a change of either. Each
  now takes the new label. (#1169)
