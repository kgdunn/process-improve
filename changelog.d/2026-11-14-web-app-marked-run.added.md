- **factori.al marks a run wherever it is shown.** Rest the pointer on a run, on a chart of the
  residuals, on a response surface, or on its row in the run sheet or in the table of each run, and
  the same run is marked on every other chart and in both tables: ringed in the warning colour on
  the charts, shaded in the tables. A press, or a tap, keeps it marked; a press again, or Escape,
  lets it go. A point out of line on one chart is then found at once on the others and in the run
  sheet. Each run on a response surface now says which run it is, and in the chart of the residuals
  in run order the line through the points no longer takes the pointer from the run under it.
- **factori.al draws how well the model knows each factor's effect.** Each line of "Each factor's
  effect on the prediction" has a band around it, the 95% confidence interval of the prediction,
  from the fit's own covariance and put back in the response's units with the line. Resting on the
  chart reads each line's value with its interval. A fit that uses every run has no estimate of the
  error, and its lines have no band.
