- **The browser app reads the spread of the results repeated at each setting.** When two settings or more are
  run twice or more, as a replicated Taguchi array is, Learn sums up each setting's readings: their mean,
  standard deviation, ln s and the signal-to-noise ratio of the response's goal (larger is better to maximise,
  smaller is better to minimise, nominal is best for a target). The response table averages each at every level
  of each factor, with its delta and rank, and says at which level the spread is smallest and the ratio highest.
  For a target it names the adjustment factor: the one that moves the mean most for its effect on the spread.
  "Show the Python" gives the same table with pandas.
