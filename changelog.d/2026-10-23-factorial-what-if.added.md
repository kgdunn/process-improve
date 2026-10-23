- **Factorial: Learn's "What if" panel predicts every response at the settings the reader chooses (#899).** A
  slider and a number box set each numeric factor inside its range, and a list sets each categorical factor's
  level. Each response's prediction, its 95% confidence interval for the mean, its 95% prediction interval for one
  new run and its leverage follow each move at once: Python sends each response's fitted model as numbers once per
  analysis, and the page predicts from them in its own JavaScript. A flag says when the settings lie beyond the runs
  a model was fitted to, and a sentence names each limit on the settings they break. The panel opens at the balanced
  settings, else the best settings, and comes back to them with one button; its Python panel prints the same
  predictions with `analyze_experiment`.
