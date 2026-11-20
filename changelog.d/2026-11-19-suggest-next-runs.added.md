- **`suggest_next_runs` proposes the next runs from a model of every run so far.** It fits a Gaussian process (with a
  noise term by default) and proposes the setting where the expected improvement over the best result so far is
  largest (Jones, Schonlau and Welch, 1998), drawn from the study's `DesignRegion`, so every run keeps its limits. A
  batch of runs to make before any result is read (`n_runs`) follows the kriging believer: each run is taken to give
  its prediction, and the next is chosen with it. It returns the runs with their prediction, standard error and
  expected improvement, the best so far, and the improvement ratio that says when to stop. `GaussianProcessFit`
  gains `with_runs`, which adds runs to a fitted model at its estimated parameters. Numeric factors only (#1374).
