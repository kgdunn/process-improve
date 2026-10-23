- **`PCA.select_n_components(return_consensus=True)` scales the block for parallel analysis
  whatever `scale_inside_folds` is (#513).** The flag was passed on as
  `parallel_analysis(scale=...)`, so with `scale_inside_folds=False` a raw block was compared in its
  own units against a unit-variance null, and parallel analysis retained every component of loud
  noise. `parallel_analysis_n_components` and `consensus` can change for calls that passed
  `scale_inside_folds=False` on a block that was not already autoscaled.
