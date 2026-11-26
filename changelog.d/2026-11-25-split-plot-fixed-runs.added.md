- **A split-plot design keeps fixed runs** (`hard_to_change` with `fixed_runs`), with the built-in search
  (`designs_split_plot.split_plot_design`, and `generate_design` by default). The fixed runs are the design's
  first runs, kept as given, a setting off the grid of levels included; a whole plot that holds one keeps its
  hard-to-change settings for all its runs, and the search chooses the rest. Until now only
  `backend="pyoptex"` did this, and the default build ignored `hard_to_change` with a warning. It was the one
  thing only pyoptex could do, so its removal in 3.0 (#1494) loses nothing.
