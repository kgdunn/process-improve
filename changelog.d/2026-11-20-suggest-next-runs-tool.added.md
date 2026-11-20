- **`suggest_next_runs` is a tool** (#1374), so an agent on the MCP server and the `doe-designer` skill can ask for
  the next runs from a Gaussian process of every run so far: 1 to 50 runs, inside the factors' ranges and any
  limits on the settings, each with its prediction, standard error and expected improvement, and `little_to_gain`
  true when the improvement ratio is below 1%. It takes at most 200 runs, as the model's cost grows with the cube
  of the runs.
