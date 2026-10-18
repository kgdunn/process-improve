- **Tool calls return the warnings they raise.** A warning raised inside a tool used to reach only the
  MCP server's stderr, which the model calling the tool never reads. `execute_tool_call` now returns
  each one with a dict result, once, under `library_warnings` as `{"category", "message"}` records.
- **`analyze_omars` is an MCP tool.** The staged analysis of a definitive screening or OMARS design
  returns its active main effects, interactions and squares, and `model_formula`, the selected model
  to pass on to `analyze_experiment` and `optimize_responses`.
- **`recommend_strategy` marks the stages the screening decides on.** An optimisation stage after a
  screening design that can test for curvature has `conditional: true` and a `run_if` naming the
  evidence that calls for it, and `minimum_total_runs` counts the plan without it.
- **`generate_design` says when a design leaves the factor ranges.** A CCD with `alpha` above 1 warns,
  naming each factor's lowest and highest setting against its range, and records them in
  `metadata["runs_beyond_ranges"]`.
- **`evaluate_design`'s power says what it assumes.** When interactions outside the model bias the
  tested coefficients, as in a Plackett-Burman design, the result adds `alias_exposure` per term and
  `power_assumes`, since the real power can then be far lower than stated.
- **`confirmation_test` gives a verdict on the mean.** `confirmed` compares the mean of the runs at
  each setting with the prediction interval for a mean of that many runs (`mean_tests`).
