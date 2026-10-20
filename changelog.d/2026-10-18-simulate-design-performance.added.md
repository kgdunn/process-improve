- **`simulate_design_performance` judges designs by the studies they would give.** It makes
  each design many times on a `TrueProcess`, the process you expect, and analyses every
  simulated study as `recommend_next_step` does. Per design it reports the share of studies
  that find each true term, all of them, and only them, the false alarms, how often nothing
  stood out, and, toward a goal, how far the predicted best settings fall short of the true
  best. A term found through an exact alias counts as found. Unlike the analytic measures of
  `compare_designs`, it sees the bias that terms outside a model put on the terms inside it.
  It is an MCP tool, and so now is `compare_designs`.
