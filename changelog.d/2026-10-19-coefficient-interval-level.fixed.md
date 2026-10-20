- **The coefficients' intervals follow `significance_level`.** With
  `analyze_experiment(..., analysis_type="coefficients", significance_level=0.10)`, each coefficient's
  `ci_low` and `ci_high` stayed at 95%, while `confidence_intervals` gave the 90% interval the level asks
  for. The `coefficients` and `coefficients_actual` intervals now use 1 - `significance_level`, and the
  result reports it as `confidence_level`. The `analyze_experiment` tool, which takes
  `significance_level`, gave the same 95% intervals.
