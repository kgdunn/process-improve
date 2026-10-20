- **`compare_designs` scores a split-plot design as one.** Its runs share whole plots, so they are
  not independent, yet every measure was computed as if each run were set on its own. When a design
  records the whole-plot variance ratio it was made for (a split-plot `DesignResult` from
  `generate_design(hard_to_change=...)` keeps it in `metadata["whole_plot_variance_ratio"]`; a
  DataFrame can carry one in `attrs["whole_plot_variance_ratio"]`), the measures now use X'V⁻¹X with
  V = I + ηZZ' over its `WholePlot` column, as `evaluate_design(random_effects=...)` does. Whole plots
  with no recorded ratio are noted (#790 C2).
