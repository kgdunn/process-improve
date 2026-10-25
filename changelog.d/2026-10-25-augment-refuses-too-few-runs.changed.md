- **`augment_design(..., "add_runs_optimal")` raises a `ValueError` when `n_additional_runs` is too few
  (#957).** When the existing runs left more model coefficients unestimated than the runs asked for, the
  augmentation added more runs than requested and said so in its explanation. It now refuses, naming
  how many new runs the target model needs, as `generate_design` refuses a budget that is too small.
