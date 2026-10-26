- **Five arguments and result keys are named for what they are, not by a letter** (no aliases, except as noted):
  - `bibd_analysis` returns `n_treatments`, `n_blocks`, `block_size`, `replicates`, `concurrences` and
    `n_runs` in place of `a`, `b`, `k`, `r`, `lambda` and `n`;
  - `spe_plot` and `t2_plot` (and the PCA and PLS methods) take `n_components` in place of `with_a`;
  - `evop_worksheet` takes `range_factor_decimals` in place of `f_decimals`;
  - `variance_components` reports each fixed effect's `t_value` and `p_value`, and each fixed term's test
    `f_value` and `p_value`, in place of `t`, `p` and `f`;
  - `ControlChart.calculate_limits` takes a known standard deviation as `sigma` in place of `s`; the fitted
    attribute is still `s`. This one keeps `s=` as a deprecated alias.
