- **`dunnett_value` takes a correlation for each treatment**, `correlation=[c_1, ..., c_k]` with
  `c_i = n_i / (n_i + n_0)`, so Dunnett's critical value is exact for treatments of unequal runs: comparisons
  i and j correlate by the root of `c_i c_j`. Its docstring gave the correlation of two comparisons as
  `n_0 / (n_0 + n)`; it is `n / (n + n_0)`, which is one half only when the control has as many runs as a
  treatment.
