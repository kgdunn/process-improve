- **`selectivity_ratio` gives `inf`, not 0, for a feature with no residual.** A feature the
  target-projected component reproduces exactly is the most selective one possible, but it got 0,
  as a feature with no signal does, and so sorted last. It now gets `inf` and sorts first. The
  residual counts as none when its sum of squares is at most machine epsilon times the explained
  one, since rounding leaves about 1e-32 of it rather than 0; every finite ratio is therefore below
  about 4.5e15. A feature with no sum of squares at all (0 / 0) still gets 0. This changes a
  published value, also in `find_predictive_descriptors`, where a lone descriptor now gets `inf`;
  its p-values do not change. The `sensory_analyze_descriptive` tool reports an infinite ratio as
  `null`, as it reports every non-finite number, because JSON cannot carry infinity. Cap the ratio
  with `Series.clip(upper=...)` where a finite number is needed. (#1085)
