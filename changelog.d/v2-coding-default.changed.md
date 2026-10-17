- **`analyze_experiment`'s `coding` defaults to `"auto"`, so coefficients on factors not already
  coded come back on the -1/+1 scale.** The change was announced in 1.98.0. A call that passes no
  `coding` now gets its `"coefficients"` and `"confidence_intervals"` refitted with each numeric
  factor mapped from its minimum and maximum to -1 and +1 and each two-level categorical factor
  from its first level to its second; in a named model a categorical factor with more levels gets
  sum-to-zero contrasts. A main effect in a model with interactions is then tested at the centre of
  the design, and the result is reported with `coding: "coded"`, its `factor_coding`, and the
  equation on the factors' own scale as `coefficients_actual`. A numeric factor already coded to
  -1/+1 is left as it is, and a Scheffe mixture model keeps `"actual"`. The other analyses do not
  use `coding` and are unchanged.
  - Pass `coding="actual"` to keep the old scale. `coding=None`, and a `null` in the agent tool, mean
    `"auto"`.
  - The `FutureWarning` and the `coding_note` key that announced the change are removed.
