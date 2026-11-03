- **A region whose limits name a categorical level now weights each allowed combination of levels by the volume
  of its settings (#783).** `DesignRegion.sample_frame` drew each allowed combination equally often, and so did the
  I-criterion of `generate_design` (`design_type="i_optimal"`, and the region sample of `"g_optimal"`) and
  `evaluate_design`'s average prediction variance, FDS curve and G-efficiency over such a region. Each combination
  is now drawn in proportion to the share of the box its limits leave (`region.given(...).volume_fraction()`), so
  every setting the region allows is equally likely: with `not (catalyst == 'B' and T > 180)` on 150 to 200 degC,
  catalyst B keeps 60% of the range and counts 0.6 times as much as A or C. A combination whose limits leave no
  volume is not drawn. I-optimal designs, their `trace_criterion`, and the averages over such a region change: the
  12-run I-optimal design over T, P, catalyst (A, B) and solvent (X, Y) with that limit now runs catalyst A 4 times
  and B twice in each solvent, not 3 times each. Rules that only remove combinations of levels leave every
  combination the same volume, and their numbers do not change. `evaluate_design` now finds the columns the rules
  leave estimable on the region's support points as well as on its sample, so a combination with little or no
  volume still counts there, as it does in `generate_design`.
