- **A region with `min_when_used` factors can be sampled and searched.** Such a region is made of
  pieces of different dimensions, one for each set of the factors in use. `DesignRegion.sample` draws
  from the measure under which each such factor is left out as often as it is used, before the
  constraints act, and is uniform between its minimum and `high` when it is used; I- and G-optimal
  designs and the prediction variance of `evaluate_design` average over those points.
  `optimize_responses` keeps, in each search, the set of factors its start uses, and from the best
  result tries the neighbouring sets (one factor left out or brought in, or one exchanged for
  another) until none is better, so the optimum has every such factor at 0 or at no less than its
  minimum. On ingredients in buckets with limits on the bucket totals, the optimum of a linear model
  matches a mixed-integer programme. A Pareto front keeps the sets its starts use.
