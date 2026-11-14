- **An optimal design over many factors with limits on group totals is found.** With 15 ingredients
  in 5 buckets, each bucket's total between 5% and 20% and the total of them all at most 60%,
  `generate_design` raised "No candidate point satisfies all the constraints", though the region is
  not empty: no point of the grid has every total inside its limits, and the region has too many
  vertices to list. When the grid and the boundary points leave too few candidates for the model,
  vertices of the region are now found by linear programming, each the furthest point of the region
  in one direction, and added with the midpoints between them; `metadata["vertices_sampled"]` says
  so. A region the grid covers keeps its candidates, so its designs do not change.
  `DesignRegion.sample` and `evaluate_design` find such a region too.
