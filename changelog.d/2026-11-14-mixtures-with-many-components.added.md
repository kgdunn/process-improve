- **Mixture designs for many components in groups.** A mixture region with more vertices than
  `extreme_vertices` lists (15 components in 5 groups, each group between 5% and 30% of the blend)
  raised "Enumerating the vertices needs ... linear solves"; some of its vertices are now found by
  linear programming, with the points between them standing in for its edges and faces, and
  `metadata["vertices_sampled"]` says so. `extreme_vertices` raises `TooManyVertices`, a
  `ValueError`, for such a region, and `sampled_vertices` returns the sample.
- **A group of components held to a fixed total no longer refuses the design.** Five groups that
  may each supply at most 20% of a blend supply exactly 20% each, which leaves one Scheffé column of
  each group but the first inestimable, and the design was refused for "too few distinct blends".
  It is now built for the columns the region can estimate, with a warning that names the others and
  `metadata["inestimable_columns"]`. Candidates the caller supplies must still support the whole
  model.
- **`min_when_used` on a mixture component.** The component is left out of a blend or makes up no
  less than that share of it: candidate blends with a trace are dropped, the design is D-optimal
  with or without a budget, `DesignRegion.feasible` is False for a trace, and `optimize_responses`
  returns a blend that leaves the component out or uses at least its minimum. Such a mixture region
  is not sampled: `DesignRegion.sample` raises, and an I- or G-optimal design averages over the
  candidate blends.
