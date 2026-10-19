- **`augment_design` no longer adds runs that are not blends to a mixture design.** On a
  mixture design, `"add_axial_points"` added star points with negative proportions,
  `"add_center_points"` a run of zeros, and `"foldover"`, `"semifold"`, `"upgrade_to_rsm"` and
  `"add_runs_optimal"` runs whose proportions do not sum to 1. The first two now add blends (see
  the axial check blends above), and the others, with `"add_blocks"`, refuse with the reason. A
  mixture design whose region is not a simplex (a component with an upper bound) is refused
  for axial and centroid runs, which are not known from its runs.
