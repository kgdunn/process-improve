- **Box and Hunter's rules for a central composite design's centre runs** (#788 A11).
  `generate_design(..., design_type="ccd", n_center_points="uniform_precision")` or `"orthogonal"`
  picks the number for a rotatable design in one block, and makes the design rotatable;
  `designs_response_surface.ccd_center_points(rule, n_cube_runs, n_factors, n_replicates=1)` gives
  the number alone. They reproduce Box and Hunter's (1957) table: uniform precision 5, 6 and 7
  centre runs for two, three and four factors and 6 for five factors on a half fraction;
  orthogonality 8, 9, 12 and 10.
