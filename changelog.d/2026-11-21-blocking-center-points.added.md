- **The center runs for orthogonal blocking at a chosen axial distance** (#788): `blocking_center_points(alpha,
  n_cube_runs, n_center_cube, n_factors, n_axial_replicates=1)` in `experiments.designs_response_surface` inverts
  `blocking_alpha`: the center runs the axial block (or blocks) needs for a central composite design to be blocked
  orthogonally at that alpha, a real number that is negative when no count does (Box and Draper 2007, Section 15.3).
  Exercise 15.8's B = 0 at the rotatable alpha with the star run twice is now checked in the reference catalog.
