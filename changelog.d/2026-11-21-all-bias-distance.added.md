- **The all-bias distance of a first-order design** (#972): `all_bias_distance(n_factors, n_cube_runs,
  n_center_points, region="spherical", expand=1.0)` in `experiments.designs_response_surface` returns the distance
  of a two-level design's corners at which its second moment equals that of a spherical or cuboidal region, so the
  bias from curvature the first-order model leaves out is smallest (Box and Draper 2007, Appendix 6B); `expand`
  spreads it, 1.15 as Box and Draper suggest. The 122 distances of their Appendix 6B tables are now checked in
  the reference catalog; two disagree by the book's working from its rounded distances.
