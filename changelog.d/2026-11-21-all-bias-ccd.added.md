- **All-bias central composite designs** (#972): `all_bias_ccd(n_factors, n_cube_runs, region="spherical",
  expand=1.0)` in `experiments.designs_response_surface` returns the center runs (real, and rounded up) and the cube
  and axial distances of the composite design whose moments up to order 5 equal those of a spherical or cuboidal
  region, so the bias from third-order terms a second-order model leaves out is smallest (Box and Draper 2007,
  Appendix 7H). `expand` spreads the design, as in `all_bias_distance`. The 84 numbers of Box and Draper's Tables
  7H.1 and 7H.2 are now checked in the reference catalog, and agree.
