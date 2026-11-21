- **Ridge analysis about a focus** (#970): `optimize_responses(..., method="ridge_analysis", focus={...})` centres the
  spheres on a coded point instead of the design centre (Box and Draper 2007, Section 12.4), for radii, multipliers
  or `n_steps`. Each radius, and `stationary_point_radius`, is the distance from the focus, and the result carries
  `focus`. Hoerl's ridge from (0.5, 0.5) on p. 400 is now checked in the reference catalogue: 35 of its 36 numbers
  agree, and its R = 0.597 at lambda = 1.6 (0.5965) is recorded as unexplained.
