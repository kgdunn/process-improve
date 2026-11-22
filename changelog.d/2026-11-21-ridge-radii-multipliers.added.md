- **Ridge analysis at chosen radii or multipliers** (#970): `optimize_responses(..., method="ridge_analysis",
  radii=[...])` reports the ridge at those radii, and `multipliers=[...]` at those Lagrange multipliers, as a ridge
  table sets it out (Box and Draper 2007, Chapter 12): each multiplier gives the point x = -(B - mu I)^-1 b / 2 and
  its radius directly, on the ridge of maxima above the largest eigenvalue of B, of minima below the smallest, and on
  one of the other stationary paths between them; an infinite multiplier gives the center. `n_steps` stays the
  default. In the reference catalog 140 rows of Box and Draper's ridge tables are now read at their printed
  multiplier, so their radii become checks and 80 disagreements that came only from the rounding of the printed radius
  are gone; the points of Hoerl's paths 2 and 3 between the eigenvalues (68 numbers) are now checked as well.
