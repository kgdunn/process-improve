- **`optimize_responses(method="canonical_analysis")` gives a finite `ridge_form` when two or more
  eigenvalues are zero.** The origin divided the slope along every axis but the ridge's by that
  axis's eigenvalue, so a surface with two flat axes, such as Box and Draper's Table 11.1 (F) and
  (G), returned a `nan` origin, origin response and offsets, with a divide-by-zero
  `RuntimeWarning`. The origin now leaves out every axis whose eigenvalue `flat_tolerance` reads as
  zero, as the stationary point does, so the plane of centres of (F) gives the origin (0, 0, 0).
  (#965)
