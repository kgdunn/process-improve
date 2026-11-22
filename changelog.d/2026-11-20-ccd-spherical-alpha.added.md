- **A spherical central composite design** (#788): `generate_design(..., design_type="ccd", alpha="spherical")` puts
  the axial runs at the square root of the number of factors, the distance of the cube's corners, so every run but
  the center runs lies on one sphere. It works on a full or a fractional cube, and `metadata["alpha_rule"]` reads
  `"spherical"`. Before, the name raised `ValueError` and the distance had to be typed as a number. The 8 spherical
  distances of Box and Draper's Table 15.2 are now checked in the reference catalog.
