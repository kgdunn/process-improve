- **Box and Draper's A canonical form in the canonical analysis** (#971): `optimize_responses(...,
  method="canonical_analysis")` returns `rotated_form`, the axes rotated about the design center: the slope
  theta_i = m_i'b along every axis (`linear`), and where the response stops changing on each axis
  (`stationary_point`, -theta_i / (2 lambda_i), `None` along a flat axis). Before, only the slope along the
  ridge's axis was returned (`ridge_form["linear"]`). 27 slopes and coordinates of Box and Draper's Chapters 10 and
  11 are now checked in the reference catalog; two disagree by the book's rounding.
