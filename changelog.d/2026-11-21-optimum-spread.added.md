- **The optimum spread against variance and bias** (#972): `optimum_spread(weight="uniform", variance_share=0.5)` in
  `experiments.designs_response_surface` returns the spread (m2/mu2)^(1/2) of a symmetric design for a straight line
  that minimizes variance plus squared bias when the truth is quadratic, for a uniform or a normal weight function
  and a given share of variance in the total (Box and Draper 2007, Section 13.5). The 12 spreads of their Table 13.1
  are now checked in the reference catalog; 11 agree, and the normal weight's 1.04 at a share of 0.2 (1.0525 by
  their Eq. 13.5.8) is recorded as unexplained.
- **Minimum-bias designs of Box and Draper's Section 13.6 checked** (#972): Example 13.1's half-side 0.447 and
  Example 13.2's a = 0.4938, n0 = 2.653 and axial distance 0.83 now check through `all_bias_distance` and
  `all_bias_ccd`.
