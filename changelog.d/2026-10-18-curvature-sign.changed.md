- **The curvature test's `difference` is now the factorial mean less the centre mean**, the sign
  Box, Hunter and Hunter (section 15.2) and Montgomery (section 6.8) print. It estimates the sum of
  the pure quadratic coefficients, b11 + ... + bkk, so a surface that bends down near a maximum
  gives a negative difference. It was the centre mean less the factorial mean. The test also
  reports the difference's `std_error`, s * sqrt(1/nF + 1/nC) with s from pure error, and its
  `t_statistic` is now the difference over that error. The F statistic and p-value do not change.
