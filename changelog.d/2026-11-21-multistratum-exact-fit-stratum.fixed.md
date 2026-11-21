- **`multistratum_anova` gives no F test in a stratum that the terms fit exactly.** Such a stratum's residual
  is rounding error only. It was divided into an F of 10^15 or more, with a p-value near 0, and on Windows with the
  lowest supported numpy it was an exact 0.0 that raised `ZeroDivisionError`. The residual sum of squares is now 0,
  and that stratum's F and p-value are NaN, as where the residual has no degrees of freedom.
