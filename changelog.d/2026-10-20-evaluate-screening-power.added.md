- **`evaluate_design(..., metric="screening_power")` runs Jones and Nachtsheim's (2011) power study.** It tests one
  term of each kind (a linear effect, a quadratic effect, an interaction) in the model an analyst would fit to find
  it: alone among the main effects, or in the full quadratic model of two factors (their Table 6). Each test's power
  is averaged over every factor or pair of factors, at an `effect_size` or at Table 7's 1, 2 and 3 times sigma. A test
  the design cannot make is marked `"aliased"` (an effect that a second-order term outside the model copies) or
  `"not estimable"`, as Table 7 marks the two-level alternatives. Random effects are taken into account through
  the generalized least squares information, as for the other metrics.
