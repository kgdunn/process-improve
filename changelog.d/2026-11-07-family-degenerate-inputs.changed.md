- **A constant Y is refused in one set of words across the family.** `PLS`, `PRM`, `AdaptivePLS` and
  `BatchPLS` refused a Y in which every response is constant or has no observed value with "There is no
  variance left in the data array for Y: cannot compute any more components beyond component 0."; `MBPLS`
  and `TPLS` said "There is no variance in Y: every response is constant or has no observed value, so
  there is nothing for X to explain.", and `OPLS` "... the response is constant, ...". All now use the
  `MBPLS` words (`TPLS` names its block, `Y["Y1"]`, and ends "for F, D and Z to explain"). The error is
  still `NotEnoughVarianceError`. (#1149)
- **One row is refused as too few rows by `BatchPCA` and `TPLS`, as across the family.** One batch, or
  one blend, was refused as an X or a Y with no variance; it is now refused with the `ValueError` the
  others raise: "Found array with 1 sample(s) ... while a minimum of 2 is required". (#1149)
- **`TPLS` refuses a component after its Y is fitted exactly in the words of `PLS` and `MBPLS`:** "There
  is no variance left in the data array for Y: cannot compute any more components beyond component 1."
  It was refused because the component's scores were not finite, in words that pointed at the groups of
  F. (#1149)
