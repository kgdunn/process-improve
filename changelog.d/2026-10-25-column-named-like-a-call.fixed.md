- **A factor named `I` no longer breaks a quadratic model with "'Series' object is not callable", nor a factor
  named `C` an ASCA.** Patsy finds a column before its own function of the same name, so a column `I` turned every
  square term `I(P ** 2)` into a call of the column, and a column `C` did the same to ASCA's `C(A, Sum)`. The formula
  check now refuses a column named like a function the formula calls, with a rename to paste:
  `.rename(columns={'I': 'I_'})`. A factor `I` in a model without square terms still works. On factori.al, the
  factor table refuses a factor named `I` as you type it, and offers `I_`.
