- **`diagnose(...).hotellings_t2` is deprecated since 2.4.0 and will be removed in 3.0; use
  `cumulative_hotellings_t2`.** The old field was a cumulative DataFrame from `PCA`, `TPLS` and `BatchPCA`,
  and the full model's T² as a Series from `PLS` (and `PLSDA`, `PRM`), `MBPLS` and `MBPCA`, so code that
  handled both had to know which it held. `cumulative_hotellings_t2` has one shape from every `diagnose`:
  a row per diagnosed row and a column per component, column `a` holding the T² over the first `a`
  components, with the labels and the values of the fitted T² (`hotellings_t2_`, `TPLS.hotellings_t2`,
  `super_hotellings_t2_`). Its last column, `.iloc[:, -1]`, is the old Series. The old field still holds
  what it did; reading it by name gives a `DeprecationWarning` at the reader's line, and reading the whole
  result, as `dict(result)` does, does not. (#983)
