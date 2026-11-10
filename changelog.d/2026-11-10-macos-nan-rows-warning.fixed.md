- On macOS, projecting rows too sparse to estimate (`transform`, `diagnose` and the contribution functions on rows
  with missing cells) no longer adds numpy's "invalid value encountered in matmul" warning beside the
  `SpecificationWarning` that counts those rows: their NaN scores and SPE are now set, not computed.
