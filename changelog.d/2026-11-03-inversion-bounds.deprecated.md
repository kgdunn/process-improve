- **`optimal_inversion` and `inversion_sensitivity` with an input that has no lower or no upper bound are
  deprecated.** Only the T² and SPE limits then hold that input, and the answer can set it to a value it cannot
  take, such as a negative amount. Give every searched input its extreme limits in `InversionRules(bounds=...)`.
  A TPLS formula's lot fractions keep their default of 0 to 1, and a fixed input needs none. Until 3.0 such a call
  warns with a `FutureWarning` that names the inputs, at the caller's own line; from 3.0 it is an error. (#1126)
