- **`hotellings_t2_limit(conf_level, 0, n_rows)` returns 0.0, not NaN.** With no components every T2 value
  is 0, and so is its limit; the formula tends to 0 as the number of components falls to 0, but scipy
  cannot evaluate an F distribution with 0 numerator degrees of freedom. Found by mutation testing.
