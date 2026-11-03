- **`OPLS.fit` checks `Y` before fitting, as `PLS.fit` does.** A `Y` with a different number of
  rows from `X` raises `PLS`'s `ValueError`, which names both counts, instead of an error from a
  matrix product. A missing or infinite `Y` cell raises a `ValueError` that says so, as the same
  cell in `X` does, instead of a singular-matrix `LinAlgError` or a message that blamed `X`. (#1078)
