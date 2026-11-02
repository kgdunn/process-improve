- **`MCA(correction="benzecri")` reports zero shares for variables with no association.** When the
  variables are unassociated (every combination of levels equally often), every eigenvalue is 1/Q,
  but rounding leaves some a few 1e-16 above it, and the corrected shares were that noise divided by
  itself: 50% or 100% on the first axis instead of 0. A corrected total no larger than the 1e-12
  below which `CA` calls an axis noise now gives zero shares, as `correction="greenacre"` also
  does. (#1029)
