- **A column whose sum passes the largest float is centered at its mean** (#1149). `MCUVScaler`, the
  autoscaling of `PLS`, `MBPLS` and the cross-validation criteria, and `TPLS` took a column's mean with numpy's or
  pandas' mean, which is `inf` once the column's sum overflows, near 1e308. `MCUVScaler` then centered the column
  at 0 without a word: on the LDPE data with Y in units of 1e303, PLS and MBPLS predictions were 12% and 16% wrong,
  and TPLS refused the data as having no variation. The mean is now taken as `column_sd` takes the standard
  deviation, on the column divided by a power of two, and is the same as before, bit for bit, wherever no sum
  overflowed.
