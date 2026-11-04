- **`TPLS` with `skip_f_matrix_preprocessing=True` no longer warns that an F column with no variance is
  excluded.** F is used as given, so such a column, for example a lot no training blend uses, stays in the fit
  and counts in every F score. The warning said it was left out. No number changes. A column of F with no
  observed value is still named, because it is left out as a missing cell is. The docstring now says that the
  caller owns F's preprocessing when it is skipped. (#1084)
