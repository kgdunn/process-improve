- **`TPLS` keeps a column measured in small units.** A column whose standard deviation was below
  1.5e-8 was taken for one with no variance and left out of the fit with a warning, so a column in
  units 1e9 times smaller moved every score; a column now has no variance when its observed values
  are all equal (or there is only one of them), as `MCUVScaler` reads one, in any units. A column
  with a single observed value is now named in the warning too, rather than left out without
  one. (#1029)
