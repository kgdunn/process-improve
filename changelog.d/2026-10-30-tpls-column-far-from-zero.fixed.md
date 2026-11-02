- **`TPLS` fits a column far from zero.** After centring and scaling, `fit` checked that every
  column had a mean within 1e-6 of 0 and an sd within 1e-6 of 1, and raised `RuntimeError` ("This
  is a bug") otherwise. A column whose values sit far from zero compared with their spread, such as
  1e11 plus noise of sd 1, has its mean known only to about 1e-5 of its sd, so the check refused
  data that fits. The check is gone: the centring and scaling are tested against a calculation
  written out by hand instead. (#1028)
