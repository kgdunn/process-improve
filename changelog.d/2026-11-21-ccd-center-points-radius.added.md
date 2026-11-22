- **Center runs for an equal variance at a chosen radius, and unrounded** (#788):
  `ccd_center_points("uniform_precision", ..., radius=r)` gives the center runs of a rotatable central composite
  design whose prediction variance at the center equals that at distance `r` from it, in the cube's coded units
  (Box and Draper 2007, Exercise 15.3); without `radius` the rule keeps Box and Hunter's unit. `rounded=False`
  returns the real number a source prints instead of whole runs. Exercise 15.3's 4.58 (rounded to 5), 5.55 and
  1.48 are now checked in the reference catalog.
