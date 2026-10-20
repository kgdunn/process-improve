- **`variance_dispersion`: the variance dispersion of a design, and whether it is rotatable**
  (`process_improve.experiments`, #790 C4). For spheres about a design's centre, in coded units,
  `variance_dispersion(design, model="quadratic", radii=21)` gives the minimum, average and maximum
  scaled prediction variance, N Var(y-hat)/sigma^2, which a variance dispersion graph plots against
  the radius, and their spread. A design is `rotatable` when the minimum and the maximum meet on
  every sphere. The average is exact: the prediction variance is written in monomials, whose
  averages over a sphere have a closed form; the minimum and the maximum are searched for from a
  seeded sample of directions. It confirms the book's statement that a Box-Behnken design is
  rotatable in four and seven factors, and not in three, five or six, and that a central composite
  design is rotatable at an axial distance of the fourth root of its factorial runs.
