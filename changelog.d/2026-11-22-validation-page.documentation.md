- **A Validation page.** `docs/validation.rst` records how the package's numbers are checked
  against references that do not depend on it: the printed numbers of the reference catalog's
  sources, the designs eight other packages build (FrF2, DoE.base, rsm and daewr in R; pydoe,
  dexpy, OApackage and definitive-screening-design in Python), the values other software
  computes for the analyses, and the companion book. Each comparison names its versions, date
  and commit, its result, what it found, and what it does not show.
  `scripts/design_crosscheck/` repeats the comparison of designs.
