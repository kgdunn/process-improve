- **U-pseudocomponents and the U-simplex check** (#973): `to_pseudocomponents(blends, upper=...)` and
  `from_pseudocomponents(..., upper=...)` convert blends with upper bounds on their components to and from
  U-pseudocomponents, u_i = (U_i - x_i)/(U - T) with U the sum of the upper bounds (Crosier 1984; Box and Draper
  2007, Section 17.3); `lower` becomes optional, and one of the two is given. The new `u_simplex_check(upper)`
  returns U, U - U_min and whether the inverted simplex the bounds cut out lies wholly inside the mixture simplex
  (U - U_min <= T). Box and Draper's Figures 17.4a and 17.4b are now checked in the reference catalogue (6 numbers).
