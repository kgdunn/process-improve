- **Three-level fractional factorials from generators.** When every factor has three levels (a
  continuous factor with three `levels`, or a categorical factor with three),
  `generate_design(..., design_type="fractional_factorial", generators=[...])` builds the
  3^(k-p) fraction in arithmetic modulo 3: `"D=A2BC2"` sets x_D = 2 x_A + x_B + 2 x_C (mod 3),
  the digits 0, 1 and 2 being each factor's levels from the lowest. The result lists the defining
  relation in Montgomery's form (`"I=AB2CD"`), its resolution, and the alias chain of each main
  effect; `resolution=` checks that the generators reach it. Montgomery's 3^(3-1), 3^(4-1) and
  3^(4-2) fractions, the L9 of NIST and of Schmidt and Launsby, and Schmidt and Launsby's Latin
  square are rebuilt from their generators in the tests. Before, three-level continuous factors
  with generators gave a two-level fraction that left out the middle level, and three-level
  categorical factors were refused. Without generators, a fractional factorial is two-level, as
  before.
- **`evaluate_design` reads a three-level fraction's own relation.** Its `defining_relation`,
  `resolution`, `alias_structure` and `clear_effects` metrics no longer read the generators with
  two-level arithmetic, which gave `I=ABCD` and `A = BCD` for `D=A2BC2`.
