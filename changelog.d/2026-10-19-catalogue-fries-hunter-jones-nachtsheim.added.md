- **The reference catalogue holds Fries and Hunter (1980) and Jones and Nachtsheim (2011), the papers that
  define minimum aberration and definitive screening designs.** `fries-hunter-1980` holds Table 1's minimum
  aberration designs (generators, defining relations, alias strings and word lengths) and 48 cells of Table 2's
  largest attainable resolution. Six of those cells are printed wrong, which an exhaustive search over every
  set of generators confirms: one 2^(12-5) cannot reach resolution V, and five designs reach one resolution more
  than printed. `jones-nachtsheim-2011` holds Figure 1's nine designs for 4 to 12 factors, the six-factor example
  with its AICc values, and the numbers of Tables 3, 4, 5, 7 and 8. Table 4's caption describes two models: its
  correlations of two squares come from the model of every main effect and every square, and the rest come from
  one three-factor projection, not an average over projections. The orthogonal 12-factor design of Table 7 is the
  12-run Plackett-Burman design folded over. The library's own definitive screening designs differ from the
  paper's for odd m (2m + 3 runs, from a conference matrix) and for m = 12 (orthogonal for main effects).
- **New check kinds in the reference catalogue:** `term_correlations` (correlations between second-order columns),
  `estimate_correlations` (correlations between estimated second-order effects, in the whole design or its
  projections), and `average_power` (the power of one coefficient's t test, averaged over factors or pairs of
  factors). `relative_efficiency` can now compare with a design the library makes and by standard errors, and a
  design the library makes can be folded over.
