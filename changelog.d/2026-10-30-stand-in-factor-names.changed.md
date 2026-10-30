- **A factor named `I` fits a model with squares, and a factor named `C` fits an ASCA.** Patsy finds a column
  before its own function of the same name, so these were refused with a rename to paste. `analyze_experiment`,
  `lm`, `evaluate_design`, `augment_design` and `ASCA` now fit such a factor under a name no column has (`I_`, or
  `I_2` when a factor is already named `I_`), and report every result under the real name: `I(I ** 2)`, `I:P`,
  `ssq_["C"]`. The numbers are those of the same factor under any other name. The same holds for `Q`, the contrast
  names (`Sum`, `Treatment`, ...) and `np`, wherever the formula calls them. `coding` ranges and `new_points`
  columns are given under the real names. `evaluate_design` keeps the refusal for a design whose generators or
  region name the factors. factori.al still refuses a factor named `I` in a design with squares. (#986)
