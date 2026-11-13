- **Model selection no longer keeps terms the runs cannot tell apart.**
  `analyze_experiment(..., analysis_type="model_selection")` started from every candidate term
  whenever the runs outnumbered them. A term aliased with another, with the intercept or with the
  blocks then could not leave, since the model fits as well with it as without, and under effect
  heredity it held its main effects in the model too. A half fraction run twice kept all ten terms
  of the interactions model whatever the data, and a two-level factorial searched as a quadratic
  model kept every square and every main effect. Now:
  - A candidate that adds nothing to the intercept and the blocks is no candidate: an interaction
    confounded with the blocks, or the square of a factor run at two levels.
  - Where the candidates are aliased with each other, the search is run twice and the better
    result kept: from none, where the first of an aliased pair to enter is the one whose main
    effects are in, and from the candidates the runs estimate together, those named first, which
    finds an interaction one of whose main effects does nothing. `direction` says which was kept.
  - Of two moves that score the same, the one that leaves the term named first in the model is
    taken, so a result does not depend on the rounding of a score.
  - A factor of `True` and `False` is a category, as patsy reads it, and has no square.
  - A formula with no intercept keeps one term at least, where the search stopped on an error when
    no term stood out.
  - Each term of a formula is scored as patsy fits it among the terms it is with. Scored from the
    columns of the whole formula, the second of two categorical factors, or of two slopes within
    levels (`P:x + Q:x`), lacked a column whenever the first was left out, and the terms chosen
    turned on the order of the levels.

  Results are unchanged where the runs estimate every candidate term and no term of a formula is
  coded by the terms beside it.
