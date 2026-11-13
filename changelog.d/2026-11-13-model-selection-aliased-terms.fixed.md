- **Model selection no longer keeps terms the runs cannot tell apart.**
  `analyze_experiment(..., analysis_type="model_selection")` started from every candidate term
  whenever the runs outnumbered them. A term aliased with another, with the intercept or with the
  blocks then could not leave, since the model fits as well with it as without, and under effect
  heredity it held its main effects in the model too. A half fraction run twice kept all ten terms
  of the interactions model whatever the data, and a two-level factorial searched as a quadratic
  model kept every square and every main effect. Now:
  - A candidate that adds nothing to the intercept and the blocks is no candidate: an interaction
    confounded with the blocks, or the square of a factor run at two levels.
  - The search starts from every candidate only when the runs can estimate them all, and from none
    otherwise: of two aliased interactions, the one whose main effects are in is the one that enters.
  - Of two moves that score the same, the one named first is taken, so a result does not depend on
    the rounding of a score.
  - A factor of `True` and `False` is a category, as patsy reads it, and has no square.
  - A formula with no intercept keeps one term at least, where the search stopped on an error when
    no term stood out, and either of two categorical factors in such a formula may leave.

  Results are unchanged where the runs estimate every candidate term.
