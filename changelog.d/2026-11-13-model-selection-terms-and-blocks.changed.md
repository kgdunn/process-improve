- **`analysis_type="model_selection"` searches a formula over its own terms, keeps the blocks, and
  knows a categorical factor.** Three changes to the stepwise search of `analyze_experiment`:
  - An explicit formula was searched as the interactions model, with a note, so a square in the
    formula could never be kept. It is now searched over its own terms, each free to enter once
    every term of the formula that it contains is in (`A:B` after `A` and `B`, `I(A ** 2)` after
    `A`). `candidate_model` is the formula, and there is no `note`.
  - The blocks of a blocked design were left out of every model the search scored, so the
    differences between blocks went to the residual and hid real effects. They are now in every
    model. `selected_terms` holds the terms chosen, without the blocks, and `selected_formula`
    names the `Block` column, so it can be given back as `model=`.
  - A categorical factor has no square in the quadratic model's search, where the search used to
    stop on one, and all its contrasts enter and leave together.

  Results change for a blocked design, and for a formula with a square or with fewer terms than
  the interactions model. (#1229)
