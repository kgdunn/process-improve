- **`AnalysisRecipe` and `RecipeStep` can be hashed, and their sequences cannot be changed in place
  (#513).** Both were `frozen=True` dataclasses holding lists, so `hash(recipe)` raised `TypeError` and
  `recipe.stages.append(...)` changed a registered recipe. `cue_phrases`, `inputs_needed`, `stages` and
  `tools` are now stored as tuples, whatever sequence is passed in; `arg_hints` stays a `dict`, copied
  from the one passed in. `to_payload()` still returns lists. `select_recipe` no longer re-canonicalises
  every cue phrase on every query.
