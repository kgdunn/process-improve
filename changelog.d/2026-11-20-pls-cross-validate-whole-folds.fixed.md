- **`PLS.cross_validate` refuses a number of folds that is not whole** (#1114). `cv=5.5`, `cv=5.0` and `cv="5"`
  were each read as 5 folds without a word; they now raise `ValueError`, as the rest of the family does for a
  count that is not whole. `"loo"`, a Python or numpy integer, and `PLSDA.cross_validate`'s class-balanced
  folds are unchanged.
