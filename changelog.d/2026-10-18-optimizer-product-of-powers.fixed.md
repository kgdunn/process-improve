- **A term such as `I(A ** 2):B` is no longer read as `I(A ** 2)`.** `optimize_responses` and
  `evaluate_model` matched a square at the start of a term name and dropped the rest of the
  product, so a model with such a term was optimised as a different model, with no error.
