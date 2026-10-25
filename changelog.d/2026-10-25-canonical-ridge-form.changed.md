- **`optimize_responses(method="canonical_analysis")` returns the ridge form as `ridge_form`, in place of
  `b_canonical`, with the same contents.** The old key named Box and Draper's B canonical form, but it held their
  ridge form of Eq. 11.1.2, y = y0 + theta X_k + sum of lambda_i X_i^2. The form centred on the stationary point,
  `canonical_axes`, is their B form; the docstring called it the A form, and now names both as Box and Draper
  (2007) do. `b_canonical` is gone, with no alias.
