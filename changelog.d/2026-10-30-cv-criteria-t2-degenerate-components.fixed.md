- **The pseudo-validation T² of `compare_cv_criteria` leaves out the components `fit` leaves out.**
  It divided each held-out score by any score standard deviation above 0, so a component with no
  spread (an sd at most 1.5e-8 of the largest) added noise over noise, and a held-out row off the
  model plane got a T² of 1057 where the two components with spread give it 0.6. It now follows the
  rule every other T² follows. Such a component needs a full-data model whose units span some eight
  orders of magnitude, as an unscaled one's can; `compare_cv_criteria` now refuses every unscaled
  model (see the next entry). (#1029)
