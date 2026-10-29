- **`t2_contributions` now counts the components that `hotellings_t2_` counts.** It treated a
  component as degenerate when its score variance was at most 1.5e-8 (a standard deviation up to
  1.2e-4), and divided its scores by 1. `fit` leaves a component out of T² only when its score
  standard deviation is at most 1.5e-8 times the largest one. So a component with a small but real
  spread (a standard deviation of 1e-6, say) was all but dropped, and the row sums missed the
  stored T²; and a degenerate component still added its squared score for a row off the model
  plane. It now uses the rule `fit` uses, and a component left out contributes 0. Models with no
  such component give the same numbers as before. (#1024)
