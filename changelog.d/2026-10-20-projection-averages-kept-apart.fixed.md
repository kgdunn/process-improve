- **`evaluate_projections` no longer averages different quantities together.** Its `averages` were keyed by the
  number's own name alone, so two metrics that return numbers under one name had them pooled: asked for `vif`
  and `power` together, each term's "average" was the mean of its variance inflation factors and its powers.
  Such numbers are now kept apart and named after their metric (`"vif.A"`, `"power.A"`). Every other key is
  unchanged: a metric's own number (`"d_efficiency"`) keeps its name, and so does a number only one metric returns.
