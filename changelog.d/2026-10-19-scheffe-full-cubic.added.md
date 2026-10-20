- **Mixture designs and analyses accept the full cubic Scheffé model, `"scheffe_cubic"` (alias `"cubic"`).**
  It adds a term `x_i x_j (x_i - x_j)` for each pair of components to the special cubic, for `q^2 + C(q, 3)`
  terms. The default design over the whole simplex is the {q, 3} simplex lattice (every blend in thirds,
  NIST/SEMATECH e-Handbook Table 5.3), plus the overall centroid for four or more components. A bounded or
  constrained region takes the points a third and two thirds of the way along each edge in place of the
  midpoints: along an edge the model is a cubic polynomial, which needs four points on that edge.
  `analyze_experiment`, `evaluate_design`, the stepwise model selection and the design catalog accept the new
  model.
