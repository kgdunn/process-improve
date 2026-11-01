- **Optimal inversion of PLS and TPLS models, with the shadow price of every rule.**
  `PLS.optimal_inversion` and `TPLS.optimal_inversion` find the inputs the model vouches for at the
  least cost, or closest to target qualities, under `InversionRules`: windows on the predicted
  qualities, bounds and fixed inputs, linear rules (a mixture that sums to 1, a safety or legal limit
  on a component), Hotelling's T² and every block's SPE within their limits, and at most `k` inputs
  of a group, each used at a minimum amount or not at all. The model is read once as exact affine
  maps checked against its own `diagnose` (`affine_model`); without counting rules the search is
  convex, and with them outer approximation over HiGHS proves the answer optimal. The answer's
  `constraints` table gives each rule's shadow price from the optimality conditions;
  `inversion_sensitivity` checks those by solving again, traces the objective as one limit moves,
  prices one input more or fewer per group, and finds how far each input's cost can move before the
  answer changes shape. For a target, the answer is the most typical (least T²) of the equally
  close points of the null space. Both functions take the fitted model or its maps, which
  `AffineModel.to_dict` stores as plain JSON and `AffineModel.from_dict` reads back. When there is no
  answer, `proven_optimal` says whether the search proved that none exists.
- **`PLS.diagnose` and `TPLS.diagnose` return the residuals** whose lengths are their SPE: a PLS
  model's in its scaled space, a TPLS model's per `"F"` and `"Z"` group, keyed like `spe`.
- **Alternatives with other inputs from `optimal_inversion`.** With counting rules,
  `optimal_inversion(..., alternatives=k, leave_out=l)` also returns the next `k` best answers, each
  leaving out at least `l` of the counted inputs that every earlier answer uses (a no-good cut on the
  outer-approximation master). Each is proven best among the input sets the earlier answers allow,
  and the list ends with the reason when no further set meets the rules.
