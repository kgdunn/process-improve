- **Mixture components and process factors in one optimal design.** `generate_design` used to refuse mixture
  components beside process factors. The optimal families (`d_optimal`, `i_optimal`, `a_optimal`,
  `e_optimal`, `g_optimal`, `k_optimal`, and `design_type=None`) now build the design from every candidate
  blend of the constrained simplex at every candidate process setting. `model_type` takes the polynomial of
  an order in all the factors (`"main_effects"`, `"interactions"`, `"quadratic"`; the last is the
  Kowalski-Cornell-Vining model), a Scheffé model added to a process model (`"scheffe_quadratic +
  main_effects"`), or the two crossed (`"scheffe_quadratic * interactions"`). The design records its formula
  in `metadata["formula"]` and its region in `metadata["region"]`. `DesignRegion` has a `"mixture_process"`
  kind, the product of the two regions, which `evaluate_design` averages and maximises over and
  `optimize_responses` searches. `analyze_experiment` keeps the components as proportions, so the blending
  coefficients read as a Scheffé model's. The 12-run design of Goos and Jones (2011), Table 6.2, is found
  again. Categorical process factors, constraints that join components and process factors,
  `hard_to_change`, `fixed_runs`, `candidates` and blocks are refused for now.
