# DOE Tool Architecture

8 tools that collectively cover all 162 questions. Each tool is agent-callable via `@tool_spec` in `process_improve/experiments/tools.py`.

## Summary

| Tool | Purpose | Touches | % of Qs |
|---|---|---|---|
| `doe_knowledge` | Retrieve DOE concepts, decision logic, interpretation guidance | 105 | 65% |
| `generate_design` | Create any experimental design matrix | 58 | 36% |
| `analyze_experiment` | Fit models, ANOVA, diagnostics, residuals | 36 | 22% |
| `optimize_responses` | Find optimal factor settings (single or multi-response) | 20 | 12% |
| `evaluate_design` | Compute design quality metrics | 15 | 9% |
| `recommend_strategy` | Recommend multi-stage experimental strategy | 14 | 9% |
| `visualize_doe` | Generate DOE plots | 10 | 6% |
| `augment_design` | Extend or modify an existing design | 9 | 6% |

---

## Tool 1: `generate_design`

Create any type of experimental design matrix.

**Inputs:**

| Parameter | Type | Description |
|---|---|---|
| `factors` | list[Factor] | Name, type (`continuous`/`categorical`/`mixture`), low, high, levels, units |
| `design_type` | str or None | `full_factorial`, `fractional_factorial`, `plackett_burman`, `box_behnken`, `ccd`, `dsd`, `d_optimal`, `i_optimal`, `mixture`, `taguchi`, `custom`. None = auto-select. |
| `budget` | int or None | Max runs the user can afford |
| `center_points` | int | Center point replicates (default 3–5) |
| `replicates` | int | Full replicates |
| `blocks` | int or None | Number of blocks |
| `resolution` | int or None | Minimum resolution (III, IV, V) |
| `generators` | list[str] or None | e.g. `["D=ABC", "E=AC"]` |
| `alpha` | str/float or None | Axial distance for CCD: `"rotatable"`, `"face_centered"`, `"orthogonal"`, or numeric |
| `constraints` | list[Constraint] or None | e.g. `"3*T + 5*D <= 600"` |
| `hard_to_change` | list[str] or None | Triggers split-plot structure |
| `random_seed` | int | For reproducible randomization |

**Outputs:** Design matrix (coded + actual), run order, generator info, defining relation.

**Handles 46 questions as primary tool.** See [mapping](tool-question-mapping.md#generate_design).

---

## Tool 2: `evaluate_design`

Compute properties and quality metrics of an existing design.

**Inputs:**

| Parameter | Type | Description |
|---|---|---|
| `design_matrix` | DataFrame | Design to evaluate |
| `model` | str or None | `"main_effects"`, `"interactions"`, `"quadratic"`, or explicit formula |
| `metric` | str or list | `"alias_structure"`, `"confounding"`, `"resolution"`, `"defining_relation"`, `"power"`, `"d_efficiency"`, `"i_efficiency"`, `"g_efficiency"`, `"prediction_variance"`, `"degrees_of_freedom"`, `"vif"`, `"condition_number"`, `"clear_effects"`, `"minimum_aberration"` |
| `effect_size` | float or None | For power calculation |
| `alpha` | float | Significance level (default 0.05) |
| `sigma` | float or None | Estimated noise SD |

**Outputs:** Requested metrics — alias chains, power curves, efficiency values, etc.

**Handles 7 questions as primary tool.** See [mapping](tool-question-mapping.md#evaluate_design).

---

## Tool 3: `analyze_experiment`

Fit models, run ANOVA, compute effects, diagnose residuals. The main analytical workhorse.

**Inputs:**

| Parameter | Type | Description |
|---|---|---|
| `design_matrix` | DataFrame | Factor settings per run |
| `responses` | DataFrame | Response column(s) |
| `model` | str or None | `"main_effects"`, `"interactions"`, `"quadratic"`, or formula. None = stepwise. |
| `analysis_type` | str or list | `"anova"`, `"effects"`, `"coefficients"`, `"significance"`, `"residual_diagnostics"`, `"lack_of_fit"`, `"curvature_test"`, `"model_selection"`, `"box_cox"`, `"lenth_method"`, `"confidence_intervals"`, `"prediction"`, `"confirmation_test"` |
| `significance_level` | float | Default 0.05 |
| `transform` | str or None | `"log"`, `"sqrt"`, `"inverse"`, `"box_cox"`, `"none"` |
| `coding` | str | `"coded"` or `"actual"` |
| `new_points` | DataFrame or None | For prediction or confirmation |
| `observed_at_new` | list[float] or None | Observed values at new_points (confirmation testing) |
| `split_plot` | dict or None | Whole-plot / subplot structure |

**Outputs:** ANOVA table, coefficients with p-values and CIs, R²/adj-R²/pred-R², adequate precision, lack-of-fit test, curvature test, residual stats, Box-Cox lambda, Lenth's PSE, predictions with PIs.

**Handles 22 questions as primary tool.** See [mapping](tool-question-mapping.md#analyze_experiment).

---

## Tool 4: `optimize_responses`

Find optimal factor settings for one or multiple responses.

**Inputs:**

| Parameter | Type | Description |
|---|---|---|
| `fitted_models` | list[FittedModel] | From `analyze_experiment` |
| `goals` | list[Goal] | Per response: `"maximize"`, `"minimize"`, `"target"` with bounds/weights |
| `method` | str | `"desirability"`, `"steepest_ascent"`, `"steepest_descent"`, `"stationary_point"`, `"ridge_analysis"`, `"canonical_analysis"`, `"pareto_front"` |
| `constraints` | list[Constraint] or None | Factor-space constraints |
| `factor_ranges` | dict or None | Exploration bounds per factor |
| `step_size` | float or None | For steepest ascent (coded units) |
| `desirability_weights` | list[float] or None | Relative importance per response |

**Outputs:** Optimal settings (coded + actual), predicted responses, overall desirability, stationary point classification, canonical form, ridge trace, steepest ascent table.

**Handles 15 questions as primary tool.** See [mapping](tool-question-mapping.md#optimize_responses).

---

## Tool 5: `augment_design`

Extend or modify an existing design — add runs, fold over, upgrade to RSM.

**Inputs:**

| Parameter | Type | Description |
|---|---|---|
| `existing_design` | DataFrame | Current design matrix |
| `augmentation_type` | str | `"foldover"`, `"semifold"`, `"add_center_points"`, `"add_axial_points"`, `"add_runs_optimal"`, `"upgrade_to_rsm"`, `"add_blocks"`, `"replicate"` |
| `target_model` | str or None | Desired model after augmentation (e.g. `"quadratic"`) |
| `n_additional_runs` | int or None | Budget for additional runs |
| `fold_on` | str or None | For semifold: which factor |
| `alpha` | str/float or None | Axial distance if adding star points |

**Outputs:** Augmented design matrix, new defining relation, updated alias structure, explanation of what changed.

**Handles 7 questions as primary tool.** See [mapping](tool-question-mapping.md#augment_design).

---

## Tool 6: `visualize_doe`

Generate DOE plots from design data or fitted models.

**Inputs:**

| Parameter | Type | Description |
|---|---|---|
| `plot_type` | str | `"main_effects"`, `"interaction"`, `"contour"`, `"surface_3d"`, `"pareto"`, `"half_normal"`, `"daniel"`, `"residuals_vs_fitted"`, `"normal_probability"`, `"residuals_vs_order"`, `"box_cox"`, `"prediction_variance"`, `"desirability_contour"`, `"cube_plot"`, `"perturbation"`, `"overlay"`, `"fds_plot"`, `"power_curve"`, `"ridge_trace"`, `"steepest_ascent_path"` |
| `data_source` | FittedModel / DataFrame | Context-dependent |
| `factors_to_plot` | list[str] or None | Which factors (2 at a time for contour/interaction) |
| `hold_values` | dict or None | Fixed values for factors not plotted |
| `response` | str or None | Which response column |
| `highlight_significant` | bool | Auto-highlight on Pareto/half-normal |
| `confidence_level` | float | For reference lines (default 0.95) |

**Outputs:** Plot object.

**Handles 3 questions as primary tool** (supports many more as secondary). See [mapping](tool-question-mapping.md#visualize_doe).

---

## Tool 7: `doe_knowledge`

Retrieve DOE knowledge — definitions, decision logic, interpretation guidance, worked examples.

This is the most-used tool (65% of questions). It provides the conceptual grounding that computation alone cannot.

**Inputs:**

| Parameter | Type | Description |
|---|---|---|
| `query` | str | Natural language or structured query |
| `topic` | str or None | `"design_selection"`, `"design_properties"`, `"analysis_methods"`, `"interpretation"`, `"rsm"`, `"screening"`, `"practical_guidance"`, `"statistical_concepts"`, `"troubleshooting"`, `"reporting"`, `"qbd"`, `"comparison"` |
| `context` | dict or None | Current problem context (factor count, budget, design type, etc.) |
| `detail_level` | str | `"novice"`, `"intermediate"`, `"expert"` |

**Knowledge graph structure:**

```
DesignType  -[SUITABLE_FOR]->     Scenario
DesignType  -[HAS_PROPERTY]->     Property
DesignType  -[REQUIRES]->         MinimumRuns
DesignType  -[SUPPORTS_MODEL]->   ModelType
DesignType  -[CAN_AUGMENT_TO]->   DesignType
Concept     -[RELATED_TO]->       Concept
Diagnostic  -[INDICATES]->        Problem
Problem     -[REMEDIED_BY]->      Action
DecisionRule -[IF_CONDITION]->    Condition
DecisionRule -[THEN_RECOMMEND]->  DesignType
WorkedExample -[DEMONSTRATES]->   Concept
```

**Outputs:** Definitions, decision trees, comparison tables, interpretation rules, worked examples, references.

**Handles 63 questions as primary tool.** See [mapping](tool-question-mapping.md#doe_knowledge).

---

## Tool 8: `recommend_strategy`

Given a problem description, recommend a multi-stage experimental strategy.

**Inputs:**

| Parameter | Type | Description |
|---|---|---|
| `factors` | list[Factor] | All candidate factors with types, ranges |
| `responses` | list[Response] | Response names with goals |
| `budget` | int or None | Total runs across all stages |
| `constraints` | list[Constraint] or None | Physical, safety, regulatory constraints |
| `hard_to_change_factors` | list[str] or None | Expensive-to-reset factors |
| `prior_knowledge` | str or None | What the user already knows |
| `existing_data` | DataFrame or None | Prior results |
| `domain` | str or None | `"pharma_formulation"`, `"fermentation"`, `"food_science"`, `"extraction"`, `"analytical_method"`, `"cell_culture"`, `"bioprocess"`, `"general"` |
| `detail_level` | str | `"novice"`, `"intermediate"` |

**Example output:**

```
Stage 1: Screening → PB design, 12 runs, factors A–G
Stage 2: Augment to resolution IV → 8 additional runs
Stage 3: RSM optimization → BBD, 3 significant factors, 17 runs
Stage 4: Confirmation → 3–5 replicates at predicted optimum
Total: ~40 runs
```

**Handles 10 questions as primary tool.** See [mapping](tool-question-mapping.md#recommend_strategy).
