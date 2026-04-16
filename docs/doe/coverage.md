# DOE Implementation Coverage

Current implementation status across every agent-facing DoE tool.

## Tool Status

| Tool | Status | Implementation | Open gaps |
|---|---|---|---|
| `generate_design` | **Implemented** | Unified dispatcher in `experiments/designs.py`; per-family handlers in `designs_factorial.py`, `designs_screening.py`, `designs_response_surface.py`, `designs_optimal.py`, `designs_mixture.py`. Covers all 11 design types (full/fractional factorial, PB, BBD, CCD, DSD, D/I/A-optimal, mixture, Taguchi). | I/A-optimal require `pyoptex`; hard-to-change factors without `pyoptex` are ignored with a warning. |
| `evaluate_design` | **Implemented** | `experiments/evaluate.py` — 14 metrics: `d/i/g_efficiency`, `prediction_variance`, `vif`, `condition_number`, `power`, `degrees_of_freedom`, `alias_structure`, `confounding`, `resolution`, `defining_relation`, `clear_effects`, `minimum_aberration`. | — |
| `analyze_experiment` | **Implemented** | `analysis.py` — 13 analysis types via statsmodels/scipy. | Split-plot ANOVA (mixed-model mapping). |
| `optimize_responses` | **Partial** | `optimization.py` — desirability, steepest ascent/descent, stationary point, canonical analysis. | Ridge analysis, Pareto front (stubs). |
| `augment_design` | **Implemented** | `augment.py` — foldover, semifold, axial, replicate, D-optimal. | — |
| `visualize_doe` | **Implemented** | `visualization/` — 20 plot types, dual Plotly/ECharts backends. | — |
| `doe_knowledge` | **Implemented** | `knowledge/` — YAML knowledge graph, in-memory query engine. | Interpretation guides and worked examples (YAML stubs). |
| `recommend_strategy` | **Implemented** | `strategy/` — deterministic rule engine, ~50 decision rules, 8 domain templates, budget allocation. | — |

## Design Family Details

| Design | Implementation | Tests |
|---|---|---|
| Full factorial 2^k | `designs_factorial.py` via `pyDOE3.ff2n`. | `tests/test_design_generation.py::TestFullFactorial`, `tests/test_design_properties.py::TestFullFactorialProperties`. |
| Fractional factorial (resolution III/IV/V, explicit generators) | `designs_screening.py::dispatch_fractional_factorial` via `pyDOE3.fracfact` / `fracfact_by_res`. | `TestFractionalFactorial`, `TestFractionalFactorialProperties`. |
| Plackett-Burman (N ∈ {8, 12, 16, 20, 24, …}) | `designs_screening.py::dispatch_plackett_burman` via `pyDOE3.pbdesign`. | `TestPlackettBurman`, `TestPlackettBurmanProperties`. |
| Box-Behnken | `designs_response_surface.py::dispatch_box_behnken` via `pyDOE3.bbdesign`. | `TestBoxBehnken`, `TestBoxBehnkenProperties`. |
| Central Composite Design (face-centered, rotatable, inscribed, orthogonal) | `designs_response_surface.py::dispatch_ccd` via `pyDOE3.ccdesign`. | `TestCCD`, `TestCCDProperties`. |
| Definitive Screening Design | `designs_response_surface.py::dispatch_dsd` — Paley conference-matrix construction (see caveat below). | `TestDSD`, `TestDSDProperties`. |
| D-optimal | `designs_optimal.py::dispatch_d_optimal` — `pyoptex` coordinate exchange when available, otherwise point exchange on a 3-level candidate set. | `TestDOptimal`. |
| I-optimal | `designs_optimal.py::dispatch_i_optimal` via `pyoptex`. | `TestIOptimal` (skipped without `pyoptex`). |
| A-optimal | `designs_optimal.py::dispatch_a_optimal` via `pyoptex`. | `TestAOptimal` (skipped without `pyoptex`). |
| Mixture (simplex-lattice, simplex-centroid) | `designs_mixture.py::dispatch_mixture` — auto-selects based on budget. | `TestMixture`. |
| Taguchi orthogonal arrays | `designs_screening.py::dispatch_taguchi` via `pyDOE3.taguchi_design`. | `TestTaguchi`. |

## `evaluate_design` Metrics

All 14 metrics live in `experiments/evaluate.py` behind the `_METRIC_REGISTRY`:

| Metric | Function | Notes |
|---|---|---|
| D-efficiency | `_compute_d_efficiency` | `100 · det(X'X)^(1/p) / N`; 100 for orthogonal full factorials. |
| I-efficiency | `_compute_i_efficiency` | `100 · p / (N · mean prediction variance over a Sobol grid)`. |
| G-efficiency | `_compute_g_efficiency` | `100 · p / (N · max prediction variance over a Sobol grid)`. |
| Prediction variance | `_compute_prediction_variance` | Leverage `x'(X'X)⁻¹x` evaluated on a Sobol grid. |
| VIF | `_compute_vif` | Per-term variance inflation factors (excluding intercept). |
| Condition number | `_compute_condition_number` | `numpy.linalg.cond(X)`. |
| Power | `_compute_power` | Non-central F; scalar or curve over effect sizes. |
| Degrees of freedom | `_compute_degrees_of_freedom` | Breakdown: model / residual / total / pure error / lack-of-fit. |
| Alias structure | `_compute_alias_structure` | GF(2) closure for fractional factorials; correlation fallback for other designs. |
| Confounding | `_compute_confounding` | Extracted from alias chains. |
| Resolution | `_compute_resolution` | Minimum word length in the defining relation. |
| Defining relation | `_compute_defining_relation` | Full closure under GF(2) multiplication. |
| Clear effects | `_compute_clear_effects` | Effects whose aliases are all higher-order. |
| Minimum aberration | `_compute_minimum_aberration` | Wordlength pattern (A_3, A_4, …). |

## Caveats

- **DSD conference matrix.** `_conference_matrix` in `designs_response_surface.py` uses Paley's construction when `m − 1` is an odd prime (covers `m ∈ {4, 6, 8, 12, 14, 18, 20, 24, 30, 32, 38, 42, 44, 48, 54, 60, 62, 68, 72, 74, 80, 84, 90, 98, …}`). For other *m* (including `m ∈ {10, 16, 22, 26, 28, 34, 36, 40, 46, 50, 52, 56, …}`) the function falls back to a cyclic approximation that does **not** satisfy `Cᵀ C = (m − 1) I`, and logs a warning. Main-effects orthogonality of the resulting DSD may be degraded in those sizes.
- **Optimal designs without `pyoptex`.** D-optimal has a point-exchange fallback; I/A-optimal raise `ImportError` instead. Hard-to-change factors (split-plot structure) are currently ignored with a `logger.warning` when `pyoptex` is not available.
- **Taguchi OA auto-selection** picks the smallest standard array that covers the requested factor count and level counts, but the underlying `pyDOE3.taguchi_design` requires the number of `levels_per_factor` entries to match the OA column count exactly, so requesting a Taguchi design with fewer factors than any available OA will fail. Users typically pick *k* to match one of the standard arrays (3, 4, 7, 11, 15, …).

## No Silent Fallbacks

`generate_design` does **not** silently substitute a different design type when the requested one is infeasible. Unknown `design_type` values raise `ValueError` in `designs.py:328-331`; individual dispatchers validate their own inputs (e.g. BBD and DSD require `k ≥ 3`). Errors surface through the tool wrapper as `{"error": ...}` and reach agent callers as HTTP 422.

## Usage Frequency

Across the 162-question benchmark suite (primary + secondary uses):

| Tool | Primary | Secondary | Total | % of Qs |
|---|---|---|---|---|
| `doe_knowledge` | 63 | 42 | 105 | 65% |
| `generate_design` | 46 | 12 | 58 | 36% |
| `analyze_experiment` | 22 | 14 | 36 | 22% |
| `optimize_responses` | 15 | 5 | 20 | 12% |
| `evaluate_design` | 7 | 8 | 15 | 9% |
| `recommend_strategy` | 10 | 4 | 14 | 9% |
| `visualize_doe` | 3 | 7 | 10 | 6% |
| `augment_design` | 7 | 2 | 9 | 6% |
