# DOE Implementation Coverage

Current implementation status across every agent-facing DoE tool.

## Tool Status

| Tool | Status | Implementation | Open gaps |
|---|---|---|---|
| `generate_design` | **Implemented** | Unified dispatcher in `experiments/designs.py`; per-family handlers in `designs_factorial.py`, `designs_screening.py`, `designs_response_surface.py`, `designs_optimal.py`, `designs_mixture.py`. Covers all 21 design types (full/fractional factorial, PB, BBD, CCD, DSD, OMARS, D/I/A/E-optimal, mixture, Taguchi, supersaturated, and six space-filling designs). | Hard-to-change factors (split-plot) need `pyoptex` and no constraints; otherwise they are ignored with a warning. Constraints are enforced by the optimal families, mixture designs and the Sobol, Halton and maximin space-filling designs (other types flag `constraints_enforced=False`); mixture constraints must be linear. Mixture-process designs are not supported. |
| `evaluate_design` | **Implemented** | `experiments/evaluate.py` - 20 metrics: `d/i/g_efficiency`, `a_optimality`, `e_optimality`, `fds`, `prediction_variance`, `vif`, `condition_number`, `correlation`, `power`, `degrees_of_freedom`, `alias_structure`, `alias_matrix`, `confounding`, `resolution`, `defining_relation`, `clear_effects`, `minimum_aberration`, `moment_aberration`. I-, G-efficiency and FDS are taken over the design's recorded (constrained) region. | - |
| `analyze_experiment` | **Implemented** | `analysis.py` - 13 analysis types via statsmodels/scipy. | Split-plot ANOVA (mixed-model mapping). |
| `optimize_responses` | **Implemented** | `optimization.py` - desirability, steepest ascent/descent, stationary point, canonical analysis, ridge analysis, Pareto front; desirability and Pareto front can be kept inside a `DesignRegion`. | - |
| `augment_design` | **Implemented** | `augment.py` - foldover, semifold, axial, replicate, D-optimal. | - |
| `visualize_doe` | **Implemented** | `visualization/` - 20 plot types, dual Plotly/ECharts backends. | - |
| `doe_knowledge` | **Implemented** | `knowledge/` - YAML knowledge graph, in-memory query engine. | Interpretation guides and worked examples (YAML stubs). |
| `recommend_strategy` | **Implemented** | `strategy/` - deterministic rule engine, ~50 decision rules, 8 domain templates, budget allocation. | - |

## Design Family Details

| Design | Implementation | Tests |
|---|---|---|
| Full factorial 2^k | `designs_factorial.py` via `pyDOE3.ff2n`. | `tests/test_design_generation.py::TestFullFactorial`, `tests/test_design_properties.py::TestFullFactorialProperties`. |
| Fractional factorial (resolution III and up, explicit generators) | `designs_screening.py::dispatch_fractional_factorial`: a table of minimum-aberration generators up to 11 factors, built with `pyDOE3.fracfact`; `pyDOE3.fracfact_by_res` beyond that, with its resolution checked. | `TestFractionalFactorial`, `TestFractionalFactorialProperties`, `tests/test_experiments_fractional_factorial.py`. |
| Plackett-Burman (N ∈ {8, 12, 16, 20, 24, …}) | `designs_screening.py::dispatch_plackett_burman` via `pyDOE3.pbdesign` where it has the order, otherwise a finite-field Hadamard matrix (every multiple of 4 up to 100 except 92). | `TestPlackettBurman`, `TestPlackettBurmanProperties`. |
| Box-Behnken | `designs_response_surface.py::dispatch_box_behnken`: the published blocks of three factors for 6 and 7 factors (48 and 56 runs); `pyDOE3.bbdesign` pairs otherwise. | `TestBoxBehnken`, `TestBoxBehnkenProperties`. |
| Central Composite Design (face-centered, rotatable, inscribed, orthogonal) | `designs_response_surface.py::dispatch_ccd` via `pyDOE3.ccdesign`. | `TestCCD`, `TestCCDProperties`. |
| Definitive Screening Design | `designs_response_surface.py::dispatch_dsd` - exact conference matrices from `_finite_fields.py` (see caveat below). | `TestDSD`, `TestDSDProperties`. |
| D-optimal | `designs_optimal.py::dispatch_d_optimal` - `pyoptex` coordinate exchange when available and no constraints are given; otherwise `designs_constrained.py::constrained_optimal_design`: Fedorov exchange over a feasible candidate set (grid plus boundary crossings) for the requested model. | `TestDOptimal`, `tests/test_designs_constrained.py`. |
| I-optimal | `designs_optimal.py::dispatch_i_optimal` - as D-optimal; the candidate exchange minimises `trace(M^-1 W)` with `W` the moment matrix of the (constrained) region, sampled uniformly. | `TestIOptimal`, `test_designs_optimal_pyoptex.py` (run in CI; `pyoptex` is in the dev dependency group). |
| A-optimal | `designs_optimal.py::dispatch_a_optimal` - as D-optimal; the candidate exchange minimises `trace(M^-1)`. | `TestAOptimal`, `test_designs_optimal_pyoptex.py` (run in CI; `pyoptex` is in the dev dependency group). |
| Mixture (simplex-lattice, simplex-centroid, extreme vertices) | `designs_mixture.py::dispatch_mixture` - auto-selects based on budget on the full simplex; with component bounds or linear constraints, `designs_mixture_constrained.py` builds the extreme-vertices design or a D-optimal subset of its candidate blends for a Scheffé model. | `TestMixture`, `tests/test_mixture_constrained.py`. |
| Taguchi orthogonal arrays | `designs_screening.py::dispatch_taguchi` via `pyDOE3.taguchi_design`. | `TestTaguchi`. |
| E-optimal | `designs_constrained.py` candidate exchange; maximises the smallest eigenvalue of `X'X`. | `tests/test_designs_constrained.py::TestEOptimal`. |
| Supersaturated | `designs_supersaturated.py::dispatch_supersaturated` - Lin's half-fraction of a Hadamard matrix (pyDOE3, or Paley `I + C`). | `tests/test_designs_supersaturated.py`. |
| Space-filling (LHS, maximin LHS, uniform, Sobol, Halton, maximin) | `designs_space_filling.py::space_filling_design`; Sobol, Halton and maximin also in constrained and mixture regions. | `tests/test_designs_space_filling.py`. |

## `evaluate_design` Metrics

All 20 metrics live in `experiments/evaluate.py` behind the `_METRIC_REGISTRY`:

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
| A-optimality | `_compute_a_optimality` | `trace((X'X)^-1)` (lower is better). |
| E-optimality | `_compute_e_optimality` | Smallest eigenvalue of `X'X` (higher is better). |
| FDS | `_compute_fds` | Fraction-of-design-space curve of the prediction variance, over the region. |
| Correlation | `_compute_correlation` | Pairwise correlation among the model's second-order terms. |
| Alias matrix | `_compute_alias_matrix` | `(X1'X1)^-1 X1' X2`: bias of the fitted terms from omitted ones. |
| Moment aberration | `_compute_moment_aberration` | Moment aberration pattern, strength and resolution (Xu 2003). |

## Caveats

- **DSD conference matrix.** `_finite_fields.conference_matrix` builds exact conference matrices (Paley over GF(q) for prime powers `q = m - 1`, and doubling of antisymmetric ones), checked against `Cᵀ C = (m − 1) I`. Orders it cannot build (22 and 34 do not exist; 36, 46, 52 are not implemented) step up to the next buildable order with fake factors, so 21-22 factors take 49 runs.
- **Optimal designs without `pyoptex`.** D-, I-, A- and E-optimal designs use the built-in candidate exchange. Hard-to-change factors (split-plot structure) are ignored with a `logger.warning` and a `hard_to_change_ignored` flag when `pyoptex` is not available or constraints are given.
- **Taguchi OA auto-selection** picks the smallest standard array that covers the requested factor count and level counts, but the underlying `pyDOE3.taguchi_design` requires the number of `levels_per_factor` entries to match the OA column count exactly, so requesting a Taguchi design with fewer factors than any available OA will fail. Users typically pick *k* to match one of the standard arrays (3, 4, 7, 11, 15, …).

## Reliance on `pyoptex`

`pyoptex` (github.com/mborn1/pyoptex) powers the high-quality optimal designs
(D/I/A-optimal coordinate exchange and split-plot structures) in
`designs_optimal.py`. For **end users** it is an optional, undeclared
dependency: the core install never pulls it in, and the integration degrades
cleanly when it is absent (D-, I-, A- and E-optimal use the built-in candidate
exchange in `designs_constrained.py`; hard-to-change factors are ignored with a
warning and a `hard_to_change_ignored` metadata flag). This keeps the coupling to a
single, well-isolated adapter module.

**How the gated tests get exercised.** In the uv-managed development
environment, `pyoptex` IS installed: it sits in the `[dependency-groups].dev`
list, and `[tool.uv] override-dependencies` relaxes its over-strict
`plotly~=5.24` and `numba~=0.61` pins to this project's own floors. Every CI
job runs `uv sync --dev --all-extras`, so the pyoptex-backed tests
(`test_designs_optimal_pyoptex.py` and the gated classes in
`test_design_generation.py`) run as blocking checks across the whole matrix.
The no-pyoptex paths stay covered too: the tests in
`test_designs_screening_optimal.py` force `_PYOPTEX_AVAILABLE = False` via
monkeypatch instead of relying on the package being absent.

**Why it is still not a published extra.** pip cannot apply uv overrides, so
declaring `pyoptex` in the `expt`/`all` extras would make combinations like
`[expt] + [plotting]` unresolvable for pip users. The upstream fix for the
plotly pin (mborn1/pyoptex#49, `plotly>=5.24,<7`) is merged but not yet in a
PyPI release; `numba~=0.61` is still pinned strictly even upstream. Until a
release ships relaxed pins, end users who want split-plot designs install
`pyoptex` in a separate environment (`pip install pyoptex`).

**Why not vendor it.** The slice used here (`doe/fixed_structure`) is
Cython-compiled. Vendoring is permitted (`pyoptex` is BSD-3-Clause) but would
add a C build toolchain to an otherwise pure-Python package and transfer the
maintenance of numerically delicate optimizer code. The friction is a
packaging release lag, not a code problem, so vendoring is the wrong trade.

**Exit criterion.** Once `pyoptex` publishes a release with the relaxed pins,
move it into the `expt`/`all` extras as a normal dependency and drop the
`[tool.uv]` overrides (the numba override can only go once upstream relaxes
`numba~=0.61`). If `pyoptex` instead goes unmaintained, the built-in candidate
exchange already covers D/I/A-optimality; what would remain is split-plot
structure, which fits the same exchange as a restriction on which swaps are
allowed (see #631 for the fixed-block version).

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
