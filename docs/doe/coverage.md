# DOE Implementation Coverage

Current implementation status across every agent-facing DoE tool.

## Tool Status

| Tool | Status | Implementation | Open gaps |
|---|---|---|---|
| `generate_design` | **Implemented** | Unified dispatcher in `experiments/designs.py`; per-family handlers in `designs_factorial.py`, `designs_screening.py`, `designs_response_surface.py`, `designs_optimal.py`, `designs_mixture.py`. Covers all 11 design types (full/fractional factorial, PB, BBD, CCD, DSD, D/I/A-optimal, mixture, Taguchi). | I/A-optimal require `pyoptex`; hard-to-change factors without `pyoptex` are ignored with a warning. |
| `evaluate_design` | **Implemented** | `experiments/evaluate.py` - 14 metrics: `d/i/g_efficiency`, `prediction_variance`, `vif`, `condition_number`, `power`, `degrees_of_freedom`, `alias_structure`, `confounding`, `resolution`, `defining_relation`, `clear_effects`, `minimum_aberration`. | - |
| `analyze_experiment` | **Implemented** | `analysis.py` - 13 analysis types via statsmodels/scipy. | Split-plot ANOVA (mixed-model mapping). |
| `optimize_responses` | **Partial** | `optimization.py` - desirability, steepest ascent/descent, stationary point, canonical analysis. | Ridge analysis, Pareto front (stubs). |
| `augment_design` | **Implemented** | `augment.py` - foldover, semifold, axial, replicate, D-optimal. | - |
| `visualize_doe` | **Implemented** | `visualization/` - 20 plot types, dual Plotly/ECharts backends. | - |
| `doe_knowledge` | **Implemented** | `knowledge/` - YAML knowledge graph, in-memory query engine. | Interpretation guides and worked examples (YAML stubs). |
| `recommend_strategy` | **Implemented** | `strategy/` - deterministic rule engine, ~50 decision rules, 8 domain templates, budget allocation. | - |

## Design Family Details

| Design | Implementation | Tests |
|---|---|---|
| Full factorial 2^k | `designs_factorial.py` via `pyDOE3.ff2n`. | `tests/test_design_generation.py::TestFullFactorial`, `tests/test_design_properties.py::TestFullFactorialProperties`. |
| Fractional factorial (resolution III/IV/V, explicit generators) | `designs_screening.py::dispatch_fractional_factorial` via `pyDOE3.fracfact` / `fracfact_by_res`. | `TestFractionalFactorial`, `TestFractionalFactorialProperties`. |
| Plackett-Burman (N ∈ {8, 12, 16, 20, 24, …}) | `designs_screening.py::dispatch_plackett_burman` via `pyDOE3.pbdesign`. | `TestPlackettBurman`, `TestPlackettBurmanProperties`. |
| Box-Behnken | `designs_response_surface.py::dispatch_box_behnken` via `pyDOE3.bbdesign`. | `TestBoxBehnken`, `TestBoxBehnkenProperties`. |
| Central Composite Design (face-centered, rotatable, inscribed, orthogonal) | `designs_response_surface.py::dispatch_ccd` via `pyDOE3.ccdesign`. | `TestCCD`, `TestCCDProperties`. |
| Definitive Screening Design | `designs_response_surface.py::dispatch_dsd` - Paley conference-matrix construction (see caveat below). | `TestDSD`, `TestDSDProperties`. |
| D-optimal | `designs_optimal.py::dispatch_d_optimal` - `pyoptex` coordinate exchange when available, otherwise point exchange on a 3-level candidate set. | `TestDOptimal`. |
| I-optimal | `designs_optimal.py::dispatch_i_optimal` via `pyoptex`. | `TestIOptimal`, `test_designs_optimal_pyoptex.py` (run in CI; `pyoptex` is in the dev dependency group). |
| A-optimal | `designs_optimal.py::dispatch_a_optimal` via `pyoptex`. | `TestAOptimal`, `test_designs_optimal_pyoptex.py` (run in CI; `pyoptex` is in the dev dependency group). |
| Mixture (simplex-lattice, simplex-centroid) | `designs_mixture.py::dispatch_mixture` - auto-selects based on budget. | `TestMixture`. |
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

- **DSD conference matrix.** `_conference_matrix` in `designs_response_surface.py` builds order *m* from Paley's construction whenever `m − 1` is an odd prime power (`m ∈ {4, 6, 8, 10, 12, 14, 18, 20, 24, 26, 28, 30, 32, 38, 42, 44, 48, 50, 54, 60, 62, …}`), and from a tabulated matrix at `m = 16`. Every matrix is verified against `Cᵀ C = (m − 1) I` before use. At the remaining orders (`m ∈ {22, 34, 36, 40, 46, 52, 56, 58, …}`) the constructor raises, and `dispatch_dsd` steps up to the next order it can build, dropping the surplus columns. That costs runs but never orthogonality: 21 and 22 factors use order 24, so 49 runs rather than the 45 a minimal design would need. Order 22 is not a gap in this module but a fact about conference matrices: one of order `m ≡ 2 (mod 4)` exists only when `m − 1` is a sum of two squares, and 21 is not. Orders 34 and 58 fail the same test; 36, 40, 46 and 52 are orders this module has no construction for, though a matrix may exist. Use `dsd_run_count(k)` to find the size a given factor count will produce.
- **DSD categorical factors.** Two-level categorical factors are supported through both column-augmentation procedures of Jones & Nachtsheim (2013), selected with `categorical_method`. Three-or-more-level categorical factors are not covered by that family and raise, pointing at `d_optimal`. `ORTH-augment` is exactly orthogonal for up to four categorical factors and only nearly orthogonal beyond that, which is the paper's own limit rather than an implementation shortcut. Note that the c = 2 panels of the paper's Figure 1 disagree with its Table 3, Equation (2) and Section 3 Step 3; we follow the procedures and the tables, and `tests/test_dsd_categorical.py::TestFigure1Errata` records the evidence.
- **Optimal designs without `pyoptex`.** D-optimal has a point-exchange fallback; I/A-optimal raise `ImportError` instead. Hard-to-change factors (split-plot structure) are currently ignored with a `logger.warning` when `pyoptex` is not available.
- **Taguchi OA auto-selection** picks the smallest standard array that covers the requested factor count and level counts, but the underlying `pyDOE3.taguchi_design` requires the number of `levels_per_factor` entries to match the OA column count exactly, so requesting a Taguchi design with fewer factors than any available OA will fail. Users typically pick *k* to match one of the standard arrays (3, 4, 7, 11, 15, …).

## Reliance on `pyoptex`

`pyoptex` (github.com/mborn1/pyoptex) powers the high-quality optimal designs
(D/I/A-optimal coordinate exchange and split-plot structures) in
`designs_optimal.py`. For **end users** it is an optional, undeclared
dependency: the core install never pulls it in, and the integration degrades
cleanly when it is absent (D-optimal falls back to point exchange; I/A-optimal
raise a clear `ImportError`; hard-to-change factors are ignored with a warning
and a `hard_to_change_ignored` metadata flag). This keeps the coupling to a
single, well-isolated adapter module.

**How the gated tests get exercised.** In the uv-managed development
environment, `pyoptex` IS installed: it sits in the `[dependency-groups].dev`
list, and `[tool.uv] override-dependencies` relaxes its over-strict
`plotly~=5.24` and `numba~=0.61` pins to this project's own floors. Every CI
job runs `uv sync --dev --all-extras`, so the pyoptex-backed tests
(`test_designs_optimal_pyoptex.py` and the gated classes in
`test_design_generation.py`) run as blocking checks across the whole matrix.
The no-pyoptex fallback and ImportError paths stay covered too: the tests in
`test_designs_screening_optimal.py` force `_PYOPTEX_AVAILABLE = False` via
monkeypatch instead of relying on the package being absent.

**Why it is still not a published extra.** pip cannot apply uv overrides, so
declaring `pyoptex` in the `expt`/`all` extras would make combinations like
`[expt] + [plotting]` unresolvable for pip users. The upstream fix for the
plotly pin (mborn1/pyoptex#49, `plotly>=5.24,<7`) is merged but not yet in a
PyPI release; `numba~=0.61` is still pinned strictly even upstream. Until a
release ships relaxed pins, end users who want I/A-optimal install `pyoptex`
in a separate environment (`pip install pyoptex`).

**Why not vendor it.** The slice used here (`doe/fixed_structure`) is
Cython-compiled. Vendoring is permitted (`pyoptex` is BSD-3-Clause) but would
add a C build toolchain to an otherwise pure-Python package and transfer the
maintenance of numerically delicate optimizer code. The friction is a
packaging release lag, not a code problem, so vendoring is the wrong trade.

**Exit criterion.** Once `pyoptex` publishes a release with the relaxed pins,
move it into the `expt`/`all` extras as a normal dependency and drop the
`[tool.uv]` overrides (the numba override can only go once upstream relaxes
`numba~=0.61`). If `pyoptex` instead goes unmaintained, the clean replacement
is a minimal numpy coordinate-exchange for I/A-optimal (reusing the existing
`point_exchange` pattern in `optimal.py`), not a transplant of the upstream
Cython.

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
