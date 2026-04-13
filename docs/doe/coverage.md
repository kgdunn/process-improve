# DOE Implementation Coverage

Current implementation status and gap analysis.

## Tool Status

| Tool | Status | Existing Code | Key Gaps |
|---|---|---|---|
| `generate_design` | **Partial** | `create_factorial_design` in `tools.py` — 2^k full factorial only | No fractional, PB, BBD, CCD, DSD, D-optimal, mixture, or Taguchi |
| `analyze_experiment` | **Implemented** | `analysis.py` — full dispatcher with 13 analysis types via statsmodels/scipy | Split-plot ANOVA (mixed model mapping) |
| `evaluate_design` | Not started | — | Everything: alias structure, power, efficiency metrics, VIF |
| `optimize_responses` | Not started | `optimization.py` has legacy MATLAB skeleton | Desirability, steepest ascent, stationary point, ridge analysis |
| `augment_design` | Not started | — | Foldover, semifold, axial points, optimal augmentation |
| `visualize_doe` | Not started | — | All DOE plot types |
| `doe_knowledge` | Not started | — | Knowledge graph, decision logic, interpretation guidance |
| `recommend_strategy` | Not started | — | Multi-stage planning, budget allocation |

## Existing Module Files

| File | What it provides | Used by tool |
|---|---|---|
| `structures.py` | `Column`, `Expt` classes | `generate_design`, `analyze_experiment` |
| `models.py` | `Model`, `lm()`, `predict()`, `summary()` | `analyze_experiment` |
| `designs_factorial.py` | `full_factorial()` | `generate_design` |
| `optimal.py` | D-optimal point exchange | `generate_design` (future) |
| `optimization.py` | Legacy MATLAB RSM code | `optimize_responses` (needs rewrite) |
| `datasets.py` | Sample datasets | Testing / examples |
| `simulations.py` | `popcorn()`, `grocery()` | Teaching / examples |
| `analysis.py` | `analyze_experiment()`, formula builder, 13 analysis types | `analyze_experiment` |
| `tools.py` | 4 tool specs registered | Agent interface |

## Usage Frequency

How often each tool is needed across all 162 questions (primary + secondary uses).

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

## Priority Order

Based on usage frequency and existing code to build on:

1. **`generate_design`** — most computational questions start here; partial implementation exists
2. **`analyze_experiment`** — the analytical workhorse; partial implementation exists
3. **`doe_knowledge`** — needed in 65% of questions; no implementation yet
4. **`optimize_responses`** — required for all RSM optimization workflows
5. **`evaluate_design`** — needed to assess design quality before running experiments
6. **`recommend_strategy`** — the orchestrator advisor for multi-stage plans
7. **`augment_design`** — extends existing designs (common in sequential experimentation)
8. **`visualize_doe`** — plotting support (often secondary to other tools)
