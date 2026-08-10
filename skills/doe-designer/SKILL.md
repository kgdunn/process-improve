---
name: doe-designer
description: Plan, generate, verify and analyse designed experiments (DOE) using the process-improve library. Use whenever the user wants to design an experiment, screen factors, optimise a process or formulation, build a response surface, choose a fractional factorial or Plackett-Burman or Box-Behnken or central composite or definitive screening design, check a design's resolution or aliasing or power, analyse experimental results, fit effects and interactions, produce a Pareto or half-normal or main-effects or interaction or contour plot, or find optimal factor settings. Also use when asked to check whether an existing design matrix is any good.
license: MIT
---

# Designed experiments

You are helping someone spend their lab time well. Every run costs money and
days, so the design has to be right *before* the first sample is made, and the
analysis has to be honest about what the data can and cannot support.

## The one rule that matters most

**Never write out a design matrix yourself. Generate it with a tool, then
verify it.**

This is not a stylistic preference. Vazquez et al. (2026, arXiv:2512.17113)
ran GPT-5.1 and Gemini 2.5 Flash across 36 two-level fractional factorial
construction tasks and found that models produce optimal designs reliably only
up to about eight factors. Beyond that they return designs of resolution 1 or
2, non-regular arrays presented as regular fractions, and tables with missing
cells, all of it looking entirely plausible. A resolution-2 design aliases one
main effect with another, so the experiment cannot answer its own question, and
nobody finds out until the results make no sense.

The catalogues in this library do not have that failure mode. So:

1. Call `generate_design` (or `create_factorial_design`) to build the matrix.
2. Run `verify_design.py` on it, or call `evaluate_design` with the
   `moment_aberration` metric.
3. Report the resolution and what it costs the user, in words, every time.

If a user pastes in a design from a chatbot, a spreadsheet or an old report,
verify it before analysing anything. `moment_aberration` works on any
two-level matrix with no generators or defining relation needed, so there is
never an excuse to skip the check.

## Workflow

Work through these in order. Do not jump to generating a design before you
know what the user is actually trying to learn.

### 1. Interview before designing

You need these before a design is meaningful. Ask for whatever is missing,
but ask in one batch, not one question at a time:

- **Response(s)**: what is measured, in what units, and is the goal to
  maximise, minimise or hit a target? How repeatable is the measurement?
- **Factors**: what can actually be *set*, with a plausible low and high for
  each. Distinguish factors from things that are merely recorded.
- **Budget**: how many runs are affordable, and over what calendar time.
- **Constraints**: combinations that are impossible or unsafe; factors that
  are expensive to change between runs (these force a split-plot).
- **Prior knowledge**: what is already known or suspected to matter. This is
  what separates a screening problem from an optimisation problem.

A common and important correction: users often arrive asking for a response
surface when they have eleven candidate factors. That is a screening problem.
Say so, and explain the staging.

### 2. Recommend a strategy

`recommend_strategy` takes the interview answers and returns a staged plan
with run counts, transition rules, budget allocation, assumptions and risks.
It applies deterministic decision rules rather than judgement, so use it
instead of reasoning from scratch about which design to pick.

```bash
python scripts/doe_tool.py call recommend_strategy --input strategy.json
```

Experiments are almost always staged: screen, then optimise, then confirm.
Spending the whole budget on one big design is the most common expensive
mistake. Aim to spend roughly a quarter to a third of the budget on the first
stage and keep the rest for what the first stage teaches you.

### 3. Generate the design

```bash
python scripts/doe_tool.py call generate_design --input design_spec.json
```

Design types available: `full_factorial`, `fractional_factorial`,
`plackett_burman`, `box_behnken`, `ccd`, `dsd`, `d_optimal`, `i_optimal`,
`a_optimal`, `mixture`, `taguchi`. Omit `design_type` to let the tool choose
from the factors and budget.

Always include center points (default 3) for continuous factors: they buy a
pure-error estimate and a curvature test for very little money.

See `references/choosing-a-design.md` for which design fits which situation.

### 4. Verify, always

```bash
python scripts/verify_design.py design.csv --require-resolution 4
```

This exits non-zero when the design does not meet the bar, so it is safe to
use as a gate. Then tell the user in plain words what the resolution means for
their experiment. `references/verification.md` has the wording for each
resolution and the other quality metrics worth reporting.

### 5. Randomise and hand over the run sheet

Give the user a run sheet in *real units*, not coded ones, with the run order
randomised and a column to write results into. Randomisation is not optional:
it is what protects the experiment against drift in ambient conditions,
operator fatigue and raw-material lots. If the user cannot randomise (a
factor is too expensive to change), say that the analysis will need to treat
it as a split-plot, and record which factor it was.

### 6. Analyse

```bash
python scripts/doe_tool.py call analyze_experiment --input results.json
```

Analysis types: `anova`, `effects`, `coefficients`, `significance`,
`residual_diagnostics`, `lack_of_fit`, `curvature_test`, `model_selection`,
`box_cox`, `lenth_method`, `confidence_intervals`, `prediction`,
`confirmation_test`.

Always run `residual_diagnostics`. A model with a good R-squared and patterned
residuals is worse than useless, because it is confidently wrong. For an
unreplicated factorial there are no degrees of freedom for error, so use
`lenth_method` rather than an F-test.

`references/analysis-workflow.md` has the full sequence and the traps.

### 7. Plot

```bash
python scripts/render_plot.py --analysis analysis.json --type pareto --output pareto.png
```

Lead with a Pareto or half-normal plot of the effects, then main-effects and
interaction plots for whatever turned out to matter. Write `.png` for a
static image or `.html` for an interactive page that needs nothing installed.

### 8. Optimise and confirm

`optimize_responses` finds settings that balance several responses via
desirability. Whatever it returns is a *prediction*, and predictions at the
edge of the design space are the least trustworthy ones. Always propose
confirmation runs and check them with `analysis_type: confirmation_test`.

## Tools

All eleven are reachable through `scripts/doe_tool.py call <name>`. Run
`python scripts/doe_tool.py spec <name>` to see the exact input schema before
calling, rather than guessing at field names.

| Tool | Use it for |
|---|---|
| `recommend_strategy` | "How should I plan this?" Staged plan from factors, budget, domain |
| `doe_knowledge` | Definitions, design-type descriptions, diagnostics, decision logic |
| `generate_design` | Build a design matrix |
| `create_factorial_design` | Quick 2^k full factorial |
| `evaluate_design` | Resolution, aliasing, moment aberration, D/I/G-efficiency, VIF, power |
| `analyze_experiment` | ANOVA, effects, diagnostics, Lenth, Box-Cox, confirmation |
| `fit_linear_model` | Explicit model formula when you need control |
| `augment_design` | Add runs: fold-over, axial points, replicates, de-aliasing |
| `optimize_responses` | Multi-response desirability optimisation |
| `visualize_doe` | 21 plot types, returns Plotly and ECharts specs |
| `trade_off_table` | Two-level runs-against-factors table: what each budget buys, and its aliasing |

`doe_knowledge` is the right first call whenever the user asks a conceptual
question ("what is aliasing?", "why center points?"). It returns curated
content from the library's knowledge base rather than a paraphrase.

## Reporting

Give the user, every time:

- The design, in real units, with the run order randomised.
- Its resolution, and one sentence on what that costs them.
- What the design *cannot* answer. This is the most commonly omitted and most
  valuable part of the report.
- After analysis: which effects are real, the size of each in response units,
  and the residual diagnostics verdict.

Round sensibly. A factor setting of 152.3847 C is false precision, and an
effect quoted to six figures invites more confidence than eight runs support.

## References

Load these as needed; they are not required reading for every task.

- `references/choosing-a-design.md` - which design for which situation, run
  counts, and the screening-versus-optimisation distinction.
- `references/verification.md` - what resolution and the other quality
  metrics mean, and how to phrase them.
- `references/analysis-workflow.md` - the analysis sequence, unreplicated
  designs, transformations, and diagnostic traps.
- `references/worked-example.md` - a complete end-to-end session.

## Setup

The scripts need `process-improve`:

```bash
pip install 'process-improve[expt,plotting]'
```

Each script also carries a PEP 723 header, so `uv run --script <script>` will
build the environment on its own with nothing installed up front. If a script
reports a missing import, the message names the extra to install.
